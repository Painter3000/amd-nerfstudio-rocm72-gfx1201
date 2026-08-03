#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shlex
import shutil
import sys
sys.dont_write_bytecode = True
import tempfile
import time
from typing import Any

from manage_public_resources_v1 import (
    PROFILE,
    ask_yes_no,
    prepare_resources,
    resource_paths,
    verify_resource_cache,
    verify_dataset,
    verify_nerfacc_wheel,
    verify_nerfstudio_source,
    verify_tcnn_runtime,
)
from public_toolchain_common import inventory_tree, json_dump, load_json, run_command, sha256, verify_manifest

SCHEMA = "amd-nerfstudio-public-fresh-env-v1"
CLASSIFICATION = "PUBLIC_REFERENCE_BINARY_FRESH_ENV_INSTALL_AND_QUICK_VALIDATION"
WHEELHOUSE_SCHEMA = "amd-nerfstudio-public-wheelhouse-lock-v1"


def resolve_python(value: str) -> Path:
    candidate = Path(value).expanduser()
    if candidate.parent != Path(".") or "/" in value:
        path = Path(os.path.abspath(os.fspath(candidate)))
    else:
        found = shutil.which(value)
        path = Path(found) if found else candidate
    return path


def probe_python(python: Path) -> dict[str, Any]:
    return run_command(
        [str(python), "-c", "import json,sys; print(json.dumps({'version':list(sys.version_info[:3]),'executable':sys.executable,'prefix':sys.prefix}))"],
        timeout=30,
    )


def create_wheelhouse_lock(wheelhouse: Path, requirements: Path, constraints: Path, manifest_path: Path) -> dict[str, Any]:
    files: dict[str, dict[str, Any]] = {}
    for path in sorted(wheelhouse.iterdir() if wheelhouse.is_dir() else []):
        if path.is_file():
            files[path.name] = {"sha256": sha256(path), "size_bytes": path.stat().st_size}
    return {
        "schema": WHEELHOUSE_SCHEMA,
        "created_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "requirements": {"path": requirements.name, "sha256": sha256(requirements)},
        "constraints": {"path": constraints.name, "sha256": sha256(constraints)},
        "resource_manifest_sha256": sha256(manifest_path),
        "file_count": len(files),
        "files": files,
    }


def verify_wheelhouse(
    wheelhouse: Path,
    lock_path: Path,
    requirements: Path,
    constraints: Path,
    manifest_path: Path,
) -> dict[str, Any]:
    if not wheelhouse.is_dir() or not lock_path.is_file():
        return {
            "passed": False,
            "error": "WHEELHOUSE_OR_LOCK_MISSING",
            "wheelhouse": str(wheelhouse),
            "lock": str(lock_path),
        }
    try:
        lock = load_json(lock_path)
    except Exception as exc:
        return {"passed": False, "error": repr(exc), "lock": str(lock_path)}
    mismatches: list[dict[str, Any]] = []
    if lock.get("schema") != WHEELHOUSE_SCHEMA:
        mismatches.append({"kind": "SCHEMA", "observed": lock.get("schema")})
    expected_inputs = {
        "requirements": sha256(requirements),
        "constraints": sha256(constraints),
        "resource_manifest_sha256": sha256(manifest_path),
    }
    if (lock.get("requirements") or {}).get("sha256") != expected_inputs["requirements"]:
        mismatches.append({"kind": "REQUIREMENTS_HASH"})
    if (lock.get("constraints") or {}).get("sha256") != expected_inputs["constraints"]:
        mismatches.append({"kind": "CONSTRAINTS_HASH"})
    if lock.get("resource_manifest_sha256") != expected_inputs["resource_manifest_sha256"]:
        mismatches.append({"kind": "RESOURCE_MANIFEST_HASH"})
    rows = lock.get("files")
    if not isinstance(rows, dict) or not rows:
        mismatches.append({"kind": "LOCK_FILES_EMPTY"})
        rows = {}
    for name, metadata in sorted(rows.items()):
        path = wheelhouse / name
        expected = metadata.get("sha256") if isinstance(metadata, dict) else None
        observed = sha256(path) if path.is_file() else None
        if observed != expected:
            mismatches.append({"kind": "WHEEL_HASH", "file": name, "expected": expected, "observed": observed})
    unlisted = sorted(path.name for path in wheelhouse.iterdir() if path.is_file() and path.name not in rows)
    if unlisted:
        mismatches.append({"kind": "UNLISTED_WHEELHOUSE_FILES", "files": unlisted})
    return {
        "passed": not mismatches,
        "wheelhouse": str(wheelhouse),
        "lock": str(lock_path),
        "file_count": len(rows),
        "mismatches": mismatches,
        "lock_payload": lock,
    }


def prepare_wheelhouse(
    *,
    python: Path,
    wheelhouse: Path,
    lock_path: Path,
    requirements: Path,
    constraints: Path,
    manifest_path: Path,
    manifest: dict[str, Any],
    auto: bool,
    offline: bool,
    replace_invalid: bool,
    timeout: int,
) -> dict[str, Any]:
    existing = verify_wheelhouse(wheelhouse, lock_path, requirements, constraints, manifest_path)
    if existing.get("passed"):
        return {"passed": True, "action": "REUSED_VERIFIED_WHEELHOUSE", "verification": existing}
    if offline:
        return {"passed": False, "action": "OFFLINE_BLOCKED", "verification": existing, "blocker": "OFFLINE_WHEELHOUSE_NOT_READY"}
    if not ask_yes_no(
        "The pinned Python wheelhouse is missing or invalid. Download the scoped package set from PyPI and the official PyTorch ROCm 7.2 index?",
        auto=auto,
    ):
        return {"passed": False, "action": "NETWORK_CONSENT_REQUIRED", "verification": existing, "blocker": "NETWORK_CONSENT_REQUIRED_FOR_WHEELHOUSE"}
    if (wheelhouse.exists() or lock_path.exists()) and not replace_invalid:
        return {"passed": False, "action": "INVALID_CACHE_REQUIRES_REPLACE", "verification": existing, "blocker": "INVALID_WHEELHOUSE_REQUIRES_REPLACE"}

    wheelhouse.parent.mkdir(parents=True, exist_ok=True)
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = wheelhouse.with_name(wheelhouse.name + f".part-{os.getpid()}")
    if temporary.exists():
        shutil.rmtree(temporary)
    temporary.mkdir(parents=True)
    indexes = manifest["public_python_index"]
    download_python = python
    bootstrap_dir: Path | None = None
    pip_probe = run_command([str(python), "-m", "pip", "--version"], timeout=30)
    bootstrap_process: dict[str, Any] | None = None
    if pip_probe.get("returncode") != 0:
        bootstrap_dir = wheelhouse.parent / f".bootstrap-pip-{os.getpid()}"
        shutil.rmtree(bootstrap_dir, ignore_errors=True)
        bootstrap_process = run_command([str(python), "-m", "venv", str(bootstrap_dir)], timeout=300)
        if bootstrap_process.get("returncode") != 0:
            shutil.rmtree(temporary, ignore_errors=True)
            shutil.rmtree(bootstrap_dir, ignore_errors=True)
            return {
                "passed": False,
                "action": "BOOTSTRAP_PIP_ENV_FAILED",
                "pip_probe": pip_probe,
                "bootstrap_process": bootstrap_process,
                "blocker": "PYTHON_PIP_OR_VENV_REQUIRED_FOR_WHEELHOUSE",
            }
        download_python = bootstrap_dir / "bin/python"
    process = run_command(
        [
            str(download_python), "-m", "pip", "download",
            "--disable-pip-version-check",
            "--only-binary=:all:",
            "--dest", str(temporary),
            "--requirement", str(requirements),
            "--constraint", str(constraints),
            "--index-url", indexes["primary"],
            "--extra-index-url", indexes["pytorch_rocm"],
        ],
        timeout=timeout,
    )
    if bootstrap_dir is not None:
        shutil.rmtree(bootstrap_dir, ignore_errors=True)
    if process.get("returncode") != 0:
        shutil.rmtree(temporary, ignore_errors=True)
        return {
            "passed": False,
            "action": "DOWNLOAD_FAILED",
            "pip_probe": pip_probe,
            "bootstrap_process": bootstrap_process,
            "process": process,
            "blocker": "WHEELHOUSE_DOWNLOAD_FAILED",
        }
    if wheelhouse.exists():
        shutil.rmtree(wheelhouse)
    os.replace(temporary, wheelhouse)
    lock = create_wheelhouse_lock(wheelhouse, requirements, constraints, manifest_path)
    temporary_lock = lock_path.with_name(lock_path.name + f".part-{os.getpid()}")
    json_dump(temporary_lock, lock)
    os.replace(temporary_lock, lock_path)
    verification = verify_wheelhouse(wheelhouse, lock_path, requirements, constraints, manifest_path)
    return {
        "passed": bool(verification.get("passed")),
        "action": "DOWNLOADED_AND_SHA256_LOCKED",
        "process": process,
        "verification": verification,
        "blocker": None if verification.get("passed") else "WHEELHOUSE_POST_DOWNLOAD_VERIFICATION_FAILED",
    }


def copy_install_inputs(resource_report: dict[str, Any], install_root: Path, manifest: dict[str, Any]) -> dict[str, Any]:
    source_paths = {name: Path(value) for name, value in resource_report["paths"].items()}
    targets = {
        "nerfstudio_source": install_root / "worktrees/nerfstudio",
        "tcnn_runtime": install_root / "runtime/tiny-rdna4-nn",
        "dataset": install_root / "data/quick-validation",
    }
    for path in targets.values():
        path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(source_paths["nerfstudio_source"], targets["nerfstudio_source"], symlinks=False)
    shutil.copytree(source_paths["tcnn_runtime"], targets["tcnn_runtime"], symlinks=False)
    shutil.copytree(source_paths["dataset"], targets["dataset"], symlinks=False)
    custom = manifest["custom_resources"]
    verification = {
        "nerfstudio_source": verify_nerfstudio_source(targets["nerfstudio_source"], manifest["nerfstudio_source"]),
        "tcnn_runtime": verify_tcnn_runtime(targets["tcnn_runtime"], custom["tiny_rdna4_runtime"]),
        "dataset": verify_dataset(targets["dataset"], custom["quick_validation_dataset"]),
    }
    return {
        "paths": {name: str(path) for name, path in targets.items()},
        "verification": verification,
        "passed": all(row.get("passed") for row in verification.values()),
    }


def create_activation_script(
    path: Path,
    *,
    venv_python: Path,
    nerfstudio: Path,
    runtime: Path,
    dataset: Path,
    output_root: Path,
) -> None:
    venv_dir = venv_python.parent.parent
    text = f'''#!/usr/bin/env bash
# Generated by setup_public_fresh_env_v1.py.
source {shlex.quote(str(venv_dir / "bin/activate"))}
export NERFSTUDIO_RDNA4_PUBLIC_PYTHON={shlex.quote(str(venv_python))}
export NERFSTUDIO_RDNA4_PUBLIC_NERFSTUDIO_WORKTREE={shlex.quote(str(nerfstudio))}
export NERFSTUDIO_RDNA4_PUBLIC_TCNN_RUNTIME={shlex.quote(str(runtime))}
export NERFSTUDIO_RDNA4_PUBLIC_DATASET={shlex.quote(str(dataset))}
export NERFSTUDIO_RDNA4_PUBLIC_OUTPUT_ROOT={shlex.quote(str(output_root))}
export PYTHONNOUSERSITE=1
export PYTHONDONTWRITEBYTECODE=1
export TCNN_RDNA4_ENABLE_PORTABLE_MLP_SHIM=1
export NERFSTUDIO_RDNA4_A5_SINGLE_SH_POLICY=TINY_RDNA4_NN_ONLY
export NERFSTUDIO_RDNA4_A5_TCNN_RUNTIME={shlex.quote(str(runtime))}
export NERFSTUDIO_RDNA4_A5_NERFSTUDIO_WORKTREE={shlex.quote(str(nerfstudio))}
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
unset TORCH_FORCE_WEIGHTS_ONLY_LOAD
export PYTHONPATH={shlex.quote(str(runtime))}:{shlex.quote(str(nerfstudio))}
'''
    path.write_text(text, encoding="utf-8")
    path.chmod(0o755)


def collect_provenance(venv_python: Path, provenance: Path) -> dict[str, Any]:
    provenance.mkdir(parents=True, exist_ok=True)
    commands = {
        "pip-freeze.txt": [str(venv_python), "-m", "pip", "freeze", "--all"],
        "pip-list.txt": [str(venv_python), "-m", "pip", "list", "--format=freeze"],
        "pip-check.txt": [str(venv_python), "-m", "pip", "check"],
        "pip-inspect.json": [str(venv_python), "-m", "pip", "inspect", "--local"],
    }
    reports: dict[str, Any] = {}
    for filename, argv in commands.items():
        result = run_command(argv, timeout=300)
        path = provenance / filename
        path.write_text(result.get("stdout", "") + result.get("stderr", ""), encoding="utf-8")
        reports[filename] = {"path": str(path), "returncode": result.get("returncode"), "sha256": sha256(path)}
    return reports


def install_environment(args: argparse.Namespace, manifest: dict[str, Any], resource_report: dict[str, Any], wheelhouse_report: dict[str, Any]) -> dict[str, Any]:
    repo_root = Path(__file__).resolve().parents[1]
    install_root = args.install_root.expanduser().resolve()
    if install_root.exists():
        if not args.force_recreate:
            return {"passed": False, "blocker": "INSTALL_ROOT_EXISTS_USE_FORCE_RECREATE", "install_root": str(install_root)}
        shutil.rmtree(install_root)
    install_root.mkdir(parents=True)
    venv_dir = install_root / "venv"
    create_venv = run_command([str(args.resolved_python), "-m", "venv", str(venv_dir)], timeout=300)
    if create_venv.get("returncode") != 0:
        return {"passed": False, "blocker": "VENV_CREATION_FAILED", "process": create_venv, "install_root": str(install_root)}
    venv_python = venv_dir / "bin/python"

    wheelhouse = Path(resource_report["paths"]["wheelhouse"])
    requirements = repo_root / "requirements/nerfacto_runtime_v1.txt"
    constraints = repo_root / "constraints/nerfacto_rocm72_py312_v1.txt"
    install_packages = run_command(
        [
            str(venv_python), "-m", "pip", "install",
            "--disable-pip-version-check",
            "--no-index",
            "--find-links", str(wheelhouse),
            "--requirement", str(requirements),
            "--constraint", str(constraints),
        ],
        timeout=args.timeout,
    )
    if install_packages.get("returncode") != 0:
        return {"passed": False, "blocker": "SCOPED_RUNTIME_INSTALL_FAILED", "process": install_packages, "install_root": str(install_root)}

    nerfacc_wheel = Path(resource_report["paths"]["nerfacc_wheel"])
    install_nerfacc = run_command(
        [str(venv_python), "-m", "pip", "install", "--disable-pip-version-check", "--no-deps", str(nerfacc_wheel)],
        timeout=600,
    )
    if install_nerfacc.get("returncode") != 0:
        return {"passed": False, "blocker": "NERFACC_INSTALL_FAILED", "process": install_nerfacc, "install_root": str(install_root)}

    copied = copy_install_inputs(resource_report, install_root, manifest)
    if not copied["passed"]:
        return {"passed": False, "blocker": "COPIED_RESOURCE_VERIFICATION_FAILED", "copied": copied, "install_root": str(install_root)}
    paths = {name: Path(value) for name, value in copied["paths"].items()}
    output_root = install_root / "evidence"
    output_root.mkdir()
    activation = install_root / "activate_rdna4_nerfacto.sh"
    create_activation_script(
        activation,
        venv_python=venv_python,
        nerfstudio=paths["nerfstudio_source"],
        runtime=paths["tcnn_runtime"],
        dataset=paths["dataset"],
        output_root=output_root,
    )

    runtime_env = os.environ.copy()
    runtime_env.update({
        "NERFSTUDIO_RDNA4_PUBLIC_PYTHON": str(venv_python),
        "NERFSTUDIO_RDNA4_PUBLIC_NERFSTUDIO_WORKTREE": str(paths["nerfstudio_source"]),
        "NERFSTUDIO_RDNA4_PUBLIC_TCNN_RUNTIME": str(paths["tcnn_runtime"]),
        "NERFSTUDIO_RDNA4_PUBLIC_DATASET": str(paths["dataset"]),
        "NERFSTUDIO_RDNA4_PUBLIC_OUTPUT_ROOT": str(output_root),
        "PYTHONNOUSERSITE": "1",
        "PYTHONDONTWRITEBYTECODE": "1",
        "TCNN_RDNA4_ENABLE_PORTABLE_MLP_SHIM": "1",
        "NERFSTUDIO_RDNA4_A5_SINGLE_SH_POLICY": "TINY_RDNA4_NN_ONLY",
        "NERFSTUDIO_RDNA4_A5_TCNN_RUNTIME": str(paths["tcnn_runtime"]),
        "NERFSTUDIO_RDNA4_A5_NERFSTUDIO_WORKTREE": str(paths["nerfstudio_source"]),
        "TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD": "1",
        "PYTHONPATH": os.pathsep.join([str(paths["tcnn_runtime"]), str(paths["nerfstudio_source"])]),
    })
    runtime_env.pop("TORCH_FORCE_WEIGHTS_ONLY_LOAD", None)

    probe_code = '''
import json, pathlib, sys
import torch
import nerfacc.csrc as nerfacc_csrc
import tinycudann.modules as tcnn_modules
from tinycudann.modules import _C
sys.path.insert(0, sys.argv[1])
from public_nerfacto_config_v1 import build_public_nerfacto_config
cfg = build_public_nerfacto_config()
print(json.dumps({
  "torch": torch.__version__, "hip": torch.version.hip,
  "cuda_available": bool(torch.cuda.is_available()),
  "gcn_arch": getattr(torch.cuda.get_device_properties(0), "gcnArchName", None) if torch.cuda.is_available() else None,
  "nerfacc_native": str(pathlib.Path(nerfacc_csrc.__file__).resolve()),
  "tcnn_modules": str(pathlib.Path(tcnn_modules.__file__).resolve()),
  "tcnn_native": str(pathlib.Path(_C.__file__).resolve()),
  "config_type": type(cfg).__name__,
}, sort_keys=True))
'''
    probe = run_command([str(venv_python), "-c", probe_code, str(repo_root / "tools")], env=runtime_env, timeout=300)
    probe_payload: dict[str, Any] = {}
    try:
        probe_payload = json.loads(probe.get("stdout", "").splitlines()[-1])
    except Exception:
        pass
    custom = manifest["custom_resources"]
    installed_native = Path(probe_payload.get("nerfacc_native", ""))
    native_hash = sha256(installed_native) if installed_native.is_file() else None
    probe_passed = bool(
        probe.get("returncode") == 0
        and probe_payload.get("torch") == manifest["target"]["torch"]
        and probe_payload.get("hip") == manifest["target"]["hip"]
        and probe_payload.get("cuda_available") is True
        and probe_payload.get("gcn_arch") == manifest["target"]["architecture"]
        and native_hash == custom["nerfacc_wheel"]["installed_native_sha256"]
    )

    quick: dict[str, Any] = {"executed": False, "passed": None}
    if not args.no_validate and probe_passed:
        argv = [str(repo_root / "scripts/run_public_quick_validation_v1.sh")]
        if args.keep_checkpoints:
            argv.append("--keep-checkpoints")
        process = run_command(argv, cwd=repo_root, env=runtime_env, timeout=args.timeout)
        quick_dir = None
        for line in process.get("stdout", "").splitlines():
            if line.startswith("PUBLIC_QUICK_VALIDATION_RUN_DIR="):
                quick_dir = Path(line.split("=", 1)[1])
        manifest_check = verify_manifest(quick_dir) if quick_dir and quick_dir.is_dir() else {"passed": False, "error": "QUICK_RUN_DIR_MISSING"}
        gate = (quick_dir / "final_gate.txt").read_text(encoding="utf-8") if quick_dir and (quick_dir / "final_gate.txt").is_file() else ""
        quick = {
            "executed": True,
            "process": process,
            "run_dir": str(quick_dir) if quick_dir else None,
            "manifest": manifest_check,
            "gate_pass": "PUBLIC_RDNA4_QUICK_VALIDATION: PASS" in gate,
            "p2_not_run": "p2_execution=NOT_RUN" in gate,
            "passed": bool(process.get("returncode") == 0 and manifest_check.get("passed") and "PUBLIC_RDNA4_QUICK_VALIDATION: PASS" in gate and "p2_execution=NOT_RUN" in gate),
        }

    provenance = install_root / "provenance"
    command_reports = collect_provenance(venv_python, provenance)
    pip_check_passed = command_reports["pip-check.txt"]["returncode"] == 0
    validation_ok = args.no_validate or quick.get("passed") is True
    passed = bool(copied["passed"] and probe_passed and pip_check_passed and validation_ok)
    decision = "FRESH_ENV_QUALIFIED" if passed and not args.no_validate else (
        "FRESH_ENV_INSTALLED_VALIDATION_NOT_RUN" if passed else "FRESH_ENV_BLOCKED"
    )
    report = {
        "schema": SCHEMA,
        "classification": CLASSIFICATION,
        "profile": PROFILE,
        "install_root": str(install_root),
        "decision": decision,
        "passed": passed,
        "blockers": [] if passed else [
            name for name, value in {
                "COPIED_RESOURCE_VERIFICATION": copied["passed"],
                "RUNTIME_IMPORT_AND_IDENTITY": probe_passed,
                "PIP_CHECK": pip_check_passed,
                "QUICK_VALIDATION": validation_ok,
            }.items() if not value
        ],
        "venv": {"path": str(venv_dir), "python": str(venv_python)},
        "activation_script": str(activation),
        "resource_report": resource_report,
        "wheelhouse_report": wheelhouse_report,
        "package_install": install_packages,
        "nerfacc_install": install_nerfacc,
        "copied_inputs": copied,
        "runtime_probe": {"process": probe, "payload": probe_payload, "nerfacc_native_sha256": native_hash, "passed": probe_passed},
        "pip_provenance": command_reports,
        "pip_check_passed": pip_check_passed,
        "quick_validation": quick,
        "nonclaims": manifest["nonclaims"],
    }
    json_dump(provenance / "fresh_env_report.json", report)
    gate = create_gate(report)
    (provenance / "final_gate.txt").write_text(gate, encoding="utf-8")
    tree = inventory_tree(provenance, exclude_names={"MANIFEST.json"})
    json_dump(provenance / "MANIFEST.json", {"schema": "amd-nerfstudio-public-fresh-env-manifest-v1", "files": tree["files"]})
    return report


def create_gate(report: dict[str, Any]) -> str:
    quick = report.get("quick_validation", {})
    lines = [
        "AMD_NERFSTUDIO_PUBLIC_FRESH_ENV_V1",
        "",
        f"classification={CLASSIFICATION}",
        f"profile={report.get('profile')}",
        f"decision={report.get('decision')}",
        f"install_root={report.get('install_root')}",
        f"quick_validation={'PASS' if quick.get('passed') else ('NOT_RUN' if not quick.get('executed') else 'FAIL')}",
        "fresh_native_build=NOT_CLAIMED",
        "p2_execution=NOT_RUN",
        "",
        "blockers=" + (",".join(report.get("blockers", [])) if report.get("blockers") else "NONE"),
        "",
        "PUBLIC_RDNA4_FRESH_ENV: PASS" if report.get("passed") else "PUBLIC_RDNA4_FRESH_ENV: FAIL",
    ]
    return "\n".join(lines) + "\n"


def self_test() -> int:
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        wheelhouse = root / "wheelhouse"
        wheelhouse.mkdir()
        wheel = wheelhouse / "fixture.whl"
        wheel.write_bytes(b"fixture")
        req = root / "requirements.txt"
        con = root / "constraints.txt"
        manifest = root / "manifest.json"
        req.write_text("fixture==1\n")
        con.write_text("fixture==1\n")
        manifest.write_text('{"schema":"fixture"}\n')
        lock_path = root / "lock.json"
        json_dump(lock_path, create_wheelhouse_lock(wheelhouse, req, con, manifest))
        clean = verify_wheelhouse(wheelhouse, lock_path, req, con, manifest)
        wheel.write_bytes(b"mutated")
        dirty = verify_wheelhouse(wheelhouse, lock_path, req, con, manifest)
        activation = root / "activate.sh"
        create_activation_script(
            activation,
            venv_python=root / "venv/bin/python",
            nerfstudio=root / "ns",
            runtime=root / "runtime",
            dataset=root / "data",
            output_root=root / "evidence",
        )
        text = activation.read_text()
        checks = {
            "wheelhouse_clean": clean["passed"],
            "wheelhouse_mutation_rejected": not dirty["passed"],
            "activation_has_five_paths": all(name in text for name in [
                "NERFSTUDIO_RDNA4_PUBLIC_PYTHON",
                "NERFSTUDIO_RDNA4_PUBLIC_NERFSTUDIO_WORKTREE",
                "NERFSTUDIO_RDNA4_PUBLIC_TCNN_RUNTIME",
                "NERFSTUDIO_RDNA4_PUBLIC_DATASET",
                "NERFSTUDIO_RDNA4_PUBLIC_OUTPUT_ROOT",
            ]),
        }
    passed = all(checks.values())
    print(json.dumps({"schema": SCHEMA, "passed": passed, "checks": checks}, indent=2, sort_keys=True))
    return 0 if passed else 2


def main() -> int:
    repo_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description="Create a scoped fresh Python environment for the qualified RDNA4 Nerfacto path")
    parser.add_argument("--mode", choices=["run", "self-test"], default="run")
    parser.add_argument("--profile", default=PROFILE)
    parser.add_argument("--manifest", type=Path, default=repo_root / "config/public_fresh_env_resources_v1.json")
    parser.add_argument("--resource-dir", type=Path)
    parser.add_argument("--install-root", type=Path)
    parser.add_argument("--python", default="python3.12")
    parser.add_argument("--nerfacc-wheel", type=Path)
    parser.add_argument("--tcnn-runtime", type=Path)
    parser.add_argument("--dataset", type=Path)
    parser.add_argument("--nerfstudio-source", type=Path)
    parser.add_argument("--auto", action="store_true")
    parser.add_argument("--offline", action="store_true")
    parser.add_argument("--download-only", action="store_true")
    parser.add_argument("--verify-resources", action="store_true")
    parser.add_argument("--replace-invalid", action="store_true")
    parser.add_argument("--force-recreate", action="store_true")
    parser.add_argument("--no-validate", action="store_true")
    parser.add_argument("--keep-checkpoints", action="store_true")
    parser.add_argument("--timeout", type=int, default=7200)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    if args.mode == "self-test":
        return self_test()
    if args.profile != PROFILE:
        print("PUBLIC_FRESH_ENV_NOT_STARTED: v1.3 supports only reference-binary-fresh-env; fresh-native-build is not claimed.", file=sys.stderr)
        return 64
    if args.resource_dir is None:
        parser.error("run mode requires --resource-dir")
    if not args.download_only and not args.verify_resources and args.install_root is None:
        parser.error("installation requires --install-root")
    if args.auto and args.offline:
        parser.error("--auto and --offline are mutually exclusive")

    args.resolved_python = resolve_python(args.python)
    python_probe = probe_python(args.resolved_python)
    try:
        python_payload = json.loads(python_probe.get("stdout", "").splitlines()[-1])
    except Exception:
        python_payload = {}
    if python_probe.get("returncode") != 0 or python_payload.get("version", [])[:2] != [3, 12]:
        print(json.dumps({"passed": False, "blocker": "PYTHON_3_12_REQUIRED", "probe": python_probe}, indent=2), file=sys.stderr)
        return 2

    manifest_path = args.manifest.expanduser().resolve()
    manifest = load_json(manifest_path)
    if args.verify_resources:
        resource_report = verify_resource_cache(args.resource_dir, manifest)
        paths = resource_paths(args.resource_dir.expanduser().resolve(), manifest)
        requirements = repo_root / "requirements/nerfacto_runtime_v1.txt"
        constraints = repo_root / "constraints/nerfacto_rocm72_py312_v1.txt"
        wheelhouse_verification = verify_wheelhouse(
            paths["wheelhouse"], paths["wheelhouse_lock"], requirements, constraints, manifest_path
        )
        result = {
            "schema": SCHEMA,
            "profile": PROFILE,
            "passed": bool(resource_report["passed"] and wheelhouse_verification.get("passed")),
            "decision": "RESOURCES_AND_WHEELHOUSE_VERIFIED" if resource_report["passed"] and wheelhouse_verification.get("passed") else "RESOURCE_VERIFICATION_BLOCKED",
            "resource_report": resource_report,
            "wheelhouse_report": wheelhouse_verification,
            "installation_executed": False,
        }
        if args.report:
            json_dump(args.report.expanduser().resolve(), result)
        print(json.dumps(result, indent=2, sort_keys=True))
        print(f"PUBLIC_RDNA4_FRESH_ENV_RESOURCES: {'PASS' if result['passed'] else 'FAIL'}")
        return 0 if result["passed"] else 2

    resource_report = prepare_resources(args, manifest)
    if not resource_report["passed"]:
        if args.report:
            json_dump(args.report.expanduser().resolve(), {"schema": SCHEMA, "passed": False, "stage": "resources", "resource_report": resource_report})
        print(json.dumps(resource_report, indent=2, sort_keys=True))
        return 2

    paths = resource_paths(args.resource_dir.expanduser().resolve(), manifest)
    requirements = repo_root / "requirements/nerfacto_runtime_v1.txt"
    constraints = repo_root / "constraints/nerfacto_rocm72_py312_v1.txt"
    wheelhouse_report = prepare_wheelhouse(
        python=args.resolved_python,
        wheelhouse=paths["wheelhouse"],
        lock_path=paths["wheelhouse_lock"],
        requirements=requirements,
        constraints=constraints,
        manifest_path=manifest_path,
        manifest=manifest,
        auto=args.auto,
        offline=args.offline,
        replace_invalid=args.replace_invalid,
        timeout=args.timeout,
    )
    if not wheelhouse_report["passed"]:
        result = {"schema": SCHEMA, "passed": False, "stage": "wheelhouse", "resource_report": resource_report, "wheelhouse_report": wheelhouse_report}
        if args.report:
            json_dump(args.report.expanduser().resolve(), result)
        print(json.dumps(result, indent=2, sort_keys=True))
        return 2

    if args.download_only:
        result = {
            "schema": SCHEMA,
            "profile": PROFILE,
            "passed": True,
            "decision": "RESOURCES_AND_WHEELHOUSE_READY",
            "resource_report": resource_report,
            "wheelhouse_report": wheelhouse_report,
            "installation_executed": False,
        }
        if args.report:
            json_dump(args.report.expanduser().resolve(), result)
        print(json.dumps(result, indent=2, sort_keys=True))
        print("PUBLIC_RDNA4_FRESH_ENV_RESOURCES: PASS")
        return 0

    report = install_environment(args, manifest, resource_report, wheelhouse_report)
    if args.report:
        json_dump(args.report.expanduser().resolve(), report)
    print(json.dumps(report, indent=2, sort_keys=True))
    if "decision" in report:
        print(create_gate(report), end="")
    return 0 if report.get("passed") else 2


if __name__ == "__main__":
    raise SystemExit(main())
