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
    resource_paths,
    verify_dataset,
    verify_nerfstudio_source,
    verify_tcnn_runtime,
)
from public_toolchain_common import (
    absolute_preserving_symlink,
    build_runtime_env,
    inventory_tree,
    json_dump,
    load_json,
    run_command,
    sha256,
    verify_manifest,
)
from setup_public_fresh_env_v1 import resolve_python

SCHEMA = "amd-nerfstudio-public-adaptive-env-v1"
CLASSIFICATION = "PUBLIC_ADAPTIVE_EXISTING_OR_FRESH_ENV_INSTALL_AND_QUICK_VALIDATION"
OWNED_MARKER = ".amd-nerfstudio-adaptive-owned-v1.json"


def python_from_env_root(env_root: Path) -> Path:
    """Return the explicit interpreter launcher for one user-selected env.

    The path is intentionally not resolved through symlinks: venv prefix
    discovery depends on executing the launcher inside ENV_ROOT/bin.
    """
    root = env_root.expanduser().absolute()
    return root / "bin/python"


def environment_kind(payload: dict[str, Any]) -> str:
    if payload.get("conda_prefix"):
        return "conda"
    prefix = payload.get("prefix")
    base_prefix = payload.get("base_prefix")
    if prefix and base_prefix and prefix != base_prefix:
        return "venv"
    return "system"


def probe_python(python: Path) -> dict[str, Any]:
    code = r'''
import json, os, sys
from importlib import metadata

def version(name):
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return None

payload = {
    "version": list(sys.version_info[:3]),
    "executable": sys.executable,
    "prefix": sys.prefix,
    "base_prefix": getattr(sys, "base_prefix", sys.prefix),
    "conda_prefix": os.environ.get("CONDA_PREFIX"),
    "virtual_env": os.environ.get("VIRTUAL_ENV"),
    "packages": {
        name: version(name)
        for name in (
            "torch", "torchvision", "nerfacc", "viser",
            "opencv-python", "opencv-python-headless",
            "pyliblzfse", "yourdfpy",
        )
    },
}
print(json.dumps(payload, sort_keys=True))
'''
    process = run_command([str(python), "-c", code], timeout=60)
    payload: dict[str, Any] = {}
    try:
        payload = json.loads(process.get("stdout", "").splitlines()[-1])
    except Exception:
        pass
    payload["kind"] = environment_kind(payload) if payload else "unavailable"
    return {"process": process, "payload": payload, "passed": process.get("returncode") == 0 and bool(payload)}


def resolve_candidate_paths(args: argparse.Namespace, manifest: dict[str, Any]) -> dict[str, Any]:
    cache: dict[str, Path] = {}
    if args.resource_dir is not None:
        cache = resource_paths(args.resource_dir.expanduser().resolve(), manifest)

    mapping = {
        "nerfstudio_worktree": (
            args.nerfstudio_worktree,
            os.environ.get("NERFSTUDIO_RDNA4_PUBLIC_NERFSTUDIO_WORKTREE"),
            cache.get("nerfstudio_source"),
        ),
        "tcnn_runtime": (
            args.tcnn_runtime,
            os.environ.get("NERFSTUDIO_RDNA4_PUBLIC_TCNN_RUNTIME"),
            cache.get("tcnn_runtime"),
        ),
        "dataset": (
            args.data,
            os.environ.get("NERFSTUDIO_RDNA4_PUBLIC_DATASET"),
            cache.get("dataset"),
        ),
    }
    result: dict[str, Any] = {}
    for name, choices in mapping.items():
        selected: Path | None = None
        source: str | None = None
        labels = ["explicit", "environment", "resource-cache"]
        for label, value in zip(labels, choices):
            if value is None:
                continue
            selected = Path(value).expanduser().resolve()
            source = label
            break
        result[name] = {"path": str(selected) if selected else None, "source": source}
    return result


def verify_candidate_inputs(paths: dict[str, Any], manifest: dict[str, Any]) -> dict[str, Any]:
    custom = manifest["custom_resources"]
    ns_value = paths["nerfstudio_worktree"]["path"]
    rt_value = paths["tcnn_runtime"]["path"]
    data_value = paths["dataset"]["path"]
    ns = Path(ns_value) if ns_value else Path("/__missing_nerfstudio__")
    rt = Path(rt_value) if rt_value else Path("/__missing_tcnn__")
    data = Path(data_value) if data_value else Path("/__missing_dataset__")
    checks = {
        "nerfstudio": verify_nerfstudio_source(ns, manifest["nerfstudio_source"]),
        "tcnn_runtime": verify_tcnn_runtime(rt, custom["tiny_rdna4_runtime"]),
        "dataset": verify_dataset(data, custom["quick_validation_dataset"]),
    }
    return {"passed": all(row.get("passed") for row in checks.values()), "checks": checks}


def probe_runtime(
    python: Path,
    paths: dict[str, Any],
    manifest: dict[str, Any],
    repo_root: Path,
) -> dict[str, Any]:
    if not all(paths[name]["path"] for name in ("nerfstudio_worktree", "tcnn_runtime", "dataset")):
        return {"passed": False, "error": "EXPLICIT_RUNTIME_PATHS_INCOMPLETE"}
    runtime = Path(paths["tcnn_runtime"]["path"])
    nerfstudio = Path(paths["nerfstudio_worktree"]["path"])
    env = build_runtime_env(runtime, nerfstudio)
    code = r'''
import json, pathlib, sys
from importlib import metadata
import torch
import nerfacc.csrc as nerfacc_csrc
import tinycudann.modules as tcnn_modules
from tinycudann.modules import _C
sys.path.insert(0, sys.argv[1])
from public_nerfacto_config_v1 import build_public_nerfacto_config

def version(name):
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return None

cfg = build_public_nerfacto_config()
print(json.dumps({
    "torch": torch.__version__,
    "hip": torch.version.hip,
    "cuda_available": bool(torch.cuda.is_available()),
    "gcn_arch": getattr(torch.cuda.get_device_properties(0), "gcnArchName", None) if torch.cuda.is_available() else None,
    "nerfacc_native": str(pathlib.Path(nerfacc_csrc.__file__).resolve()),
    "tcnn_modules": str(pathlib.Path(tcnn_modules.__file__).resolve()),
    "tcnn_native": str(pathlib.Path(_C.__file__).resolve()),
    "config_vis": cfg.vis,
    "viser": version("viser"),
    "pyliblzfse": version("pyliblzfse"),
    "yourdfpy": version("yourdfpy"),
}, sort_keys=True))
'''
    process = run_command([str(python), "-c", code, str(repo_root / "tools")], env=env, timeout=300)
    payload: dict[str, Any] = {}
    try:
        payload = json.loads(process.get("stdout", "").splitlines()[-1])
    except Exception:
        pass
    custom = manifest["custom_resources"]
    nerfacc_native = Path(payload.get("nerfacc_native", ""))
    nerfacc_hash = sha256(nerfacc_native) if nerfacc_native.is_file() else None
    expected_viser = manifest["target"].get("viser_math", "1.0.0")
    expected_runtime = Path(paths["tcnn_runtime"]["path"]).resolve()
    observed_tcnn = Path(payload.get("tcnn_modules", "/__missing__"))
    tcnn_origin_ok = observed_tcnn.is_file() and expected_runtime in observed_tcnn.resolve().parents
    checks = {
        "process": process.get("returncode") == 0,
        "torch": payload.get("torch") == manifest["target"]["torch"],
        "hip": payload.get("hip") == manifest["target"]["hip"],
        "cuda_available": payload.get("cuda_available") is True,
        "gcn_arch": payload.get("gcn_arch") == manifest["target"]["architecture"],
        "nerfacc_native": nerfacc_hash == custom["nerfacc_wheel"]["installed_native_sha256"],
        "tcnn_origin": tcnn_origin_ok,
        "config_vis": payload.get("config_vis") == "tensorboard",
        "viser_math": payload.get("viser") == expected_viser,
        "viewer_extras_absent": payload.get("pyliblzfse") is None and payload.get("yourdfpy") is None,
    }
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "process": process,
        "payload": payload,
        "nerfacc_native_sha256": nerfacc_hash,
    }


def collect_pip_state(python: Path, directory: Path, label: str) -> dict[str, Any]:
    directory.mkdir(parents=True, exist_ok=True)
    commands = {
        "freeze": [str(python), "-m", "pip", "freeze", "--all"],
        "check": [str(python), "-m", "pip", "check"],
    }
    rows: dict[str, Any] = {}
    for name, argv in commands.items():
        result = run_command(argv, timeout=300)
        path = directory / f"pip-{label}-{name}.txt"
        path.write_text(result.get("stdout", "") + result.get("stderr", ""), encoding="utf-8")
        rows[name] = {
            "path": str(path),
            "sha256": sha256(path),
            "returncode": result.get("returncode"),
        }
    rows["pip_check_advisory"] = True
    return rows


def choose_action(args: argparse.Namespace, compatible: bool) -> dict[str, Any]:
    if args.env_policy == "new":
        if args.install_root is None:
            return {"action": "BLOCKED", "reason": "INSTALL_ROOT_REQUIRED_FOR_NEW_ENV"}
        if getattr(args, "resource_dir", None) is None:
            return {"action": "BLOCKED", "reason": "RESOURCE_DIR_REQUIRED_FOR_NEW_ENV"}
        return {"action": "CREATE_NEW_ENV", "reason": "ENV_POLICY_NEW"}
    if compatible:
        return {"action": "REUSE_EXISTING_ENV", "reason": "CANDIDATE_COMPATIBLE"}
    if args.env_policy in {"current", "reuse"}:
        return {"action": "BLOCKED", "reason": "REQUESTED_ENV_NOT_COMPATIBLE"}
    if args.no_build:
        return {"action": "BLOCKED", "reason": "NO_BUILD_AND_CANDIDATE_NOT_COMPATIBLE"}
    if args.install_root is None:
        return {"action": "BLOCKED", "reason": "INSTALL_ROOT_REQUIRED_FOR_NEW_ENV"}
    if getattr(args, "resource_dir", None) is None:
        return {"action": "BLOCKED", "reason": "RESOURCE_DIR_REQUIRED_FOR_NEW_ENV"}
    if args.repair:
        return {"action": "CREATE_NEW_ENV", "reason": "SAFE_REPAIR_BY_ISOLATED_REPLACEMENT"}
    return {"action": "CREATE_NEW_ENV", "reason": "AUTO_FALLBACK_TO_ISOLATED_ENV"}


def create_activation_script(path: Path, python: Path, paths: dict[str, Any], output_root: Path) -> None:
    runtime = paths["tcnn_runtime"]["path"]
    nerfstudio = paths["nerfstudio_worktree"]["path"]
    dataset = paths["dataset"]["path"]
    lines = [
        "#!/usr/bin/env bash",
        "# Generated by setup_public_adaptive_env_v1.py.",
    ]
    activate = python.parent / "activate"
    if activate.is_file():
        lines.append(f"source {shlex.quote(str(activate))}")
    lines.extend([
        f"export NERFSTUDIO_RDNA4_PUBLIC_PYTHON={shlex.quote(str(python))}",
        f"export NERFSTUDIO_RDNA4_PUBLIC_NERFSTUDIO_WORKTREE={shlex.quote(str(nerfstudio))}",
        f"export NERFSTUDIO_RDNA4_PUBLIC_TCNN_RUNTIME={shlex.quote(str(runtime))}",
        f"export NERFSTUDIO_RDNA4_PUBLIC_DATASET={shlex.quote(str(dataset))}",
        f"export NERFSTUDIO_RDNA4_PUBLIC_OUTPUT_ROOT={shlex.quote(str(output_root))}",
        "export PYTHONNOUSERSITE=1",
        "export PYTHONDONTWRITEBYTECODE=1",
        "export TCNN_RDNA4_ENABLE_PORTABLE_MLP_SHIM=1",
        "export NERFSTUDIO_RDNA4_A5_SINGLE_SH_POLICY=TINY_RDNA4_NN_ONLY",
        f"export NERFSTUDIO_RDNA4_A5_TCNN_RUNTIME={shlex.quote(str(runtime))}",
        f"export NERFSTUDIO_RDNA4_A5_NERFSTUDIO_WORKTREE={shlex.quote(str(nerfstudio))}",
        "export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1",
        "unset TORCH_FORCE_WEIGHTS_ONLY_LOAD",
        f"export PYTHONPATH={shlex.quote(str(runtime))}:{shlex.quote(str(nerfstudio))}",
        "",
    ])
    path.write_text("\n".join(lines), encoding="utf-8")
    path.chmod(0o755)


def run_p0(python: Path, paths: dict[str, Any], output_root: Path, repo_root: Path, args: argparse.Namespace) -> dict[str, Any]:
    run_id = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime()) + f"_{os.getpid()}_adaptive_p0"
    cmd = [
        str(python), str(repo_root / "tools/run_public_a5p0_preflight_v1.py"),
        "--python", str(python),
        "--nerfstudio-worktree", paths["nerfstudio_worktree"]["path"],
        "--tcnn-runtime", paths["tcnn_runtime"]["path"],
        "--dataset", paths["dataset"]["path"],
        "--output-root", str(output_root),
        "--reference", str(args.reference.expanduser().resolve()),
        "--run-id", run_id,
    ]
    process = run_command(cmd, cwd=Path("/tmp"), timeout=args.timeout)
    run_dir = output_root / "public_a5p0_preflight_v1" / run_id
    manifest = verify_manifest(run_dir) if run_dir.is_dir() else {"passed": False, "error": "P0_RUN_DIR_MISSING"}
    gate = (run_dir / "final_gate.txt").read_text(encoding="utf-8") if (run_dir / "final_gate.txt").is_file() else ""
    return {
        "executed": True,
        "kind": "P0_ONLY",
        "process": process,
        "run_dir": str(run_dir),
        "manifest": manifest,
        "passed": process.get("returncode") == 0 and manifest.get("passed") and "PUBLIC_RDNA4_A5_P0_PREFLIGHT: PASS" in gate,
    }


def run_quick(python: Path, paths: dict[str, Any], output_root: Path, repo_root: Path, args: argparse.Namespace) -> dict[str, Any]:
    cmd = [
        str(python), str(repo_root / "tools/run_public_quick_validation_v1.py"),
        "--python", str(python),
        "--nerfstudio-worktree", paths["nerfstudio_worktree"]["path"],
        "--tcnn-runtime", paths["tcnn_runtime"]["path"],
        "--data", paths["dataset"]["path"],
        "--output-root", str(output_root),
        "--reference", str(args.reference.expanduser().resolve()),
        "--timeout", str(args.timeout),
    ]
    if args.keep_checkpoints:
        cmd.append("--keep-checkpoints")
    process = run_command(cmd, cwd=Path("/tmp"), timeout=args.timeout * 4)
    run_dir: Path | None = None
    for line in process.get("stdout", "").splitlines():
        if line.startswith("PUBLIC_QUICK_VALIDATION_RUN_DIR="):
            run_dir = Path(line.split("=", 1)[1])
    manifest = verify_manifest(run_dir) if run_dir and run_dir.is_dir() else {"passed": False, "error": "QUICK_RUN_DIR_MISSING"}
    gate = (run_dir / "final_gate.txt").read_text(encoding="utf-8") if run_dir and (run_dir / "final_gate.txt").is_file() else ""
    return {
        "executed": True,
        "kind": "P0_PLUS_P1_QUICK",
        "process": process,
        "run_dir": str(run_dir) if run_dir else None,
        "manifest": manifest,
        "passed": bool(
            process.get("returncode") == 0
            and manifest.get("passed")
            and "PUBLIC_RDNA4_QUICK_VALIDATION: PASS" in gate
            and "p2_execution=NOT_RUN" in gate
        ),
    }


def run_unit_tests(python: Path, repo_root: Path, timeout: int) -> dict[str, Any]:
    process = run_command(
        [str(python), "-m", "unittest", "discover", "-s", "tests", "-p", "test_*.py", "-v"],
        cwd=repo_root,
        timeout=timeout,
    )
    return {"executed": True, "process": process, "passed": process.get("returncode") == 0}


def fresh_command(args: argparse.Namespace, repo_root: Path, report_path: Path) -> list[str]:
    bootstrap = resolve_python(args.bootstrap_python)
    cmd = [
        str(bootstrap), str(repo_root / "tools/setup_public_fresh_env_v1.py"),
        "--resource-dir", str(args.resource_dir.expanduser().resolve()),
        "--install-root", str(args.install_root.expanduser().resolve()),
        "--python", str(bootstrap),
        "--timeout", str(args.timeout),
        "--report", str(report_path),
    ]
    for flag, value in (
        ("--nerfacc-wheel", args.nerfacc_wheel),
        ("--tcnn-runtime", args.tcnn_runtime),
        ("--dataset", args.data),
        ("--nerfstudio-source", args.nerfstudio_worktree),
    ):
        if value is not None:
            cmd.extend([flag, str(Path(value).expanduser().resolve())])
    if args.auto:
        cmd.append("--auto")
    if args.offline:
        cmd.append("--offline")
    if args.replace_invalid:
        cmd.append("--replace-invalid")
    if args.force_recreate or args.repair:
        cmd.append("--force-recreate")
    if args.validation in {"none", "verify"}:
        cmd.append("--no-validate")
    if args.keep_checkpoints:
        cmd.append("--keep-checkpoints")
    return cmd


def installed_paths(install_root: Path) -> dict[str, Any]:
    return {
        "nerfstudio_worktree": {"path": str(install_root / "worktrees/nerfstudio"), "source": "fresh-install"},
        "tcnn_runtime": {"path": str(install_root / "runtime/tiny-rdna4-nn"), "source": "fresh-install"},
        "dataset": {"path": str(install_root / "data/quick-validation"), "source": "fresh-install"},
    }


def create_gate(report: dict[str, Any]) -> str:
    checks = report.get("checks", {})
    lines = [
        "AMD_NERFSTUDIO_PUBLIC_ADAPTIVE_ENV_V1",
        "",
        f"classification={CLASSIFICATION}",
        f"decision={report.get('decision')}",
        f"action={report.get('action')}",
        f"environment_kind={report.get('environment_kind')}",
        f"environment_mutated={str(bool(report.get('environment_mutated'))).upper()}",
        f"validation={report.get('validation')}",
        f"pip_check_policy=ADVISORY",
        "p2_execution=NOT_RUN",
        "",
        f"PUBLIC_RDNA4_ADAPTIVE_PLAN: {'PASS' if checks.get('PLAN') else 'FAIL'}",
        f"PUBLIC_RDNA4_ADAPTIVE_RUNTIME: {'PASS' if checks.get('RUNTIME') else 'FAIL'}",
        f"PUBLIC_RDNA4_ADAPTIVE_VALIDATION: {'PASS' if checks.get('VALIDATION') else 'FAIL'}",
        f"PUBLIC_RDNA4_ADAPTIVE_PROVENANCE: {'PASS' if checks.get('PROVENANCE') else 'FAIL'}",
        "",
        "blockers=" + (",".join(report.get("blockers", [])) if report.get("blockers") else "NONE"),
        "",
        "PUBLIC_RDNA4_ADAPTIVE_ENV: PASS" if report.get("passed") else "PUBLIC_RDNA4_ADAPTIVE_ENV: FAIL",
    ]
    return "\n".join(lines) + "\n"


def cleanup_owned(root: Path) -> dict[str, Any]:
    removed: list[str] = []
    skipped: list[str] = []
    if not root.is_dir():
        return {"passed": True, "root": str(root), "removed": removed, "skipped": skipped}
    for marker in sorted(root.rglob(OWNED_MARKER)):
        run_dir = marker.parent
        work = run_dir / "work"
        if work.is_dir():
            shutil.rmtree(work)
            removed.append(str(work))
        else:
            skipped.append(str(run_dir))
    return {"passed": True, "root": str(root), "removed": removed, "skipped": skipped}


def self_test() -> int:
    fixtures = {
        "venv": {"prefix": "/x/venv", "base_prefix": "/usr", "conda_prefix": None},
        "conda": {"prefix": "/x/conda", "base_prefix": "/usr", "conda_prefix": "/x/conda"},
        "system": {"prefix": "/usr", "base_prefix": "/usr", "conda_prefix": None},
    }
    checks = {
        "detect_venv": environment_kind(fixtures["venv"]) == "venv",
        "detect_conda": environment_kind(fixtures["conda"]) == "conda",
        "detect_system": environment_kind(fixtures["system"]) == "system",
        "explicit_env_maps_to_bin_python": python_from_env_root(Path("/x/selected-env")) == Path("/x/selected-env/bin/python"),
    }
    ns = argparse.Namespace(env_policy="auto", no_build=False, install_root=Path("/new"), resource_dir=Path("/cache"), repair=False)
    checks["compatible_reused"] = choose_action(ns, True)["action"] == "REUSE_EXISTING_ENV"
    checks["incompatible_auto_new"] = choose_action(ns, False)["action"] == "CREATE_NEW_ENV"
    ns.no_build = True
    checks["no_build_blocks"] = choose_action(ns, False)["action"] == "BLOCKED"
    ns.env_policy = "reuse"
    checks["reuse_incompatible_blocks"] = choose_action(ns, False)["action"] == "BLOCKED"
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        run = root / "run"
        work = run / "work"
        work.mkdir(parents=True)
        json_dump(run / OWNED_MARKER, {"schema": SCHEMA})
        result = cleanup_owned(root)
        checks["cleanup_owned_work_only"] = result["passed"] and not work.exists() and run.exists()
    passed = all(checks.values())
    print(json.dumps({"schema": SCHEMA, "passed": passed, "checks": checks}, indent=2, sort_keys=True))
    return 0 if passed else 2


def main() -> int:
    repo_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description="Adaptively reuse a compatible environment or create a new isolated RDNA4 Nerfacto environment")
    parser.add_argument("--mode", choices=["run", "plan", "self-test", "cleanup-only"], default="run")
    parser.add_argument("--env-policy", choices=["auto", "current", "reuse", "new"], default="auto")
    candidate = parser.add_mutually_exclusive_group()
    candidate.add_argument(
        "--env", type=Path,
        help="Exact user-selected environment root. Existing environments use ENV/bin/python; no disk search is performed.",
    )
    candidate.add_argument(
        "--python",
        help="Advanced alternative: exact candidate Python launcher. No implicit system-Python fallback is used.",
    )
    parser.add_argument("--bootstrap-python", default="python3.12", help="Python 3.12 used only for a new isolated environment")
    parser.add_argument("--resource-dir", type=Path)
    parser.add_argument("--install-root", type=Path)
    parser.add_argument("--output-root", type=Path, default=repo_root / "evidence/adaptive-env-v1")
    parser.add_argument("--manifest", type=Path, default=repo_root / "config/public_fresh_env_resources_v1.json")
    parser.add_argument("--reference", type=Path, default=repo_root / "config/reference_gfx1201_rocm72.json")
    parser.add_argument("--nerfacc-wheel", type=Path)
    parser.add_argument("--tcnn-runtime", type=Path)
    parser.add_argument("--data", type=Path)
    parser.add_argument("--nerfstudio-worktree", type=Path)
    parser.add_argument("--repair", action="store_true", help="Repair safely by creating a new isolated environment; never mutate the candidate")
    parser.add_argument("--no-build", action="store_true", help="Never create a new environment; reuse or fail closed")
    parser.add_argument("--validation", choices=["quick", "verify", "none", "full"], default="quick")
    validation = parser.add_mutually_exclusive_group()
    validation.add_argument("--quick", action="store_true")
    validation.add_argument("--verify", action="store_true")
    validation.add_argument("--no-test", action="store_true")
    validation.add_argument("--full-test", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--auto", action="store_true", help="Approve pinned public network/cache operations for a new environment")
    parser.add_argument("--offline", action="store_true")
    parser.add_argument("--replace-invalid", action="store_true")
    parser.add_argument("--force-recreate", action="store_true")
    parser.add_argument("--keep-checkpoints", action="store_true")
    parser.add_argument("--keep-work", action="store_true")
    parser.add_argument("--no-cleanup", action="store_true")
    parser.add_argument("--keep-built-wheels", action="store_true", help="Compatibility flag; reference-binary v1 performs no native wheel build")
    parser.add_argument("--timeout", type=int, default=1800)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()

    if args.mode == "self-test":
        return self_test()
    if args.mode in {"run", "plan"} and args.env is None and args.python is None:
        parser.error("one of --env or --python is required; the installer never searches the disk or silently falls back to system Python")
    if args.dry_run:
        args.mode = "plan"
    if args.quick:
        args.validation = "quick"
    elif args.verify:
        args.validation = "verify"
    elif args.no_test:
        args.validation = "none"
    elif args.full_test:
        args.validation = "full"
    if args.auto and args.offline:
        parser.error("--auto and --offline are mutually exclusive")
    if args.mode == "cleanup-only":
        result = cleanup_owned(args.output_root.expanduser().resolve())
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0

    manifest = load_json(args.manifest.expanduser().resolve())
    if manifest.get("profile") != PROFILE:
        parser.error(f"unsupported manifest profile: {manifest.get('profile')}")
    output_root = args.output_root.expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    run_id = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime()) + f"_{os.getpid()}"
    run_dir = output_root / "adaptive_env_v1" / run_id
    run_dir.mkdir(parents=True)
    json_dump(run_dir / OWNED_MARKER, {"schema": SCHEMA, "run_id": run_id})
    work = run_dir / "work"
    work.mkdir()

    selected_env_root = args.env.expanduser().absolute() if args.env is not None else None
    candidate_python = (
        python_from_env_root(selected_env_root)
        if selected_env_root is not None
        else resolve_python(args.python)
    )
    if candidate_python.is_file():
        python_report = probe_python(candidate_python)
    else:
        python_report = {
            "passed": False,
            "process": {"returncode": 127, "stdout": "", "stderr": f"candidate Python missing: {candidate_python}"},
            "payload": {},
            "error": "CANDIDATE_ENV_OR_PYTHON_MISSING",
        }
    paths = resolve_candidate_paths(args, manifest)
    inputs = verify_candidate_inputs(paths, manifest)
    runtime = probe_runtime(candidate_python, paths, manifest, repo_root) if python_report.get("passed") and inputs.get("passed") else {"passed": False, "skipped": True}
    python_version_ok = (python_report.get("payload") or {}).get("version", [])[:2] == [3, 12]
    compatible = bool(python_version_ok and inputs.get("passed") and runtime.get("passed"))
    plan = choose_action(args, compatible)

    before = collect_pip_state(candidate_python, run_dir / "provenance", "before") if python_report.get("passed") else {"skipped": True}
    report: dict[str, Any] = {
        "schema": SCHEMA,
        "classification": CLASSIFICATION,
        "run_id": run_id,
        "mode": args.mode,
        "validation": args.validation,
        "profile": PROFILE,
        "candidate_python": str(candidate_python),
        "selected_env_root": str(selected_env_root) if selected_env_root is not None else None,
        "candidate_selection": "EXPLICIT_ENV_ROOT" if selected_env_root is not None else "EXPLICIT_PYTHON",
        "disk_environment_search": False,
        "implicit_system_python_fallback": False,
        "environment_kind": (python_report.get("payload") or {}).get("kind"),
        "environment_mutated": False,
        "python_probe": python_report,
        "candidate_paths": paths,
        "input_verification": inputs,
        "runtime_probe": runtime,
        "candidate_compatible": compatible,
        "plan": plan,
        "action": plan["action"],
        "pip_before": before,
        "pip_check_policy": "ADVISORY_NOT_COMPATIBILITY_GATE",
        "keep_built_wheels_requested": bool(args.keep_built_wheels),
        "nonclaims": manifest.get("nonclaims", []) + [
            "IN_PLACE_MUTATION_OF_EXISTING_SYSTEM_OR_SHARED_ENVIRONMENTS",
            "AUTOMATIC_P2_EXECUTION",
            "FRESH_NATIVE_CUSTOM_RUNTIME_BUILD",
        ],
    }

    if args.mode == "plan":
        passed = plan["action"] != "BLOCKED"
        report.update({
            "passed": passed,
            "decision": "ADAPTIVE_INSTALL_PLAN_READY" if passed else "ADAPTIVE_INSTALL_PLAN_BLOCKED",
            "blockers": [] if passed else [plan["reason"]],
            "checks": {"PLAN": passed, "RUNTIME": compatible or plan["action"] == "CREATE_NEW_ENV", "VALIDATION": True, "PROVENANCE": bool(before)},
        })
    elif plan["action"] == "BLOCKED":
        report.update({
            "passed": False,
            "decision": "ADAPTIVE_ENV_BLOCKED",
            "blockers": [plan["reason"]],
            "checks": {"PLAN": False, "RUNTIME": False, "VALIDATION": False, "PROVENANCE": bool(before)},
        })
    elif plan["action"] == "REUSE_EXISTING_ENV":
        activation = run_dir / "activate_adaptive_env.sh"
        create_activation_script(activation, candidate_python, paths, output_root)
        if args.validation == "none":
            validation_report = {"executed": False, "passed": True, "kind": "NOT_RUN"}
        elif args.validation == "verify":
            validation_report = run_p0(candidate_python, paths, output_root, repo_root, args)
        else:
            validation_report = run_quick(candidate_python, paths, output_root, repo_root, args)
            if args.validation == "full":
                validation_report["unit_tests"] = run_unit_tests(candidate_python, repo_root, args.timeout)
                validation_report["passed"] = bool(validation_report["passed"] and validation_report["unit_tests"]["passed"])
        after = collect_pip_state(candidate_python, run_dir / "provenance", "after")
        freeze_unchanged = before.get("freeze", {}).get("sha256") == after.get("freeze", {}).get("sha256")
        passed = bool(runtime.get("passed") and validation_report.get("passed") and freeze_unchanged)
        report.update({
            "activation_script": str(activation),
            "validation_result": validation_report,
            "pip_after": after,
            "pip_freeze_unchanged": freeze_unchanged,
            "passed": passed,
            "decision": "EXISTING_ENV_REUSED_AND_QUALIFIED" if passed else "EXISTING_ENV_REUSE_BLOCKED",
            "blockers": [] if passed else [name for name, value in {
                "RUNTIME": runtime.get("passed"),
                "VALIDATION": validation_report.get("passed"),
                "UNEXPECTED_ENV_MUTATION": freeze_unchanged,
            }.items() if not value],
            "checks": {"PLAN": True, "RUNTIME": bool(runtime.get("passed")), "VALIDATION": bool(validation_report.get("passed")), "PROVENANCE": freeze_unchanged},
        })
    else:
        if args.resource_dir is None or args.install_root is None:
            report.update({
                "passed": False,
                "decision": "NEW_ENV_INPUTS_INCOMPLETE",
                "blockers": [name for name, value in {"RESOURCE_DIR": args.resource_dir, "INSTALL_ROOT": args.install_root}.items() if value is None],
                "checks": {"PLAN": False, "RUNTIME": False, "VALIDATION": False, "PROVENANCE": bool(before)},
            })
        else:
            fresh_report_path = work / "fresh_env_report.json"
            command = fresh_command(args, repo_root, fresh_report_path)
            process = run_command(command, cwd=repo_root, timeout=args.timeout * 6)
            fresh_report = load_json(fresh_report_path) if fresh_report_path.is_file() else {}
            new_python = args.install_root.expanduser().resolve() / "venv/bin/python"
            new_paths = installed_paths(args.install_root.expanduser().resolve())
            validation_report: dict[str, Any]
            if args.validation == "none":
                validation_report = {"executed": False, "passed": True, "kind": "NOT_RUN"}
            elif args.validation == "verify":
                validation_report = run_p0(new_python, new_paths, output_root, repo_root, args) if new_python.is_file() else {"passed": False, "error": "NEW_PYTHON_MISSING"}
            else:
                validation_report = fresh_report.get("quick_validation", {"passed": False, "error": "FRESH_QUICK_RESULT_MISSING"})
                if args.validation == "full" and new_python.is_file():
                    validation_report["unit_tests"] = run_unit_tests(new_python, repo_root, args.timeout)
                    validation_report["passed"] = bool(validation_report.get("passed") and validation_report["unit_tests"]["passed"])
            after = collect_pip_state(new_python, run_dir / "provenance", "after") if new_python.is_file() else {"skipped": True}
            fresh_ok = bool(process.get("returncode") == 0 and fresh_report.get("passed"))
            passed = bool(fresh_ok and validation_report.get("passed"))
            report.update({
                "fresh_command": command,
                "fresh_process": process,
                "fresh_report": fresh_report,
                "new_python": str(new_python),
                "new_paths": new_paths,
                "environment_kind": "venv",
                "environment_mutated": False,
                "validation_result": validation_report,
                "pip_after": after,
                "passed": passed,
                "decision": "NEW_ISOLATED_ENV_CREATED_AND_QUALIFIED" if passed else "NEW_ISOLATED_ENV_BLOCKED",
                "blockers": [] if passed else [name for name, value in {"FRESH_ENV": fresh_ok, "VALIDATION": validation_report.get("passed")}.items() if not value],
                "checks": {"PLAN": True, "RUNTIME": fresh_ok, "VALIDATION": bool(validation_report.get("passed")), "PROVENANCE": bool(after)},
            })

    report["run_dir"] = str(run_dir)
    report["p2"] = {"executed": False, "policy": "MAINTAINER_ONLY"}
    json_dump(run_dir / "final_aggregate.json", report)
    (run_dir / "final_gate.txt").write_text(create_gate(report), encoding="utf-8")
    manifest_tree = inventory_tree(run_dir, exclude_names={"MANIFEST.json"})
    json_dump(run_dir / "MANIFEST.json", {"schema": SCHEMA + "-manifest", "run_id": run_id, "files": manifest_tree["files"]})
    (output_root / "adaptive_env_v1.latest").write_text(str(run_dir) + "\n", encoding="utf-8")
    if not (args.keep_work or args.no_cleanup):
        shutil.rmtree(work, ignore_errors=True)
    if args.report:
        json_dump(args.report.expanduser().resolve(), report)
    print(create_gate(report), end="")
    print(f"PUBLIC_ADAPTIVE_ENV_RUN_DIR={run_dir}")
    return 0 if report.get("passed") else 2


if __name__ == "__main__":
    raise SystemExit(main())
