#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
sys.dont_write_bytecode = True
import tempfile
from typing import Any

from public_toolchain_common import inspect_dataset, json_dump, load_json, run_command, sha256

SCHEMA = "amd-nerfstudio-public-resource-manager-v1"
PROFILE = "reference-binary-fresh-env"


def ask_yes_no(prompt: str, *, auto: bool) -> bool:
    if auto:
        return True
    if not sys.stdin.isatty():
        return False
    answer = input(f"{prompt} [y/N] ").strip().lower()
    return answer in {"y", "yes", "j", "ja"}


def atomic_copy_file(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + f".part-{os.getpid()}")
    if temporary.exists():
        temporary.unlink()
    shutil.copy2(source, temporary)
    os.replace(temporary, destination)


def atomic_copy_tree(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + f".part-{os.getpid()}")
    if temporary.exists():
        shutil.rmtree(temporary)
    shutil.copytree(source, temporary, symlinks=False)
    os.replace(temporary, destination)


def resource_paths(resource_dir: Path, manifest: dict[str, Any]) -> dict[str, Path]:
    custom = manifest["custom_resources"]
    layout = manifest["cache_layout"]
    return {
        "nerfacc_wheel": resource_dir / custom["nerfacc_wheel"]["cache_relative_path"],
        "tcnn_runtime": resource_dir / custom["tiny_rdna4_runtime"]["cache_relative_path"],
        "dataset": resource_dir / custom["quick_validation_dataset"]["cache_relative_path"],
        "nerfstudio_source": resource_dir / layout["nerfstudio_source"],
        "wheelhouse": resource_dir / layout["wheelhouse"],
        "wheelhouse_lock": resource_dir / layout["wheelhouse_lock"],
    }


def verify_nerfacc_wheel(path: Path, spec: dict[str, Any]) -> dict[str, Any]:
    observed = sha256(path) if path.is_file() else None
    return {
        "path": str(path),
        "exists": path.is_file(),
        "sha256": observed,
        "expected_sha256": spec["sha256"],
        "passed": observed == spec["sha256"],
    }


def verify_tcnn_runtime(path: Path, spec: dict[str, Any]) -> dict[str, Any]:
    modules = path / spec["modules_relative_path"]
    native_matches = sorted(path.glob(spec["native_glob"])) if path.is_dir() else []
    native = native_matches[0] if len(native_matches) == 1 else None
    modules_hash = sha256(modules) if modules.is_file() else None
    native_hash = sha256(native) if native and native.is_file() else None
    return {
        "path": str(path),
        "modules": {
            "path": str(modules),
            "exists": modules.is_file(),
            "sha256": modules_hash,
            "expected_sha256": spec["modules_sha256"],
            "passed": modules_hash == spec["modules_sha256"],
        },
        "native": {
            "glob": spec["native_glob"],
            "match_count": len(native_matches),
            "path": str(native) if native else None,
            "sha256": native_hash,
            "expected_sha256": spec["native_sha256"],
            "passed": len(native_matches) == 1 and native_hash == spec["native_sha256"],
        },
        "passed": bool(modules_hash == spec["modules_sha256"] and len(native_matches) == 1 and native_hash == spec["native_sha256"]),
    }


def verify_dataset(path: Path, spec: dict[str, Any]) -> dict[str, Any]:
    inspection = inspect_dataset(path)
    images: dict[str, Any] = {}
    for name, expected in sorted(spec["images"].items()):
        image = path / name
        observed = sha256(image) if image.is_file() else None
        images[name] = {
            "path": str(image),
            "sha256": observed,
            "expected_sha256": expected,
            "passed": observed == expected,
        }
    passed = bool(
        inspection.get("passed")
        and inspection.get("transforms_sha256") == spec["transforms_sha256"]
        and inspection.get("path_size_manifest_sha256") == spec["path_size_manifest_sha256"]
        and all(row["passed"] for row in images.values())
    )
    return {
        "path": str(path),
        "inspection": inspection,
        "expected_transforms_sha256": spec["transforms_sha256"],
        "expected_path_size_manifest_sha256": spec["path_size_manifest_sha256"],
        "images": images,
        "passed": passed,
    }


def verify_nerfstudio_source(path: Path, spec: dict[str, Any]) -> dict[str, Any]:
    if not path.is_dir():
        return {"path": str(path), "exists": False, "passed": False}
    head = run_command(["git", "-C", str(path), "rev-parse", "HEAD"], timeout=30)
    tree = run_command(["git", "-C", str(path), "rev-parse", "HEAD^{tree}"], timeout=30)
    status = run_command(["git", "-C", str(path), "status", "--porcelain", "--untracked-files=no"], timeout=30)
    mlp = path / spec["mlp_relative_path"]
    mlp_hash = sha256(mlp) if mlp.is_file() else None
    passed = bool(
        head.get("stdout", "").strip() == spec["commit"]
        and tree.get("stdout", "").strip() == spec["tree"]
        and status.get("returncode") == 0
        and not status.get("stdout", "").strip()
        and mlp_hash == spec["mlp_sha256"]
    )
    return {
        "path": str(path),
        "exists": True,
        "head": head,
        "tree": tree,
        "tracked_status": status,
        "mlp": {
            "path": str(mlp),
            "sha256": mlp_hash,
            "expected_sha256": spec["mlp_sha256"],
        },
        "passed": passed,
    }


def clone_source(source: str, destination: Path, commit: str) -> dict[str, Any]:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + f".part-{os.getpid()}")
    if temporary.exists():
        shutil.rmtree(temporary)
    attempts: list[dict[str, Any]] = []
    clone = run_command(["git", "clone", "--no-hardlinks", source, str(temporary)], timeout=900)
    attempts.append({"source": source, "process": clone})
    if clone.get("returncode") != 0 and Path(source).is_dir():
        shutil.rmtree(temporary, ignore_errors=True)
        common = run_command(
            ["git", "-C", source, "rev-parse", "--path-format=absolute", "--git-common-dir"],
            timeout=30,
        )
        common_dir = common.get("stdout", "").strip()
        if common.get("returncode") == 0 and common_dir:
            clone = run_command(["git", "clone", "--no-hardlinks", common_dir, str(temporary)], timeout=900)
            attempts.append({"source": common_dir, "common_dir_probe": common, "process": clone})
    checkout: dict[str, Any] = {}
    if clone.get("returncode") == 0:
        checkout = run_command(["git", "-C", str(temporary), "checkout", "--detach", commit], timeout=120)
    if clone.get("returncode") == 0 and checkout.get("returncode") == 0:
        os.replace(temporary, destination)
    elif temporary.exists():
        shutil.rmtree(temporary)
    return {
        "clone_attempts": attempts,
        "checkout": checkout,
        "passed": clone.get("returncode") == 0 and checkout.get("returncode") == 0,
    }


def remove_invalid(path: Path, replace_invalid: bool) -> bool:
    if not path.exists():
        return True
    if not replace_invalid:
        return False
    if path.is_dir():
        shutil.rmtree(path)
    else:
        path.unlink()
    return True


def verify_resource_cache(resource_dir: Path, manifest: dict[str, Any]) -> dict[str, Any]:
    resource_dir = resource_dir.expanduser().resolve()
    paths = resource_paths(resource_dir, manifest)
    custom = manifest["custom_resources"]
    verification = {
        "nerfacc_wheel": verify_nerfacc_wheel(paths["nerfacc_wheel"], custom["nerfacc_wheel"]),
        "tcnn_runtime": verify_tcnn_runtime(paths["tcnn_runtime"], custom["tiny_rdna4_runtime"]),
        "dataset": verify_dataset(paths["dataset"], custom["quick_validation_dataset"]),
        "nerfstudio_source": verify_nerfstudio_source(paths["nerfstudio_source"], manifest["nerfstudio_source"]),
    }
    blockers = [f"RESOURCE_VERIFICATION_FAILED_{name.upper()}" for name, row in verification.items() if not row.get("passed")]
    return {
        "schema": SCHEMA,
        "profile": manifest["profile"],
        "resource_dir": str(resource_dir),
        "paths": {name: str(path) for name, path in paths.items()},
        "actions": [],
        "verification": verification,
        "blockers": blockers,
        "passed": not blockers,
        "verification_only": True,
    }


def prepare_resources(args: argparse.Namespace, manifest: dict[str, Any]) -> dict[str, Any]:
    resource_dir = args.resource_dir.expanduser().resolve()
    resource_dir.mkdir(parents=True, exist_ok=True)
    paths = resource_paths(resource_dir, manifest)
    actions: list[dict[str, Any]] = []
    blockers: list[str] = []
    custom = manifest["custom_resources"]

    explicit = {
        "nerfacc_wheel": args.nerfacc_wheel,
        "tcnn_runtime": args.tcnn_runtime,
        "dataset": args.dataset,
    }
    verifiers = {
        "nerfacc_wheel": lambda p: verify_nerfacc_wheel(p, custom["nerfacc_wheel"]),
        "tcnn_runtime": lambda p: verify_tcnn_runtime(p, custom["tiny_rdna4_runtime"]),
        "dataset": lambda p: verify_dataset(p, custom["quick_validation_dataset"]),
    }

    for name in ["nerfacc_wheel", "tcnn_runtime", "dataset"]:
        current = verifiers[name](paths[name])
        if current.get("passed"):
            continue
        supplied = explicit[name]
        if supplied is None:
            blockers.append(f"MISSING_CUSTOM_RESOURCE_{name.upper()}")
            continue
        source = supplied.expanduser().resolve()
        source_check = verifiers[name](source)
        if not source_check.get("passed"):
            blockers.append(f"INVALID_SUPPLIED_RESOURCE_{name.upper()}")
            actions.append({"resource": name, "action": "SUPPLIED_RESOURCE_REJECTED", "verification": source_check})
            continue
        if not remove_invalid(paths[name], args.replace_invalid):
            blockers.append(f"INVALID_CACHE_REQUIRES_REPLACE_{name.upper()}")
            continue
        if name == "nerfacc_wheel":
            atomic_copy_file(source, paths[name])
        else:
            atomic_copy_tree(source, paths[name])
        actions.append({"resource": name, "action": "STAGED_FROM_EXPLICIT_LOCAL_PATH", "source": str(source), "destination": str(paths[name])})

    source_spec = manifest["nerfstudio_source"]
    source_check = verify_nerfstudio_source(paths["nerfstudio_source"], source_spec)
    if not source_check.get("passed"):
        if args.nerfstudio_source is not None:
            source_value = str(args.nerfstudio_source.expanduser().resolve())
            if not remove_invalid(paths["nerfstudio_source"], args.replace_invalid):
                blockers.append("INVALID_CACHE_REQUIRES_REPLACE_NERFSTUDIO_SOURCE")
            else:
                action = clone_source(source_value, paths["nerfstudio_source"], source_spec["commit"])
                actions.append({"resource": "nerfstudio_source", "action": "CLONED_FROM_EXPLICIT_LOCAL_PATH", **action})
                if not action["passed"]:
                    blockers.append("NERFSTUDIO_LOCAL_CLONE_FAILED")
        elif args.offline:
            blockers.append("OFFLINE_MISSING_NERFSTUDIO_SOURCE")
        elif ask_yes_no(
            f"Pinned Nerfstudio source is missing. Clone commit {source_spec['commit']} from the official repository?",
            auto=args.auto,
        ):
            if not remove_invalid(paths["nerfstudio_source"], args.replace_invalid):
                blockers.append("INVALID_CACHE_REQUIRES_REPLACE_NERFSTUDIO_SOURCE")
            else:
                action = clone_source(source_spec["url"], paths["nerfstudio_source"], source_spec["commit"])
                actions.append({"resource": "nerfstudio_source", "action": "CLONED_FROM_PINNED_PUBLIC_SOURCE", **action})
                if not action["passed"]:
                    blockers.append("NERFSTUDIO_NETWORK_CLONE_FAILED")
        else:
            blockers.append("NETWORK_CONSENT_REQUIRED_FOR_NERFSTUDIO_SOURCE")

    verification = {
        "nerfacc_wheel": verify_nerfacc_wheel(paths["nerfacc_wheel"], custom["nerfacc_wheel"]),
        "tcnn_runtime": verify_tcnn_runtime(paths["tcnn_runtime"], custom["tiny_rdna4_runtime"]),
        "dataset": verify_dataset(paths["dataset"], custom["quick_validation_dataset"]),
        "nerfstudio_source": verify_nerfstudio_source(paths["nerfstudio_source"], source_spec),
    }
    for name, row in verification.items():
        if not row.get("passed"):
            marker = f"RESOURCE_VERIFICATION_FAILED_{name.upper()}"
            if marker not in blockers:
                blockers.append(marker)

    return {
        "schema": SCHEMA,
        "profile": manifest["profile"],
        "resource_dir": str(resource_dir),
        "paths": {name: str(path) for name, path in paths.items()},
        "actions": actions,
        "verification": verification,
        "blockers": blockers,
        "passed": not blockers and all(row.get("passed") for row in verification.values()),
    }


def self_test() -> int:
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        wheel = root / "x.whl"
        wheel.write_bytes(b"wheel")
        wheel_spec = {"sha256": sha256(wheel)}
        runtime = root / "runtime"
        (runtime / "tinycudann").mkdir(parents=True)
        (runtime / "tinycudann_bindings").mkdir()
        modules = runtime / "tinycudann/modules.py"
        native = runtime / "tinycudann_bindings/_120_C.fixture.so"
        modules.write_text("fixture\n")
        native.write_bytes(b"native")
        runtime_spec = {
            "modules_relative_path": "tinycudann/modules.py",
            "modules_sha256": sha256(modules),
            "native_glob": "tinycudann_bindings/_120_C*.so",
            "native_sha256": sha256(native),
        }
        dataset = root / "dataset"
        dataset.mkdir()
        (dataset / "000.png").write_bytes(b"image")
        (dataset / "transforms.json").write_text('{"frames":[{"file_path":"000.png"}]}\n')
        inspected = inspect_dataset(dataset)
        dataset_spec = {
            "transforms_sha256": inspected["transforms_sha256"],
            "path_size_manifest_sha256": inspected["path_size_manifest_sha256"],
            "images": {"000.png": sha256(dataset / "000.png")},
        }
        checks = {
            "wheel": verify_nerfacc_wheel(wheel, wheel_spec)["passed"],
            "runtime": verify_tcnn_runtime(runtime, runtime_spec)["passed"],
            "dataset": verify_dataset(dataset, dataset_spec)["passed"],
        }
    passed = all(checks.values())
    print(json.dumps({"schema": SCHEMA, "passed": passed, "checks": checks}, indent=2, sort_keys=True))
    return 0 if passed else 2


def create_gate(report: dict[str, Any]) -> str:
    lines = [
        "AMD_NERFSTUDIO_PUBLIC_RESOURCE_MANAGER_V1",
        "",
        f"profile={report.get('profile')}",
        f"resource_dir={report.get('resource_dir')}",
        f"decision={'RESOURCES_READY' if report.get('passed') else 'RESOURCES_BLOCKED'}",
        "blockers=" + (",".join(report.get("blockers", [])) if report.get("blockers") else "NONE"),
        "",
        "PUBLIC_RDNA4_RESOURCES: PASS" if report.get("passed") else "PUBLIC_RDNA4_RESOURCES: FAIL",
    ]
    return "\n".join(lines) + "\n"


def main() -> int:
    repo_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description="Prepare and verify pinned resources for the public fresh environment")
    parser.add_argument("--mode", choices=["run", "self-test"], default="run")
    parser.add_argument("--profile", default=PROFILE)
    parser.add_argument("--manifest", type=Path, default=repo_root / "config/public_fresh_env_resources_v1.json")
    parser.add_argument("--resource-dir", type=Path)
    parser.add_argument("--nerfacc-wheel", type=Path)
    parser.add_argument("--tcnn-runtime", type=Path)
    parser.add_argument("--dataset", type=Path)
    parser.add_argument("--nerfstudio-source", type=Path)
    parser.add_argument("--auto", action="store_true")
    parser.add_argument("--offline", action="store_true")
    parser.add_argument("--replace-invalid", action="store_true")
    parser.add_argument("--verify-only", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.mode == "self-test":
        return self_test()
    if args.profile != PROFILE:
        print(f"Unsupported profile: {args.profile}. fresh-native-build is not claimed by v1.3.", file=sys.stderr)
        return 64
    if args.resource_dir is None:
        parser.error("run mode requires --resource-dir")
    if args.auto and args.offline:
        parser.error("--auto and --offline are mutually exclusive")
    manifest = load_json(args.manifest.expanduser().resolve())
    report = verify_resource_cache(args.resource_dir, manifest) if args.verify_only else prepare_resources(args, manifest)
    if args.output:
        output = args.output.expanduser().resolve()
        json_dump(output, report)
        gate = output.with_name(output.stem + "_gate.txt")
        gate.write_text(create_gate(report), encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    print(create_gate(report), end="")
    return 0 if report["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
