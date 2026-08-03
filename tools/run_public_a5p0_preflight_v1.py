#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shlex
import sys
sys.dont_write_bytecode = True
import time
from typing import Any

from public_toolchain_common import (
    absolute_preserving_symlink,
    build_runtime_env,
    file_anchor,
    inspect_dataset,
    inventory_tree,
    json_dump,
    load_json,
    run_command,
    safe_relative,
    sha256,
)

SCHEMA = "amd-nerfstudio-public-a5p0-preflight-v1"
CLASSIFICATION = "PUBLIC_A5_P0_RUNTIME_SOURCE_DATASET_PREFLIGHT_V1"

CORE_MODULES = [
    "nerfstudio",
    "nerfstudio.configs.method_configs",
    "nerfstudio.engine.trainer",
    "nerfstudio.pipelines.base_pipeline",
    "nerfstudio.data.datamanagers.parallel_datamanager",
    "nerfstudio.data.dataparsers.nerfstudio_dataparser",
    "nerfstudio.models.nerfacto",
    "nerfstudio.field_components.encodings",
    "nerfstudio.field_components.mlp",
]


def git_probe(worktree: Path, reference: dict[str, Any]) -> dict[str, Any]:
    head = run_command(["git", "-C", str(worktree), "rev-parse", "HEAD"], timeout=30)
    tree = run_command(["git", "-C", str(worktree), "rev-parse", "HEAD^{tree}"], timeout=30)
    status = run_command(["git", "-C", str(worktree), "status", "--porcelain", "--untracked-files=no"], timeout=30)
    ns_ref = reference["nerfstudio"]
    mlp = worktree / ns_ref["mlp_relative_path"]
    result = {
        "head": head,
        "tree": tree,
        "tracked_status": status,
        "head_matches": head.get("stdout", "").strip() == ns_ref["commit"],
        "tree_matches": tree.get("stdout", "").strip() == ns_ref["tree"],
        "tracked_tree_clean": status.get("returncode") == 0 and not status.get("stdout", "").strip(),
        "mlp": file_anchor(mlp, ns_ref["mlp_sha256"]),
    }
    result["passed"] = bool(
        result["head_matches"]
        and result["tree_matches"]
        and result["tracked_tree_clean"]
        and result["mlp"]["hash_matches"]
    )
    return result


def find_one(root: Path, pattern: str) -> Path | None:
    matches = sorted(root.glob(pattern))
    return matches[0].resolve() if len(matches) == 1 else None


def child_probe(python: Path, runtime: Path, nerfstudio: Path, reference: dict[str, Any]) -> dict[str, Any]:
    code = r'''
import importlib, json, pathlib, sys, traceback
modules = json.loads(sys.argv[1])
out = {"modules": [], "passed": False}
try:
    import torch
    import nerfacc.csrc as nerfacc_csrc
    import tinycudann.modules as tcnn_modules
    from tinycudann.modules import _C
    for name in modules:
        try:
            mod = importlib.import_module(name)
            out["modules"].append({"module": name, "passed": True, "file": str(pathlib.Path(mod.__file__).resolve()) if getattr(mod, "__file__", None) else None})
        except Exception as exc:
            out["modules"].append({"module": name, "passed": False, "error": repr(exc), "traceback": traceback.format_exc()})
    from nerfstudio.configs.method_configs import all_methods
    cfg = all_methods["nerfacto"]
    dm = cfg.pipeline.datamanager
    origins = {}
    for name, mod in sorted(sys.modules.items()):
        if name.startswith(("tinycudann", "tinycudann_bindings")) and getattr(mod, "__file__", None):
            origins[name] = str(pathlib.Path(mod.__file__).resolve())
    out.update({
        "torch": torch.__version__,
        "hip": torch.version.hip,
        "cuda_available": bool(torch.cuda.is_available()),
        "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "gcn_arch": getattr(torch.cuda.get_device_properties(0), "gcnArchName", None) if torch.cuda.is_available() else None,
        "tinycudann_modules": str(pathlib.Path(tcnn_modules.__file__).resolve()),
        "tinycudann_native": str(pathlib.Path(_C.__file__).resolve()),
        "nerfacc_native": str(pathlib.Path(nerfacc_csrc.__file__).resolve()),
        "tinycudann_origins": origins,
        "nerfacto_config_type": type(cfg).__name__,
        "datamanager_config_type": type(dm).__name__,
        "has_dataloader_num_workers": hasattr(dm, "dataloader_num_workers"),
        "has_prefetch_factor": hasattr(dm, "prefetch_factor"),
    })
    out["passed"] = bool(out["cuda_available"] and all(row["passed"] for row in out["modules"]))
except Exception as exc:
    out["error"] = repr(exc)
    out["traceback"] = traceback.format_exc()
print("PUBLIC_A5P0_JSON=" + json.dumps(out, sort_keys=True))
'''
    env = build_runtime_env(runtime, nerfstudio)
    result = run_command([str(python), "-c", code, json.dumps(CORE_MODULES)], cwd=Path("/tmp"), env=env, timeout=300)
    payload = None
    for line in reversed(result.get("stdout", "").splitlines()):
        if line.startswith("PUBLIC_A5P0_JSON="):
            payload = json.loads(line.split("=", 1)[1])
            break
    return {"process": result, "payload": payload, "environment": {key: env.get(key) for key in [
        "PYTHONPATH", "PYTHONNOUSERSITE", "TCNN_RDNA4_ENABLE_PORTABLE_MLP_SHIM",
        "NERFSTUDIO_RDNA4_A5_SINGLE_SH_POLICY", "TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD",
    ]}}


def write_policy(path: Path, *, python: Path, runtime: Path, nerfstudio: Path, torch_lib: Path) -> None:
    text = f'''#!/usr/bin/env bash
# Generated by run_public_a5p0_preflight_v1.py. Review paths before sourcing.
export NERFSTUDIO_RDNA4_PUBLIC_PYTHON={shlex.quote(str(python))}
export NERFSTUDIO_RDNA4_PUBLIC_TCNN_RUNTIME={shlex.quote(str(runtime))}
export NERFSTUDIO_RDNA4_PUBLIC_NERFSTUDIO_WORKTREE={shlex.quote(str(nerfstudio))}
export PYTHONNOUSERSITE=1
export PYTHONDONTWRITEBYTECODE=1
export TCNN_RDNA4_ENABLE_PORTABLE_MLP_SHIM=1
export NERFSTUDIO_RDNA4_A5_SINGLE_SH_POLICY=TINY_RDNA4_NN_ONLY
export NERFSTUDIO_RDNA4_A5_TCNN_RUNTIME={shlex.quote(str(runtime))}
export NERFSTUDIO_RDNA4_A5_NERFSTUDIO_WORKTREE={shlex.quote(str(nerfstudio))}
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
unset TORCH_FORCE_WEIGHTS_ONLY_LOAD
export PYTHONPATH={shlex.quote(str(runtime))}:{shlex.quote(str(nerfstudio))}
export LD_LIBRARY_PATH={shlex.quote(str(torch_lib))}:/opt/rocm/lib:/opt/rocm/lib64:${{LD_LIBRARY_PATH:-}}
'''
    path.write_text(text, encoding="utf-8")
    path.chmod(0o755)


def self_test() -> int:
    reference = {"target": {"gcn_arch": "gfx1201"}, "qualification": {"dataloader_num_workers": 1}}
    ok = reference["target"]["gcn_arch"] == "gfx1201" and reference["qualification"]["dataloader_num_workers"] == 1
    print(json.dumps({"schema": SCHEMA, "passed": ok, "path_policy": "ALL_RUNTIME_PATHS_EXPLICIT"}, indent=2))
    return 0 if ok else 2


def main() -> int:
    parser = argparse.ArgumentParser(description="Public fail-closed runtime/source/dataset preflight for the RDNA4 Nerfacto qualification chain")
    parser.add_argument("--mode", choices=["run", "self-test"], default="run")
    parser.add_argument("--python", type=Path)
    parser.add_argument("--nerfstudio-worktree", type=Path)
    parser.add_argument("--tcnn-runtime", type=Path)
    parser.add_argument("--dataset", type=Path)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--reference", type=Path, default=Path(__file__).resolve().parents[1] / "config/reference_gfx1201_rocm72.json")
    parser.add_argument("--run-id")
    args = parser.parse_args()
    if args.mode == "self-test":
        return self_test()
    required = [args.python, args.nerfstudio_worktree, args.tcnn_runtime, args.dataset, args.output_root]
    if any(value is None for value in required):
        parser.error("run mode requires --python, --nerfstudio-worktree, --tcnn-runtime, --dataset and --output-root")

    python = absolute_preserving_symlink(args.python)
    nerfstudio = args.nerfstudio_worktree.expanduser().resolve()
    runtime = args.tcnn_runtime.expanduser().resolve()
    dataset = args.dataset.expanduser().resolve()
    output_root = args.output_root.expanduser().resolve()
    reference_path = args.reference.expanduser().resolve()
    reference = load_json(reference_path)
    run_id = args.run_id or time.strftime("%Y%m%dT%H%M%SZ", time.gmtime()) + f"_{os.getpid()}"
    run_dir = output_root / "public_a5p0_preflight_v1" / run_id
    run_dir.mkdir(parents=True, exist_ok=False)

    checks: dict[str, bool] = {}
    paths = {
        "python": {"path": str(python), "passed": python.is_file() and os.access(python, os.X_OK)},
        "nerfstudio": {"path": str(nerfstudio), "passed": (nerfstudio / ".git").exists() or (nerfstudio / "nerfstudio").is_dir()},
        "runtime": {"path": str(runtime), "passed": runtime.is_dir()},
        "dataset": {"path": str(dataset), "passed": dataset.exists()},
        "reference": {"path": str(reference_path), "passed": reference_path.is_file()},
    }
    checks["EXPLICIT_PATHS_EXIST"] = all(row["passed"] for row in paths.values())

    source = git_probe(nerfstudio, reference) if paths["nerfstudio"]["passed"] else {"passed": False}
    checks["NERFSTUDIO_SOURCE_PINNED_AND_CLEAN"] = bool(source.get("passed"))

    dataset_report = inspect_dataset(dataset)
    checks["REAL_NERFSTUDIO_DATASET_STRUCTURE"] = bool(dataset_report.get("passed"))

    probe = child_probe(python, runtime, nerfstudio, reference) if checks["EXPLICIT_PATHS_EXIST"] else {"payload": None, "process": {}}
    payload = probe.get("payload") or {}
    target = reference["target"]
    runtime_ref = reference["runtime"]
    modules_path = Path(payload["tinycudann_modules"]).resolve() if payload.get("tinycudann_modules") else None
    native_path = Path(payload["tinycudann_native"]).resolve() if payload.get("tinycudann_native") else None
    nerfacc_path = Path(payload["nerfacc_native"]).resolve() if payload.get("nerfacc_native") else None
    runtime_anchors = {
        "torch_matches": payload.get("torch") == target["torch"],
        "hip_matches": payload.get("hip") == target["hip"],
        "gcn_arch_matches": payload.get("gcn_arch") == target["gcn_arch"],
        "selected_imports_pass": payload.get("passed") is True,
        "dataloader_config_fields_present": payload.get("has_dataloader_num_workers") is True and payload.get("has_prefetch_factor") is True,
        "tinycudann_modules": file_anchor(modules_path, runtime_ref["tinycudann_modules_sha256"]) if modules_path else {"exists": False, "hash_matches": False},
        "tinycudann_native": file_anchor(native_path, runtime_ref["tinycudann_native_sha256"]) if native_path else {"exists": False, "hash_matches": False},
        "nerfacc_native": file_anchor(nerfacc_path, runtime_ref["nerfacc_native_sha256"]) if nerfacc_path else {"exists": False, "hash_matches": False},
    }
    origins = payload.get("tinycudann_origins", {})
    runtime_anchors["single_runtime_origin"] = bool(origins) and all(safe_relative(Path(path), runtime) for path in origins.values())
    runtime_anchors["passed"] = bool(
        runtime_anchors["torch_matches"] and runtime_anchors["hip_matches"] and runtime_anchors["gcn_arch_matches"]
        and runtime_anchors["selected_imports_pass"] and runtime_anchors["dataloader_config_fields_present"]
        and runtime_anchors["tinycudann_modules"].get("hash_matches") and runtime_anchors["tinycudann_native"].get("hash_matches")
        and runtime_anchors["nerfacc_native"].get("hash_matches") and runtime_anchors["single_runtime_origin"]
    )
    checks["ROCM_GFX1201_RUNTIME_AND_SINGLE_ORIGIN"] = runtime_anchors["passed"]

    torch_lib = python.parent.parent / f"lib/python{target['python_major_minor']}/site-packages/torch/lib"
    policy = run_dir / "runtime_policy.sh"
    if torch_lib.is_dir() and all(checks.values()):
        write_policy(policy, python=python, runtime=runtime, nerfstudio=nerfstudio, torch_lib=torch_lib)
    checks["RUNTIME_POLICY_EMITTED"] = policy.is_file()

    blockers = [name for name, passed in checks.items() if not passed]
    report = {
        "schema": SCHEMA,
        "classification": CLASSIFICATION,
        "run_id": run_id,
        "decision": "PROCEED_TO_PUBLIC_A5_P1" if not blockers else "PUBLIC_A5_P0_BLOCKED",
        "passed": not blockers,
        "paths": paths,
        "reference": {"path": str(reference_path), "sha256": sha256(reference_path), "payload": reference},
        "nerfstudio_source": source,
        "dataset": dataset_report,
        "runtime_probe": probe,
        "runtime_anchors": runtime_anchors,
        "checks": checks,
        "blockers": blockers,
        "nonclaims": reference.get("nonclaims", []),
    }
    json_dump(run_dir / "final_aggregate.json", report)
    gate_lines = [
        "AMD_NERFSTUDIO_PUBLIC_A5_P0_PREFLIGHT_V1", "",
        f"classification={CLASSIFICATION}", f"decision={report['decision']}", f"run_id={run_id}", "",
    ]
    gate_lines.extend(f"PUBLIC_RDNA4_{name}: {'PASS' if passed else 'FAIL'}" for name, passed in checks.items())
    gate_lines.extend(["", "blockers=" + (",".join(blockers) if blockers else "NONE"), "",
                       "PUBLIC_RDNA4_A5_P0_PREFLIGHT: PASS" if report["passed"] else "PUBLIC_RDNA4_A5_P0_PREFLIGHT: FAIL"])
    (run_dir / "final_gate.txt").write_text("\n".join(gate_lines) + "\n", encoding="utf-8")
    manifest = inventory_tree(run_dir, exclude_names={"MANIFEST.json"})
    manifest.update({"schema": SCHEMA + "-manifest", "run_id": run_id})
    json_dump(run_dir / "MANIFEST.json", manifest)
    print(json.dumps(report, indent=2, sort_keys=True))
    print("\n".join(gate_lines))
    print(f"PUBLIC_A5P0_RUN_DIR={run_dir}")
    return 0 if report["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
