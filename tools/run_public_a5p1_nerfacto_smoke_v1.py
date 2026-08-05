#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import dataclasses
import hashlib
import json
import math
import os
from pathlib import Path
import random
import shlex
import shutil
import subprocess
import sys
sys.dont_write_bytecode = True
import time
import traceback
from typing import Any, Iterable

from public_toolchain_common import absolute_preserving_symlink, verify_manifest

SCHEMA = "amd-nerfstudio-public-a5p1-nerfacto-smoke-v1"
CLASSIFICATION = "PUBLIC_A5_P1_NERFACTO_DATALOADER_FORWARD_BACKWARD_CHECKPOINT_V1"

EXPECTED = {
    'nerfstudio_commit': '50e0e3c70c775e89333256213363badbf074f29d',
    'nerfstudio_tree': '9d5ff468eeff89b66995e9984acaa378c37dc07e',
    'nerfstudio_mlp': '4939a5a6901d82d8e310d93e2a135ca57ccc1bd79be79a7f67e2740e730c44ad',
    'tinycudann_native': '4a561cc605bb7a6353d0eca1f9effc5ac9fcdfa3a9cb605a8cf36e1ae25b1917',
    'tinycudann_modules': 'b4df43b54f64fe2b31272a997aafd50137aecac411d59b05251acedcd5512d12',
    'nerfacc_native': 'd3beee150cfa3a9ad3038a3283ff0a46953c345634d8cb6109449c5e3d04d1e2',
    'torch': '2.13.0+rocm7.2',
    'hip': '7.2.53211',
    'gcn_arch': 'gfx1201',
}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def json_dump(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")


def run_command(
    argv: list[str],
    *,
    cwd: Path | None = None,
    env: dict[str, str] | None = None,
    timeout: int = 120,
) -> dict[str, Any]:
    started = time.time()
    try:
        proc = subprocess.run(
            argv,
            cwd=str(cwd) if cwd else None,
            env=env,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=timeout,
            check=False,
        )
        return {
            "argv": argv,
            "command": shlex.join(argv),
            "cwd": str(cwd) if cwd else None,
            "returncode": proc.returncode,
            "stdout": proc.stdout,
            "stderr": proc.stderr,
            "timed_out": False,
            "duration_seconds": time.time() - started,
        }
    except subprocess.TimeoutExpired as exc:
        return {
            "argv": argv,
            "command": shlex.join(argv),
            "cwd": str(cwd) if cwd else None,
            "returncode": None,
            "stdout": exc.stdout or "",
            "stderr": exc.stderr or "",
            "timed_out": True,
            "duration_seconds": time.time() - started,
        }


def safe_relative(path: Path, parent: Path) -> bool:
    try:
        path.resolve().relative_to(parent.resolve())
        return True
    except ValueError:
        return False


def discover_datasets(search_roots: Iterable[Path], max_depth: int = 7, limit: int = 100) -> list[str]:
    found: list[str] = []
    skip_names = {".git", "node_modules", "site-packages", "__pycache__", ".cache", "outputs", "evidence", "logs"}
    for root in search_roots:
        root = root.expanduser()
        if not root.is_dir():
            continue
        root_depth = len(root.parts)
        for current, dirs, files in os.walk(root):
            here = Path(current)
            depth = len(here.parts) - root_depth
            dirs[:] = [d for d in dirs if d not in skip_names and depth < max_depth]
            if "transforms.json" in files:
                found.append(str(here.resolve()))
                if len(found) >= limit:
                    return sorted(set(found))
    return sorted(set(found))


def resolve_image_path(dataset_dir: Path, raw: str) -> Path | None:
    candidate = Path(raw)
    if not candidate.is_absolute():
        candidate = dataset_dir / candidate
    candidate = candidate.expanduser()
    options = [candidate]
    if candidate.suffix == "":
        options.extend(candidate.with_suffix(ext) for ext in [".png", ".jpg", ".jpeg", ".JPG", ".PNG", ".JPEG"])
    for option in options:
        if option.is_file():
            return option.resolve()
    return None


def inspect_dataset(data: Path) -> dict[str, Any]:
    resolved = data.expanduser().resolve()
    if resolved.is_dir():
        transforms = resolved / "transforms.json"
        dataset_dir = resolved
    elif resolved.is_file() and resolved.suffix.lower() == ".json":
        transforms = resolved
        dataset_dir = resolved.parent
    else:
        return {"passed": False, "error": "DATA_PATH_IS_NOT_A_NERFSTUDIO_DIRECTORY_OR_JSON", "data": str(resolved)}
    if not transforms.is_file():
        return {
            "passed": False,
            "error": "TRANSFORMS_JSON_MISSING",
            "data": str(resolved),
            "expected": str(transforms),
        }
    try:
        payload = json.loads(transforms.read_text(encoding="utf-8"))
    except Exception as exc:
        return {"passed": False, "error": repr(exc), "transforms": str(transforms), "traceback": traceback.format_exc()}
    frames = payload.get("frames")
    if not isinstance(frames, list) or not frames:
        return {"passed": False, "error": "FRAMES_MISSING_OR_EMPTY", "transforms": str(transforms)}

    image_rows: list[dict[str, Any]] = []
    missing: list[str] = []
    for index, frame in enumerate(frames):
        raw = frame.get("file_path") if isinstance(frame, dict) else None
        if not isinstance(raw, str) or not raw:
            missing.append(f"frame[{index}]:missing_file_path")
            continue
        image = resolve_image_path(dataset_dir, raw)
        if image is None:
            missing.append(raw)
            continue
        image_rows.append({"index": index, "raw": raw, "path": str(image), "size_bytes": image.stat().st_size})

    sample_indices: list[int] = []
    if image_rows:
        sample_indices = sorted(set([0, len(image_rows) // 2, len(image_rows) - 1]))
    sample_hashes = []
    for index in sample_indices:
        row = image_rows[index]
        path = Path(row["path"])
        sample_hashes.append({**row, "sha256": sha256(path)})

    path_manifest = hashlib.sha256()
    for row in image_rows:
        path_manifest.update(f"{row['index']}\0{row['raw']}\0{row['size_bytes']}\n".encode("utf-8"))

    return {
        "passed": bool(not missing and len(image_rows) == len(frames)),
        "data_argument": str(resolved),
        "dataset_dir": str(dataset_dir),
        "transforms": str(transforms),
        "transforms_sha256": sha256(transforms),
        "frame_count": len(frames),
        "resolved_image_count": len(image_rows),
        "missing_count": len(missing),
        "missing": missing[:50],
        "sample_image_hashes": sample_hashes,
        "path_size_manifest_sha256": path_manifest.hexdigest(),
        "full_dataset_content_hash": "NOT_CLAIMED",
    }


def file_anchor(path: Path, expected: str) -> dict[str, Any]:
    observed = sha256(path) if path.is_file() else None
    return {
        "path": str(path),
        "exists": path.is_file(),
        "sha256": observed,
        "expected_sha256": expected,
        "hash_matches": observed == expected,
    }



def prerequisite_report(preflight_dir: Path, dataset: Path, nerfstudio: Path, runtime: Path, python: Path) -> dict[str, Any]:
    preflight_manifest = verify_manifest(preflight_dir)
    preflight_payload: dict[str, Any] = {}
    try:
        preflight_payload = json.loads((preflight_dir / "final_aggregate.json").read_text(encoding="utf-8"))
    except Exception as exc:
        preflight_payload = {"passed": False, "error": repr(exc)}
    p0_semantics = {
        "passed": bool(
            preflight_manifest.get("passed")
            and preflight_payload.get("passed") is True
            and preflight_payload.get("decision") == "PROCEED_TO_PUBLIC_A5_P1"
            and not preflight_payload.get("blockers")
        ),
        "decision": preflight_payload.get("decision"),
        "blockers": preflight_payload.get("blockers"),
        "classification": preflight_payload.get("classification"),
        "run_id": preflight_payload.get("run_id"),
    }

    git_head = run_command(["git", "-C", str(nerfstudio), "rev-parse", "HEAD"], timeout=30)
    git_tree = run_command(["git", "-C", str(nerfstudio), "rev-parse", "HEAD^{tree}"], timeout=30)
    git_status = run_command(["git", "-C", str(nerfstudio), "status", "--porcelain", "--untracked-files=no"], timeout=30)
    source = {
        "head": git_head,
        "tree": git_tree,
        "tracked_status": git_status,
        "head_matches": git_head.get("stdout", "").strip() == EXPECTED["nerfstudio_commit"],
        "tree_matches": git_tree.get("stdout", "").strip() == EXPECTED["nerfstudio_tree"],
        "tracked_tree_clean": git_status.get("returncode") == 0 and not git_status.get("stdout", "").strip(),
        "mlp": file_anchor(nerfstudio / "nerfstudio/field_components/mlp.py", EXPECTED["nerfstudio_mlp"]),
    }
    source["passed"] = bool(source["head_matches"] and source["tree_matches"] and source["tracked_tree_clean"] and source["mlp"]["hash_matches"])

    runtime_from_preflight = preflight_payload.get("runtime_anchors", {})
    runtime_anchors = {
        "passed": bool(runtime_from_preflight.get("passed")),
        "source": "PUBLIC_A5P0_MANIFEST_VERIFIED_PREFLIGHT",
        "details": runtime_from_preflight,
        "runtime_root": str(runtime),
        "python": str(python),
    }
    dataset_report = inspect_dataset(dataset)
    env_policy = {
        "PYTHONNOUSERSITE": os.environ.get("PYTHONNOUSERSITE"),
        "TCNN_RDNA4_ENABLE_PORTABLE_MLP_SHIM": os.environ.get("TCNN_RDNA4_ENABLE_PORTABLE_MLP_SHIM"),
        "NERFSTUDIO_RDNA4_A5_SINGLE_SH_POLICY": os.environ.get("NERFSTUDIO_RDNA4_A5_SINGLE_SH_POLICY"),
        "NERFSTUDIO_RDNA4_A5_TCNN_RUNTIME": os.environ.get("NERFSTUDIO_RDNA4_A5_TCNN_RUNTIME"),
        "NERFSTUDIO_RDNA4_A5_NERFSTUDIO_WORKTREE": os.environ.get("NERFSTUDIO_RDNA4_A5_NERFSTUDIO_WORKTREE"),
        "TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD": os.environ.get("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD"),
        "TORCH_FORCE_WEIGHTS_ONLY_LOAD": os.environ.get("TORCH_FORCE_WEIGHTS_ONLY_LOAD"),
    }
    env_policy["passed"] = bool(
        env_policy["PYTHONNOUSERSITE"] == "1"
        and env_policy["TCNN_RDNA4_ENABLE_PORTABLE_MLP_SHIM"] == "1"
        and env_policy["NERFSTUDIO_RDNA4_A5_SINGLE_SH_POLICY"] == "TINY_RDNA4_NN_ONLY"
        and Path(env_policy["NERFSTUDIO_RDNA4_A5_TCNN_RUNTIME"] or "").resolve() == runtime.resolve()
        and Path(env_policy["NERFSTUDIO_RDNA4_A5_NERFSTUDIO_WORKTREE"] or "").resolve() == nerfstudio.resolve()
        and env_policy["TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD"] == "1"
        and not env_policy["TORCH_FORCE_WEIGHTS_ONLY_LOAD"]
    )
    passed = bool(p0_semantics["passed"] and source["passed"] and runtime_anchors["passed"] and dataset_report.get("passed") and env_policy["passed"])
    return {
        "passed": passed,
        "p0_hash_chain": preflight_manifest,
        "p0_semantics": p0_semantics,
        "nerfstudio_source": source,
        "runtime_anchors": runtime_anchors,
        "dataset": dataset_report,
        "environment_policy": env_policy,
        "paths": {
            "preflight_dir": str(preflight_dir),
            "nerfstudio_worktree": str(nerfstudio),
            "tcnn_runtime": str(runtime),
            "python": str(python),
        },
    }

def seed_everything(seed: int) -> None:
    random.seed(seed)
    try:
        import numpy as np
        np.random.seed(seed)
    except Exception:
        pass
    import torch
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def tensor_hash(tensor: Any) -> str:
    import torch
    if not torch.is_tensor(tensor):
        raise TypeError("tensor_hash expects torch.Tensor")
    value = tensor.detach()
    if value.is_sparse:
        value = value.to_dense()
    value = value.cpu().contiguous()
    raw = value.reshape(-1).view(torch.uint8).numpy().tobytes()
    h = hashlib.sha256()
    h.update(str(value.dtype).encode("utf-8"))
    h.update(json.dumps(list(value.shape)).encode("utf-8"))
    h.update(raw)
    return h.hexdigest()


def update_canonical_hash(h: Any, value: Any) -> None:
    import torch
    if torch.is_tensor(value):
        h.update(b"T")
        h.update(tensor_hash(value).encode("ascii"))
    elif isinstance(value, dict):
        h.update(b"D")
        for key in sorted(value.keys(), key=lambda x: repr(x)):
            update_canonical_hash(h, key)
            update_canonical_hash(h, value[key])
    elif isinstance(value, (list, tuple)):
        h.update(b"L" if isinstance(value, list) else b"U")
        for item in value:
            update_canonical_hash(h, item)
    elif isinstance(value, Path):
        h.update(b"P" + str(value).encode("utf-8"))
    elif dataclasses.is_dataclass(value):
        update_canonical_hash(h, dataclasses.asdict(value))
    else:
        h.update(type(value).__name__.encode("utf-8"))
        h.update(repr(value).encode("utf-8"))


def canonical_hash(value: Any) -> str:
    h = hashlib.sha256()
    update_canonical_hash(h, value)
    return h.hexdigest()


def parameter_hashes(module: Any) -> dict[str, str]:
    return {name: tensor_hash(param) for name, param in sorted(module.named_parameters())}


def summarize_value(value: Any, depth: int = 0) -> Any:
    import torch
    if depth > 4:
        return {"type": type(value).__name__, "truncated": True}
    if torch.is_tensor(value):
        finite = None
        if value.is_floating_point() or value.is_complex():
            finite = bool(torch.isfinite(value).all().item())
        return {
            "type": "Tensor",
            "shape": list(value.shape),
            "dtype": str(value.dtype),
            "device": str(value.device),
            "requires_grad": bool(value.requires_grad),
            "finite": finite,
        }
    if isinstance(value, dict):
        return {str(k): summarize_value(v, depth + 1) for k, v in sorted(value.items(), key=lambda item: str(item[0]))}
    if isinstance(value, (list, tuple)):
        return [summarize_value(v, depth + 1) for v in value[:32]]
    for attr in ["origins", "directions", "pixel_area", "camera_indices", "nears", "fars"]:
        if hasattr(value, attr):
            return {
                "type": type(value).__name__,
                **{name: summarize_value(getattr(value, name), depth + 1) for name in ["origins", "directions", "pixel_area", "camera_indices", "nears", "fars"] if hasattr(value, name)},
            }
    return {"type": type(value).__name__, "repr": repr(value)[:500]}


def scalarize(value: Any) -> Any:
    import torch
    if torch.is_tensor(value):
        if value.numel() == 1:
            return float(value.detach().float().cpu().item())
        return summarize_value(value)
    if isinstance(value, dict):
        return {str(k): scalarize(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [scalarize(v) for v in value]
    if isinstance(value, (int, float, str, bool)) or value is None:
        return value
    return repr(value)


def finite_scalar_mapping(mapping: dict[str, Any]) -> bool:
    import torch
    for value in mapping.values():
        if torch.is_tensor(value):
            if not bool(torch.isfinite(value).all().item()):
                return False
        elif isinstance(value, (float, int)):
            if not math.isfinite(float(value)):
                return False
    return True


def proc_status() -> dict[str, str]:
    result: dict[str, str] = {}
    try:
        for line in Path("/proc/self/status").read_text(encoding="utf-8", errors="replace").splitlines():
            if line.startswith(("VmPeak:", "VmSize:", "VmHWM:", "VmRSS:", "RssAnon:", "RssFile:")):
                key, value = line.split(":", 1)
                result[key] = value.strip()
    except Exception as exc:
        result["error"] = repr(exc)
    return result


def memory_snapshot(label: str) -> dict[str, Any]:
    import torch
    out: dict[str, Any] = {"label": label, "process": proc_status()}
    if not torch.cuda.is_available():
        out["cuda_available"] = False
        return out
    torch.cuda.synchronize()
    free_bytes, total_bytes = torch.cuda.mem_get_info()
    stats = torch.cuda.memory_stats()
    keys = [
        "allocated_bytes.all.current",
        "allocated_bytes.all.peak",
        "reserved_bytes.all.current",
        "reserved_bytes.all.peak",
        "active_bytes.all.current",
        "active_bytes.all.peak",
        "inactive_split_bytes.all.current",
        "num_alloc_retries",
        "num_ooms",
    ]
    out.update({
        "cuda_available": True,
        "device": torch.cuda.get_device_name(0),
        "gcnArchName": getattr(torch.cuda.get_device_properties(0), "gcnArchName", None),
        "free_bytes": int(free_bytes),
        "total_bytes": int(total_bytes),
        "allocated_bytes": int(torch.cuda.memory_allocated()),
        "reserved_bytes": int(torch.cuda.memory_reserved()),
        "max_allocated_bytes": int(torch.cuda.max_memory_allocated()),
        "max_reserved_bytes": int(torch.cuda.max_memory_reserved()),
        "selected_memory_stats": {key: int(stats.get(key, 0)) for key in keys},
    })
    return out


def loaded_tcnn_origins() -> dict[str, str]:
    result: dict[str, str] = {}
    for name, module in sorted(sys.modules.items()):
        if name.startswith(("tinycudann", "tinycudann_bindings")):
            path = getattr(module, "__file__", None)
            if path:
                result[name] = str(Path(path).resolve())
    return result



def configure_trainer(data: Path, output_dir: Path, run_name: str, seed: int, rays: int, checkpoint: Path | None = None) -> tuple[Any, dict[str, Any]]:
    from public_nerfacto_config_v1 import build_public_nerfacto_config
    cfg = copy.deepcopy(build_public_nerfacto_config())
    cfg.data = data
    cfg.output_dir = output_dir
    cfg.experiment_name = run_name
    cfg.timestamp = run_name
    cfg.method_name = "nerfacto"
    cfg.vis = "tensorboard"
    cfg.max_num_iterations = 2
    cfg.steps_per_save = 1
    cfg.save_only_latest_checkpoint = True
    cfg.log_gradients = False
    if hasattr(cfg.viewer, "quit_on_train_completion"):
        cfg.viewer.quit_on_train_completion = True
    if hasattr(cfg.machine, "seed"):
        cfg.machine.seed = seed
    if hasattr(cfg.machine, "num_devices"):
        cfg.machine.num_devices = 1
    if hasattr(cfg.machine, "device_type"):
        cfg.machine.device_type = "cuda"

    dm = cfg.pipeline.datamanager
    dm.data = data
    if hasattr(dm, "dataparser") and hasattr(dm.dataparser, "data"):
        dm.dataparser.data = data
    if hasattr(dm, "train_num_rays_per_batch"):
        dm.train_num_rays_per_batch = rays
    if hasattr(dm, "eval_num_rays_per_batch"):
        dm.eval_num_rays_per_batch = min(rays, 1024)
    if not hasattr(dm, "dataloader_num_workers"):
        raise RuntimeError("pinned ParallelDataManagerConfig lacks dataloader_num_workers")
    dm.dataloader_num_workers = 1
    if not hasattr(dm, "prefetch_factor"):
        raise RuntimeError("pinned ParallelDataManagerConfig lacks prefetch_factor")
    dm.prefetch_factor = 2
    if checkpoint is not None:
        cfg.load_checkpoint = checkpoint
        cfg.load_dir = None
        cfg.load_step = None

    summary = {
        "method_name": cfg.method_name,
        "vis": cfg.vis,
        "mixed_precision": bool(cfg.mixed_precision),
        "use_grad_scaler": bool(cfg.use_grad_scaler),
        "data": str(data),
        "output_dir": str(output_dir),
        "experiment_name": cfg.experiment_name,
        "timestamp": cfg.timestamp,
        "train_num_rays_per_batch": getattr(dm, "train_num_rays_per_batch", None),
        "eval_num_rays_per_batch": getattr(dm, "eval_num_rays_per_batch", None),
        "datamanager_type": type(dm).__name__,
        "datamanager_target": repr(getattr(dm, "_target", None)),
        "dataloader_num_workers": getattr(dm, "dataloader_num_workers", None),
        "prefetch_factor": getattr(dm, "prefetch_factor", None),
        "loader_policy": "PUBLIC_SINGLE_WORKER_FAIL_CLOSED",
        "model_implementation": getattr(cfg.pipeline.model, "implementation", None),
        "load_checkpoint": str(checkpoint) if checkpoint else None,
    }
    return cfg, summary

def gradient_report(module: Any) -> dict[str, Any]:
    import torch
    rows: list[dict[str, Any]] = []
    finite_all = True
    nonzero_count = 0
    total_with_grad = 0
    for name, param in sorted(module.named_parameters()):
        if param.grad is None:
            continue
        total_with_grad += 1
        grad = param.grad.detach()
        finite = bool(torch.isfinite(grad).all().item())
        nonzero = int(torch.count_nonzero(grad).item())
        finite_all &= finite
        nonzero_count += nonzero
        rows.append({
            "name": name,
            "shape": list(grad.shape),
            "dtype": str(grad.dtype),
            "finite": finite,
            "nonzero": nonzero,
            "norm": float(grad.float().norm().cpu().item()) if finite else None,
            "max_abs": float(grad.float().abs().max().cpu().item()) if grad.numel() and finite else None,
        })
    return {
        "passed": bool(total_with_grad > 0 and finite_all and nonzero_count > 0),
        "parameters_with_grad": total_with_grad,
        "total_nonzero_gradient_elements": nonzero_count,
        "all_present_gradients_finite": finite_all,
        "rows": rows,
    }


def parameter_change_report(before: dict[str, str], after: dict[str, str]) -> dict[str, Any]:
    common = sorted(set(before) & set(after))
    changed = [name for name in common if before[name] != after[name]]
    return {
        "passed": bool(common and changed),
        "parameter_count_before": len(before),
        "parameter_count_after": len(after),
        "common_parameter_count": len(common),
        "changed_parameter_count": len(changed),
        "changed_parameters": changed,
        "missing_after": sorted(set(before) - set(after)),
        "new_after": sorted(set(after) - set(before)),
    }


def runtime_identity() -> dict[str, Any]:
    import torch
    import nerfacc.csrc as nerfacc_csrc
    import tinycudann.modules as modules
    from tinycudann.modules import _C
    runtime = Path(os.environ["NERFSTUDIO_RDNA4_A5_TCNN_RUNTIME"]).resolve()
    loaded = loaded_tcnn_origins()
    return {
        "passed": bool(
            torch.__version__ == EXPECTED["torch"]
            and torch.version.hip == EXPECTED["hip"]
            and getattr(torch.cuda.get_device_properties(0), "gcnArchName", None) == EXPECTED["gcn_arch"]
            and sha256(Path(modules.__file__).resolve()) == EXPECTED["tinycudann_modules"]
            and sha256(Path(_C.__file__).resolve()) == EXPECTED["tinycudann_native"]
            and sha256(Path(nerfacc_csrc.__file__).resolve()) == EXPECTED["nerfacc_native"]
            and loaded
            and all(safe_relative(Path(path), runtime) for path in loaded.values())
        ),
        "torch": torch.__version__,
        "hip": torch.version.hip,
        "device": torch.cuda.get_device_name(0),
        "gcnArchName": getattr(torch.cuda.get_device_properties(0), "gcnArchName", None),
        "tinycudann_modules": {"path": str(Path(modules.__file__).resolve()), "sha256": sha256(Path(modules.__file__).resolve())},
        "tinycudann_native": {"path": str(Path(_C.__file__).resolve()), "sha256": sha256(Path(_C.__file__).resolve())},
        "nerfacc_native": {"path": str(Path(nerfacc_csrc.__file__).resolve()), "sha256": sha256(Path(nerfacc_csrc.__file__).resolve())},
        "loaded_tinycudann_modules": loaded,
        "runtime_root": str(runtime),
    }


def install_single_sh_guard() -> None:
    import nerfstudio.field_components.encodings as encodings
    def forbidden_torch_sh(*args: Any, **kwargs: Any) -> Any:
        raise RuntimeError("A5 single-SH policy violation: Nerfstudio Torch SH helper executed")
    encodings.components_from_spherical_harmonics = forbidden_torch_sh


def child_execute(mode: str, root: Path, data: Path, run_dir: Path, seed: int, rays: int, checkpoint: Path | None, output: Path) -> int:
    import torch
    started = time.time()
    report: dict[str, Any] = {
        "schema": SCHEMA,
        "classification": CLASSIFICATION,
        "mode": mode,
        "pid": os.getpid(),
        "ppid": os.getppid(),
        "seed": seed,
        "rays": rays,
        "data": str(data),
        "checkpoint_input": str(checkpoint) if checkpoint else None,
        "passed": False,
    }
    trainer = None
    try:
        seed_everything(seed)
        if not torch.cuda.is_available():
            raise RuntimeError("torch.cuda.is_available() is false")
        torch.cuda.set_device(0)
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        # NERFSTUDIO_RDNA4_P1_QUARANTINE_ORDER_V1_BEGIN
        from public_nerfacto_config_v1 import (
            install_viewer_free_import_quarantine,
        )
        viewer_import_policy = install_viewer_free_import_quarantine()
        report["viewer_import_policy"] = viewer_import_policy
        # NERFSTUDIO_RDNA4_P1_QUARANTINE_ORDER_V1_END
        install_single_sh_guard()
        identity_before = runtime_identity()
        report["runtime_identity_before_setup"] = identity_before
        if not identity_before["passed"]:
            raise RuntimeError("runtime identity/pinning failed before setup")

        output_base = run_dir / mode
        cfg, cfg_summary = configure_trainer(
            data=data,
            output_dir=output_base,
            run_name=f"a5p1_{mode}",
            seed=seed,
            rays=rays,
            checkpoint=checkpoint,
        )
        report["config"] = cfg_summary
        report["memory_before_setup"] = memory_snapshot("before_setup")
        setup_started = time.time()
        trainer = cfg.setup(local_rank=0, world_size=1)
        trainer.setup(test_mode="val")
        torch.cuda.synchronize()
        report["setup_seconds"] = time.time() - setup_started
        report["memory_after_setup"] = memory_snapshot("after_setup")

        dm = trainer.pipeline.datamanager
        train_dataset = getattr(dm, "train_dataset", None)
        eval_dataset = getattr(dm, "eval_dataset", None)
        report["dataset_runtime"] = {
            "train_dataset_type": type(train_dataset).__name__ if train_dataset is not None else None,
            "train_length": len(train_dataset) if train_dataset is not None else None,
            "eval_dataset_type": type(eval_dataset).__name__ if eval_dataset is not None else None,
            "eval_length": len(eval_dataset) if eval_dataset is not None else None,
            "train_rays_per_batch": dm.get_train_rays_per_batch(),
            "datapath": str(dm.get_datapath()) if hasattr(dm, "get_datapath") else None,
        }
        if train_dataset is None or len(train_dataset) <= 0:
            raise RuntimeError("real train dataset is absent or empty")
        train_loader = getattr(dm, "train_ray_dataloader", None)
        report["dataset_runtime"]["dataloader_num_workers"] = getattr(train_loader, "num_workers", None)
        report["dataset_runtime"]["prefetch_factor"] = getattr(train_loader, "prefetch_factor", None)
        report["dataset_runtime"]["single_worker_topology"] = bool(getattr(train_loader, "num_workers", None) == 1)
        if not report["dataset_runtime"]["single_worker_topology"]:
            raise RuntimeError(f"runtime DataLoader worker count is not 1: {getattr(train_loader, 'num_workers', None)}")


        captured: dict[str, Any] = {}
        original_next_train = dm.next_train
        def wrapped_next_train(step: int):
            result = original_next_train(step)
            try:
                ray_bundle, batch = result
                captured["step"] = step
                captured["ray_bundle"] = summarize_value(ray_bundle)
                captured["batch"] = summarize_value(batch)
            except Exception as exc:
                captured["capture_error"] = repr(exc)
            return result
        dm.next_train = wrapped_next_train

        loaded_start_step = int(getattr(trainer, "_start_step", 0))
        report["loaded_start_step"] = loaded_start_step
        expected_step = 0 if mode == "producer" else 1
        if mode == "producer" and loaded_start_step != 0:
            raise RuntimeError(f"producer unexpectedly loaded start step {loaded_start_step}")
        if mode == "reload" and loaded_start_step != 1:
            raise RuntimeError(f"fresh reload expected start step 1, got {loaded_start_step}")

        pipeline_hash_before = canonical_hash(trainer.pipeline.state_dict())
        optimizer_hash_before = canonical_hash({k: v.state_dict() for k, v in trainer.optimizers.optimizers.items()})
        scheduler_hash_before = canonical_hash({k: v.state_dict() for k, v in trainer.optimizers.schedulers.items()})
        scaler_hash_before = canonical_hash(trainer.grad_scaler.state_dict())
        params_before = parameter_hashes(trainer.pipeline)

        if mode == "reload":
            producer_path = run_dir / "producer/producer.json"
            producer = json.loads(producer_path.read_text(encoding="utf-8"))
            expected_state = producer["checkpoint_state_hashes"]
            reload_equivalence = {
                "pipeline_exact": pipeline_hash_before == expected_state["pipeline"],
                "optimizers_exact": optimizer_hash_before == expected_state["optimizers"],
                "schedulers_exact": scheduler_hash_before == expected_state["schedulers"],
                "scaler_exact": scaler_hash_before == expected_state["scaler"],
            }
            reload_equivalence["passed"] = all(reload_equivalence.values())
            report["fresh_reload_exact_state_equivalence"] = reload_equivalence
            if not reload_equivalence["passed"]:
                raise RuntimeError("fresh-process checkpoint state equivalence failed")

        torch.cuda.reset_peak_memory_stats()
        report["memory_before_iteration"] = memory_snapshot("before_iteration")
        iteration_started = time.time()
        loss, loss_dict, metrics_dict = trainer.train_iteration(step=expected_step)
        torch.cuda.synchronize()
        iteration_seconds = time.time() - iteration_started
        report["iteration_seconds"] = iteration_seconds
        report["captured_real_batch"] = captured
        report["loss"] = float(loss.detach().float().cpu().item())
        report["loss_dict"] = scalarize(loss_dict)
        report["metrics_dict"] = scalarize(metrics_dict)
        report["loss_checks"] = {
            "loss_finite": bool(torch.isfinite(loss).all().item()),
            "loss_positive": float(loss.detach().float().cpu().item()) > 0.0,
            "loss_dict_finite": finite_scalar_mapping(loss_dict),
            "metrics_dict_finite": finite_scalar_mapping(metrics_dict),
        }
        report["gradient_report"] = gradient_report(trainer.pipeline)
        params_after = parameter_hashes(trainer.pipeline)
        report["parameter_change"] = parameter_change_report(params_before, params_after)
        report["memory_after_iteration"] = memory_snapshot("after_iteration")

        checkpoint_step = expected_step
        trainer.save_checkpoint(checkpoint_step)
        torch.cuda.synchronize()
        checkpoint_path = trainer.checkpoint_dir / f"step-{checkpoint_step:09d}.ckpt"
        if not checkpoint_path.is_file():
            raise RuntimeError(f"checkpoint was not created: {checkpoint_path}")
        report["checkpoint"] = {
            "path": str(checkpoint_path.resolve()),
            "size_bytes": checkpoint_path.stat().st_size,
            "sha256": sha256(checkpoint_path),
            "step": checkpoint_step,
        }
        report["checkpoint_state_hashes"] = {
            "pipeline": canonical_hash(trainer.pipeline.state_dict()),
            "optimizers": canonical_hash({k: v.state_dict() for k, v in trainer.optimizers.optimizers.items()}),
            "schedulers": canonical_hash({k: v.state_dict() for k, v in trainer.optimizers.schedulers.items()}),
            "scaler": canonical_hash(trainer.grad_scaler.state_dict()),
        }
        identity_after = runtime_identity()
        report["runtime_identity_after_iteration"] = identity_after
        report["memory_final"] = memory_snapshot("final")

        memory = report["memory_after_iteration"]
        memory_ok = bool(
            memory.get("cuda_available")
            and memory.get("max_allocated_bytes", 0) > 0
            and memory.get("max_allocated_bytes", 0) < memory.get("total_bytes", 0)
            and memory.get("selected_memory_stats", {}).get("num_ooms", 0) == 0
        )
        batch_ok = bool(captured.get("ray_bundle") and captured.get("batch") and not captured.get("capture_error"))
        loss_ok = all(report["loss_checks"].values())
        report["checks"] = {
            "runtime_identity": bool(identity_before["passed"] and identity_after["passed"]),
            "real_dataset_loaded": bool(report["dataset_runtime"]["train_length"] and report["dataset_runtime"]["train_length"] > 0),
            "real_batch_captured": batch_ok,
            "forward_loss_finite_positive": loss_ok,
            "backward_gradients_finite_nonzero": report["gradient_report"]["passed"],
            "optimizer_step_changed_parameters": report["parameter_change"]["passed"],
            "memory_telemetry_no_oom": memory_ok,
            "checkpoint_written": report["checkpoint"]["size_bytes"] > 0,
            "fresh_reload_exact_state": report.get("fresh_reload_exact_state_equivalence", {"passed": True})["passed"],
        }
        report["passed"] = all(report["checks"].values())
    except Exception as exc:
        report["error"] = repr(exc)
        report["traceback"] = traceback.format_exc()
        try:
            report["memory_on_error"] = memory_snapshot("on_error")
        except Exception as mem_exc:
            report["memory_on_error"] = {"error": repr(mem_exc)}
    finally:
        report["duration_seconds"] = time.time() - started
        report["finished_at_unix"] = time.time()
        if trainer is not None:
            try:
                trainer.shutdown()
            except Exception as exc:
                report["shutdown_error"] = repr(exc)
        json_dump(output, report)
        print(f"A5P1_CHILD_JSON={output}")
        print(f"A5P1_CHILD_MODE={mode}")
        print(f"A5P1_CHILD_PASSED={'YES' if report.get('passed') else 'NO'}")
    return 0 if report.get("passed") else 2


def create_gate(report: dict[str, Any]) -> str:
    checks = report.get("checks", {})
    ordered = [
        "PUBLIC_A5_P0_MANIFEST_PREREQUISITE",
        "NERFSTUDIO_SOURCE_AND_RUNTIME_PINNING",
        "REAL_NERFSTUDIO_DATASET_STRUCTURE",
        "PRODUCER_FRESH_PROCESS",
        "REAL_DATALOADER_BATCH",
        "REAL_NERFACTO_FORWARD_LOSS",
        "REAL_NERFACTO_BACKWARD_GRADIENTS",
        "REAL_OPTIMIZER_PARAMETER_UPDATE",
        "VRAM_VMM_TELEMETRY_NO_OOM",
        "CHECKPOINT_WRITE_AND_HASH",
        "FRESH_PROCESS_CHECKPOINT_EXACT_RELOAD",
        "FRESH_PROCESS_RESUME_STEP",
        "TINY_RDNA4_NN_SINGLE_SH_AND_NO_MIXED_ORIGINS",
        "CHECKPOINT_RETENTION_POLICY",
        "A5_P1_REAL_MECHANICS",
    ]
    lines = [
        "AMD_NERFSTUDIO_PUBLIC_A5_P1_NERFACTO_SMOKE_V1",
        "",
        f"classification={CLASSIFICATION}",
        f"decision={report.get('decision')}",
        f"run_id={report.get('run_id')}",
        f"dataset={report.get('dataset')}",
        f"rays_per_batch={report.get('rays')}",
        "training_scope=PRODUCER_STEP_0_PLUS_FRESH_RESUME_STEP_1",
        "long_run_training_stability=NOT_CLAIMED",
        "full_shared_venv_global_consistency=NOT_CLAIMED",
        f"checkpoint_policy={report.get('checkpoint_retention', {}).get('policy', 'UNKNOWN')}",
        "",
    ]
    for name in ordered:
        lines.append(f"PUBLIC_RDNA4_{name}: {'PASS' if checks.get(name) else 'FAIL'}")
    lines.extend([
        "",
        "blockers=" + (",".join(report.get("blockers", [])) if report.get("blockers") else "NONE"),
        "",
        "PUBLIC_RDNA4_A5_P1_NERFACTO_SMOKE: PASS" if report.get("passed") else "PUBLIC_RDNA4_A5_P1_NERFACTO_SMOKE: FAIL",
        "PUBLIC_RDNA4_A5_P1_PROCEED_TO_P2: PASS" if report.get("passed") else "PUBLIC_RDNA4_A5_P1_PROCEED_TO_P2: BLOCKED",
    ])
    return "\n".join(lines) + "\n"


def apply_checkpoint_retention(report: dict[str, Any], *, keep_checkpoints: bool) -> dict[str, Any]:
    """Verify and either retain or delete the two temporary P1 checkpoints.

    Failed P1 runs always retain any checkpoint that exists so recovery evidence
    is not destroyed. Successful runs delete verified checkpoints by default;
    ``--keep-checkpoints`` retains them explicitly.
    """
    core_passed = bool(report.get("checks", {}).get("A5_P1_REAL_MECHANICS"))
    requested_policy = "KEEP" if keep_checkpoints else "DELETE_AFTER_VERIFICATION"
    effective_policy = requested_policy if core_passed else "RETAIN_ON_FAILURE"
    rows: list[dict[str, Any]] = []
    passed = True

    for role in ("producer", "reload"):
        checkpoint = report.get(role, {}).get("checkpoint", {})
        raw_path = checkpoint.get("path")
        expected_sha = checkpoint.get("sha256")
        expected_size = checkpoint.get("size_bytes")
        row: dict[str, Any] = {
            "role": role,
            "path": raw_path,
            "expected_sha256": expected_sha,
            "expected_size_bytes": expected_size,
            "action": "NOT_PRESENT",
            "passed": False,
        }
        if not raw_path:
            row["error"] = "CHECKPOINT_PATH_MISSING_FROM_REPORT"
            rows.append(row)
            passed = False
            continue

        path = Path(raw_path)
        if not path.is_file():
            row["error"] = "CHECKPOINT_FILE_MISSING"
            rows.append(row)
            passed = False
            continue

        observed_sha = sha256(path)
        observed_size = path.stat().st_size
        row.update({
            "observed_sha256": observed_sha,
            "observed_size_bytes": observed_size,
            "hash_matches": observed_sha == expected_sha,
            "size_matches": observed_size == expected_size,
        })
        if not row["hash_matches"] or not row["size_matches"]:
            row["action"] = "RETAINED_MISMATCH"
            row["error"] = "CHECKPOINT_VERIFICATION_FAILED"
            rows.append(row)
            passed = False
            continue

        if effective_policy == "DELETE_AFTER_VERIFICATION":
            try:
                path.chmod(0o600)
                path.unlink()
                row["action"] = "DELETED_AFTER_VERIFICATION"
                row["deleted"] = not path.exists()
                row["passed"] = bool(row["deleted"])
            except OSError as exc:
                row["action"] = "DELETE_FAILED"
                row["error"] = repr(exc)
                row["passed"] = False
        else:
            row["action"] = "RETAINED"
            row["retained"] = path.is_file()
            row["passed"] = bool(row["retained"])

        rows.append(row)
        passed = passed and bool(row["passed"])

    return {
        "requested_policy": requested_policy,
        "policy": effective_policy,
        "core_mechanics_passed": core_passed,
        "files": rows,
        "passed": passed,
    }


def chmod_tree_readonly(root: Path) -> None:
    for path in sorted(root.rglob("*"), reverse=True):
        try:
            if path.is_file():
                path.chmod(0o444)
            elif path.is_dir():
                path.chmod(0o555)
        except OSError:
            pass
    try:
        root.chmod(0o555)
    except OSError:
        pass



def orchestrate(args: argparse.Namespace) -> int:
    output_root = args.output_root.expanduser().resolve()
    python = absolute_preserving_symlink(args.python)
    dataset = args.data.expanduser().resolve()
    nerfstudio = args.nerfstudio_worktree.expanduser().resolve()
    runtime = args.tcnn_runtime.expanduser().resolve()
    preflight_dir = args.preflight_run_dir.expanduser().resolve()
    run_id = args.run_id or time.strftime("%Y%m%dT%H%M%SZ", time.gmtime()) + f"_{os.getpid()}"
    run_dir = output_root / "public_a5p1_nerfacto_smoke_v1" / run_id
    if run_dir.exists():
        raise RuntimeError(f"run directory already exists: {run_dir}")
    run_dir.mkdir(parents=True, exist_ok=False)

    torch_lib = python.parent.parent / "lib/python3.12/site-packages/torch/lib"
    child_env = os.environ.copy()
    child_env.pop("TORCH_FORCE_WEIGHTS_ONLY_LOAD", None)
    child_env["PYTHONNOUSERSITE"] = "1"
    child_env["PYTHONDONTWRITEBYTECODE"] = "1"
    child_env["TCNN_RDNA4_ENABLE_PORTABLE_MLP_SHIM"] = "1"
    child_env["NERFSTUDIO_RDNA4_A5_SINGLE_SH_POLICY"] = "TINY_RDNA4_NN_ONLY"
    child_env["NERFSTUDIO_RDNA4_A5_TCNN_RUNTIME"] = str(runtime)
    child_env["NERFSTUDIO_RDNA4_A5_NERFSTUDIO_WORKTREE"] = str(nerfstudio)
    child_env["TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD"] = "1"
    child_env["PYTHONPATH"] = os.pathsep.join([str(runtime), str(nerfstudio)])
    current_ld = child_env.get("LD_LIBRARY_PATH", "")
    child_env["LD_LIBRARY_PATH"] = os.pathsep.join([str(torch_lib), "/opt/rocm/lib", "/opt/rocm/lib64"] + ([current_ld] if current_ld else []))

    old_env = os.environ.copy()
    os.environ.update(child_env)
    try:
        prereq = prerequisite_report(preflight_dir, dataset, nerfstudio, runtime, python)
    finally:
        os.environ.clear(); os.environ.update(old_env)
    json_dump(run_dir / "prerequisites.json", prereq)

    report: dict[str, Any] = {
        "schema": SCHEMA,
        "classification": CLASSIFICATION,
        "run_id": run_id,
        "output_root": str(output_root),
        "dataset": str(dataset),
        "rays": args.rays,
        "seed": args.seed,
        "run_dir": str(run_dir),
        "prerequisites": prereq,
        "passed": False,
        "decision": "PUBLIC_A5_P1_BLOCKED",
        "processes": {},
        "nonclaims": [
            "LONG_RUN_TRAINING_STABILITY_OR_MEMORY_LEAK_FREEDOM",
            "MULTI_GPU_OR_DISTRIBUTED_TRAINING",
            "VIEWER_EVAL_EXPORT_AND_UNUSED_OPTIONAL_FEATURES",
            "A5_P2_SUSTAINED_TRAINING_QUALIFICATION",
        ],
    }
    if not prereq.get("passed"):
        report["blockers"] = ["PUBLIC_A5_P0_OR_SOURCE_RUNTIME_DATASET_PREREQUISITE"]
        report["checks"] = {"PUBLIC_A5_P0_MANIFEST_PREREQUISITE": False}
        json_dump(run_dir / "final_aggregate.json", report)
        (run_dir / "final_gate.txt").write_text(create_gate(report), encoding="utf-8")
        print(json.dumps(report, indent=2, sort_keys=True)); print(create_gate(report))
        return 2

    common = [
        str(python), str(Path(__file__).resolve()),
        "--root", str(output_root), "--python", str(python), "--data", str(dataset),
        "--run-dir", str(run_dir), "--seed", str(args.seed), "--rays", str(args.rays),
    ]
    producer_json = run_dir / "producer/producer.json"
    producer = run_command(common + ["--mode", "producer", "--child-output", str(producer_json)], cwd=Path("/tmp"), env=child_env, timeout=args.timeout)
    producer_log = run_dir / "producer/producer_process.log"; producer_log.parent.mkdir(parents=True, exist_ok=True)
    producer_log.write_text(producer.get("stdout", "") + "\n--- STDERR ---\n" + producer.get("stderr", ""), encoding="utf-8")
    report["processes"]["producer"] = {**producer, "log": str(producer_log), "json": str(producer_json)}
    producer_payload = json.loads(producer_json.read_text(encoding="utf-8")) if producer_json.is_file() else {}
    report["producer"] = producer_payload

    reload_payload: dict[str, Any] = {}
    if producer.get("returncode") == 0 and producer_payload.get("passed") and producer_payload.get("checkpoint", {}).get("path"):
        checkpoint = Path(producer_payload["checkpoint"]["path"])
        reload_json = run_dir / "reload/reload.json"
        reload_result = run_command(common + ["--mode", "reload", "--checkpoint", str(checkpoint), "--child-output", str(reload_json)], cwd=Path("/tmp"), env=child_env, timeout=args.timeout)
        reload_log = run_dir / "reload/reload_process.log"; reload_log.parent.mkdir(parents=True, exist_ok=True)
        reload_log.write_text(reload_result.get("stdout", "") + "\n--- STDERR ---\n" + reload_result.get("stderr", ""), encoding="utf-8")
        report["processes"]["reload"] = {**reload_result, "log": str(reload_log), "json": str(reload_json)}
        reload_payload = json.loads(reload_json.read_text(encoding="utf-8")) if reload_json.is_file() else {}
        report["reload"] = reload_payload
    else:
        report["processes"]["reload"] = {"skipped": True, "reason": "producer failed"}

    producer_pid = producer_payload.get("pid"); reload_pid = reload_payload.get("pid")
    producer_checks = producer_payload.get("checks", {}); reload_checks = reload_payload.get("checks", {})
    exact_reload = reload_payload.get("fresh_reload_exact_state_equivalence", {}).get("passed") is True
    origins_ok = bool(producer_payload.get("runtime_identity_after_iteration", {}).get("passed") and reload_payload.get("runtime_identity_after_iteration", {}).get("passed"))
    checks = {
        "PUBLIC_A5_P0_MANIFEST_PREREQUISITE": bool(prereq.get("p0_hash_chain", {}).get("passed") and prereq.get("p0_semantics", {}).get("passed")),
        "NERFSTUDIO_SOURCE_AND_RUNTIME_PINNING": bool(prereq.get("nerfstudio_source", {}).get("passed") and prereq.get("runtime_anchors", {}).get("passed")),
        "REAL_NERFSTUDIO_DATASET_STRUCTURE": bool(prereq.get("dataset", {}).get("passed")),
        "PRODUCER_FRESH_PROCESS": bool(producer.get("returncode") == 0 and producer_pid and producer_pid != os.getpid()),
        "REAL_DATALOADER_BATCH": bool(producer_checks.get("real_batch_captured") and reload_checks.get("real_batch_captured")),
        "REAL_NERFACTO_FORWARD_LOSS": bool(producer_checks.get("forward_loss_finite_positive") and reload_checks.get("forward_loss_finite_positive")),
        "REAL_NERFACTO_BACKWARD_GRADIENTS": bool(producer_checks.get("backward_gradients_finite_nonzero") and reload_checks.get("backward_gradients_finite_nonzero")),
        "REAL_OPTIMIZER_PARAMETER_UPDATE": bool(producer_checks.get("optimizer_step_changed_parameters") and reload_checks.get("optimizer_step_changed_parameters")),
        "VRAM_VMM_TELEMETRY_NO_OOM": bool(producer_checks.get("memory_telemetry_no_oom") and reload_checks.get("memory_telemetry_no_oom")),
        "CHECKPOINT_WRITE_AND_HASH": bool(producer_checks.get("checkpoint_written") and producer_payload.get("checkpoint", {}).get("sha256")),
        "FRESH_PROCESS_CHECKPOINT_EXACT_RELOAD": bool(reload_pid and reload_pid != producer_pid and exact_reload),
        "FRESH_PROCESS_RESUME_STEP": bool(reload_payload.get("passed") and reload_payload.get("loaded_start_step") == 1 and reload_payload.get("checkpoint", {}).get("step") == 1),
        "TINY_RDNA4_NN_SINGLE_SH_AND_NO_MIXED_ORIGINS": origins_ok,
    }
    checks["A5_P1_REAL_MECHANICS"] = all(checks.values())
    report["checks"] = checks
    report["checkpoint_retention"] = apply_checkpoint_retention(report, keep_checkpoints=args.keep_checkpoints)
    checks["CHECKPOINT_RETENTION_POLICY"] = bool(report["checkpoint_retention"]["passed"])
    checks["A5_P1_REAL_MECHANICS"] = all(
        passed for name, passed in checks.items() if name != "A5_P1_REAL_MECHANICS"
    )
    blockers = [name for name, passed in checks.items() if not passed and name != "A5_P1_REAL_MECHANICS"]
    report.update({"checks": checks, "blockers": blockers, "passed": checks["A5_P1_REAL_MECHANICS"], "decision": "PROCEED_TO_PUBLIC_A5_P2" if checks["A5_P1_REAL_MECHANICS"] else "PUBLIC_A5_P1_BLOCKED"})
    report["process_identity"] = {"orchestrator_pid": os.getpid(), "producer_pid": producer_pid, "reload_pid": reload_pid, "producer_and_reload_distinct": bool(producer_pid and reload_pid and producer_pid != reload_pid)}

    final_json = run_dir / "final_aggregate.json"; final_gate = run_dir / "final_gate.txt"
    json_dump(final_json, report); final_gate.write_text(create_gate(report), encoding="utf-8")
    manifest = {"schema": SCHEMA + "-manifest", "run_id": run_id, "files": {}}
    for path in sorted(run_dir.rglob("*")):
        if path.is_file() and path.name != "MANIFEST.json":
            manifest["files"][str(path.relative_to(run_dir))] = {"sha256": sha256(path), "size_bytes": path.stat().st_size}
    json_dump(run_dir / "MANIFEST.json", manifest)
    (output_root / "public_a5p1_nerfacto_smoke_v1.latest").write_text(str(run_dir) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True)); print(create_gate(report)); print(f"PUBLIC_A5P1_RUN_DIR={run_dir}")
    chmod_tree_readonly(run_dir)
    return 0 if report["passed"] else 2

def self_test() -> int:
    import torch
    payload = {
        "tensor": torch.arange(12, dtype=torch.float32).reshape(3, 4),
        "nested": {"a": [1, 2, 3], "b": "x"},
    }
    h1 = canonical_hash(payload)
    h2 = canonical_hash(copy.deepcopy(payload))
    before = {"a": "1", "b": "2"}
    after = {"a": "1", "b": "3"}
    ok = h1 == h2 and parameter_change_report(before, after)["passed"]
    print(json.dumps({"passed": ok, "canonical_hash": h1}, indent=2))
    return 0 if ok else 2



def main() -> int:
    parser = argparse.ArgumentParser(description="Public P1 real Nerfacto mechanics and fresh-process checkpoint smoke")
    parser.add_argument("--mode", choices=["orchestrate", "producer", "reload", "self-test"], default="orchestrate")
    parser.add_argument("--root", type=Path)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--preflight-run-dir", type=Path)
    parser.add_argument("--nerfstudio-worktree", type=Path)
    parser.add_argument("--tcnn-runtime", type=Path)
    parser.add_argument("--python", type=Path)
    parser.add_argument("--data", type=Path)
    parser.add_argument("--run-dir", type=Path)
    parser.add_argument("--run-id")
    parser.add_argument("--seed", type=int, default=20260803)
    parser.add_argument("--rays", type=int, default=1024)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--child-output", type=Path)
    parser.add_argument("--timeout", type=int, default=1800)
    parser.add_argument("--keep-checkpoints", action="store_true", help="retain temporary producer and reload checkpoints after successful verification")
    args = parser.parse_args()
    if args.mode == "self-test":
        return self_test()
    if args.data is None or args.python is None:
        parser.error("all run modes require --data and --python")
    if args.rays <= 0:
        parser.error("--rays must be positive")
    if args.mode in {"producer", "reload"}:
        if args.root is None or args.run_dir is None or args.child_output is None:
            parser.error("child modes require --root, --run-dir and --child-output")
        if args.mode == "reload" and args.checkpoint is None:
            parser.error("reload mode requires --checkpoint")
        return child_execute(
            mode=args.mode,
            root=args.root.expanduser().resolve(),
            data=args.data.expanduser().resolve(),
            run_dir=args.run_dir.expanduser().resolve(),
            seed=args.seed,
            rays=args.rays,
            checkpoint=args.checkpoint.expanduser().resolve() if args.checkpoint else None,
            output=args.child_output.expanduser().resolve(),
        )
    required = [args.output_root, args.preflight_run_dir, args.nerfstudio_worktree, args.tcnn_runtime]
    if any(value is None for value in required):
        parser.error("orchestrate mode requires --output-root, --preflight-run-dir, --nerfstudio-worktree and --tcnn-runtime")
    return orchestrate(args)


if __name__ == "__main__":
    raise SystemExit(main())
