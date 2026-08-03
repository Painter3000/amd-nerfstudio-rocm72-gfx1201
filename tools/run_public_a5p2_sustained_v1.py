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
import statistics
import subprocess
import sys
sys.dont_write_bytecode = True
import time
import traceback
from typing import Any, Iterable

from public_toolchain_common import absolute_preserving_symlink, inspect_dataset as inspect_dataset_common, verify_manifest

SCHEMA = "amd-nerfstudio-public-a5p2-sustained-v1"
CLASSIFICATION = "PUBLIC_A5_P2_SUSTAINED_REAL_TRAINING_AND_RESUME_TRAJECTORY_V1"

EXPECTED = {
    'nerfstudio_commit': '50e0e3c70c775e89333256213363badbf074f29d',
    'nerfstudio_tree': '9d5ff468eeff89b66995e9984acaa378c37dc07e',
    'nerfstudio_mlp': '4939a5a6901d82d8e310d93e2a135ca57ccc1bd79be79a7f67e2740e730c44ad',
    'tinycudann_native': '883f89efdad7bb909a4a3899ab79b2defe9713fdb5c7cf22cf4882c626b3efc4',
    'tinycudann_modules': 'b4df43b54f64fe2b31272a997aafd50137aecac411d59b05251acedcd5512d12',
    'nerfacc_native': 'd3beee150cfa3a9ad3038a3283ff0a46953c345634d8cb6109449c5e3d04d1e2',
    'torch': '2.13.0+rocm7.2',
    'hip': '7.2.53211',
    'gcn_arch': 'gfx1201',
}

# These limits are fixed before the qualification run.
TOLERANCE_POLICY = {
    "reference_multiplier": 4.0,
    "state": {
        "fp16_bf16": {"max_abs_floor": 2.0e-3, "relative_l2_floor": 5.0e-4},
        "fp32": {"max_abs_floor": 1.0e-4, "relative_l2_floor": 1.0e-5},
        "fp64": {"max_abs_floor": 1.0e-8, "relative_l2_floor": 1.0e-9},
    },
    "scalar_trajectory": {"max_abs_floor": 1.0e-4, "relative_rmse_floor": 2.0e-3},
    "memory": {
        "allocated_slope_bytes_per_step_max": 524288.0,
        "allocated_drift_bytes_max": 134217728.0,
        "reserved_slope_bytes_per_step_max": 1048576.0,
        "reserved_drift_bytes_max": 268435456.0,
        "rss_slope_bytes_per_step_max": 2097152.0,
        "rss_drift_bytes_max": 536870912.0,
    },
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
    argv: list[str], *, cwd: Path | None = None, env: dict[str, str] | None = None, timeout: int = 7200
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


def file_anchor(path: Path, expected: str) -> dict[str, Any]:
    observed = sha256(path) if path.is_file() else None
    return {
        "path": str(path),
        "exists": path.is_file(),
        "sha256": observed,
        "expected_sha256": expected,
        "hash_matches": observed == expected,
    }


def resolve_image_path(dataset_dir: Path, raw: str) -> Path | None:
    candidate = Path(raw)
    if not candidate.is_absolute():
        candidate = dataset_dir / candidate
    options = [candidate]
    if candidate.suffix == "":
        options.extend(candidate.with_suffix(ext) for ext in [".png", ".jpg", ".jpeg", ".JPG", ".PNG", ".JPEG"])
    for option in options:
        if option.is_file():
            return option.resolve()
    return None


def inspect_dataset(data: Path) -> dict[str, Any]:
    return inspect_dataset_common(data)


def prerequisite_report(p1_run_dir: Path, dataset: Path, nerfstudio: Path, runtime: Path, python: Path) -> dict[str, Any]:
    p1_manifest = verify_manifest(p1_run_dir)
    try:
        p1_payload = json.loads((p1_run_dir / "final_aggregate.json").read_text(encoding="utf-8"))
    except Exception as exc:
        p1_payload = {"passed": False, "error": repr(exc)}
    p1_semantics = {
        "passed": bool(
            p1_manifest.get("passed")
            and p1_payload.get("passed") is True
            and p1_payload.get("decision") == "PROCEED_TO_PUBLIC_A5_P2"
            and not p1_payload.get("blockers")
        ),
        "decision": p1_payload.get("decision"),
        "blockers": p1_payload.get("blockers"),
        "run_id": p1_payload.get("run_id"),
        "classification": p1_payload.get("classification"),
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
    p1_prereq = p1_payload.get("prerequisites", {})
    runtime_anchors = {
        "passed": bool(p1_prereq.get("runtime_anchors", {}).get("passed")),
        "source": "PUBLIC_A5P1_MANIFEST_VERIFIED_PREREQUISITE",
        "runtime_root": str(runtime),
        "python": str(python),
        "details": p1_prereq.get("runtime_anchors", {}),
    }
    dataset_report = inspect_dataset(dataset)
    p1_dataset = p1_payload.get("dataset")
    dataset_report["same_dataset_path_as_p1"] = bool(p1_dataset and Path(p1_dataset).resolve() == dataset.resolve())
    dataset_report["pinned_passed"] = bool(dataset_report.get("passed") and dataset_report["same_dataset_path_as_p1"])
    env_policy = {
        "passed": bool(
            os.environ.get("PYTHONNOUSERSITE") == "1"
            and os.environ.get("TCNN_RDNA4_ENABLE_PORTABLE_MLP_SHIM") == "1"
            and os.environ.get("NERFSTUDIO_RDNA4_A5_SINGLE_SH_POLICY") == "TINY_RDNA4_NN_ONLY"
            and Path(os.environ.get("NERFSTUDIO_RDNA4_A5_TCNN_RUNTIME", "/nonexistent")).resolve() == runtime.resolve()
            and Path(os.environ.get("NERFSTUDIO_RDNA4_A5_NERFSTUDIO_WORKTREE", "/nonexistent")).resolve() == nerfstudio.resolve()
            and os.environ.get("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD") == "1"
            and not os.environ.get("TORCH_FORCE_WEIGHTS_ONLY_LOAD")
        )
    }
    passed = bool(p1_semantics["passed"] and source["passed"] and runtime_anchors["passed"] and dataset_report["pinned_passed"] and env_policy["passed"])
    return {
        "passed": passed,
        "p1_hash_chain": p1_manifest,
        "p1_semantics": p1_semantics,
        "nerfstudio_source": source,
        "runtime_anchors": runtime_anchors,
        "dataset": dataset_report,
        "environment_policy": env_policy,
        "paths": {"p1_run_dir": str(p1_run_dir), "nerfstudio_worktree": str(nerfstudio), "tcnn_runtime": str(runtime), "python": str(python)},
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


def scalarize(value: Any) -> Any:
    import torch
    if torch.is_tensor(value):
        if value.numel() == 1:
            return float(value.detach().float().cpu().item())
        return {"type": "Tensor", "shape": list(value.shape), "dtype": str(value.dtype)}
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
        elif isinstance(value, (float, int)) and not math.isfinite(float(value)):
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


def parse_kib(value: str | None) -> int | None:
    if not value:
        return None
    parts = value.split()
    try:
        return int(parts[0]) * 1024
    except Exception:
        return None


def memory_snapshot(label: str, step: int | None = None) -> dict[str, Any]:
    import torch
    out: dict[str, Any] = {"label": label, "step": step, "process": proc_status()}
    out["rss_bytes"] = parse_kib(out["process"].get("VmRSS"))
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


def percentile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * q
    low = int(math.floor(position))
    high = int(math.ceil(position))
    if low == high:
        return ordered[low]
    fraction = position - low
    return ordered[low] * (1.0 - fraction) + ordered[high] * fraction


def latency_summary(records: list[dict[str, Any]], warmup: int, start_step: int) -> dict[str, Any]:
    cutoff = start_step + warmup
    values = [float(row["iteration_seconds"]) for row in records if row["step"] >= cutoff and math.isfinite(float(row["iteration_seconds"]))]
    return {
        "warmup_excluded_steps": warmup,
        "sample_count": len(values),
        "minimum": min(values) if values else None,
        "median": statistics.median(values) if values else None,
        "p90": percentile(values, 0.90),
        "p95": percentile(values, 0.95),
        "p99": percentile(values, 0.99),
        "maximum": max(values) if values else None,
        "performance_gate": "NOT_APPLIED_IN_A5_P2",
    }


def theil_sen_slope(points: list[tuple[float, float]]) -> float | None:
    slopes: list[float] = []
    for i in range(len(points)):
        x1, y1 = points[i]
        for j in range(i + 1, len(points)):
            x2, y2 = points[j]
            if x2 != x1:
                slopes.append((y2 - y1) / (x2 - x1))
    return statistics.median(slopes) if slopes else None


def drift_median(points: list[tuple[float, float]], window: int = 3) -> float | None:
    if len(points) < 2:
        return None
    k = min(window, max(1, len(points) // 3))
    return statistics.median(y for _, y in points[-k:]) - statistics.median(y for _, y in points[:k])


def memory_trend(telemetry: list[dict[str, Any]], start_step: int, warmup: int) -> dict[str, Any]:
    cutoff = start_step + warmup
    filtered = [row for row in telemetry if isinstance(row.get("step"), int) and row["step"] >= cutoff]
    def metric(name: str) -> dict[str, Any]:
        pts = [(float(row["step"]), float(row[name])) for row in filtered if row.get(name) is not None]
        return {"points": len(pts), "theil_sen_slope_bytes_per_step": theil_sen_slope(pts), "median_endpoint_drift_bytes": drift_median(pts)}
    allocated = metric("allocated_bytes")
    reserved = metric("reserved_bytes")
    rss = metric("rss_bytes")
    limits = TOLERANCE_POLICY["memory"]
    def below(value: float | None, limit: float) -> bool:
        return value is not None and value <= limit
    allocated["passed"] = below(allocated["theil_sen_slope_bytes_per_step"], limits["allocated_slope_bytes_per_step_max"]) and below(allocated["median_endpoint_drift_bytes"], limits["allocated_drift_bytes_max"])
    reserved["passed"] = below(reserved["theil_sen_slope_bytes_per_step"], limits["reserved_slope_bytes_per_step_max"]) and below(reserved["median_endpoint_drift_bytes"], limits["reserved_drift_bytes_max"])
    rss["passed"] = below(rss["theil_sen_slope_bytes_per_step"], limits["rss_slope_bytes_per_step_max"]) and below(rss["median_endpoint_drift_bytes"], limits["rss_drift_bytes_max"])
    return {"cutoff_step": cutoff, "telemetry_points": len(filtered), "allocated": allocated, "reserved": reserved, "rss": rss, "policy": limits, "gpu_passed": allocated["passed"] and reserved["passed"], "host_passed": rss["passed"]}


def loaded_tcnn_origins() -> dict[str, str]:
    result: dict[str, str] = {}
    for name, module in sorted(sys.modules.items()):
        if name.startswith(("tinycudann", "tinycudann_bindings")):
            path = getattr(module, "__file__", None)
            if path:
                result[name] = str(Path(path).resolve())
    return result


def runtime_identity() -> dict[str, Any]:
    import torch
    import nerfacc.csrc as nerfacc_csrc
    import tinycudann.modules as modules
    from tinycudann.modules import _C
    runtime = Path(os.environ["NERFSTUDIO_RDNA4_A5_TCNN_RUNTIME"]).resolve()
    loaded = loaded_tcnn_origins()
    passed = bool(
        torch.__version__ == EXPECTED["torch"]
        and torch.version.hip == EXPECTED["hip"]
        and getattr(torch.cuda.get_device_properties(0), "gcnArchName", None) == EXPECTED["gcn_arch"]
        and sha256(Path(modules.__file__).resolve()) == EXPECTED["tinycudann_modules"]
        and sha256(Path(_C.__file__).resolve()) == EXPECTED["tinycudann_native"]
        and sha256(Path(nerfacc_csrc.__file__).resolve()) == EXPECTED["nerfacc_native"]
        and loaded
        and all(safe_relative(Path(path), runtime) for path in loaded.values())
    )
    return {
        "passed": passed,
        "torch": torch.__version__, "hip": torch.version.hip,
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


def configure_trainer(data: Path, output_dir: Path, run_name: str, seed: int, rays: int, end_step: int, checkpoint: Path | None) -> tuple[Any, dict[str, Any]]:
    from nerfstudio.configs.method_configs import all_methods
    cfg = copy.deepcopy(all_methods["nerfacto"])
    cfg.data = data
    cfg.output_dir = output_dir
    cfg.experiment_name = run_name
    cfg.timestamp = run_name
    cfg.method_name = "nerfacto"
    cfg.vis = "tensorboard"
    cfg.max_num_iterations = end_step
    cfg.steps_per_save = end_step + 1
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
    # Pinned Nerfstudio commit 50e0e3c uses ParallelDataManagerConfig.dataloader_num_workers.
    # The v1 runner probed obsolete field names, leaving the default of 4 workers active.
    # With the six-image P1 dataset, ceil(6/4)=2 leaves worker 3 with an empty slice.
    if not hasattr(dm, "dataloader_num_workers"):
        raise RuntimeError("pinned ParallelDataManagerConfig lacks dataloader_num_workers")
    dm.dataloader_num_workers = 1
    if hasattr(dm, "prefetch_factor"):
        dm.prefetch_factor = 2
    if checkpoint is not None:
        cfg.load_checkpoint = checkpoint
        cfg.load_dir = None
        cfg.load_step = None
    summary = {
        "method_name": cfg.method_name,
        "data": str(data),
        "output_dir": str(output_dir),
        "run_name": run_name,
        "max_num_iterations": cfg.max_num_iterations,
        "train_num_rays_per_batch": getattr(dm, "train_num_rays_per_batch", None),
        "eval_num_rays_per_batch": getattr(dm, "eval_num_rays_per_batch", None),
        "datamanager_type": type(dm).__name__,
        "datamanager_target": repr(getattr(dm, "_target", None)),
        "dataloader_num_workers": getattr(dm, "dataloader_num_workers", None),
        "prefetch_factor": getattr(dm, "prefetch_factor", None),
        "loader_policy": "SINGLE_WORKER_FOR_SIX_IMAGE_QUALIFICATION_DATASET",
        "model_implementation": getattr(cfg.pipeline.model, "implementation", None),
        "load_checkpoint": str(checkpoint) if checkpoint else None,
        "mixed_precision": bool(cfg.mixed_precision),
        "use_grad_scaler": bool(cfg.use_grad_scaler),
    }
    return cfg, summary


def batch_fingerprint(batch: dict[str, Any]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key in ["indices", "image"]:
        value = batch.get(key)
        if value is not None:
            try:
                result[key] = {"sha256": tensor_hash(value), "shape": list(value.shape), "dtype": str(value.dtype)}
            except Exception as exc:
                result[key] = {"error": repr(exc)}
    result["combined_sha256"] = canonical_hash(result)
    return result


def gradient_probe(module: Any) -> dict[str, Any]:
    import torch
    finite_all = True
    nonzero = 0
    with_grad = 0
    for _, param in module.named_parameters():
        if param.grad is None:
            continue
        with_grad += 1
        grad = param.grad.detach()
        finite = bool(torch.isfinite(grad).all().item())
        finite_all &= finite
        if finite:
            nonzero += int(torch.count_nonzero(grad).item())
    return {"passed": bool(with_grad > 0 and finite_all and nonzero > 0), "parameters_with_grad": with_grad, "total_nonzero_gradient_elements": nonzero, "all_present_gradients_finite": finite_all}


def parameter_finite_probe(module: Any) -> dict[str, Any]:
    import torch
    failures: list[str] = []
    count = 0
    for name, param in module.named_parameters():
        count += 1
        if (param.is_floating_point() or param.is_complex()) and not bool(torch.isfinite(param.detach()).all().item()):
            failures.append(name)
    return {"passed": not failures, "parameter_count": count, "nonfinite_parameters": failures}


def capture_rng_state(path: Path) -> dict[str, Any]:
    import numpy as np
    import torch
    payload = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
        "torch_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)
    return {"path": str(path.resolve()), "sha256": sha256(path), "size_bytes": path.stat().st_size}


def restore_rng_state(path: Path) -> dict[str, Any]:
    import numpy as np
    import torch
    payload = torch.load(path, map_location="cpu", weights_only=False)
    random.setstate(payload["python"])
    np.random.set_state(payload["numpy"])
    torch.set_rng_state(payload["torch_cpu"])
    if torch.cuda.is_available() and payload.get("torch_cuda"):
        torch.cuda.set_rng_state_all(payload["torch_cuda"])
    return {"path": str(path.resolve()), "sha256": sha256(path), "restored": True}


def child_series(args: argparse.Namespace) -> int:
    import torch
    started = time.time()
    report: dict[str, Any] = {
        "schema": SCHEMA,
        "classification": CLASSIFICATION,
        "series_name": args.series_name,
        "pid": os.getpid(), "ppid": os.getppid(),
        "seed": args.seed, "rays": args.rays,
        "start_step_argument": args.start_step, "end_step": args.end_step,
        "checkpoint_input": str(args.checkpoint) if args.checkpoint else None,
        "replay_batches": args.replay_batches,
        "passed": False,
    }
    trainer = None
    try:
        seed_everything(args.seed)
        if not torch.cuda.is_available():
            raise RuntimeError("torch.cuda.is_available() is false")
        torch.cuda.set_device(0)
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        install_single_sh_guard()
        identity_before = runtime_identity()
        report["runtime_identity_before_setup"] = identity_before
        if not identity_before["passed"]:
            raise RuntimeError("runtime identity failed before setup")

        cfg, config_summary = configure_trainer(
            args.data, args.output_base, args.series_name, args.seed, args.rays, args.end_step, args.checkpoint
        )
        report["config"] = config_summary
        report["memory_before_setup"] = memory_snapshot("before_setup")
        setup_started = time.time()
        trainer = cfg.setup(local_rank=0, world_size=1)
        trainer.setup(test_mode="val")
        torch.cuda.synchronize()
        report["setup_seconds"] = time.time() - setup_started
        report["memory_after_setup"] = memory_snapshot("after_setup")

        loaded_start = int(getattr(trainer, "_start_step", 0))
        report["loaded_start_step"] = loaded_start
        if loaded_start != args.start_step:
            raise RuntimeError(f"expected loaded start step {args.start_step}, got {loaded_start}")
        dm = trainer.pipeline.datamanager
        train_dataset = getattr(dm, "train_dataset", None)
        report["dataset_runtime"] = {
            "train_dataset_type": type(train_dataset).__name__ if train_dataset is not None else None,
            "train_length": len(train_dataset) if train_dataset is not None else None,
            "train_rays_per_batch": dm.get_train_rays_per_batch(),
            "datapath": str(dm.get_datapath()) if hasattr(dm, "get_datapath") else None,
        }
        if train_dataset is None or len(train_dataset) <= 0:
            raise RuntimeError("real train dataset is absent or empty")
        train_loader = getattr(dm, "train_ray_dataloader", None)
        loader_workers = getattr(train_loader, "num_workers", None)
        loader_prefetch = getattr(train_loader, "prefetch_factor", None)
        report["dataloader_runtime_topology"] = {
            "loader_type": type(train_loader).__name__ if train_loader is not None else None,
            "num_workers": loader_workers,
            "prefetch_factor": loader_prefetch,
            "dataset_length": len(train_dataset),
            "passed": train_loader is not None and loader_workers == 1,
        }
        if not report["dataloader_runtime_topology"]["passed"]:
            raise RuntimeError(
                f"expected one runtime dataloader worker, got {loader_workers!r}"
            )

        state_before = {
            "pipeline": canonical_hash(trainer.pipeline.state_dict()),
            "optimizers": canonical_hash({k: v.state_dict() for k, v in trainer.optimizers.optimizers.items()}),
            "schedulers": canonical_hash({k: v.state_dict() for k, v in trainer.optimizers.schedulers.items()}),
            "scalers": canonical_hash(trainer.grad_scaler.state_dict()),
        }
        report["state_hashes_before_training"] = state_before
        if args.expected_state_json:
            expected_payload = json.loads(args.expected_state_json.read_text(encoding="utf-8"))
            expected = expected_payload["checkpoint_state_hashes"]
            exact = {key + "_exact": state_before[key] == expected[key] for key in ["pipeline", "optimizers", "schedulers", "scalers"]}
            exact["passed"] = all(exact.values())
            report["fresh_reload_exact_state_equivalence"] = exact
            if not exact["passed"]:
                raise RuntimeError("fresh reload exact state equivalence failed")

        original_next_train = dm.next_train
        replay_fingerprints: list[dict[str, Any]] = []
        if args.replay_batches:
            for replay_step in range(args.replay_batches):
                _, batch = original_next_train(replay_step)
                replay_fingerprints.append({"step": replay_step, "batch_fingerprint": batch_fingerprint(batch)})
            report["replay_fingerprints"] = replay_fingerprints
            report["replay_completed"] = len(replay_fingerprints) == args.replay_batches
        if args.rng_sidecar_in:
            report["rng_restore_after_replay"] = restore_rng_state(args.rng_sidecar_in)

        captured: dict[str, Any] = {}
        def wrapped_next_train(step: int):
            ray_bundle, batch = original_next_train(step)
            captured.clear()
            captured.update({"step": step, "batch_fingerprint": batch_fingerprint(batch)})
            return ray_bundle, batch
        dm.next_train = wrapped_next_train

        params_before = parameter_hashes(trainer.pipeline)
        records: list[dict[str, Any]] = []
        telemetry: list[dict[str, Any]] = []
        gradient_probes: list[dict[str, Any]] = []
        parameter_probes: list[dict[str, Any]] = []
        all_losses_finite = True
        all_mappings_finite = True
        all_steps_completed = True
        torch.cuda.reset_peak_memory_stats()
        telemetry.append(memory_snapshot("training_start", args.start_step))

        for step in range(args.start_step, args.end_step):
            captured.clear()
            iter_started = time.time()
            loss, loss_dict, metrics_dict = trainer.train_iteration(step=step)
            torch.cuda.synchronize()
            iter_seconds = time.time() - iter_started
            loss_value = float(loss.detach().float().cpu().item())
            finite_loss = math.isfinite(loss_value)
            mappings_finite = finite_scalar_mapping(loss_dict) and finite_scalar_mapping(metrics_dict)
            all_losses_finite &= finite_loss
            all_mappings_finite &= mappings_finite
            if captured.get("step") != step:
                all_steps_completed = False
            record = {
                "step": step,
                "iteration_seconds": iter_seconds,
                "loss": loss_value,
                "loss_dict": scalarize(loss_dict),
                "metrics_dict": scalarize(metrics_dict),
                "loss_finite": finite_loss,
                "mappings_finite": mappings_finite,
                "batch_fingerprint": captured.get("batch_fingerprint"),
            }
            records.append(record)
            probe_now = step == args.start_step or step == args.end_step - 1 or ((step - args.start_step + 1) % args.gradient_interval == 0)
            if probe_now:
                gradient_probes.append({"step": step, **gradient_probe(trainer.pipeline)})
                parameter_probes.append({"step": step, **parameter_finite_probe(trainer.pipeline)})
            telemetry_now = step == args.end_step - 1 or ((step - args.start_step + 1) % args.telemetry_interval == 0)
            if telemetry_now:
                telemetry.append(memory_snapshot("training", step))

        params_after = parameter_hashes(trainer.pipeline)
        changed = [name for name in sorted(set(params_before) & set(params_after)) if params_before[name] != params_after[name]]
        report["parameter_change"] = {"passed": bool(changed), "changed_parameter_count": len(changed), "changed_parameters": changed}
        report["records"] = records
        report["telemetry"] = telemetry
        report["gradient_probes"] = gradient_probes
        report["parameter_finite_probes"] = parameter_probes
        report["latency_distribution"] = latency_summary(records, args.warmup, args.start_step)
        report["memory_trend"] = memory_trend(telemetry, args.start_step, args.warmup)

        if args.rng_sidecar_out:
            report["rng_sidecar"] = capture_rng_state(args.rng_sidecar_out)

        checkpoint_step = args.end_step - 1
        trainer.save_checkpoint(checkpoint_step)
        torch.cuda.synchronize()
        checkpoint_path = trainer.checkpoint_dir / f"step-{checkpoint_step:09d}.ckpt"
        if not checkpoint_path.is_file():
            raise RuntimeError(f"checkpoint missing: {checkpoint_path}")
        report["checkpoint"] = {"path": str(checkpoint_path.resolve()), "sha256": sha256(checkpoint_path), "size_bytes": checkpoint_path.stat().st_size, "step": checkpoint_step}
        report["checkpoint_state_hashes"] = {
            "pipeline": canonical_hash(trainer.pipeline.state_dict()),
            "optimizers": canonical_hash({k: v.state_dict() for k, v in trainer.optimizers.optimizers.items()}),
            "schedulers": canonical_hash({k: v.state_dict() for k, v in trainer.optimizers.schedulers.items()}),
            "scalers": canonical_hash(trainer.grad_scaler.state_dict()),
        }
        report["runtime_identity_after_training"] = runtime_identity()
        report["memory_final"] = memory_snapshot("final", checkpoint_step)

        final_stats = report["memory_final"].get("selected_memory_stats", {})
        no_oom_retry = final_stats.get("num_ooms", 0) == 0 and final_stats.get("num_alloc_retries", 0) == 0
        checks = {
            "runtime_identity": bool(identity_before["passed"] and report["runtime_identity_after_training"]["passed"]),
            "real_dataset_loaded": bool(report["dataset_runtime"]["train_length"] and report["dataset_runtime"]["train_length"] > 0),
            "single_worker_dataloader_topology": report.get("dataloader_runtime_topology", {}).get("passed") is True,
            "all_steps_completed": bool(all_steps_completed and len(records) == args.end_step - args.start_step),
            "loss_and_metrics_finite": bool(all_losses_finite and all_mappings_finite),
            "gradient_probes_finite_nonzero": bool(gradient_probes and all(row["passed"] for row in gradient_probes)),
            "parameters_finite": bool(parameter_probes and all(row["passed"] for row in parameter_probes)),
            "parameters_changed": report["parameter_change"]["passed"],
            "no_oom_or_alloc_retry": no_oom_retry,
            "gpu_memory_no_positive_ramp": report["memory_trend"]["gpu_passed"],
            "host_rss_no_positive_ramp": report["memory_trend"]["host_passed"],
            "latency_distribution_recorded": report["latency_distribution"]["sample_count"] > 0,
            "checkpoint_written_and_hashed": bool(report["checkpoint"]["size_bytes"] > 0 and report["checkpoint"]["sha256"]),
            "fresh_reload_exact_state": report.get("fresh_reload_exact_state_equivalence", {"passed": True})["passed"],
            "replay_completed": (not args.replay_batches) or report.get("replay_completed") is True,
        }
        report["checks"] = checks
        report["passed"] = all(checks.values())
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
        json_dump(args.child_output, report)
        print(f"A5P2_CHILD_JSON={args.child_output}")
        print(f"A5P2_SERIES={args.series_name}")
        print(f"A5P2_CHILD_PASSED={'YES' if report.get('passed') else 'NO'}")
    return 0 if report.get("passed") else 2


def flatten_scalars(prefix: str, value: Any, out: dict[str, float]) -> None:
    if isinstance(value, dict):
        for key, item in value.items():
            flatten_scalars(f"{prefix}.{key}" if prefix else str(key), item, out)
    elif isinstance(value, (int, float)) and not isinstance(value, bool):
        out[prefix] = float(value)


def record_scalar_map(record: dict[str, Any]) -> dict[str, float]:
    out = {"loss": float(record["loss"])}
    flatten_scalars("loss_dict", record.get("loss_dict", {}), out)
    flatten_scalars("metrics_dict", record.get("metrics_dict", {}), out)
    return out


def records_by_step(payloads: list[dict[str, Any]]) -> dict[int, dict[str, Any]]:
    result: dict[int, dict[str, Any]] = {}
    for payload in payloads:
        for row in payload.get("records", []):
            result[int(row["step"])] = row
    return result


def compare_batches(anchor: dict[int, dict[str, Any]], other: dict[int, dict[str, Any]], start: int, end: int) -> dict[str, Any]:
    mismatches: list[int] = []
    missing: list[int] = []
    for step in range(start, end):
        if step not in anchor or step not in other:
            missing.append(step)
            continue
        a = anchor[step].get("batch_fingerprint", {}).get("combined_sha256")
        b = other[step].get("batch_fingerprint", {}).get("combined_sha256")
        if not a or not b or a != b:
            mismatches.append(step)
    return {"passed": not missing and not mismatches, "steps": end - start, "missing_steps": missing, "mismatch_count": len(mismatches), "mismatch_steps": mismatches[:50]}


def scalar_pair_metrics(anchor: dict[int, dict[str, Any]], other: dict[int, dict[str, Any]], start: int, end: int) -> dict[str, Any]:
    anchor_maps = {step: record_scalar_map(anchor[step]) for step in range(start, end) if step in anchor}
    other_maps = {step: record_scalar_map(other[step]) for step in range(start, end) if step in other}
    missing_steps = [step for step in range(start, end) if step not in anchor_maps or step not in other_maps]
    keys = sorted(set.intersection(*(set(m.keys()) for m in list(anchor_maps.values()) + list(other_maps.values())))) if anchor_maps and other_maps else []
    per_key: dict[str, Any] = {}
    for key in keys:
        a_values = [anchor_maps[step][key] for step in range(start, end) if step in anchor_maps and step in other_maps]
        b_values = [other_maps[step][key] for step in range(start, end) if step in anchor_maps and step in other_maps]
        diffs = [abs(a - b) for a, b in zip(a_values, b_values)]
        rmse = math.sqrt(sum((a - b) ** 2 for a, b in zip(a_values, b_values)) / len(diffs)) if diffs else None
        anchor_rms = math.sqrt(sum(a * a for a in a_values) / len(a_values)) if a_values else None
        relative_rmse = rmse / max(anchor_rms or 0.0, 1.0e-12) if rmse is not None else None
        per_key[key] = {"max_abs": max(diffs) if diffs else None, "rmse": rmse, "relative_rmse": relative_rmse, "samples": len(diffs)}
    return {"missing_steps": missing_steps, "keys": per_key, "structure_passed": not missing_steps and bool(per_key)}


def scalar_envelope(reference: dict[str, Any], resume: dict[str, Any]) -> dict[str, Any]:
    policy = TOLERANCE_POLICY["scalar_trajectory"]
    multiplier = TOLERANCE_POLICY["reference_multiplier"]
    rows: dict[str, Any] = {}
    passed = reference.get("structure_passed") and resume.get("structure_passed")
    common = sorted(set(reference.get("keys", {})) & set(resume.get("keys", {})))
    if set(reference.get("keys", {})) != set(resume.get("keys", {})):
        passed = False
    for key in common:
        ref = reference["keys"][key]
        obs = resume["keys"][key]
        max_abs_limit = max(policy["max_abs_floor"], multiplier * float(ref["max_abs"] or 0.0) + 1.0e-12)
        rel_limit = max(policy["relative_rmse_floor"], multiplier * float(ref["relative_rmse"] or 0.0) + 1.0e-12)
        row_pass = bool(float(obs["max_abs"] or 0.0) <= max_abs_limit and float(obs["relative_rmse"] or 0.0) <= rel_limit)
        rows[key] = {"reference": ref, "resume": obs, "max_abs_limit": max_abs_limit, "relative_rmse_limit": rel_limit, "passed": row_pass}
        passed &= row_pass
    return {"passed": bool(passed), "policy": policy, "reference_multiplier": multiplier, "keys": rows}


def dtype_bucket(value: Any) -> str | None:
    import torch
    if not torch.is_tensor(value):
        return None
    if value.dtype in (torch.float16, torch.bfloat16):
        return "fp16_bf16"
    if value.dtype == torch.float32:
        return "fp32"
    if value.dtype == torch.float64:
        return "fp64"
    return None


def flatten_state(prefix: str, value: Any, out: dict[str, tuple[str, Any]]) -> None:
    import torch
    if torch.is_tensor(value):
        out[prefix] = ("tensor", value.detach().cpu().contiguous())
    elif isinstance(value, dict):
        for key in sorted(value.keys(), key=lambda item: repr(item)):
            flatten_state(f"{prefix}/{repr(key)}", value[key], out)
    elif isinstance(value, (list, tuple)):
        out[prefix + "/__container__"] = ("meta", type(value).__name__)
        for index, item in enumerate(value):
            flatten_state(f"{prefix}/{index}", item, out)
    elif isinstance(value, float):
        out[prefix] = ("float", float(value))
    else:
        out[prefix] = ("exact", (type(value).__name__, repr(value)))


def state_pair_metrics(anchor_state: Any, other_state: Any) -> dict[str, Any]:
    import torch
    a: dict[str, tuple[str, Any]] = {}
    b: dict[str, tuple[str, Any]] = {}
    flatten_state("root", anchor_state, a)
    flatten_state("root", other_state, b)
    missing = sorted(set(a) - set(b))
    extra = sorted(set(b) - set(a))
    kind_mismatches: list[str] = []
    descriptor_mismatches: list[str] = []
    exact_mismatches: list[str] = []
    buckets: dict[str, dict[str, float]] = {
        key: {"max_abs": 0.0, "sum_sq_diff": 0.0, "sum_sq_anchor": 0.0, "count": 0.0}
        for key in ["fp16_bf16", "fp32", "fp64"]
    }
    for key in sorted(set(a) & set(b)):
        kind_a, value_a = a[key]
        kind_b, value_b = b[key]
        if kind_a != kind_b:
            kind_mismatches.append(key)
            continue
        if kind_a == "tensor":
            if value_a.shape != value_b.shape or value_a.dtype != value_b.dtype:
                descriptor_mismatches.append(key)
                continue
            bucket = dtype_bucket(value_a)
            if bucket is None:
                if not torch.equal(value_a, value_b):
                    exact_mismatches.append(key)
                continue
            av = value_a.to(torch.float64)
            bv = value_b.to(torch.float64)
            diff = av - bv
            stats = buckets[bucket]
            if diff.numel():
                stats["max_abs"] = max(stats["max_abs"], float(diff.abs().max().item()))
                stats["sum_sq_diff"] += float((diff * diff).sum().item())
                stats["sum_sq_anchor"] += float((av * av).sum().item())
                stats["count"] += float(diff.numel())
        elif kind_a == "float":
            diff = abs(float(value_a) - float(value_b))
            stats = buckets["fp64"]
            stats["max_abs"] = max(stats["max_abs"], diff)
            stats["sum_sq_diff"] += diff * diff
            stats["sum_sq_anchor"] += float(value_a) * float(value_a)
            stats["count"] += 1.0
        elif value_a != value_b:
            exact_mismatches.append(key)
    summary: dict[str, Any] = {}
    for bucket, stats in buckets.items():
        count = int(stats["count"])
        rel = math.sqrt(stats["sum_sq_diff"]) / max(math.sqrt(stats["sum_sq_anchor"]), 1.0e-30) if count else 0.0
        summary[bucket] = {"max_abs": stats["max_abs"], "relative_l2": rel, "element_count": count}
    structure_passed = not missing and not extra and not kind_mismatches and not descriptor_mismatches
    exact_passed = not exact_mismatches
    return {
        "structure_passed": structure_passed,
        "exact_discrete_passed": exact_passed,
        "missing": missing[:50], "extra": extra[:50],
        "kind_mismatches": kind_mismatches[:50], "descriptor_mismatches": descriptor_mismatches[:50],
        "exact_mismatches": exact_mismatches[:50],
        "buckets": summary,
    }


def state_envelope(reference: dict[str, Any], resume: dict[str, Any]) -> dict[str, Any]:
    multiplier = TOLERANCE_POLICY["reference_multiplier"]
    passed = bool(reference.get("structure_passed") and resume.get("structure_passed") and reference.get("exact_discrete_passed") and resume.get("exact_discrete_passed"))
    rows: dict[str, Any] = {}
    for bucket, policy in TOLERANCE_POLICY["state"].items():
        ref = reference["buckets"][bucket]
        obs = resume["buckets"][bucket]
        max_abs_limit = max(policy["max_abs_floor"], multiplier * ref["max_abs"] + 1.0e-15)
        relative_l2_limit = max(policy["relative_l2_floor"], multiplier * ref["relative_l2"] + 1.0e-15)
        row_pass = bool(obs["max_abs"] <= max_abs_limit and obs["relative_l2"] <= relative_l2_limit)
        rows[bucket] = {"reference": ref, "resume": obs, "max_abs_limit": max_abs_limit, "relative_l2_limit": relative_l2_limit, "passed": row_pass}
        passed &= row_pass
    return {"passed": passed, "reference_multiplier": multiplier, "policy": TOLERANCE_POLICY["state"], "buckets": rows, "reference_structure": {k: reference[k] for k in ["structure_passed", "exact_discrete_passed"]}, "resume_structure": {k: resume[k] for k in ["structure_passed", "exact_discrete_passed"]}}


def load_checkpoint(path: Path) -> dict[str, Any]:
    import torch
    return torch.load(path, map_location="cpu", weights_only=False)


def compare_final_checkpoints(a_path: Path, c_path: Path, b_path: Path) -> dict[str, Any]:
    a = load_checkpoint(a_path)
    c = load_checkpoint(c_path)
    b = load_checkpoint(b_path)
    required = ["pipeline", "optimizers", "schedulers", "scalers"]
    missing = {name: [key for key in required if key not in state] for name, state in [("A", a), ("C", c), ("B", b)]}
    sections: dict[str, Any] = {}
    passed = all(not values for values in missing.values())
    for section in required:
        if any(section not in state for state in [a, c, b]):
            continue
        reference = state_pair_metrics(a[section], c[section])
        resume = state_pair_metrics(a[section], b[section])
        envelope = state_envelope(reference, resume)
        sections[section] = {"reference_A_vs_C": reference, "resume_A_vs_B": resume, "envelope": envelope}
        passed &= envelope["passed"]
    steps = {"A": int(a.get("step", -1)), "C": int(c.get("step", -1)), "B": int(b.get("step", -1))}
    step_exact = len(set(steps.values())) == 1
    passed &= step_exact
    return {"passed": passed, "missing_required_sections": missing, "steps": steps, "step_exact": step_exact, "sections": sections}


def create_gate(report: dict[str, Any]) -> str:
    checks = report.get("checks", {})
    ordered = [
        "PUBLIC_A5_P1_MANIFEST_PREREQUISITE",
        "NERFSTUDIO_SOURCE_RUNTIME_DATASET_PINNING",
        "SINGLE_WORKER_REAL_DATALOADER_TOPOLOGY",
        "THREE_TRAJECTORY_FRESH_PROCESS_TOPOLOGY",
        "SUSTAINED_ALL_STEPS_COMPLETED",
        "SUSTAINED_LOSS_METRICS_FINITE",
        "SUSTAINED_GRADIENT_PARAMETER_FINITE",
        "SUSTAINED_NO_OOM_OR_ALLOC_RETRY",
        "SUSTAINED_GPU_MEMORY_NO_POSITIVE_RAMP",
        "SUSTAINED_HOST_RSS_NO_POSITIVE_RAMP",
        "STEP_LATENCY_DISTRIBUTION_RECORDED",
        "CHECKPOINT_CHAIN_WRITE_AND_HASH",
        "SPLIT_FRESH_RELOAD_EXACT_STATE",
        "RESUME_DATASTREAM_REPLAY_ALIGNED",
        "RESUME_TRAJECTORY_WITHIN_REFERENCE_ENVELOPE",
        "TINY_RDNA4_NN_SINGLE_SH_AND_NO_MIXED_ORIGINS",
        "A5_P2_SUSTAINED_REAL_TRAINING_QUALIFICATION",
    ]
    lines = [
        "AMD_NERFSTUDIO_PUBLIC_A5_P2_SUSTAINED_V1",
        "",
        f"classification={report.get('classification')}",
        f"decision={report.get('decision')}",
        f"run_id={report.get('run_id')}",
        f"dataset={report.get('dataset')}",
        f"steps={report.get('steps')}",
        f"split_step={report.get('split_step')}",
        f"rays_per_batch={report.get('rays')}",
        "trajectory_design=A_CONTINUOUS_PLUS_C_CONTINUOUS_REFERENCE_PLUS_B_SPLIT_FRESH_RESUME",
        "resume_datastream_alignment=QUALIFICATION_REPLAY_WITH_RNG_RESTORE",
        "performance_benchmark=NOT_CLAIMED",
        "vmm_fallback_performance=NOT_CLAIMED",
        "infinite_horizon_leak_freedom=NOT_CLAIMED",
        "",
    ]
    for name in ordered:
        lines.append(f"PUBLIC_RDNA4_{name}: {'PASS' if checks.get(name) else 'FAIL'}")
    lines.extend([
        "",
        "blockers=" + (",".join(report.get("blockers", [])) if report.get("blockers") else "NONE"),
        "",
        "PUBLIC_RDNA4_A5_P2_SUSTAINED_AND_RESUME: PASS" if report.get("passed") else "PUBLIC_RDNA4_A5_P2_SUSTAINED_AND_RESUME: FAIL",
        "PUBLIC_RDNA4_A5_P2_QUALIFIED: PASS" if report.get("passed") else "PUBLIC_RDNA4_A5_P2_QUALIFIED: BLOCKED",
    ])
    return "\n".join(lines) + "\n"


def chmod_tree_readonly(root: Path) -> None:
    for path in sorted(root.rglob("*"), reverse=True):
        try:
            path.chmod(0o444 if path.is_file() else 0o555)
        except OSError:
            pass
    try:
        root.chmod(0o555)
    except OSError:
        pass


def orchestrate(args: argparse.Namespace) -> int:
    root = args.output_root.expanduser().resolve()
    python = absolute_preserving_symlink(args.python)
    dataset = args.data.expanduser().resolve()
    nerfstudio = args.nerfstudio_worktree.expanduser().resolve()
    runtime = args.tcnn_runtime.expanduser().resolve()
    p1_run_dir = args.p1_run_dir.expanduser().resolve()
    run_id = args.run_id or time.strftime("%Y%m%dT%H%M%SZ", time.gmtime()) + f"_{os.getpid()}"
    run_dir = root / "public_a5p2_sustained_v1" / run_id
    if run_dir.exists():
        raise RuntimeError(f"run directory already exists: {run_dir}")
    run_dir.mkdir(parents=True)
    torch_lib = python.parent.parent / "lib/python3.12/site-packages/torch/lib"
    policy_env = os.environ.copy()
    policy_env.pop("TORCH_FORCE_WEIGHTS_ONLY_LOAD", None)
    policy_env["PYTHONNOUSERSITE"] = "1"
    policy_env["PYTHONDONTWRITEBYTECODE"] = "1"
    policy_env["TCNN_RDNA4_ENABLE_PORTABLE_MLP_SHIM"] = "1"
    policy_env["NERFSTUDIO_RDNA4_A5_SINGLE_SH_POLICY"] = "TINY_RDNA4_NN_ONLY"
    policy_env["NERFSTUDIO_RDNA4_A5_TCNN_RUNTIME"] = str(runtime)
    policy_env["NERFSTUDIO_RDNA4_A5_NERFSTUDIO_WORKTREE"] = str(nerfstudio)
    policy_env["TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD"] = "1"
    policy_env["PYTHONPATH"] = os.pathsep.join([str(runtime), str(nerfstudio)])
    current_ld = policy_env.get("LD_LIBRARY_PATH", "")
    policy_env["LD_LIBRARY_PATH"] = os.pathsep.join([str(torch_lib), "/opt/rocm/lib", "/opt/rocm/lib64"] + ([current_ld] if current_ld else []))
    old_env = os.environ.copy()
    os.environ.update(policy_env)
    try:
        prereq = prerequisite_report(p1_run_dir, dataset, nerfstudio, runtime, python)
    finally:
        os.environ.clear(); os.environ.update(old_env)
    json_dump(run_dir / "prerequisites.json", prereq)
    report: dict[str, Any] = {
        "schema": SCHEMA, "classification": CLASSIFICATION,
        "run_id": run_id, "run_dir": str(run_dir), "root": str(root), "dataset": str(dataset),
        "steps": args.steps, "split_step": args.split_step, "rays": args.rays, "seed": args.seed,
        "warmup": args.warmup, "telemetry_interval": args.telemetry_interval, "gradient_interval": args.gradient_interval,
        "tolerance_policy": TOLERANCE_POLICY,
        "prerequisites": prereq,
        "processes": {}, "series": {}, "passed": False, "decision": "PUBLIC_A5_P2_BLOCKED",
        "nonclaims": [
            "FULL_SHARED_VENV_GLOBAL_CONSISTENCY",
            "INFINITE_HORIZON_MEMORY_LEAK_FREEDOM",
            "VMM_FALLBACK_PERFORMANCE_OR_P99_CAUSAL_ATTRIBUTION",
            "MULTI_GPU_OR_DISTRIBUTED_TRAINING",
            "VIEWER_EVAL_EXPORT_AND_UNUSED_OPTIONAL_FEATURES",
            "NATIVE_DATALOADER_WORKER_STATE_SERIALIZATION_IN_NERFSTUDIO_CHECKPOINT",
            "PRODUCTION_RECONSTRUCTION_QUALITY_OF_THE_SIX_IMAGE_P1_DATASET",
        ],
    }
    if not prereq.get("passed"):
        report["checks"] = {"PUBLIC_A5_P1_MANIFEST_PREREQUISITE": False}
        report["blockers"] = ["PUBLIC_A5_P1_OR_SOURCE_RUNTIME_DATASET_PREREQUISITE"]
        json_dump(run_dir / "final_aggregate.json", report)
        (run_dir / "final_gate.txt").write_text(create_gate(report), encoding="utf-8")
        print(json.dumps(report, indent=2, sort_keys=True))
        print(create_gate(report))
        return 2

    child_env = policy_env.copy()
    tool = Path(__file__).resolve()

    def launch(name: str, start: int, end: int, checkpoint: Path | None = None, replay: int = 0, rng_in: Path | None = None, rng_out: Path | None = None, expected_state_json: Path | None = None) -> dict[str, Any]:
        out_dir = run_dir / name
        child_json = out_dir / f"{name}.json"
        cmd = [
            str(python), str(tool), "--mode", "series", "--root", str(root), "--python", str(python),
            "--data", str(dataset), "--output-base", str(out_dir), "--series-name", name,
            "--start-step", str(start), "--end-step", str(end), "--steps", str(args.steps), "--split-step", str(args.split_step),
            "--seed", str(args.seed), "--rays", str(args.rays), "--warmup", str(args.warmup),
            "--telemetry-interval", str(args.telemetry_interval), "--gradient-interval", str(args.gradient_interval),
            "--replay-batches", str(replay), "--child-output", str(child_json),
        ]
        if checkpoint:
            cmd += ["--checkpoint", str(checkpoint)]
        if rng_in:
            cmd += ["--rng-sidecar-in", str(rng_in)]
        if rng_out:
            cmd += ["--rng-sidecar-out", str(rng_out)]
        if expected_state_json:
            cmd += ["--expected-state-json", str(expected_state_json)]
        result = run_command(cmd, cwd=Path("/tmp"), env=child_env, timeout=args.timeout)
        log = out_dir / f"{name}_process.log"
        log.parent.mkdir(parents=True, exist_ok=True)
        log.write_text(result.get("stdout", "") + "\n--- STDERR ---\n" + result.get("stderr", ""), encoding="utf-8")
        payload = json.loads(child_json.read_text(encoding="utf-8")) if child_json.is_file() else {}
        report["processes"][name] = {**result, "log": str(log), "json": str(child_json)}
        report["series"][name] = payload
        return payload

    a = launch("A_continuous", 0, args.steps)
    c = launch("C_continuous_reference", 0, args.steps)
    b0_rng = run_dir / "B_split/rng_at_split.pt"
    b0 = launch("B_split", 0, args.split_step, rng_out=b0_rng)
    b1: dict[str, Any] = {}
    if b0.get("passed") and b0.get("checkpoint", {}).get("path") and b0_rng.is_file():
        b1 = launch(
            "B_resume", args.split_step, args.steps,
            checkpoint=Path(b0["checkpoint"]["path"]), replay=args.split_step,
            rng_in=b0_rng, expected_state_json=run_dir / "B_split/B_split.json",
        )
    else:
        report["processes"]["B_resume"] = {"skipped": True, "reason": "B_split failed"}

    payloads = [a, c, b0, b1]
    process_pass = all(payload.get("passed") for payload in payloads)
    pids = [payload.get("pid") for payload in payloads]
    fresh_topology = bool(process_pass and all(pids) and len(set(pids)) == 4 and os.getpid() not in pids)
    a_records = records_by_step([a])
    c_records = records_by_step([c])
    b_records = records_by_step([b0, b1])
    batch_ref = compare_batches(a_records, c_records, 0, args.steps)
    batch_resume = compare_batches(a_records, b_records, 0, args.steps)
    replay_records = {int(row["step"]): {"batch_fingerprint": row["batch_fingerprint"]} for row in b1.get("replay_fingerprints", [])}
    replay_alignment = compare_batches(a_records, replay_records, 0, args.split_step)
    scalar_ref_pre = scalar_pair_metrics(a_records, c_records, 0, args.split_step)
    scalar_resume_pre = scalar_pair_metrics(a_records, b_records, 0, args.split_step)
    scalar_ref_post = scalar_pair_metrics(a_records, c_records, args.split_step, args.steps)
    scalar_resume_post = scalar_pair_metrics(a_records, b_records, args.split_step, args.steps)
    scalar_envelope_pre = scalar_envelope(scalar_ref_pre, scalar_resume_pre)
    scalar_envelope_post = scalar_envelope(scalar_ref_post, scalar_resume_post)
    report["datastream_comparison"] = {"A_vs_C_reference": batch_ref, "A_vs_B_resume": batch_resume, "A_vs_B_resume_replay_prefix": replay_alignment}
    report["scalar_trajectory_comparison"] = {
        "pre_split": {"reference_A_vs_C": scalar_ref_pre, "resume_A_vs_B": scalar_resume_pre, "envelope": scalar_envelope_pre},
        "post_split": {"reference_A_vs_C": scalar_ref_post, "resume_A_vs_B": scalar_resume_post, "envelope": scalar_envelope_post},
    }

    checkpoint_comparison: dict[str, Any] = {"passed": False, "error": "checkpoints unavailable"}
    try:
        checkpoint_comparison = compare_final_checkpoints(
            Path(a["checkpoint"]["path"]), Path(c["checkpoint"]["path"]), Path(b1["checkpoint"]["path"])
        )
    except Exception as exc:
        checkpoint_comparison = {"passed": False, "error": repr(exc), "traceback": traceback.format_exc()}
    report["final_checkpoint_trajectory_comparison"] = checkpoint_comparison

    child_checks = [payload.get("checks", {}) for payload in payloads]
    all_check = lambda key: bool(child_checks and all(check.get(key) for check in child_checks))
    checkpoint_chain = bool(
        all(payload.get("checkpoint", {}).get("sha256") and payload.get("checkpoint", {}).get("size_bytes", 0) > 0 for payload in payloads)
        and b0_rng.is_file() and sha256(b0_rng) == b0.get("rng_sidecar", {}).get("sha256")
    )
    exact_reload = b1.get("fresh_reload_exact_state_equivalence", {}).get("passed") is True and b1.get("loaded_start_step") == args.split_step
    origins_ok = all(payload.get("runtime_identity_after_training", {}).get("passed") for payload in payloads)
    trajectory_ok = bool(scalar_envelope_pre["passed"] and scalar_envelope_post["passed"] and checkpoint_comparison.get("passed"))
    datastream_ok = bool(batch_ref["passed"] and batch_resume["passed"] and replay_alignment["passed"])
    checks = {
        "PUBLIC_A5_P1_MANIFEST_PREREQUISITE": bool(prereq.get("p1_hash_chain", {}).get("passed") and prereq.get("p1_semantics", {}).get("passed")),
        "NERFSTUDIO_SOURCE_RUNTIME_DATASET_PINNING": bool(prereq.get("nerfstudio_source", {}).get("passed") and prereq.get("runtime_anchors", {}).get("passed") and prereq.get("dataset", {}).get("pinned_passed")),
        "SINGLE_WORKER_REAL_DATALOADER_TOPOLOGY": all_check("single_worker_dataloader_topology"),
        "THREE_TRAJECTORY_FRESH_PROCESS_TOPOLOGY": fresh_topology,
        "SUSTAINED_ALL_STEPS_COMPLETED": all_check("all_steps_completed"),
        "SUSTAINED_LOSS_METRICS_FINITE": all_check("loss_and_metrics_finite"),
        "SUSTAINED_GRADIENT_PARAMETER_FINITE": bool(all_check("gradient_probes_finite_nonzero") and all_check("parameters_finite") and all_check("parameters_changed")),
        "SUSTAINED_NO_OOM_OR_ALLOC_RETRY": all_check("no_oom_or_alloc_retry"),
        "SUSTAINED_GPU_MEMORY_NO_POSITIVE_RAMP": all_check("gpu_memory_no_positive_ramp"),
        "SUSTAINED_HOST_RSS_NO_POSITIVE_RAMP": all_check("host_rss_no_positive_ramp"),
        "STEP_LATENCY_DISTRIBUTION_RECORDED": all_check("latency_distribution_recorded"),
        "CHECKPOINT_CHAIN_WRITE_AND_HASH": checkpoint_chain,
        "SPLIT_FRESH_RELOAD_EXACT_STATE": exact_reload,
        "RESUME_DATASTREAM_REPLAY_ALIGNED": datastream_ok,
        "RESUME_TRAJECTORY_WITHIN_REFERENCE_ENVELOPE": trajectory_ok,
        "TINY_RDNA4_NN_SINGLE_SH_AND_NO_MIXED_ORIGINS": origins_ok,
    }
    core = list(checks)
    checks["A5_P2_SUSTAINED_REAL_TRAINING_QUALIFICATION"] = all(checks[name] for name in core)
    blockers = [name for name, passed in checks.items() if not passed and name != "A5_P2_SUSTAINED_REAL_TRAINING_QUALIFICATION"]
    report["checks"] = checks
    report["blockers"] = blockers
    report["passed"] = checks["A5_P2_SUSTAINED_REAL_TRAINING_QUALIFICATION"]
    report["decision"] = "PUBLIC_A5_P2_QUALIFIED" if report["passed"] else "PUBLIC_A5_P2_BLOCKED"
    report["process_identity"] = {"orchestrator_pid": os.getpid(), "series_pids": dict(zip(["A", "C", "B_split", "B_resume"], pids)), "all_distinct": fresh_topology}

    final_json = run_dir / "final_aggregate.json"
    final_gate = run_dir / "final_gate.txt"
    json_dump(final_json, report)
    final_gate.write_text(create_gate(report), encoding="utf-8")
    manifest = {"schema": SCHEMA + "-manifest", "run_id": run_id, "files": {}}
    for path in sorted(run_dir.rglob("*")):
        if path.is_file() and path.name != "MANIFEST.json":
            manifest["files"][str(path.relative_to(run_dir))] = {"sha256": sha256(path), "size_bytes": path.stat().st_size}
    json_dump(run_dir / "MANIFEST.json", manifest)
    latest = root / "public_a5p2_sustained_v1.latest"
    latest.write_text(str(run_dir) + "\n", encoding="utf-8")
    canonical_gate = root / "public_a5p2_sustained_v1_gate.txt"
    canonical_result = root / "public_a5p2_sustained_v1.result"
    shutil.copy2(final_gate, canonical_gate)
    canonical_result.write_text(f"exit_code={0 if report['passed'] else 2}\nclassification={CLASSIFICATION}\ndecision={report['decision']}\nrun_id={run_id}\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    print(create_gate(report))
    print("A5-P2 artifact inventory:")
    for path in [final_json, final_gate, run_dir / "MANIFEST.json", canonical_gate, canonical_result]:
        print(f"{sha256(path)}  {path}")
    chmod_tree_readonly(run_dir)
    return 0 if report["passed"] else 2


def self_test() -> int:
    import torch
    slope = theil_sen_slope([(0, 10), (1, 11), (2, 12), (3, 13)])
    pct = percentile([1.0, 2.0, 3.0, 4.0], 0.5)
    a = {"x": torch.tensor([1.0, 2.0], dtype=torch.float32), "n": 3}
    c = {"x": torch.tensor([1.0, 2.000001], dtype=torch.float32), "n": 3}
    b = {"x": torch.tensor([1.0, 2.000002], dtype=torch.float32), "n": 3}
    ref = state_pair_metrics(a, c)
    obs = state_pair_metrics(a, b)
    envelope = state_envelope(ref, obs)
    loader_policy = {"dataloader_num_workers": 1, "prefetch_factor": 2}
    ok = (
        slope == 1.0
        and pct == 2.5
        and envelope["passed"]
        and canonical_hash(a) == canonical_hash(copy.deepcopy(a))
        and loader_policy["dataloader_num_workers"] == 1
    )
    print(json.dumps({"passed": ok, "slope": slope, "p50": pct, "loader_policy": loader_policy, "state_envelope": envelope}, indent=2, sort_keys=True))
    return 0 if ok else 2



def main() -> int:
    parser = argparse.ArgumentParser(description="Public A5-P2 sustained real Nerfacto training and resume trajectory qualification")
    parser.add_argument("--mode", choices=["orchestrate", "series", "self-test"], default="orchestrate")
    parser.add_argument("--root", type=Path)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--p1-run-dir", type=Path)
    parser.add_argument("--nerfstudio-worktree", type=Path)
    parser.add_argument("--tcnn-runtime", type=Path)
    parser.add_argument("--python", type=Path)
    parser.add_argument("--data", type=Path)
    parser.add_argument("--output-base", type=Path)
    parser.add_argument("--series-name")
    parser.add_argument("--start-step", type=int, default=0)
    parser.add_argument("--end-step", type=int)
    parser.add_argument("--steps", type=int, default=192)
    parser.add_argument("--split-step", type=int, default=96)
    parser.add_argument("--seed", type=int, default=20260803)
    parser.add_argument("--rays", type=int, default=1024)
    parser.add_argument("--warmup", type=int, default=16)
    parser.add_argument("--telemetry-interval", type=int, default=4)
    parser.add_argument("--gradient-interval", type=int, default=32)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--replay-batches", type=int, default=0)
    parser.add_argument("--rng-sidecar-in", type=Path)
    parser.add_argument("--rng-sidecar-out", type=Path)
    parser.add_argument("--expected-state-json", type=Path)
    parser.add_argument("--child-output", type=Path)
    parser.add_argument("--run-id")
    parser.add_argument("--timeout", type=int, default=7200)
    args = parser.parse_args()
    if args.mode == "self-test":
        return self_test()
    if args.data is None or args.python is None:
        parser.error("run modes require --data and --python")
    if args.steps < 64 or args.split_step <= 0 or args.split_step >= args.steps:
        parser.error("require steps >= 64 and 0 < split-step < steps")
    if args.warmup < 0 or args.telemetry_interval <= 0 or args.gradient_interval <= 0 or args.rays <= 0:
        parser.error("invalid warmup/interval/rays")
    if args.mode == "series":
        required = [args.root, args.output_base, args.series_name, args.end_step, args.child_output]
        if any(value is None for value in required):
            parser.error("series mode missing root/output/name/end/output-json")
        return child_series(args)
    required = [args.output_root, args.p1_run_dir, args.nerfstudio_worktree, args.tcnn_runtime]
    if any(value is None for value in required):
        parser.error("orchestrate mode requires --output-root, --p1-run-dir, --nerfstudio-worktree and --tcnn-runtime")
    return orchestrate(args)


if __name__ == "__main__":
    raise SystemExit(main())
