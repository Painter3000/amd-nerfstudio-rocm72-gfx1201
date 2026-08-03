#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import shlex
import subprocess
import time
from typing import Any, Iterable


def absolute_preserving_symlink(path: Path) -> Path:
    """Return an absolute path without resolving the final symlink.

    Virtual-environment launchers such as ``venv/bin/python`` are commonly
    symlinks to the system interpreter. Resolving that symlink changes Python's
    prefix discovery and silently drops the virtual environment.
    """
    return Path(os.path.abspath(os.fspath(path.expanduser())))


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def json_dump(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")


def load_json(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"expected JSON object: {path}")
    return data


def run_command(argv: list[str], *, cwd: Path | None = None, env: dict[str, str] | None = None, timeout: int = 300) -> dict[str, Any]:
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


def file_anchor(path: Path, expected: str | None = None) -> dict[str, Any]:
    observed = sha256(path) if path.is_file() else None
    return {
        "path": str(path),
        "exists": path.is_file(),
        "sha256": observed,
        "expected_sha256": expected,
        "hash_matches": observed == expected if expected else None,
    }


def verify_manifest(run_dir: Path, manifest_name: str = "MANIFEST.json") -> dict[str, Any]:
    manifest_path = run_dir / manifest_name
    if not manifest_path.is_file():
        return {"passed": False, "error": "MANIFEST_MISSING", "path": str(manifest_path)}
    try:
        manifest = load_json(manifest_path)
    except Exception as exc:
        return {"passed": False, "error": repr(exc), "path": str(manifest_path)}
    rows = manifest.get("files", {})
    if not isinstance(rows, dict):
        return {"passed": False, "error": "MANIFEST_FILES_NOT_OBJECT", "path": str(manifest_path)}
    mismatches: list[dict[str, Any]] = []
    for rel, expected in sorted(rows.items()):
        path = run_dir / rel
        expected_sha = expected.get("sha256") if isinstance(expected, dict) else None
        observed = sha256(path) if path.is_file() else None
        if observed != expected_sha:
            mismatches.append({"path": str(path), "expected": expected_sha, "observed": observed})
    return {"passed": not mismatches, "path": str(manifest_path), "file_count": len(rows), "mismatches": mismatches}


def inventory_tree(root: Path, *, exclude_names: Iterable[str] = ()) -> dict[str, Any]:
    excluded = set(exclude_names)
    files: dict[str, dict[str, Any]] = {}
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.name in excluded:
            continue
        rel = str(path.relative_to(root))
        files[rel] = {"sha256": sha256(path), "size_bytes": path.stat().st_size}
    return {"root": str(root), "file_count": len(files), "files": files}


def build_runtime_env(runtime: Path, nerfstudio: Path, torch_lib: Path | None = None) -> dict[str, str]:
    env = os.environ.copy()
    env.pop("PYTHONPATH", None)
    env.pop("TORCH_FORCE_WEIGHTS_ONLY_LOAD", None)
    env["PYTHONNOUSERSITE"] = "1"
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    env["PYTHONPATH"] = os.pathsep.join([str(runtime.resolve()), str(nerfstudio.resolve())])
    env["TCNN_RDNA4_ENABLE_PORTABLE_MLP_SHIM"] = "1"
    env["NERFSTUDIO_RDNA4_A5_SINGLE_SH_POLICY"] = "TINY_RDNA4_NN_ONLY"
    env["NERFSTUDIO_RDNA4_A5_TCNN_RUNTIME"] = str(runtime.resolve())
    env["NERFSTUDIO_RDNA4_A5_NERFSTUDIO_WORKTREE"] = str(nerfstudio.resolve())
    env["TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD"] = "1"
    if torch_lib:
        current = env.get("LD_LIBRARY_PATH", "")
        parts = [str(torch_lib.resolve()), "/opt/rocm/lib", "/opt/rocm/lib64"]
        if current:
            parts.append(current)
        env["LD_LIBRARY_PATH"] = os.pathsep.join(parts)
    return env


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
    resolved = data.expanduser().resolve()
    transforms = resolved / "transforms.json" if resolved.is_dir() else resolved
    dataset_dir = resolved if resolved.is_dir() else resolved.parent
    if not transforms.is_file():
        return {"passed": False, "error": "TRANSFORMS_JSON_MISSING", "path": str(transforms)}
    try:
        payload = load_json(transforms)
    except Exception as exc:
        return {"passed": False, "error": repr(exc), "path": str(transforms)}
    frames = payload.get("frames")
    if not isinstance(frames, list) or not frames:
        return {"passed": False, "error": "FRAMES_MISSING_OR_EMPTY", "path": str(transforms)}
    rows: list[dict[str, Any]] = []
    missing: list[str] = []
    for index, frame in enumerate(frames):
        raw = frame.get("file_path") if isinstance(frame, dict) else None
        if not isinstance(raw, str) or not raw:
            missing.append(f"frame[{index}]")
            continue
        image = resolve_image_path(dataset_dir, raw)
        if image is None:
            missing.append(raw)
            continue
        rows.append({"index": index, "raw": raw, "path": str(image), "size_bytes": image.stat().st_size, "sha256": sha256(image)})
    path_manifest = hashlib.sha256()
    for row in rows:
        path_manifest.update(f"{row['index']}\0{row['raw']}\0{row['size_bytes']}\n".encode("utf-8"))
    return {
        "passed": not missing and len(rows) == len(frames),
        "dataset_dir": str(dataset_dir),
        "transforms": str(transforms),
        "transforms_sha256": sha256(transforms),
        "frame_count": len(frames),
        "resolved_image_count": len(rows),
        "missing": missing,
        "path_size_manifest_sha256": path_manifest.hexdigest(),
        "images": rows,
    }
