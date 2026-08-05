#!/usr/bin/env python3
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
sys.dont_write_bytecode = True
import tarfile
import time
from typing import Any

SCHEMA = "amd-nerfstudio-public-dev5-p0-p1-v1"
CLASSIFICATION = "PUBLIC_DEV5_DATASET_DEPLOYMENT_PLUS_REAL_NERFACTO_P0_P1"
EXPECTED_NERFSTUDIO_COMMIT = "50e0e3c70c775e89333256213363badbf074f29d"
EXPECTED_NERFSTUDIO_TREE = "9d5ff468eeff89b66995e9984acaa378c37dc07e"
EXPECTED_TINY_NATIVE_SHA = "4a561cc605bb7a6353d0eca1f9effc5ac9fcdfa3a9cb605a8cf36e1ae25b1917"
EXPECTED_NERFACC_NATIVE_SHA = "d3beee150cfa3a9ad3038a3283ff0a46953c345634d8cb6109449c5e3d04d1e2"

P1_CHECKS = [
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

CHECK_ORDER = [
    "DATASET_ARCHIVE_HASH",
    "DATASET_ARCHIVE_SAFE_MEMBERS",
    "DATASET_REQUIRED_FILE_SET",
    "DATASET_FILE_HASHES",
    "DATASET_SEMANTICS",
    "REPOSITORY_TRACKED_WORKTREE_CLEAN",
    "NERFSTUDIO_SOURCE_PINNED",
    "TINY_NATIVE_IDENTITY",
    "NERFACC_NATIVE_IDENTITY",
    "QUICK_P0_PREFLIGHT",
    "QUICK_P1_REAL_MECHANICS",
    "QUICK_CHECKPOINT_POLICY",
    "QUICK_MANIFEST_CHAIN",
    *[f"P1_{name}" for name in P1_CHECKS],
]

if len(CHECK_ORDER) != 28:
    raise RuntimeError(f"dev5 gate contract must contain exactly 28 checks, got {len(CHECK_ORDER)}")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def json_dump(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")


def run_command(argv: list[str], *, cwd: Path | None = None, env: dict[str, str] | None = None, timeout: int = 1800) -> dict[str, Any]:
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


def git_identity(repo: Path) -> dict[str, Any]:
    head = run_command(["git", "-C", str(repo), "rev-parse", "HEAD"], timeout=30)
    tree = run_command(["git", "-C", str(repo), "rev-parse", "HEAD^{tree}"], timeout=30)
    status = run_command(["git", "-C", str(repo), "status", "--porcelain=v1", "--untracked-files=no"], timeout=30)
    return {
        "head": head.get("stdout", "").strip(),
        "tree": tree.get("stdout", "").strip(),
        "status": status.get("stdout", ""),
        "passed": bool(
            head.get("returncode") == 0
            and tree.get("returncode") == 0
            and status.get("returncode") == 0
            and not status.get("stdout", "").strip()
        ),
        "processes": {"head": head, "tree": tree, "status": status},
    }


def find_exact_one(root: Path, pattern: str) -> Path | None:
    matches = sorted(root.glob(pattern))
    return matches[0].resolve() if len(matches) == 1 else None


def probe_nerfacc(python: Path, nerfstudio: Path, tcnn_runtime: Path) -> dict[str, Any]:
    """Attest nerfacc after loading the owning torch/ROCm runtime first."""

    code = r"""
import hashlib
import json
import pathlib
import traceback

out = {"no_error": False}

try:
    import torch
    import nerfacc.csrc as nerfacc_csrc

    path = pathlib.Path(nerfacc_csrc.__file__).resolve()
    out.update(
        {
            "no_error": True,
            "torch": torch.__version__,
            "hip": torch.version.hip,
            "cuda_available": bool(torch.cuda.is_available()),
            "gcn_arch": (
                getattr(
                    torch.cuda.get_device_properties(0),
                    "gcnArchName",
                    None,
                )
                if torch.cuda.is_available()
                else None
            ),
            "path": str(path),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }
    )
except Exception as exc:
    out["error"] = repr(exc)
    out["traceback"] = traceback.format_exc()

print("DEV5_NERFACC_JSON=" + json.dumps(out, sort_keys=True))
"""

    env = os.environ.copy()
    env["PYTHONNOUSERSITE"] = "1"
    env["PYTHONDONTWRITEBYTECODE"] = "1"

    previous_pythonpath = env.get("PYTHONPATH", "")
    pythonpath_parts = [str(tcnn_runtime), str(nerfstudio)]
    if previous_pythonpath:
        pythonpath_parts.append(previous_pythonpath)
    env["PYTHONPATH"] = os.pathsep.join(pythonpath_parts)

    pyver = f"python{sys.version_info.major}.{sys.version_info.minor}"
    torch_lib = python.parent.parent / f"lib/{pyver}/site-packages/torch/lib"
    previous_ld = env.get("LD_LIBRARY_PATH", "")
    ld_parts = [
        str(torch_lib),
        "/opt/rocm/lib",
        "/opt/rocm/lib64",
    ]
    if previous_ld:
        ld_parts.append(previous_ld)
    env["LD_LIBRARY_PATH"] = os.pathsep.join(ld_parts)

    result = run_command(
        [str(python), "-c", code],
        cwd=Path("/tmp"),
        env=env,
        timeout=120,
    )

    payload: dict[str, Any] = {}
    for line in reversed(result.get("stdout", "").splitlines()):
        if line.startswith("DEV5_NERFACC_JSON="):
            try:
                payload = json.loads(line.split("=", 1)[1])
            except Exception as exc:
                payload = {"error": repr(exc)}
            break

    if not payload:
        payload = {
            "error": "DEV5_NERFACC_JSON_MISSING",
            "stdout": result.get("stdout", ""),
            "stderr": result.get("stderr", ""),
        }

    payload["process"] = result
    payload["environment"] = {
        "PYTHONPATH": env["PYTHONPATH"],
        "LD_LIBRARY_PATH": env["LD_LIBRARY_PATH"],
        "torch_lib": str(torch_lib),
        "torch_lib_exists": torch_lib.is_dir(),
        "load_order": "torch_then_nerfacc_csrc",
    }
    payload["checks"] = {
        "process": result.get("returncode") == 0,
        "no_error": payload.get("no_error") is True,
        "torch": payload.get("torch") == "2.13.0+rocm7.2",
        "hip": payload.get("hip") == "7.2.53211",
        "cuda_available": payload.get("cuda_available") is True,
        "gcn_arch": payload.get("gcn_arch") == "gfx1201",
        "native_hash": payload.get("sha256") == EXPECTED_NERFACC_NATIVE_SHA,
        "torch_lib": torch_lib.is_dir(),
    }
    payload["passed"] = all(payload["checks"].values())
    return payload

def create_gate(report: dict[str, Any]) -> str:
    checks = report.get("checks", {})
    lines = [
        "AMD_NERFSTUDIO_PUBLIC_DEV5_P0_P1_V1",
        "",
        f"classification={CLASSIFICATION}",
        f"decision={report.get('decision')}",
        f"run_id={report.get('run_id')}",
        f"dataset={report.get('dataset')}",
        f"quick_run_dir={report.get('quick_run_dir')}",
        "declared_gate_count=28",
        "p2_execution=NOT_RUN",
        "p2_policy=MAINTAINER_ONLY",
        "replacement_runs=NONE",
        "",
    ]
    for name in CHECK_ORDER:
        lines.append(f"PUBLIC_RDNA4_DEV5_{name}: {'PASS' if checks.get(name) else 'FAIL'}")
    lines.extend([
        "",
        "blockers=" + (",".join(report.get("blockers", [])) if report.get("blockers") else "NONE"),
        "",
        "PUBLIC_RDNA4_DEV5_P0_P1: PASS" if report.get("passed") else "PUBLIC_RDNA4_DEV5_P0_P1: FAIL",
    ])
    return "\n".join(lines) + "\n"


def inventory(root: Path, exclude: set[str] | None = None) -> dict[str, dict[str, Any]]:
    excluded = exclude or set()
    rows: dict[str, dict[str, Any]] = {}
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.name in excluded:
            continue
        relative = str(path.relative_to(root))
        rows[relative] = {"sha256": sha256(path), "size_bytes": path.stat().st_size}
    return rows


def write_deterministic_archive(source: Path, destination: Path) -> str:
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("wb") as raw:
        with gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as gz:
            with tarfile.open(fileobj=gz, mode="w") as tf:
                for path in sorted(source.rglob("*")):
                    relative = Path(source.name) / path.relative_to(source)
                    info = tf.gettarinfo(str(path), arcname=str(relative))
                    info.uid = 0
                    info.gid = 0
                    info.uname = "root"
                    info.gname = "root"
                    info.mtime = 0
                    if path.is_file():
                        with path.open("rb") as handle:
                            tf.addfile(info, handle)
                    else:
                        tf.addfile(info)
    return sha256(destination)


def orchestrate(args: argparse.Namespace) -> int:
    repo_root = Path(__file__).resolve().parents[1]
    python = args.python.expanduser().resolve()
    nerfstudio = args.nerfstudio_worktree.expanduser().resolve()
    tcnn_runtime = args.tcnn_runtime.expanduser().resolve()
    dataset_archive = args.dataset_archive.expanduser().resolve()
    dataset = args.dataset.expanduser().resolve()
    output_root = args.output_root.expanduser().resolve()
    reference = args.reference.expanduser().resolve()
    run_id = args.run_id or time.strftime("%Y%m%dT%H%M%SZ", time.gmtime()) + f"_{os.getpid()}"
    run_dir = output_root / "public_dev5_p0_p1_v1" / run_id
    if run_dir.exists():
        raise RuntimeError(f"run directory already exists: {run_dir}")
    run_dir.mkdir(parents=True)

    dataset_report_path = run_dir / "dataset_deployment.json"
    deploy_cmd = [
        str(python), str(repo_root / "tools/deploy_public_quick_dataset_v2.py"),
        "--archive", str(dataset_archive),
        "--destination", str(dataset),
        "--contract", str(repo_root / "config/quick_validation_dataset_v2.json"),
        "--report", str(dataset_report_path),
    ]
    deploy_process = run_command(deploy_cmd, cwd=Path("/tmp"), timeout=300)
    (run_dir / "dataset_deployment.log").write_text(
        deploy_process.get("stdout", "") + "\n--- STDERR ---\n" + deploy_process.get("stderr", ""), encoding="utf-8"
    )
    dataset_report = json.loads(dataset_report_path.read_text(encoding="utf-8")) if dataset_report_path.is_file() else {}

    repo_identity = git_identity(repo_root)
    ns_identity = git_identity(nerfstudio)
    ns_identity["commit_matches"] = ns_identity.get("head") == EXPECTED_NERFSTUDIO_COMMIT
    ns_identity["tree_matches"] = ns_identity.get("tree") == EXPECTED_NERFSTUDIO_TREE
    ns_identity["pinned"] = bool(ns_identity.get("passed") and ns_identity["commit_matches"] and ns_identity["tree_matches"])

    tiny_native = find_exact_one(tcnn_runtime, "tinycudann_bindings/_120_C*.so")
    tiny_identity = {
        "path": str(tiny_native) if tiny_native else None,
        "sha256": sha256(tiny_native) if tiny_native else None,
    }
    tiny_identity["passed"] = tiny_identity["sha256"] == EXPECTED_TINY_NATIVE_SHA
    nerfacc_identity = probe_nerfacc(python, nerfstudio, tcnn_runtime)

    quick_process: dict[str, Any] = {"skipped": True, "reason": "DATASET_OR_RUNTIME_PREFLIGHT_FAILED"}
    quick_report: dict[str, Any] = {}
    quick_dir: Path | None = None
    preconditions = bool(
        dataset_report.get("passed")
        and repo_identity.get("passed")
        and ns_identity.get("pinned")
        and tiny_identity.get("passed")
        and nerfacc_identity.get("passed")
    )
    if preconditions:
        quick_run_id = f"{run_id}_quick"
        quick_dir = output_root / "public_quick_validation_v1" / quick_run_id
        quick_cmd = [
            str(python), str(repo_root / "tools/run_public_quick_validation_v1.py"),
            "--python", str(python),
            "--nerfstudio-worktree", str(nerfstudio),
            "--tcnn-runtime", str(tcnn_runtime),
            "--data", str(dataset),
            "--output-root", str(output_root),
            "--reference", str(reference),
            "--run-id", quick_run_id,
            "--seed", str(args.seed),
            "--rays", str(args.rays),
            "--timeout", str(args.timeout),
        ]
        if args.keep_checkpoints:
            quick_cmd.append("--keep-checkpoints")
        quick_process = run_command(quick_cmd, cwd=Path("/tmp"), timeout=args.timeout * 4)
        (run_dir / "quick_validation.log").write_text(
            quick_process.get("stdout", "") + "\n--- STDERR ---\n" + quick_process.get("stderr", ""), encoding="utf-8"
        )
        aggregate = quick_dir / "final_aggregate.json"
        if aggregate.is_file():
            quick_report = json.loads(aggregate.read_text(encoding="utf-8"))

    quick_checks = quick_report.get("checks", {})
    p1_checks = quick_report.get("p1_result", {}).get("checks", {})
    dataset_checks = dataset_report.get("checks", {})
    checks: dict[str, bool] = {
        "DATASET_ARCHIVE_HASH": bool(dataset_checks.get("DATASET_ARCHIVE_HASH")),
        "DATASET_ARCHIVE_SAFE_MEMBERS": bool(dataset_checks.get("DATASET_ARCHIVE_SAFE_MEMBERS")),
        "DATASET_REQUIRED_FILE_SET": bool(dataset_checks.get("DATASET_REQUIRED_FILE_SET")),
        "DATASET_FILE_HASHES": bool(dataset_checks.get("DATASET_FILE_HASHES")),
        "DATASET_SEMANTICS": bool(dataset_checks.get("DATASET_SEMANTICS")),
        "REPOSITORY_TRACKED_WORKTREE_CLEAN": bool(repo_identity.get("passed")),
        "NERFSTUDIO_SOURCE_PINNED": bool(ns_identity.get("pinned")),
        "TINY_NATIVE_IDENTITY": bool(tiny_identity.get("passed")),
        "NERFACC_NATIVE_IDENTITY": bool(nerfacc_identity.get("passed")),
        "QUICK_P0_PREFLIGHT": bool(quick_checks.get("P0_PREFLIGHT")),
        "QUICK_P1_REAL_MECHANICS": bool(quick_checks.get("P1_REAL_MECHANICS")),
        "QUICK_CHECKPOINT_POLICY": bool(quick_checks.get("CHECKPOINT_POLICY")),
        "QUICK_MANIFEST_CHAIN": bool(quick_checks.get("MANIFEST_CHAIN")),
    }
    for name in P1_CHECKS:
        checks[f"P1_{name}"] = bool(p1_checks.get(name))

    blockers = [name for name in CHECK_ORDER if not checks.get(name)]
    passed = not blockers
    report = {
        "schema": SCHEMA,
        "classification": CLASSIFICATION,
        "run_id": run_id,
        "passed": passed,
        "decision": "DEV5_P0_P1_QUALIFIED" if passed else "DEV5_P0_P1_BLOCKED",
        "blockers": blockers,
        "declared_gate_count": len(CHECK_ORDER),
        "checks": checks,
        "paths": {
            "python": str(python),
            "repository": str(repo_root),
            "nerfstudio": str(nerfstudio),
            "tcnn_runtime": str(tcnn_runtime),
            "dataset_archive": str(dataset_archive),
            "dataset": str(dataset),
            "output_root": str(output_root),
            "reference": str(reference),
        },
        "dataset": str(dataset),
        "dataset_deployment": dataset_report,
        "repository_identity": repo_identity,
        "nerfstudio_identity": ns_identity,
        "tiny_identity": tiny_identity,
        "nerfacc_identity": nerfacc_identity,
        "quick_process": quick_process,
        "quick_run_dir": str(quick_dir) if quick_dir else None,
        "quick_result": quick_report,
        "p2": {"executed": False, "policy": "MAINTAINER_ONLY", "automatic_transition": False},
        "replacement_runs": [],
        "nonclaims": [
            "FULL_NERFSTUDIO_FEATURE_COVERAGE",
            "VIEWER_OR_EXPORT_QUALIFICATION",
            "SPLATFACTO_QUALIFICATION",
            "MULTI_GPU_OR_DISTRIBUTED_TRAINING",
            "LONG_RUN_STABILITY_FROM_P0_P1",
            "PERFORMANCE_SUPERIORITY_OVER_CUDA",
        ],
    }
    json_dump(run_dir / "final_aggregate.json", report)
    gate = create_gate(report)
    (run_dir / "final_gate.txt").write_text(gate, encoding="utf-8")
    files = inventory(run_dir, exclude={"MANIFEST.json", "DEV5_EVIDENCE_SHA256SUMS.txt"})
    json_dump(run_dir / "MANIFEST.json", {"schema": SCHEMA + "-manifest", "run_id": run_id, "files": files})
    sums = [f"{row['sha256']}  {name}" for name, row in sorted(inventory(run_dir, exclude={"DEV5_EVIDENCE_SHA256SUMS.txt"}).items())]
    (run_dir / "DEV5_EVIDENCE_SHA256SUMS.txt").write_text("\n".join(sums) + "\n", encoding="utf-8")

    archive = output_root / "public_dev5_p0_p1_archives" / f"{run_id}.tar.gz"
    archive_sha = write_deterministic_archive(run_dir, archive)
    attestation = {
        "schema": SCHEMA + "-archive-attestation",
        "run_id": run_id,
        "source_run_dir": str(run_dir),
        "source_manifest_sha256": sha256(run_dir / "MANIFEST.json"),
        "archive": str(archive),
        "archive_sha256": archive_sha,
        "archive_size_bytes": archive.stat().st_size,
        "passed": True,
    }
    attestation_path = archive.with_suffix(archive.suffix + ".json")
    json_dump(attestation_path, attestation)
    (output_root / "public_dev5_p0_p1_v1.latest").write_text(str(run_dir) + "\n", encoding="utf-8")

    print(gate, end="")
    print(f"PUBLIC_DEV5_RUN_DIR={run_dir}")
    print(f"PUBLIC_DEV5_EVIDENCE_ARCHIVE={archive}")
    print(f"PUBLIC_DEV5_EVIDENCE_ARCHIVE_SHA256={archive_sha}")
    return 0 if passed else 2


def self_test() -> int:
    checks = {name: True for name in CHECK_ORDER}
    report = {
        "run_id": "fixture",
        "decision": "DEV5_P0_P1_QUALIFIED",
        "dataset": "/fixture/data",
        "quick_run_dir": "/fixture/quick",
        "checks": checks,
        "blockers": [],
        "passed": True,
    }
    gate = create_gate(report)
    passed = bool(
        len(CHECK_ORDER) == 28
        and gate.count("PUBLIC_RDNA4_DEV5_") == 29
        and "declared_gate_count=28" in gate
        and "p2_execution=NOT_RUN" in gate
        and "replacement_runs=NONE" in gate
        and "PUBLIC_RDNA4_DEV5_P0_P1: PASS" in gate
    )
    print(json.dumps({"schema": SCHEMA, "passed": passed, "gate_count": len(CHECK_ORDER), "gate": gate}, indent=2))
    return 0 if passed else 2


def main() -> int:
    parser = argparse.ArgumentParser(description="Dev5: deploy the pinned quick dataset and execute real Public P0+P1")
    parser.add_argument("--mode", choices=["run", "self-test"], default="run")
    parser.add_argument("--python", type=Path)
    parser.add_argument("--nerfstudio-worktree", type=Path)
    parser.add_argument("--tcnn-runtime", type=Path)
    parser.add_argument("--dataset-archive", type=Path)
    parser.add_argument("--dataset", type=Path)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--reference", type=Path, default=Path(__file__).resolve().parents[1] / "config/reference_gfx1201_rocm72.json")
    parser.add_argument("--run-id")
    parser.add_argument("--seed", type=int, default=20260803)
    parser.add_argument("--rays", type=int, default=1024)
    parser.add_argument("--timeout", type=int, default=1800)
    parser.add_argument("--keep-checkpoints", action="store_true")
    args = parser.parse_args()
    if args.mode == "self-test":
        return self_test()
    required = [args.python, args.nerfstudio_worktree, args.tcnn_runtime, args.dataset_archive, args.dataset, args.output_root]
    if any(value is None for value in required):
        parser.error("run mode requires --python, --nerfstudio-worktree, --tcnn-runtime, --dataset-archive, --dataset and --output-root")
    if args.rays <= 0 or args.timeout <= 0:
        parser.error("--rays and --timeout must be positive")
    return orchestrate(args)


if __name__ == "__main__":
    raise SystemExit(main())
