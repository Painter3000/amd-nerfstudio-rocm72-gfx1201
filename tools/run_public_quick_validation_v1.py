#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
sys.dont_write_bytecode = True
import time
from typing import Any

from public_toolchain_common import (
    absolute_preserving_symlink,
    inventory_tree,
    json_dump,
    run_command,
    verify_manifest,
)

SCHEMA = "amd-nerfstudio-public-quick-validation-v1"
CLASSIFICATION = "PUBLIC_NERFACTO_RDNA4_QUICK_VALIDATION_P0_PLUS_P1"


def create_gate(report: dict[str, Any]) -> str:
    checks = report.get("checks", {})
    lines = [
        "AMD_NERFSTUDIO_PUBLIC_QUICK_VALIDATION_V1",
        "",
        f"classification={CLASSIFICATION}",
        f"decision={report.get('decision')}",
        f"run_id={report.get('run_id')}",
        f"p0_run_dir={report.get('p0_run_dir')}",
        f"p1_run_dir={report.get('p1_run_dir')}",
        f"checkpoint_policy={report.get('checkpoint_policy')}",
        "p2_execution=NOT_RUN",
        "p2_policy=MAINTAINER_ONLY",
        "",
        f"PUBLIC_RDNA4_QUICK_P0_PREFLIGHT: {'PASS' if checks.get('P0_PREFLIGHT') else 'FAIL'}",
        f"PUBLIC_RDNA4_QUICK_P1_REAL_MECHANICS: {'PASS' if checks.get('P1_REAL_MECHANICS') else 'FAIL'}",
        f"PUBLIC_RDNA4_QUICK_CHECKPOINT_POLICY: {'PASS' if checks.get('CHECKPOINT_POLICY') else 'FAIL'}",
        f"PUBLIC_RDNA4_QUICK_MANIFEST_CHAIN: {'PASS' if checks.get('MANIFEST_CHAIN') else 'FAIL'}",
        "",
        "blockers=" + (",".join(report.get("blockers", [])) if report.get("blockers") else "NONE"),
        "",
        "PUBLIC_RDNA4_QUICK_VALIDATION: PASS" if report.get("passed") else "PUBLIC_RDNA4_QUICK_VALIDATION: FAIL",
    ]
    return "\n".join(lines) + "\n"


def read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected object: {path}")
    return payload


def orchestrate(args: argparse.Namespace) -> int:
    repo_root = Path(__file__).resolve().parents[1]
    output_root = args.output_root.expanduser().resolve()
    python = absolute_preserving_symlink(args.python)
    nerfstudio = args.nerfstudio_worktree.expanduser().resolve()
    runtime = args.tcnn_runtime.expanduser().resolve()
    dataset = args.data.expanduser().resolve()
    reference = args.reference.expanduser().resolve()
    run_id = args.run_id or time.strftime("%Y%m%dT%H%M%SZ", time.gmtime()) + f"_{os.getpid()}"
    quick_dir = output_root / "public_quick_validation_v1" / run_id
    if quick_dir.exists():
        raise RuntimeError(f"run directory already exists: {quick_dir}")
    quick_dir.mkdir(parents=True)

    p0_run_id = f"{run_id}_p0"
    p0_dir = output_root / "public_a5p0_preflight_v1" / p0_run_id
    p0_cmd = [
        str(python), str(repo_root / "tools/run_public_a5p0_preflight_v1.py"),
        "--python", str(python),
        "--nerfstudio-worktree", str(nerfstudio),
        "--tcnn-runtime", str(runtime),
        "--dataset", str(dataset),
        "--output-root", str(output_root),
        "--reference", str(reference),
        "--run-id", p0_run_id,
    ]
    p0 = run_command(p0_cmd, cwd=Path("/tmp"), timeout=args.timeout)
    (quick_dir / "p0_process.log").write_text(
        p0.get("stdout", "") + "\n--- STDERR ---\n" + p0.get("stderr", ""), encoding="utf-8"
    )
    p0_payload = read_json(p0_dir / "final_aggregate.json") if (p0_dir / "final_aggregate.json").is_file() else {}

    p1: dict[str, Any] = {"skipped": True, "reason": "P0_FAILED"}
    p1_payload: dict[str, Any] = {}
    p1_dir: Path | None = None
    if p0.get("returncode") == 0 and p0_payload.get("passed"):
        p1_run_id = f"{run_id}_p1"
        p1_dir = output_root / "public_a5p1_nerfacto_smoke_v1" / p1_run_id
        p1_cmd = [
            str(python), str(repo_root / "tools/run_public_a5p1_nerfacto_smoke_v1.py"),
            "--python", str(python),
            "--nerfstudio-worktree", str(nerfstudio),
            "--tcnn-runtime", str(runtime),
            "--data", str(dataset),
            "--output-root", str(output_root),
            "--preflight-run-dir", str(p0_dir),
            "--run-id", p1_run_id,
            "--seed", str(args.seed),
            "--rays", str(args.rays),
            "--timeout", str(args.timeout),
        ]
        if args.keep_checkpoints:
            p1_cmd.append("--keep-checkpoints")
        p1 = run_command(p1_cmd, cwd=Path("/tmp"), timeout=args.timeout * 3)
        (quick_dir / "p1_process.log").write_text(
            p1.get("stdout", "") + "\n--- STDERR ---\n" + p1.get("stderr", ""), encoding="utf-8"
        )
        if (p1_dir / "final_aggregate.json").is_file():
            p1_payload = read_json(p1_dir / "final_aggregate.json")

    p0_manifest = verify_manifest(p0_dir) if p0_dir.is_dir() else {"passed": False, "error": "P0_RUN_DIR_MISSING"}
    p1_manifest = verify_manifest(p1_dir) if p1_dir and p1_dir.is_dir() else {"passed": False, "error": "P1_RUN_DIR_MISSING"}
    checkpoint_report = p1_payload.get("checkpoint_retention", {})
    checks = {
        "P0_PREFLIGHT": bool(p0.get("returncode") == 0 and p0_payload.get("passed")),
        "P1_REAL_MECHANICS": bool(p1.get("returncode") == 0 and p1_payload.get("passed")),
        "CHECKPOINT_POLICY": bool(checkpoint_report.get("passed")),
        "MANIFEST_CHAIN": bool(p0_manifest.get("passed") and p1_manifest.get("passed")),
    }
    blockers = [name for name, passed in checks.items() if not passed]
    passed = all(checks.values())
    report = {
        "schema": SCHEMA,
        "classification": CLASSIFICATION,
        "run_id": run_id,
        "passed": passed,
        "decision": "PUBLIC_QUICK_VALIDATION_QUALIFIED" if passed else "PUBLIC_QUICK_VALIDATION_BLOCKED",
        "blockers": blockers,
        "checks": checks,
        "python": str(python),
        "nerfstudio_worktree": str(nerfstudio),
        "tcnn_runtime": str(runtime),
        "dataset": str(dataset),
        "output_root": str(output_root),
        "p0_run_dir": str(p0_dir),
        "p1_run_dir": str(p1_dir) if p1_dir else None,
        "checkpoint_policy": checkpoint_report.get("policy"),
        "keep_checkpoints_requested": bool(args.keep_checkpoints),
        "p0_process": p0,
        "p1_process": p1,
        "p0_manifest": p0_manifest,
        "p1_manifest": p1_manifest,
        "p0_result": p0_payload,
        "p1_result": p1_payload,
        "p2": {
            "executed": False,
            "policy": "MAINTAINER_ONLY",
            "automatic_transition": False,
        },
    }
    json_dump(quick_dir / "final_aggregate.json", report)
    (quick_dir / "final_gate.txt").write_text(create_gate(report), encoding="utf-8")
    manifest = inventory_tree(quick_dir, exclude_names={"MANIFEST.json"})
    json_dump(quick_dir / "MANIFEST.json", {"schema": SCHEMA + "-manifest", "run_id": run_id, "files": manifest["files"]})
    (output_root / "public_quick_validation_v1.latest").write_text(str(quick_dir) + "\n", encoding="utf-8")
    print(create_gate(report), end="")
    print(f"PUBLIC_QUICK_VALIDATION_RUN_DIR={quick_dir}")
    return 0 if passed else 2


def self_test() -> int:
    report = {
        "run_id": "fixture",
        "decision": "PUBLIC_QUICK_VALIDATION_QUALIFIED",
        "p0_run_dir": "/fixture/p0",
        "p1_run_dir": "/fixture/p1",
        "checkpoint_policy": "DELETE_AFTER_VERIFICATION",
        "checks": {
            "P0_PREFLIGHT": True,
            "P1_REAL_MECHANICS": True,
            "CHECKPOINT_POLICY": True,
            "MANIFEST_CHAIN": True,
        },
        "blockers": [],
        "passed": True,
    }
    gate = create_gate(report)
    ok = (
        "PUBLIC_RDNA4_QUICK_VALIDATION: PASS" in gate
        and "p2_execution=NOT_RUN" in gate
        and "p2_policy=MAINTAINER_ONLY" in gate
    )
    print(json.dumps({"schema": SCHEMA, "passed": ok, "gate": gate}, indent=2))
    return 0 if ok else 2


def main() -> int:
    parser = argparse.ArgumentParser(description="Public quick user validation: P0 plus P1 only; P2 is never launched")
    parser.add_argument("--mode", choices=["run", "self-test"], default="run")
    parser.add_argument("--python", type=Path)
    parser.add_argument("--nerfstudio-worktree", type=Path)
    parser.add_argument("--tcnn-runtime", type=Path)
    parser.add_argument("--data", type=Path)
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
    required = [args.python, args.nerfstudio_worktree, args.tcnn_runtime, args.data, args.output_root]
    if any(value is None for value in required):
        parser.error("run mode requires --python, --nerfstudio-worktree, --tcnn-runtime, --data and --output-root")
    if args.rays <= 0:
        parser.error("--rays must be positive")
    return orchestrate(args)


if __name__ == "__main__":
    raise SystemExit(main())
