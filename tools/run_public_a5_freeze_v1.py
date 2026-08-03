#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import tarfile
import time
from typing import Any

import sys
sys.dont_write_bytecode = True

from public_toolchain_common import inventory_tree, json_dump, load_json, sha256, verify_manifest

SCHEMA = "amd-nerfstudio-public-a5-freeze-v1"
CLASSIFICATION = "PUBLIC_A5_NERFACTO_RDNA4_REQUALIFICATION_FREEZE_V1"
SCOPE = "NERFACTO_TRAINING_CHAIN_NOT_FULL_NERFSTUDIO"


def semantic(run_dir: Path, expected_decision: str) -> dict[str, Any]:
    manifest = verify_manifest(run_dir)
    try:
        payload = load_json(run_dir / "final_aggregate.json")
    except Exception as exc:
        return {"passed": False, "manifest": manifest, "error": repr(exc)}
    result = {
        "manifest": manifest,
        "schema": payload.get("schema"),
        "classification": payload.get("classification"),
        "run_id": payload.get("run_id"),
        "decision": payload.get("decision"),
        "blockers": payload.get("blockers"),
        "payload_passed": payload.get("passed"),
        "dataset": payload.get("dataset"),
        "payload_sha256": sha256(run_dir / "final_aggregate.json"),
        "gate_sha256": sha256(run_dir / "final_gate.txt") if (run_dir / "final_gate.txt").is_file() else None,
        "manifest_sha256": sha256(run_dir / "MANIFEST.json") if (run_dir / "MANIFEST.json").is_file() else None,
    }
    result["passed"] = bool(
        manifest.get("passed")
        and payload.get("passed") is True
        and payload.get("decision") == expected_decision
        and not payload.get("blockers")
    )
    return result


def _dataset_path(payload: dict[str, Any]) -> str | None:
    value = payload.get("dataset")
    if isinstance(value, str):
        return str(Path(value).resolve())
    if isinstance(value, dict):
        candidate = value.get("dataset_dir") or value.get("data_argument") or value.get("path")
        if candidate:
            return str(Path(candidate).resolve())
    paths = payload.get("paths")
    if isinstance(paths, dict):
        row = paths.get("dataset")
        if isinstance(row, dict) and row.get("path"):
            return str(Path(row["path"]).resolve())
    return None


def same_anchor(p0: dict[str, Any], p1: dict[str, Any], p2: dict[str, Any]) -> dict[str, Any]:
    values = [_dataset_path(payload) for payload in [p0, p1, p2]]
    present = [value for value in values if value]
    return {"passed": len(present) == 3 and len(set(present)) == 1, "dataset_values": values}


def self_test() -> int:
    ok = SCOPE == "NERFACTO_TRAINING_CHAIN_NOT_FULL_NERFSTUDIO"
    print(json.dumps({"schema": SCHEMA, "passed": ok, "scope": SCOPE, "archive_default": False}, indent=2))
    return 0 if ok else 2


def main() -> int:
    parser = argparse.ArgumentParser(description="Freeze a successful public P0/P1/P2 requalification chain without modifying the private canonical A5 freeze")
    parser.add_argument("--mode", choices=["run", "self-test"], default="run")
    parser.add_argument("--p0-run-dir", type=Path)
    parser.add_argument("--p1-run-dir", type=Path)
    parser.add_argument("--p2-run-dir", type=Path)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--reference", type=Path, default=Path(__file__).resolve().parents[1] / "config/reference_gfx1201_rocm72.json")
    parser.add_argument("--run-id")
    parser.add_argument("--create-archive", action="store_true")
    args = parser.parse_args()
    if args.mode == "self-test":
        return self_test()
    required = [args.p0_run_dir, args.p1_run_dir, args.p2_run_dir, args.output_root]
    if any(value is None for value in required):
        parser.error("run mode requires --p0-run-dir, --p1-run-dir, --p2-run-dir and --output-root")

    p0_dir = args.p0_run_dir.expanduser().resolve()
    p1_dir = args.p1_run_dir.expanduser().resolve()
    p2_dir = args.p2_run_dir.expanduser().resolve()
    output_root = args.output_root.expanduser().resolve()
    reference = args.reference.expanduser().resolve()
    run_id = args.run_id or time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    freeze_dir = output_root / "public_a5_freeze_v1" / run_id
    freeze_dir.mkdir(parents=True, exist_ok=False)

    p0 = semantic(p0_dir, "PROCEED_TO_PUBLIC_A5_P1")
    p1 = semantic(p1_dir, "PROCEED_TO_PUBLIC_A5_P2")
    p2 = semantic(p2_dir, "PUBLIC_A5_P2_QUALIFIED")
    try:
        p0_payload = load_json(p0_dir / "final_aggregate.json")
        p1_payload = load_json(p1_dir / "final_aggregate.json")
        p2_payload = load_json(p2_dir / "final_aggregate.json")
    except Exception:
        p0_payload, p1_payload, p2_payload = {}, {}, {}
    dataset_chain = same_anchor(p0_payload, p1_payload, p2_payload)
    reference_row = {"path": str(reference), "exists": reference.is_file(), "sha256": sha256(reference) if reference.is_file() else None}

    checks = {
        "PUBLIC_A5_P0_MANIFEST_AND_SEMANTICS": p0.get("passed", False),
        "PUBLIC_A5_P1_MANIFEST_AND_SEMANTICS": p1.get("passed", False),
        "PUBLIC_A5_P2_MANIFEST_AND_SEMANTICS": p2.get("passed", False),
        "PUBLIC_A5_DATASET_CHAIN_CONSISTENT": dataset_chain.get("passed", False),
        "PUBLIC_REFERENCE_MANIFEST_PRESENT": reference_row["exists"],
        "PUBLIC_SCOPE_AND_NONCLAIMS_EMITTED": True,
    }
    blockers = [name for name, passed in checks.items() if not passed]
    report = {
        "schema": SCHEMA,
        "classification": CLASSIFICATION,
        "freeze_id": run_id,
        "scope": SCOPE,
        "decision": "PUBLIC_A5_FROZEN" if not blockers else "PUBLIC_A5_FREEZE_BLOCKED",
        "passed": not blockers,
        "checks": checks,
        "blockers": blockers,
        "chain": {"p0": p0, "p1": p1, "p2": p2, "dataset": dataset_chain},
        "reference": reference_row,
        "private_canonical_a5_freeze_modified": False,
        "nonclaims": [
            "FULL_NERFSTUDIO_FEATURE_COVERAGE",
            "INFINITE_HORIZON_LEAK_FREEDOM",
            "VMM_FALLBACK_PERFORMANCE_PARITY",
            "PERFORMANCE_SUPERIORITY_OVER_CUDA",
        ],
    }
    json_dump(freeze_dir / "PUBLIC_FREEZE_REPORT.json", report)
    json_dump(freeze_dir / "PUBLIC_CLAIM_SCOPE.json", {"schema": SCHEMA + "-scope", "scope": SCOPE, "nonclaims": report["nonclaims"]})
    gate = [
        "AMD_NERFSTUDIO_PUBLIC_A5_FREEZE_V1", "",
        f"classification={CLASSIFICATION}", f"decision={report['decision']}", f"freeze_id={run_id}", f"scope={SCOPE}", "",
    ]
    gate.extend(f"{name}: {'PASS' if passed else 'FAIL'}" for name, passed in checks.items())
    gate.extend(["", "blockers=" + (",".join(blockers) if blockers else "NONE"), "",
                 "PUBLIC_RDNA4_A5_REQUALIFICATION_FREEZE: PASS" if report["passed"] else "PUBLIC_RDNA4_A5_REQUALIFICATION_FREEZE: FAIL"])
    (freeze_dir / "PUBLIC_FREEZE_GATE.txt").write_text("\n".join(gate) + "\n", encoding="utf-8")
    manifest = inventory_tree(freeze_dir, exclude_names={"MANIFEST.json"})
    manifest.update({"schema": SCHEMA + "-manifest", "freeze_id": run_id})
    json_dump(freeze_dir / "MANIFEST.json", manifest)

    archive = None
    archive_row = None
    if args.create_archive and report["passed"]:
        archive = freeze_dir.with_suffix(".tar.gz")
        with tarfile.open(archive, "w:gz") as tar:
            tar.add(freeze_dir, arcname=freeze_dir.name)
        archive_row = {"path": str(archive), "sha256": sha256(archive), "size_bytes": archive.stat().st_size}
        archive.with_suffix(archive.suffix + ".sha256").write_text(f"{archive_row['sha256']}  {archive.name}\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    print("\n".join(gate))
    print(f"PUBLIC_A5_FREEZE_DIR={freeze_dir}")
    if archive:
        print(f"PUBLIC_A5_FREEZE_ARCHIVE={archive}")
        print("PUBLIC_A5_FREEZE_ARCHIVE_SHA256=" + archive_row["sha256"])
    return 0 if report["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
