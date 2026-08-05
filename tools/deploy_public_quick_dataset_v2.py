#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import sys
sys.dont_write_bytecode = True
import tarfile
import tempfile
import time
from typing import Any

SCHEMA = "amd-nerfstudio-public-quick-dataset-deployment-v2"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def json_dump(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def load_contract(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema") != "amd-nerfstudio-public-quick-dataset-contract-v2":
        raise ValueError("unexpected dataset contract schema")
    if not isinstance(payload.get("required_files"), dict) or not payload["required_files"]:
        raise ValueError("dataset contract has no required_files")
    return payload


def inspect_archive(archive: Path, contract: dict[str, Any]) -> dict[str, Any]:
    root = contract["root_dir"]
    expected_files = {f"{root}/{name}" for name in contract["required_files"]}
    expected_dirs = {root}
    observed_files: set[str] = set()
    observed_dirs: set[str] = set()
    unsafe: list[dict[str, Any]] = []

    with tarfile.open(archive, "r:gz") as tf:
        for member in tf.getmembers():
            name = member.name.rstrip("/")
            pure = PurePosixPath(name)
            if pure.is_absolute() or ".." in pure.parts or not pure.parts or pure.parts[0] != root:
                unsafe.append({"name": member.name, "reason": "PATH_TRAVERSAL_OR_WRONG_ROOT"})
                continue
            if member.issym() or member.islnk() or member.isdev():
                unsafe.append({"name": member.name, "reason": "LINK_OR_DEVICE_MEMBER"})
                continue
            if member.isdir():
                observed_dirs.add(name)
            elif member.isfile():
                observed_files.add(name)
            else:
                unsafe.append({"name": member.name, "reason": "UNSUPPORTED_MEMBER_TYPE"})

    extras = sorted(observed_files - expected_files)
    missing = sorted(expected_files - observed_files)
    unexpected_dirs = sorted(observed_dirs - expected_dirs)
    passed = not unsafe and not extras and not missing and not unexpected_dirs
    return {
        "passed": passed,
        "unsafe": unsafe,
        "missing": missing,
        "extra": extras,
        "unexpected_dirs": unexpected_dirs,
        "file_count": len(observed_files),
    }


def verify_tree(root: Path, contract: dict[str, Any]) -> dict[str, Any]:
    expected = contract["required_files"]
    observed_files = {
        str(path.relative_to(root))
        for path in root.rglob("*")
        if path.is_file()
    }
    expected_files = set(expected)
    missing = sorted(expected_files - observed_files)
    extra = sorted(observed_files - expected_files)
    hash_rows: dict[str, dict[str, Any]] = {}
    for relative, expected_sha in sorted(expected.items()):
        path = root / relative
        observed_sha = sha256(path) if path.is_file() else None
        hash_rows[relative] = {
            "path": str(path),
            "expected_sha256": expected_sha,
            "observed_sha256": observed_sha,
            "passed": observed_sha == expected_sha,
        }

    semantics = contract["semantics"]
    semantic_report: dict[str, Any] = {"passed": False}
    try:
        transforms = json.loads((root / "transforms.json").read_text(encoding="utf-8"))
        provenance = json.loads((root / "DATASET_PROVENANCE.json").read_text(encoding="utf-8"))
        frames = transforms.get("frames", [])
        semantic_checks = {
            "classification": provenance.get("classification") == semantics["classification"],
            "license": provenance.get("license") == semantics["license"],
            "image_count": len(frames) == semantics["image_count"] == provenance.get("image_count"),
            "width": transforms.get("w") == semantics["width"] == provenance.get("width"),
            "height": transforms.get("h") == semantics["height"] == provenance.get("height"),
            "camera_model": transforms.get("camera_model") == semantics["camera_model"] == provenance.get("camera_model"),
            "quality_nonclaim": provenance.get("training_quality_claim") == semantics["training_quality_claim"],
            "geometry_nonclaim": provenance.get("geometric_accuracy_claim") == semantics["geometric_accuracy_claim"],
            "frame_paths": all(
                isinstance(frame, dict)
                and isinstance(frame.get("file_path"), str)
                and (root / frame["file_path"]).is_file()
                for frame in frames
            ),
        }
        semantic_report = {
            "checks": semantic_checks,
            "passed": all(semantic_checks.values()),
            "frame_count": len(frames),
            "width": transforms.get("w"),
            "height": transforms.get("h"),
            "camera_model": transforms.get("camera_model"),
        }
    except Exception as exc:
        semantic_report = {"passed": False, "error": repr(exc)}

    return {
        "root": str(root),
        "missing": missing,
        "extra": extra,
        "hashes": hash_rows,
        "file_set_passed": not missing and not extra,
        "file_hashes_passed": all(row["passed"] for row in hash_rows.values()),
        "semantics": semantic_report,
        "passed": bool(
            not missing
            and not extra
            and all(row["passed"] for row in hash_rows.values())
            and semantic_report.get("passed")
        ),
    }


def deploy(archive: Path, destination: Path, contract_path: Path, report_path: Path) -> int:
    started = time.time()
    archive = archive.expanduser().resolve()
    destination = destination.expanduser().resolve()
    contract_path = contract_path.expanduser().resolve()
    report_path = report_path.expanduser().resolve()
    contract = load_contract(contract_path)
    expected_archive_sha = contract["archive"]["sha256"]
    observed_archive_sha = sha256(archive) if archive.is_file() else None
    archive_identity = {
        "path": str(archive),
        "expected_sha256": expected_archive_sha,
        "observed_sha256": observed_archive_sha,
        "passed": observed_archive_sha == expected_archive_sha,
    }
    archive_safety = inspect_archive(archive, contract) if archive_identity["passed"] else {"passed": False, "error": "ARCHIVE_IDENTITY_FAILED"}

    existing = verify_tree(destination, contract) if destination.is_dir() else {"passed": False, "reason": "DESTINATION_ABSENT"}
    action = "REUSED_VERIFIED_DATASET" if existing.get("passed") else "DEPLOYED_ATOMICALLY"
    deployed = existing
    backup: Path | None = None
    stage_parent: Path | None = None

    if archive_identity["passed"] and archive_safety.get("passed") and not existing.get("passed"):
        destination.parent.mkdir(parents=True, exist_ok=True)
        stage_parent = Path(tempfile.mkdtemp(prefix=".quick-dataset-v2-stage-", dir=destination.parent))
        try:
            with tarfile.open(archive, "r:gz") as tf:
                tf.extractall(stage_parent, filter="data")
            staged_root = stage_parent / contract["root_dir"]
            staged_report = verify_tree(staged_root, contract)
            if not staged_report["passed"]:
                deployed = staged_report
                action = "STAGED_DATASET_VERIFICATION_FAILED"
            else:
                if destination.exists():
                    backup = destination.with_name(destination.name + f".backup-{os.getpid()}")
                    if backup.exists():
                        shutil.rmtree(backup)
                    os.replace(destination, backup)
                os.replace(staged_root, destination)
                deployed = verify_tree(destination, contract)
                if not deployed["passed"]:
                    failed = destination.with_name(destination.name + f".failed-{os.getpid()}")
                    os.replace(destination, failed)
                    if backup and backup.exists():
                        os.replace(backup, destination)
                    action = "DEPLOYED_DATASET_POSTVERIFY_FAILED_ROLLED_BACK"
                elif backup and backup.exists():
                    shutil.rmtree(backup)
        finally:
            if stage_parent.exists():
                shutil.rmtree(stage_parent)

    checks = {
        "DATASET_ARCHIVE_HASH": bool(archive_identity["passed"]),
        "DATASET_ARCHIVE_SAFE_MEMBERS": bool(archive_safety.get("passed")),
        "DATASET_REQUIRED_FILE_SET": bool(deployed.get("file_set_passed")),
        "DATASET_FILE_HASHES": bool(deployed.get("file_hashes_passed")),
        "DATASET_SEMANTICS": bool(deployed.get("semantics", {}).get("passed")),
    }
    passed = all(checks.values())
    report = {
        "schema": SCHEMA,
        "passed": passed,
        "action": action,
        "archive": archive_identity,
        "archive_safety": archive_safety,
        "destination": str(destination),
        "contract": {"path": str(contract_path), "sha256": sha256(contract_path), "payload": contract},
        "existing_before": existing,
        "deployed": deployed,
        "checks": checks,
        "duration_seconds": time.time() - started,
    }
    json_dump(report_path, report)
    print(json.dumps(report, indent=2, sort_keys=True))
    print(f"PUBLIC_QUICK_DATASET_V2_DEPLOYMENT: {'PASS' if passed else 'FAIL'}")
    print(f"DATASET_DEPLOYMENT_REPORT={report_path}")
    return 0 if passed else 2


def self_test() -> int:
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        source = root / "fixture"
        source.mkdir()
        (source / "a.txt").write_text("alpha\n", encoding="utf-8")
        contract = {
            "schema": "amd-nerfstudio-public-quick-dataset-contract-v2",
            "archive": {"filename": "fixture.tar.gz", "sha256": ""},
            "root_dir": "fixture",
            "required_files": {"a.txt": sha256(source / "a.txt")},
            "semantics": {
                "classification": "fixture",
                "license": "CC0-1.0",
                "image_count": 0,
                "width": 0,
                "height": 0,
                "camera_model": "fixture",
                "training_quality_claim": "NOT_CLAIMED",
                "geometric_accuracy_claim": "NOT_CLAIMED",
            },
        }
        archive = root / "fixture.tar.gz"
        with tarfile.open(archive, "w:gz") as tf:
            tf.add(source, arcname="fixture")
        contract["archive"]["sha256"] = sha256(archive)
        safety = inspect_archive(archive, contract)

        bad = root / "bad.tar.gz"
        payload = root / "bad.txt"
        payload.write_text("bad\n", encoding="utf-8")
        with tarfile.open(bad, "w:gz") as tf:
            tf.add(payload, arcname="../bad.txt")
        bad_contract = dict(contract)
        bad_contract["archive"] = {"filename": "bad.tar.gz", "sha256": sha256(bad)}
        bad_safety = inspect_archive(bad, bad_contract)

        passed = bool(safety["passed"] and not bad_safety["passed"])
        print(json.dumps({"schema": SCHEMA, "passed": passed, "safe": safety, "unsafe": bad_safety}, indent=2, sort_keys=True))
        return 0 if passed else 2


def main() -> int:
    parser = argparse.ArgumentParser(description="Deploy and attest quick-validation-dataset-v2 atomically")
    parser.add_argument("--mode", choices=["run", "self-test"], default="run")
    parser.add_argument("--archive", type=Path)
    parser.add_argument("--destination", type=Path)
    parser.add_argument("--contract", type=Path, default=Path(__file__).resolve().parents[1] / "config/quick_validation_dataset_v2.json")
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    if args.mode == "self-test":
        return self_test()
    if args.archive is None or args.destination is None or args.report is None:
        parser.error("run mode requires --archive, --destination and --report")
    return deploy(args.archive, args.destination, args.contract, args.report)


if __name__ == "__main__":
    raise SystemExit(main())
