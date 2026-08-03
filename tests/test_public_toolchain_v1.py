#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
TOOLS = ROOT / "tools"
sys.path.insert(0, str(TOOLS))

import run_public_a5p1_nerfacto_smoke_v1 as p1_tool


class PublicToolchainSelfTests(unittest.TestCase):
    def run_self_test(self, script: str) -> dict:
        proc = subprocess.run(
            [sys.executable, str(TOOLS / script), "--mode", "self-test"],
            cwd=str(ROOT), text=True, capture_output=True, check=False,
        )
        self.assertEqual(0, proc.returncode, msg=proc.stdout + "\n" + proc.stderr)
        return json.loads(proc.stdout)

    def test_preflight_self_test(self):
        self.assertTrue(self.run_self_test("run_public_a5p0_preflight_v1.py")["passed"])

    def test_p1_self_test(self):
        self.assertTrue(self.run_self_test("run_public_a5p1_nerfacto_smoke_v1.py")["passed"])

    def test_p2_self_test(self):
        self.assertTrue(self.run_self_test("run_public_a5p2_sustained_v1.py")["passed"])

    def test_quick_validation_self_test(self):
        self.assertTrue(self.run_self_test("run_public_quick_validation_v1.py")["passed"])

    def test_freeze_self_test(self):
        self.assertTrue(self.run_self_test("run_public_a5_freeze_v1.py")["passed"])

    def test_tree_audit_self_test(self):
        self.assertTrue(self.run_self_test("audit_public_tree_v1.py")["passed"])

    def test_preflight_fails_closed_on_incomplete_runtime(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            ns = root / "ns"
            runtime = root / "runtime"
            data = root / "data"
            output = root / "output"
            (ns / "nerfstudio").mkdir(parents=True)
            runtime.mkdir()
            data.mkdir()
            (data / "transforms.json").write_text('{"frames": []}\n')
            venv_bin = root / "venv" / "bin"
            venv_bin.mkdir(parents=True)
            python_link = venv_bin / "python"
            python_link.symlink_to(Path(sys.executable).resolve())
            proc = subprocess.run(
                [
                    sys.executable,
                    str(TOOLS / "run_public_a5p0_preflight_v1.py"),
                    "--python", str(python_link),
                    "--nerfstudio-worktree", str(ns),
                    "--tcnn-runtime", str(runtime),
                    "--dataset", str(data),
                    "--output-root", str(output),
                    "--run-id", "fixture",
                ],
                cwd=str(ROOT), text=True, capture_output=True, check=False,
            )
            self.assertEqual(2, proc.returncode, msg=proc.stdout + "\n" + proc.stderr)
            report = json.loads((output / "public_a5p0_preflight_v1/fixture/final_aggregate.json").read_text())
            self.assertFalse(report["passed"])
            self.assertEqual("PUBLIC_A5_P0_BLOCKED", report["decision"])
            self.assertTrue(report["blockers"])
            self.assertEqual(str(python_link.absolute()), report["paths"]["python"]["path"])
            self.assertNotEqual(str(Path(sys.executable).resolve()), report["paths"]["python"]["path"])

    def test_public_freeze_fixture(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            dataset = (root / "dataset").resolve()
            dataset.mkdir()
            payloads = {
                "p0": {
                    "schema": "fixture", "classification": "p0", "run_id": "r0",
                    "passed": True, "decision": "PROCEED_TO_PUBLIC_A5_P1", "blockers": [],
                    "dataset": {"dataset_dir": str(dataset)},
                },
                "p1": {
                    "schema": "fixture", "classification": "p1", "run_id": "r1",
                    "passed": True, "decision": "PROCEED_TO_PUBLIC_A5_P2", "blockers": [],
                    "dataset": str(dataset),
                },
                "p2": {
                    "schema": "fixture", "classification": "p2", "run_id": "r2",
                    "passed": True, "decision": "PUBLIC_A5_P2_QUALIFIED", "blockers": [],
                    "dataset": str(dataset),
                },
            }
            for name, payload in payloads.items():
                run_dir = root / name
                run_dir.mkdir()
                aggregate = run_dir / "final_aggregate.json"
                gate = run_dir / "final_gate.txt"
                aggregate.write_text(json.dumps(payload, indent=2) + "\n")
                gate.write_text("PASS\n")
                files = {}
                for path in [aggregate, gate]:
                    files[path.name] = {
                        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                        "size_bytes": path.stat().st_size,
                    }
                (run_dir / "MANIFEST.json").write_text(json.dumps({"files": files}, indent=2) + "\n")
            output = root / "output"
            proc = subprocess.run(
                [
                    sys.executable,
                    str(TOOLS / "run_public_a5_freeze_v1.py"),
                    "--p0-run-dir", str(root / "p0"),
                    "--p1-run-dir", str(root / "p1"),
                    "--p2-run-dir", str(root / "p2"),
                    "--output-root", str(output),
                    "--run-id", "fixture",
                ],
                cwd=str(ROOT), text=True, capture_output=True, check=False,
            )
            self.assertEqual(0, proc.returncode, msg=proc.stdout + "\n" + proc.stderr)
            report = json.loads((output / "public_a5_freeze_v1/fixture/PUBLIC_FREEZE_REPORT.json").read_text())
            self.assertTrue(report["passed"])
            self.assertEqual("PUBLIC_A5_FROZEN", report["decision"])
            self.assertFalse(report["private_canonical_a5_freeze_modified"])

    def test_successful_p1_deletes_verified_checkpoints_by_default(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            producer = root / "producer.ckpt"
            reload = root / "reload.ckpt"
            producer.write_bytes(b"producer-checkpoint")
            reload.write_bytes(b"reload-checkpoint")
            report = {
                "checks": {"A5_P1_REAL_MECHANICS": True},
                "producer": {
                    "checkpoint": {
                        "path": str(producer),
                        "size_bytes": producer.stat().st_size,
                        "sha256": hashlib.sha256(producer.read_bytes()).hexdigest(),
                    }
                },
                "reload": {
                    "checkpoint": {
                        "path": str(reload),
                        "size_bytes": reload.stat().st_size,
                        "sha256": hashlib.sha256(reload.read_bytes()).hexdigest(),
                    }
                },
            }
            retention = p1_tool.apply_checkpoint_retention(report, keep_checkpoints=False)
            self.assertTrue(retention["passed"])
            self.assertEqual("DELETE_AFTER_VERIFICATION", retention["policy"])
            self.assertFalse(producer.exists())
            self.assertFalse(reload.exists())

    def test_p2_wrapper_requires_maintainer_confirmation(self):
        proc = subprocess.run(
            ["bash", str(ROOT / "scripts/run_public_a5p2_sustained_v1.sh")],
            cwd=str(ROOT), text=True, capture_output=True, check=False,
        )
        self.assertEqual(64, proc.returncode)
        self.assertIn("PUBLIC_A5_P2_NOT_STARTED", proc.stderr)
        self.assertIn("run_public_quick_validation_v1.sh", proc.stderr)


if __name__ == "__main__":
    unittest.main()
