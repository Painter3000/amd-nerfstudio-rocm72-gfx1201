#!/usr/bin/env python3
from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import subprocess
import sys
sys.dont_write_bytecode = True
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
DEPLOY = ROOT / "tools/deploy_public_quick_dataset_v2.py"
DEV5 = ROOT / "tools/run_public_dev5_p0_p1_v1.py"
CONTRACT = ROOT / "config/quick_validation_dataset_v2.json"
WRAPPER = ROOT / "scripts/run_public_dev5_p0_p1_v1.sh"
REFERENCE = ROOT / "config/reference_gfx1201_rocm72.json"
FRESH_RESOURCES = ROOT / "config/public_fresh_env_resources_v1.json"
P1_RUNNER = ROOT / "tools/run_public_a5p1_nerfacto_smoke_v1.py"


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class PublicDev5Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.deploy = load_module(DEPLOY, "deploy_public_quick_dataset_v2")
        cls.dev5 = load_module(DEV5, "run_public_dev5_p0_p1_v1")
        cls.contract = json.loads(CONTRACT.read_text(encoding="utf-8"))

    def test_contract_schema(self):
        self.assertEqual("amd-nerfstudio-public-quick-dataset-contract-v2", self.contract["schema"])

    def test_archive_identity_is_pinned(self):
        self.assertEqual(
            "0a968da041884f1f815bc9176aef1a13dc72beb7531e25c5c98cf24db1db25ac",
            self.contract["archive"]["sha256"],
        )

    def test_dataset_has_six_images(self):
        pngs = sorted(name for name in self.contract["required_files"] if name.endswith(".png"))
        self.assertEqual([f"{index:03d}.png" for index in range(6)], pngs)
        self.assertEqual(6, self.contract["semantics"]["image_count"])

    def test_dataset_nonclaims_are_explicit(self):
        semantics = self.contract["semantics"]
        self.assertEqual("NOT_CLAIMED", semantics["training_quality_claim"])
        self.assertEqual("NOT_CLAIMED", semantics["geometric_accuracy_claim"])

    def test_dev5_declares_exactly_28_gates(self):
        self.assertEqual(28, len(self.dev5.CHECK_ORDER))
        self.assertEqual(28, len(set(self.dev5.CHECK_ORDER)))

    def test_dev5_never_launches_p2(self):
        source = DEV5.read_text(encoding="utf-8")
        self.assertIn('"executed": False', source)
        self.assertIn('"policy": "MAINTAINER_ONLY"', source)
        self.assertNotIn("run_public_a5p2_sustained_v1.py", source)

    def test_dev5_uses_existing_quick_runner(self):
        source = DEV5.read_text(encoding="utf-8")
        self.assertIn("run_public_quick_validation_v1.py", source)
        self.assertIn("p1_result", source)

    def test_dev5_native_hashes_match_dev4e_runtime(self):
        self.assertEqual(
            "4a561cc605bb7a6353d0eca1f9effc5ac9fcdfa3a9cb605a8cf36e1ae25b1917",
            self.dev5.EXPECTED_TINY_NATIVE_SHA,
        )
        self.assertEqual(
            "6555845d9483f672feefeef3b7ca5a264737ffe0e43ead1bbdebb661d6a3663a",
            self.dev5.EXPECTED_TINY_MODULES_SHA,
        )
        self.assertEqual(
            "d3beee150cfa3a9ad3038a3283ff0a46953c345634d8cb6109449c5e3d04d1e2",
            self.dev5.EXPECTED_NERFACC_NATIVE_SHA,
        )

    def test_active_tiny_modules_hashes_are_aligned(self):
        expected = "6555845d9483f672feefeef3b7ca5a264737ffe0e43ead1bbdebb661d6a3663a"
        reference = json.loads(REFERENCE.read_text(encoding="utf-8"))
        fresh = json.loads(FRESH_RESOURCES.read_text(encoding="utf-8"))
        p1 = P1_RUNNER.read_text(encoding="utf-8")

        self.assertEqual(
            expected,
            reference["runtime"]["tinycudann_modules_sha256"],
        )
        self.assertEqual(
            expected,
            fresh["custom_resources"]["tiny_rdna4_runtime"]["modules_sha256"],
        )
        self.assertEqual(1, p1.count(expected))
        self.assertNotIn(
            "b4df43b54f64fe2b31272a997aafd50137aecac411d59b05251acedcd5512d12",
            p1,
        )

    def test_nerfacc_probe_uses_qualified_load_order_and_library_path(self):
        source = DEV5.read_text(encoding="utf-8")
        probe_start = source.index("def probe_nerfacc")
        probe_end = source.index("\ndef create_gate", probe_start)
        probe = source[probe_start:probe_end]

        self.assertLess(
            probe.index("import torch"),
            probe.index("import nerfacc.csrc"),
        )
        self.assertIn('env["LD_LIBRARY_PATH"]', probe)
        self.assertIn("/opt/rocm/lib", probe)
        self.assertIn("DEV5_NERFACC_JSON=", probe)
        self.assertIn('"load_order": "torch_then_nerfacc_csrc"', probe)

    def test_dev5_preserves_managed_python_symlink_path(self):
        source = DEV5.read_text(encoding="utf-8")
        self.assertNotIn(
            "python = args.python.expanduser().resolve()",
            source,
        )
        self.assertIn(
            "python = Path(os.path.abspath(os.path.expanduser(str(args.python))))",
            source,
        )
        probe_start = source.index("def probe_nerfacc")
        probe_end = source.index("\ndef create_gate", probe_start)
        probe = source[probe_start:probe_end]
        self.assertIn("python_version_probe", probe)
        self.assertIn("managed_python_path_preserved", probe)
        self.assertIn("python.parent.parent", probe)
        self.assertIn('pyver == "python3.12"', probe)

    def test_tools_self_tests(self):
        for script in (DEPLOY, DEV5):
            proc = subprocess.run(
                [sys.executable, str(script), "--mode", "self-test"],
                cwd=ROOT,
                text=True,
                capture_output=True,
                check=False,
                env={**__import__("os").environ, "PYTHONDONTWRITEBYTECODE": "1"},
            )
            self.assertEqual(0, proc.returncode, msg=proc.stdout + "\n" + proc.stderr)
            payload = json.loads(proc.stdout)
            self.assertTrue(payload["passed"])

    def test_wrapper_is_fail_closed(self):
        source = WRAPPER.read_text(encoding="utf-8")
        self.assertIn("set -euo pipefail", source)
        self.assertIn("NERFSTUDIO_RDNA4_PUBLIC_DATASET_ARCHIVE", source)
        self.assertIn("run_public_dev5_p0_p1_v1.py", source)

    def test_gate_contains_no_replacement_runs(self):
        report = {
            "run_id": "fixture",
            "decision": "DEV5_P0_P1_QUALIFIED",
            "dataset": "/fixture/data",
            "quick_run_dir": "/fixture/quick",
            "checks": {name: True for name in self.dev5.CHECK_ORDER},
            "blockers": [],
            "passed": True,
        }
        gate = self.dev5.create_gate(report)
        self.assertIn("replacement_runs=NONE", gate)
        self.assertIn("p2_execution=NOT_RUN", gate)

    def test_contract_json_is_canonical_object(self):
        with tempfile.TemporaryDirectory() as td:
            copied = Path(td) / "contract.json"
            copied.write_text(json.dumps(self.contract, indent=2, sort_keys=True) + "\n", encoding="utf-8")
            self.assertEqual(self.contract, json.loads(copied.read_text(encoding="utf-8")))


if __name__ == "__main__":
    unittest.main(verbosity=2)
