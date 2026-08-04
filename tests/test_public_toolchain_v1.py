#!/usr/bin/env python3
from __future__ import annotations

import ast
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
import setup_public_fresh_env_v1 as fresh_env_tool


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


    def test_scoped_nerfacto_config_avoids_global_method_registry(self):
        forbidden = "from nerfstudio.configs.method_configs import"
        for rel in [
            "tools/run_public_a5p0_preflight_v1.py",
            "tools/run_public_a5p1_nerfacto_smoke_v1.py",
            "tools/run_public_a5p2_sustained_v1.py",
            "tools/public_nerfacto_config_v1.py",
        ]:
            self.assertNotIn(forbidden, (ROOT / rel).read_text(encoding="utf-8"), msg=rel)
        builder = (ROOT / "tools/public_nerfacto_config_v1.py").read_text(encoding="utf-8")
        p0_source = (ROOT / "tools/run_public_a5p0_preflight_v1.py").read_text(encoding="utf-8")
        self.assertLess(
            p0_source.index("install_viewer_free_import_quarantine()"),
            p0_source.index("for name in modules:"),
        )
        for anchor in [
            'method_name="nerfacto"',
            "steps_per_save=2000",
            "max_num_iterations=30000",
            "train_num_rays_per_batch=4096",
            'CameraOptimizerConfig(mode="SO3xR3")',
        ]:
            self.assertIn(anchor, builder)

    def test_fresh_env_manifest_has_no_unqualified_custom_urls(self):
        manifest = json.loads((ROOT / "config/public_fresh_env_resources_v1.json").read_text(encoding="utf-8"))
        self.assertEqual("reference-binary-fresh-env", manifest["profile"])
        for spec in manifest["custom_resources"].values():
            self.assertIsNone(spec.get("download_url"))
            self.assertEqual("CACHE_OR_EXPLICIT_LOCAL_PATH", spec.get("availability"))
        requirements = (ROOT / "requirements/nerfacto_runtime_v1.txt").read_text(encoding="utf-8").lower()
        self.assertIn("torch==2.13.0+rocm7.2", requirements)
        self.assertIn("torchvision==0.28.0+rocm7.2", requirements)
        for excluded in ["gsplat", "open3d", "jupyterlab", "wandb", "comet-ml", "nuscenes-devkit", "viser", "pyliblzfse", "yourdfpy"]:
            self.assertNotIn(excluded, requirements)

    def test_fresh_env_contract_is_viewer_free(self):
        requirements = (ROOT / "requirements/nerfacto_runtime_v1.txt").read_text(encoding="utf-8").lower()
        constraints = (ROOT / "constraints/nerfacto_rocm72_py312_v1.txt").read_text(encoding="utf-8").lower()
        builder = (ROOT / "tools/public_nerfacto_config_v1.py").read_text(encoding="utf-8")
        for forbidden in ["viser", "pyliblzfse", "yourdfpy"]:
            self.assertNotIn(forbidden, requirements)
            self.assertNotIn(forbidden, constraints)
        self.assertIn('vis="tensorboard"', builder)
        self.assertIn("install_viewer_free_import_quarantine", builder)
        self.assertIn("VISER_VIEWER_DISABLED_BY_PUBLIC_P0_P1_CONTRACT", builder)

    def test_viewer_free_import_quarantine_is_fail_closed(self):
        code = f"""
import sys, types
sys.path.insert(0, {str(TOOLS)!r})
root = types.ModuleType('nerfstudio')
root.__path__ = []
sys.modules['nerfstudio'] = root
import public_nerfacto_config_v1 as scoped
report = scoped.install_viewer_free_import_quarantine()
import viser
from nerfstudio.viewer.viewer import Viewer
from nerfstudio.viewer_legacy.server.viewer_state import ViewerLegacyState
assert report['policy'] == 'TENSORBOARD_ONLY_VIEWER_IMPORT_QUARANTINE'
assert getattr(viser, scoped.VIEWER_FREE_STUB_MARKER) is True
for cls in (Viewer, ViewerLegacyState):
    try:
        cls()
    except scoped.ViewerDisabledError:
        pass
    else:
        raise AssertionError('viewer stub did not fail closed')
print('VIEWER_FREE_IMPORT_QUARANTINE: PASS')
"""
        proc = subprocess.run([sys.executable, "-c", code], cwd=str(ROOT), text=True, capture_output=True, check=False)
        self.assertEqual(0, proc.returncode, msg=proc.stdout + "\n" + proc.stderr)
        self.assertIn("VIEWER_FREE_IMPORT_QUARANTINE: PASS", proc.stdout)

    def test_viewer_free_quarantine_preserves_viser_transforms_bridge(self):
        code = f"""
import sys, types
sys.path.insert(0, {str(TOOLS)!r})

root = types.ModuleType('nerfstudio')
root.__path__ = []
sys.modules['nerfstudio'] = root

import public_nerfacto_config_v1 as scoped

first = scoped.install_viewer_free_import_quarantine()
second = scoped.install_viewer_free_import_quarantine()

import viser
import viser.transforms as vtf

assert first['viewer_construction'] == 'FAIL_CLOSED'
assert second['viewer_construction'] == 'FAIL_CLOSED'
assert getattr(viser, scoped.VIEWER_FREE_STUB_MARKER) is True
assert getattr(viser, '__file__', None) is None
assert hasattr(viser, '__path__')
assert viser.transforms is vtf
assert hasattr(vtf, 'SO3')

assert viser.ViserServer is scoped._ViserServerUnavailable

print('VIEWER_TRANSFORMS_BRIDGE_REGRESSION: PASS')
"""
        proc = subprocess.run(
            [sys.executable, "-c", code],
            cwd=str(ROOT),
            text=True,
            capture_output=True,
            check=False,
        )
        self.assertEqual(
            0,
            proc.returncode,
            msg=proc.stdout + "\n" + proc.stderr,
        )
        self.assertIn(
            "VIEWER_TRANSFORMS_BRIDGE_REGRESSION: PASS",
            proc.stdout,
        )

    def test_p1_quarantine_precedes_single_sh_guard(self):
        source = (
            ROOT / "tools/run_public_a5p1_nerfacto_smoke_v1.py"
        ).read_text(encoding="utf-8")

        tree = ast.parse(source)

        children = [
            node
            for node in tree.body
            if isinstance(node, ast.FunctionDef)
            and node.name == "child_execute"
        ]
        self.assertEqual(1, len(children))

        child = children[0]
        quarantine_lines = []
        sh_guard_lines = []

        for node in ast.walk(child):
            if not isinstance(node, ast.Call):
                continue
            if not isinstance(node.func, ast.Name):
                continue

            if node.func.id == "install_viewer_free_import_quarantine":
                quarantine_lines.append(node.lineno)
            elif node.func.id == "install_single_sh_guard":
                sh_guard_lines.append(node.lineno)

        self.assertEqual(
            1,
            len(quarantine_lines),
            msg="Expected one quarantine activation in child_execute()",
        )
        self.assertEqual(
            1,
            len(sh_guard_lines),
            msg="Expected one single-SH guard in child_execute()",
        )
        self.assertLess(
            quarantine_lines[0],
            sh_guard_lines[0],
            msg=(
                "Viewer quarantine must run before the SH guard imports "
                "Nerfstudio encodings"
            ),
        )
        self.assertIn(
            'report["viewer_import_policy"] = viewer_import_policy',
            source,
        )


    def test_wheelhouse_rejects_viewer_dependency_chain(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            wheelhouse = root / "wheelhouse"
            wheelhouse.mkdir()
            for name in [
                "opencv_python_headless-4.10.0.84-py3-none-any.whl",
                "viser-0.2.7-py3-none-any.whl",
                "pyliblzfse-0.4.1-cp312-cp312-linux_x86_64.whl",
                "yourdfpy-0.0.60-py3-none-any.whl",
            ]:
                (wheelhouse / name).write_bytes(name.encode("utf-8"))
            requirements = root / "requirements.txt"
            constraints = root / "constraints.txt"
            manifest = root / "manifest.json"
            requirements.write_text("opencv-python-headless==4.10.0.84\n", encoding="utf-8")
            constraints.write_text("opencv-python-headless==4.10.0.84\n", encoding="utf-8")
            manifest.write_text("{}\n", encoding="utf-8")
            lock_path = root / "lock.json"
            lock_path.write_text(json.dumps(fresh_env_tool.create_wheelhouse_lock(wheelhouse, requirements, constraints, manifest)), encoding="utf-8")
            report = fresh_env_tool.verify_wheelhouse(wheelhouse, lock_path, requirements, constraints, manifest)
            self.assertFalse(report["passed"])
            kinds = {row["kind"] for row in report["mismatches"]}
            self.assertIn("FORBIDDEN_VISER_WHEEL", kinds)
            self.assertIn("FORBIDDEN_VIEWER_CODEC_WHEEL", kinds)
            self.assertIn("FORBIDDEN_VIEWER_URDF_WHEEL", kinds)

    def test_wheelhouse_accepts_headless_opencv_without_viewer(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            wheelhouse = root / "wheelhouse"
            wheelhouse.mkdir()
            name = "opencv_python_headless-4.10.0.84-py3-none-any.whl"
            (wheelhouse / name).write_bytes(name.encode("utf-8"))
            requirements = root / "requirements.txt"
            constraints = root / "constraints.txt"
            manifest = root / "manifest.json"
            requirements.write_text("opencv-python-headless==4.10.0.84\n", encoding="utf-8")
            constraints.write_text("opencv-python-headless==4.10.0.84\n", encoding="utf-8")
            manifest.write_text("{}\n", encoding="utf-8")
            lock_path = root / "lock.json"
            lock_path.write_text(json.dumps(fresh_env_tool.create_wheelhouse_lock(wheelhouse, requirements, constraints, manifest)), encoding="utf-8")
            report = fresh_env_tool.verify_wheelhouse(wheelhouse, lock_path, requirements, constraints, manifest)
            self.assertTrue(report["passed"], msg=json.dumps(report, indent=2))

    def test_wheelhouse_rejects_duplicate_cv2_distribution_providers(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            wheelhouse = root / "wheelhouse"
            wheelhouse.mkdir()
            for name in [
                "opencv_python_headless-4.10.0.84-py3-none-any.whl",
                "opencv_python-4.14.0.94-py3-none-any.whl",
            ]:
                (wheelhouse / name).write_bytes(name.encode("utf-8"))
            requirements = root / "requirements.txt"
            constraints = root / "constraints.txt"
            manifest = root / "manifest.json"
            requirements.write_text("opencv-python-headless==4.10.0.84\n", encoding="utf-8")
            constraints.write_text("opencv-python-headless==4.10.0.84\n", encoding="utf-8")
            manifest.write_text("{}\n", encoding="utf-8")
            lock_path = root / "lock.json"
            lock_path.write_text(json.dumps(fresh_env_tool.create_wheelhouse_lock(wheelhouse, requirements, constraints, manifest)), encoding="utf-8")
            report = fresh_env_tool.verify_wheelhouse(wheelhouse, lock_path, requirements, constraints, manifest)
            self.assertFalse(report["passed"])
            kinds = {row["kind"] for row in report["mismatches"]}
            self.assertIn("DUPLICATE_CV2_DISTRIBUTION_PROVIDERS", kinds)
            self.assertIn("FORBIDDEN_GUI_OPENCV_WHEEL", kinds)

    def test_resource_manager_self_test(self):
        self.assertTrue(self.run_self_test("manage_public_resources_v1.py")["passed"])

    def test_fresh_env_installer_self_test(self):
        self.assertTrue(self.run_self_test("setup_public_fresh_env_v1.py")["passed"])

    def test_resource_manager_offline_missing_resources_fails_closed(self):
        with tempfile.TemporaryDirectory() as td:
            proc = subprocess.run(
                [
                    sys.executable,
                    str(TOOLS / "manage_public_resources_v1.py"),
                    "--resource-dir", td,
                    "--offline",
                ],
                cwd=str(ROOT), text=True, capture_output=True, check=False,
            )
            self.assertEqual(2, proc.returncode, msg=proc.stdout + "\n" + proc.stderr)
            self.assertIn("OFFLINE_MISSING_NERFSTUDIO_SOURCE", proc.stdout)
            self.assertIn("MISSING_CUSTOM_RESOURCE_NERFACC_WHEEL", proc.stdout)

    def test_fresh_native_profile_is_rejected(self):
        proc = subprocess.run(
            [
                sys.executable,
                str(TOOLS / "setup_public_fresh_env_v1.py"),
                "--profile", "fresh-native-build",
                "--resource-dir", "/tmp/not-used",
                "--download-only",
            ],
            cwd=str(ROOT), text=True, capture_output=True, check=False,
        )
        self.assertEqual(64, proc.returncode)
        self.assertIn("fresh-native-build is not claimed", proc.stderr)

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
