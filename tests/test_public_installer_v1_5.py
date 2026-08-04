#!/usr/bin/env python3
from __future__ import annotations

import importlib.util
from pathlib import Path
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "amd_nerfstudio_setup.py"
SPEC = importlib.util.spec_from_file_location("amd_nerfstudio_setup", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
installer = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(installer)


class PublicInstallerV15Tests(unittest.TestCase):
    def test_default_workdir_is_parent_when_script_is_in_repo(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            repo = root / installer.REPO_NAME
            (repo / ".git").mkdir(parents=True)
            script = repo / "amd_nerfstudio_setup.py"
            script.write_text("#!/usr/bin/env python3\n", encoding="utf-8")
            self.assertEqual(root, installer.infer_default_workdir(script))

    def test_managed_paths_are_under_workdir(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            script = root / "amd_nerfstudio_setup.py"
            script.write_text("#!/usr/bin/env python3\n", encoding="utf-8")
            workdir = root / "install"
            paths = installer.derive_paths(script, workdir, None)
            self.assertEqual(workdir / "venv", paths["env"])
            self.assertEqual(workdir / "sources" / "tiny-rdna4-nn", paths["tiny_source"])
            self.assertEqual(workdir / "runtime" / "tiny-rdna4-nn", paths["tiny_runtime"])

    def test_only_managed_env_is_autodetected(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            script = root / "amd_nerfstudio_setup.py"
            script.write_text("#!/usr/bin/env python3\n", encoding="utf-8")
            workdir = root / "install"
            (workdir / ".venv").mkdir(parents=True)
            paths = installer.derive_paths(script, workdir, None)
            selection = installer.select_environment(paths, explicit_env=False)
            self.assertEqual("CREATE_NEW_ENV", selection["action"])
            self.assertEqual(workdir / "venv", Path(selection["path"]))

    def test_missing_explicit_env_is_blocked(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            script = root / "amd_nerfstudio_setup.py"
            script.write_text("#!/usr/bin/env python3\n", encoding="utf-8")
            workdir = root / "install"
            explicit = root / "external-env"
            paths = installer.derive_paths(script, workdir, explicit)
            selection = installer.select_environment(paths, explicit_env=True)
            self.assertFalse(selection["passed"])
            self.assertEqual("EXPLICIT_ENV_NOT_FOUND", selection["reason"])

    def test_apt_command_preserves_contract_order(self) -> None:
        text = installer.format_apt_command(["python3.12-dev", "git", "cmake"])
        self.assertLess(text.index("cmake"), text.index("git"))
        self.assertLess(text.index("git"), text.index("python3.12-dev"))
        self.assertIn("sudo apt update", text)
        self.assertIn("sudo apt install --no-install-recommends", text)


    def test_build_package_requirements_are_pinned(self) -> None:
        self.assertEqual(
            [f"{name}=={version}" for name, version in installer.BUILD_PACKAGE_PINS.items()],
            installer.build_package_requirements(),
        )

    def test_managed_marker_contract(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            env = Path(td) / "venv"
            payload = installer.managed_marker_payload(env, "READY")
            self.assertEqual("amd-nerfstudio-managed-env-v1", payload["schema"])
            self.assertEqual(str(env), payload["environment"])
            self.assertEqual("READY", payload["state"])

    def test_external_env_preparation_is_verify_only(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            report = {
                "passed": True,
                "environment_selection": {
                    "action": "REUSE_EXISTING_ENV",
                    "path": str(root / "external"),
                },
            }
            original = installer.probe_python_packages
            try:
                installer.probe_python_packages = lambda python: {"passed": False}
                result = installer.prepare_environment(report)
            finally:
                installer.probe_python_packages = original
            self.assertFalse(result["passed"])
            self.assertFalse(result["modified"])
            self.assertEqual("EXPLICIT_ENV_BUILD_BASE_INCOMPLETE", result["reason"])

    def test_self_test(self) -> None:
        self.assertEqual(0, installer.self_test())


if __name__ == "__main__":
    unittest.main(verbosity=2)
