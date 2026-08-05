#!/usr/bin/env python3
from __future__ import annotations

import importlib.util
from pathlib import Path
import tempfile
import zipfile
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
            self.assertEqual(workdir / "sources" / "nerfstudio", paths["nerfstudio_source"])
            self.assertEqual(workdir / "runtime" / "tiny-rdna4-nn", paths["tiny_runtime"])
            self.assertEqual(workdir / "runtime" / installer.VISER_MATH_RUNTIME_DIRNAME, paths["viser_math_runtime"])

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

    def test_reused_build_refresh_defers_global_pip_check(self) -> None:
        calls = []
        original_run_command = installer.run_command
        original_probe = installer.probe_python_packages

        def fake_run_command(argv, timeout=120, env=None):
            calls.append(list(argv))
            return {
                "argv": list(argv),
                "returncode": 0,
                "stdout": "",
                "stderr": "",
            }

        try:
            installer.run_command = fake_run_command
            installer.probe_python_packages = lambda python: {
                "passed": True,
                "packages": {},
            }
            result = installer.install_build_packages(
                Path("/managed/bin/python"),
                require_global_pip_check=False,
            )
        finally:
            installer.run_command = original_run_command
            installer.probe_python_packages = original_probe

        self.assertTrue(result["passed"])
        self.assertTrue(result["pip_check"]["skipped"])
        self.assertFalse(
            any(command[-2:] == ["pip", "check"] for command in calls)
        )

    def test_stale_full_viser_is_removed_before_strict_checks(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            report = {
                "environment_selection": {
                    "ownership": "MANAGED_AUTODETECTED",
                    "path": str(root / "venv"),
                },
                "paths": {
                    "logs": str(root / "logs"),
                },
            }
            probes = iter(
                (
                    {
                        "passed": True,
                        "payload": {
                            "installed": True,
                            "name": "viser",
                            "version": "1.0.0",
                        },
                        "process": {"returncode": 0},
                    },
                    {
                        "passed": True,
                        "payload": {
                            "installed": False,
                            "name": "viser",
                            "version": None,
                        },
                        "process": {"returncode": 0},
                    },
                )
            )
            original_probe = installer.probe_distribution
            original_logged = installer.run_command_logged
            try:
                installer.probe_distribution = lambda python, name: next(probes)
                installer.run_command_logged = lambda *args, **kwargs: {
                    "returncode": 0,
                    "stdout": "",
                    "stderr": "",
                }
                result = installer.remove_stale_full_viser_distribution(
                    report
                )
            finally:
                installer.probe_distribution = original_probe
                installer.run_command_logged = original_logged

        self.assertTrue(result["passed"])
        self.assertTrue(result["modified"])
        self.assertEqual(
            "STALE_FULL_VISER_DISTRIBUTION_REMOVED",
            result["reason"],
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

    def test_torch_requirements_are_pinned(self) -> None:
        self.assertEqual(
            [f"{name}=={version}" for name, version in installer.TORCH_PINS.items()],
            installer.torch_requirements(),
        )
        self.assertEqual("7.2.53211", installer.EXPECTED_TORCH_HIP)

    def test_tiny_source_is_locked_to_qualified_public_commit(self) -> None:
        self.assertEqual("phase4a2-model-b-public-gfx1201-pass", installer.TINY_TAG)
        self.assertEqual("b98bdcc6b2878f6cb6c10a2141e50867cec6d96a", installer.TINY_COMMIT)

    def test_external_env_torch_install_is_verify_only(self) -> None:
        report = {
            "passed": True,
            "environment_selection": {
                "ownership": "EXTERNAL_EXPLICIT",
                "path": "/tmp/external-env",
            },
        }
        result = installer.install_torch_stack(report)
        self.assertFalse(result["passed"])
        self.assertFalse(result["modified"])
        self.assertEqual("EXPLICIT_EXTERNAL_ENV_IS_VERIFY_ONLY", result["reason"])

    def test_runtime_library_path_is_environment_aware(self) -> None:
        value = installer.compose_runtime_library_path(
            Path("/env/torch/lib"), Path("/opt/rocm"), "/custom/lib:/env/torch/lib"
        )
        self.assertEqual(
            ["/env/torch/lib", "/opt/rocm/lib", "/opt/rocm/lib64", "/custom/lib"],
            value.split(installer.os.pathsep),
        )

    def test_tiny_build_requires_passing_preflight(self) -> None:
        result = installer.build_tiny_runtime({"passed": False})
        self.assertFalse(result["passed"])
        self.assertEqual("PREFLIGHT_NOT_PASSED", result["reason"])

    def test_clean_submodule_status_keeps_leading_marker(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            source = Path(td)
            (source / ".git").mkdir()
            for path in (
                source / "bindings" / "torch" / "setup.py",
                source / "src" / "rocwmma_width64_mlp.cu",
                source / "scripts" / "phase4a_hipcc_compat.sh",
            ):
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text("x\n", encoding="utf-8")
            (source / "scripts" / "phase4a_hipcc_compat.sh").chmod(0o755)
            for dep in ("cutlass", "fmt", "cmrc"):
                (source / "dependencies" / dep / ".git").mkdir(parents=True)
            values = {
                ("rev-parse", "HEAD"): installer.TINY_COMMIT,
                ("rev-parse", "HEAD^{tree}"): "tree-sha",
                ("status", "--porcelain=v1", "--untracked-files=all"): "",
                ("rev-list", "-n", "1", installer.TINY_TAG): installer.TINY_COMMIT,
                ("submodule", "status", "--recursive"): " abc dependencies/cutlass\n def dependencies/fmt",
            }
            original = installer.git_output
            try:
                installer.git_output = lambda repo, *args: values[args]
                result = installer.verify_tiny_source(source)
            finally:
                installer.git_output = original
            self.assertTrue(result["passed"])
            self.assertTrue(result["checks"]["submodules_clean"])

    def test_nerfacc_contract_is_fully_locked(self) -> None:
        self.assertEqual(
            "252ec63319461889319a3bc535c4076c3c84bfc1ff6ddb5d64e1bb8b18032e00",
            installer.NERFACC_WHEEL_SHA256,
        )
        self.assertEqual(
            "d3beee150cfa3a9ad3038a3283ff0a46953c345634d8cb6109449c5e3d04d1e2",
            installer.NERFACC_NATIVE_SHA256,
        )
        self.assertEqual(
            "d84cdf3afd7dcfc42150e0f0506db58a5ce62812",
            installer.NERFACC_SOURCE_COMMIT,
        )

    def test_nerfacc_python_dependencies_are_pinned(self) -> None:
        self.assertEqual(
            [f"{name}=={version}" for name, version in installer.NERFACC_RICH_PINS.items()],
            installer.nerfacc_python_requirements(),
        )

    def test_missing_authorized_nerfacc_wheel_is_blocked(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            report = {"paths": {"cache": str(Path(td) / "cache")}}
            result = installer.resolve_nerfacc_wheel(report, None)
            self.assertFalse(result["passed"])
            self.assertEqual("AUTHORIZED_NERFACC_WHEEL_NOT_FOUND", result["reason"])

    def test_external_env_nerfacc_install_is_verify_only(self) -> None:
        report = {
            "passed": True,
            "environment_selection": {
                "ownership": "EXTERNAL_EXPLICIT",
                "path": "/tmp/external-env",
            },
        }
        result = installer.install_nerfacc_stack(report)
        self.assertFalse(result["passed"])
        self.assertFalse(result["modified"])
        self.assertEqual("EXPLICIT_EXTERNAL_ENV_IS_VERIFY_ONLY", result["reason"])

    def test_scoped_pip_check_is_strict(self) -> None:
        clean = installer.evaluate_pip_check({
            "returncode": 0,
            "stdout": "No broken requirements found.\n",
            "stderr": "",
        })
        broken = installer.evaluate_pip_check({
            "returncode": 1,
            "stdout": "viser 1.0.0 requires yourdfpy, which is not installed.\n",
            "stderr": "",
        })
        self.assertTrue(clean["passed"])
        self.assertFalse(broken["passed"])

    def test_viser_math_member_selection_is_viewer_free(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            wheel = Path(td) / "viser.whl"
            with zipfile.ZipFile(wheel, "w") as archive:
                for name in (
                    "viser/transforms/__init__.py",
                    "viser/transforms/_base.py",
                    "viser/transforms/_se2.py",
                    "viser/transforms/_se3.py",
                    "viser/transforms/_so2.py",
                    "viser/transforms/_so3.py",
                ):
                    archive.writestr(name, "# test\n")
                archive.writestr("viser/_viser.py", "raise RuntimeError('viewer')\n")
                archive.writestr("viser-1.0.0.dist-info/licenses/LICENSE", "MIT\n")
            selection = installer.viser_math_member_names(wheel)
            self.assertTrue(selection["passed"])
            self.assertNotIn("viser/_viser.py", selection["members"])
            self.assertIn("viser/transforms/_so3.py", selection["members"])

    def test_viser_math_runtime_deploys_from_locked_wheel(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            wheel = root / "viser.whl"
            with zipfile.ZipFile(wheel, "w") as archive:
                for name in (
                    "viser/transforms/__init__.py",
                    "viser/transforms/_base.py",
                    "viser/transforms/_se2.py",
                    "viser/transforms/_se3.py",
                    "viser/transforms/_so2.py",
                    "viser/transforms/_so3.py",
                ):
                    archive.writestr(name, "# test\n")
                archive.writestr("viser/_viser.py", "raise RuntimeError('viewer')\n")
                archive.writestr("viser-1.0.0.dist-info/licenses/LICENSE", "MIT\n")
            report = {
                "paths": {
                    "viser_math_runtime": str(root / "runtime" / "viser-math-only-v1"),
                }
            }
            original_hash = installer.VISER_WHEEL_SHA256
            try:
                installer.VISER_WHEEL_SHA256 = installer.sha256_file(wheel)
                result = installer.deploy_viser_math_runtime(report, wheel)
            finally:
                installer.VISER_WHEEL_SHA256 = original_hash
            self.assertTrue(result["passed"])
            runtime = Path(result["runtime"])
            self.assertTrue((runtime / installer.VISER_MATH_MARKER).is_file())
            self.assertTrue((runtime / "viser" / "transforms" / "_so3.py").is_file())
            self.assertFalse((runtime / "viser" / "_viser.py").exists())

    def test_viser_math_runtime_replaces_unverified_existing_tree(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            wheel = root / "viser.whl"
            with zipfile.ZipFile(wheel, "w") as archive:
                for name in (
                    "viser/transforms/__init__.py",
                    "viser/transforms/_base.py",
                    "viser/transforms/_se2.py",
                    "viser/transforms/_se3.py",
                    "viser/transforms/_so2.py",
                    "viser/transforms/_so3.py",
                ):
                    archive.writestr(name, "# fixture\n")
                archive.writestr("viser-1.0.0.dist-info/licenses/LICENSE", "MIT\n")
            runtime = root / "runtime" / installer.VISER_MATH_RUNTIME_DIRNAME
            runtime.mkdir(parents=True)
            (runtime / "stale.txt").write_text("stale\n", encoding="utf-8")
            report = {
                "paths": {
                    "viser_math_runtime": str(runtime),
                }
            }
            original_hash = installer.VISER_WHEEL_SHA256
            try:
                installer.VISER_WHEEL_SHA256 = installer.sha256_file(wheel)
                result = installer.deploy_viser_math_runtime(report, wheel)
            finally:
                installer.VISER_WHEEL_SHA256 = original_hash
            self.assertTrue(result["passed"])
            self.assertTrue(result["replaced_unverified_runtime"])
            self.assertTrue(result["unverified_backup_removed"])
            self.assertFalse((runtime / "stale.txt").exists())
            self.assertTrue((runtime / installer.VISER_MATH_MARKER).is_file())

    def test_runtime_probe_installs_viewer_quarantine_before_trainer_import(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            project_repo = root / "project"
            source = root / "sources" / "nerfstudio"
            tiny_runtime = root / "runtime" / "tiny-rdna4-nn"
            viser_runtime = root / "runtime" / installer.VISER_MATH_RUNTIME_DIRNAME
            tools = project_repo / "tools"
            tools.mkdir(parents=True)
            (tools / "public_nerfacto_config_v1.py").write_text(
                "# fixture\n",
                encoding="utf-8",
            )

            module_names = [
                "nerfstudio",
                "nerfstudio.engine.trainer",
                "nerfstudio.pipelines.base_pipeline",
                "nerfstudio.data.datamanagers.parallel_datamanager",
                "nerfstudio.data.dataparsers.nerfstudio_dataparser",
                "nerfstudio.models.nerfacto",
                "nerfstudio.field_components.encodings",
                "nerfstudio.field_components.mlp",
                "viser.transforms",
            ]
            module_rows = []
            for name in module_names:
                if name == "viser.transforms":
                    path = viser_runtime / "viser" / "transforms" / "__init__.py"
                else:
                    suffix = "__init__.py" if name == "nerfstudio" else name.split(".")[-1] + ".py"
                    path = source / suffix
                module_rows.append({"name": name, "file": str(path)})

            payload = {
                "modules": module_rows,
                "python": str(root / "venv" / "bin" / "python"),
                "torch": installer.TORCH_PINS["torch"],
                "hip": installer.EXPECTED_TORCH_HIP,
                "cuda_available": True,
                "gcn_arch": installer.SUPPORTED_ARCH,
                "nerfacc_native": str(root / "nerfacc" / "csrc.so"),
                "nerfacc_native_sha256": installer.NERFACC_NATIVE_SHA256,
                "tiny_module": str(tiny_runtime / "tinycudann" / "modules.py"),
                "tiny_native": str(tiny_runtime / "tinycudann_bindings" / "_120_C.so"),
                "tiny_native_sha256": "4a561cc605bb7a6353d0eca1f9effc5ac9fcdfa3a9cb605a8cf36e1ae25b1917",
                "viewer_policy": {
                    "policy": "TENSORBOARD_ONLY_VIEWER_IMPORT_QUARANTINE",
                    "viewer_construction": "FAIL_CLOSED",
                },
                "viser_stub": True,
                "viser_module": None,
                "viser_transforms": str(
                    viser_runtime / "viser" / "transforms" / "__init__.py"
                ),
                "viser_distribution": None,
                "rich_version": installer.NERFACC_RICH_PINS["rich"],
                "pyyaml_version": installer.SCOPED_RUNTIME_IMPORT_PINS["PyYAML"],
                "rawpy_version": installer.SCOPED_RUNTIME_IMPORT_PINS["rawpy"],
                "yaml_module": str(root / "venv" / "site-packages" / "yaml" / "__init__.py"),
                "rawpy_module": str(root / "venv" / "site-packages" / "rawpy" / "__init__.py"),
                "opencv_headless": "4.10.0.84",
                "opencv_gui": None,
            }

            captured = {}
            original_run = installer.run_command

            def fake_run(argv, timeout=120, env=None):
                captured["code"] = argv[2]
                captured["env"] = dict(env or {})
                return {
                    "argv": list(argv),
                    "returncode": 0,
                    "stdout": installer.json.dumps(payload, sort_keys=True) + "\n",
                    "stderr": "",
                }

            report = {
                "arch": installer.SUPPORTED_ARCH,
                "paths": {
                    "project_repo": str(project_repo),
                    "tiny_runtime": str(tiny_runtime),
                    "viser_math_runtime": str(viser_runtime),
                },
            }

            try:
                installer.run_command = fake_run
                result = installer.probe_nerfstudio_runtime(
                    report,
                    source,
                    root / "venv" / "bin" / "python",
                )
            finally:
                installer.run_command = original_run

            self.assertTrue(result["passed"])
            code = captured["code"]
            self.assertLess(
                code.index("viewer_policy = install_viewer_free_import_quarantine()"),
                code.index("import viser\n"),
            )
            self.assertIn("VIEWER_FREE_STUB_MARKER", code)
            self.assertIn("import yaml", code)
            self.assertIn("import rawpy", code)
            self.assertEqual(
                str(tools.resolve()),
                captured["env"]["AMD_NERFSTUDIO_PROJECT_TOOLS"],
            )

    def test_scoped_runtime_import_dependencies_are_pinned(self) -> None:
        requirements = (ROOT / installer.SCOPED_RUNTIME_REQUIREMENTS).read_text(encoding="utf-8")
        constraints = (ROOT / installer.SCOPED_RUNTIME_CONSTRAINTS).read_text(encoding="utf-8")
        self.assertIn("PyYAML==6.0.3", requirements)
        self.assertIn("rawpy==0.27.0", requirements)
        self.assertIn("PyYAML==6.0.3", constraints)
        self.assertIn("rawpy==0.27.0", constraints)
        self.assertEqual(
            {"PyYAML": "6.0.3", "rawpy": "0.27.0"},
            installer.SCOPED_RUNTIME_IMPORT_PINS,
        )

    def test_nerfstudio_source_is_locked(self) -> None:
        self.assertEqual(
            "50e0e3c70c775e89333256213363badbf074f29d",
            installer.NERFSTUDIO_COMMIT,
        )
        self.assertEqual(
            "9d5ff468eeff89b66995e9984acaa378c37dc07e",
            installer.NERFSTUDIO_TREE,
        )
        self.assertEqual("1.0.0", installer.VISER_VERSION)
        self.assertEqual(
            "3be881a60f0295efd8a93df97646bbc04d070ccf8d16d8faf284eb3b70eda6eb",
            installer.VISER_WHEEL_SHA256,
        )
        self.assertEqual("amd-nerfstudio-viser-math-only-v1", installer.VISER_MATH_MARKER_SCHEMA)
        self.assertEqual("viser/transforms/", installer.VISER_MATH_PREFIX)


    def test_build_report_exposes_requested_arch(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            script = root / "amd_nerfstudio_setup.py"
            script.write_text("#!/usr/bin/env python3\n", encoding="utf-8")
            args = installer.argparse.Namespace(
                workdir=root / "install",
                env=None,
                rocm_path=Path("/opt/rocm"),
                arch=installer.SUPPORTED_ARCH,
                validation="quick",
                prepare_env=False,
                install_torch=False,
                build_tiny=False,
                install_nerfacc=False,
                nerfacc_wheel=None,
                install_nerfstudio=False,
                max_jobs=8,
                self_test=False,
                json_report=None,
            )
            originals = (
                installer.select_environment,
                installer.host_package_probes,
                installer.rocm_probe,
            )
            try:
                installer.select_environment = lambda paths, explicit_env: {
                    "passed": True,
                    "path": str(paths["env"]),
                    "ownership": "MANAGED_NEW",
                    "action": "CREATE_NEW_ENV",
                    "reason": "MANAGED_ENV_ABSENT",
                }
                installer.host_package_probes = lambda: []
                installer.rocm_probe = lambda path: {
                    "passed": True,
                    "requested": str(path),
                }
                report = installer.build_report(args, script)
            finally:
                (
                    installer.select_environment,
                    installer.host_package_probes,
                    installer.rocm_probe,
                ) = originals
            self.assertEqual(installer.SUPPORTED_ARCH, report["arch"])

    def test_nerfstudio_install_requires_passing_preflight(self) -> None:
        result = installer.install_nerfstudio_runtime({"passed": False})
        self.assertFalse(result["passed"])
        self.assertEqual("PREFLIGHT_NOT_PASSED", result["reason"])

    def test_self_test(self) -> None:
        self.assertEqual(0, installer.self_test())


if __name__ == "__main__":
    unittest.main(verbosity=2)
