#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import sysconfig
import tempfile
from datetime import datetime, timezone
from typing import Any, Callable

VERSION = "1.5.0-dev2"
SCHEMA = "amd-nerfstudio-public-installer-v1-5-env-preparation"
REPO_NAME = "amd-nerfstudio-rocm72-gfx1201"
MANAGED_ENV_NAME = "venv"
SUPPORTED_ARCH = "gfx1201"
DEFAULT_ROCM_PATH = Path("/opt/rocm")

APT_PACKAGE_ORDER = [
    "build-essential",
    "cmake",
    "ninja-build",
    "git",
    "pkg-config",
    "python3.12",
    "python3.12-venv",
    "python3.12-dev",
    "ca-certificates",
    "curl",
    "tar",
    "gzip",
    "unzip",
]

BUILD_PACKAGE_PINS = {
    "pip": "26.1.2",
    "setuptools": "83.0.0",
    "wheel": "0.47.0",
    "packaging": "26.2",
    "ninja": "1.13.0",
    "cmake": "4.4.0",
}
MANAGED_ENV_MARKER = ".amd-nerfstudio-managed-v1.json"
PYPI_INDEX = "https://pypi.org/simple"


def run_command(
    argv: list[str],
    timeout: int = 30,
    *,
    cwd: Path | None = None,
    env: dict[str, str] | None = None,
) -> dict[str, Any]:
    try:
        proc = subprocess.run(
            argv,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=timeout,
            check=False,
            cwd=str(cwd) if cwd is not None else None,
            env=env,
        )
    except Exception as exc:
        return {
            "argv": argv,
            "returncode": None,
            "stdout": "",
            "stderr": repr(exc),
        }
    return {
        "argv": argv,
        "returncode": proc.returncode,
        "stdout": proc.stdout,
        "stderr": proc.stderr,
    }


def absolute(path: Path) -> Path:
    return path.expanduser().absolute()


def infer_default_workdir(script_path: Path) -> Path:
    script_dir = script_path.resolve().parent
    if (script_dir / ".git").exists() and script_dir.name == REPO_NAME:
        return script_dir.parent
    return script_dir


def derive_paths(script_path: Path, workdir: Path, env_path: Path | None) -> dict[str, Path]:
    script_dir = script_path.resolve().parent
    workdir = absolute(workdir)
    repo_here = script_dir if (script_dir / ".git").exists() and script_dir.name == REPO_NAME else None
    repo = repo_here if repo_here is not None else workdir / REPO_NAME
    env = absolute(env_path) if env_path is not None else workdir / MANAGED_ENV_NAME
    return {
        "installer": script_path.resolve(),
        "workdir": workdir,
        "project_repo": repo,
        "nerfstudio_source": workdir / "sources" / "nerfstudio",
        "tiny_source": workdir / "sources" / "tiny-rdna4-nn",
        "tiny_build": workdir / "build" / "tiny-rdna4-nn",
        "tiny_runtime": workdir / "runtime" / "tiny-rdna4-nn",
        "env": env,
        "dataset": workdir / "datasets" / "quick-validation-dataset-v2",
        "cache": workdir / "cache",
        "reports": workdir / "reports",
        "logs": workdir / "logs",
    }


def probe_env_python(env_root: Path) -> dict[str, Any]:
    python = env_root / "bin" / "python"
    if not python.is_file():
        return {"passed": False, "python": str(python), "reason": "ENV_PYTHON_MISSING"}
    code = (
        "import json,sys; "
        "print(json.dumps({'version':list(sys.version_info[:3]),"
        "'prefix':sys.prefix,'base_prefix':getattr(sys,'base_prefix',sys.prefix)}))"
    )
    result = run_command([str(python), "-c", code], timeout=30)
    payload: dict[str, Any] = {}
    if result["returncode"] == 0:
        try:
            payload = json.loads(result["stdout"].splitlines()[-1])
        except Exception:
            payload = {}
    checks = {
        "process": result["returncode"] == 0,
        "python_3_12": payload.get("version", [0, 0])[:2] == [3, 12],
        "isolated_prefix": payload.get("prefix") not in (None, payload.get("base_prefix")),
    }
    return {
        "passed": all(checks.values()),
        "python": str(python),
        "checks": checks,
        "payload": payload,
        "process": result,
    }


def select_environment(paths: dict[str, Path], explicit_env: bool) -> dict[str, Any]:
    env = paths["env"]
    managed = paths["workdir"] / MANAGED_ENV_NAME
    if explicit_env:
        if not env.exists():
            return {
                "passed": False,
                "action": "BLOCKED",
                "reason": "EXPLICIT_ENV_NOT_FOUND",
                "ownership": "EXTERNAL_EXPLICIT" if env != managed else "MANAGED_EXPLICIT",
                "path": str(env),
            }
        probe = probe_env_python(env)
        return {
            "passed": probe["passed"],
            "action": "REUSE_EXISTING_ENV" if probe["passed"] else "BLOCKED",
            "reason": "EXPLICIT_ENV_COMPATIBLE" if probe["passed"] else "EXPLICIT_ENV_INCOMPATIBLE",
            "ownership": "EXTERNAL_EXPLICIT" if env != managed else "MANAGED_EXPLICIT",
            "path": str(env),
            "probe": probe,
        }
    if managed.exists():
        probe = probe_env_python(managed)
        return {
            "passed": probe["passed"],
            "action": "REUSE_MANAGED_ENV" if probe["passed"] else "BLOCKED",
            "reason": "MANAGED_ENV_COMPATIBLE" if probe["passed"] else "MANAGED_ENV_INCOMPATIBLE",
            "ownership": "MANAGED_AUTODETECTED",
            "path": str(managed),
            "probe": probe,
        }
    return {
        "passed": True,
        "action": "CREATE_NEW_ENV",
        "reason": "MANAGED_ENV_NOT_FOUND",
        "ownership": "MANAGED_NEW",
        "path": str(managed),
    }


def command_all(names: tuple[str, ...]) -> tuple[bool, dict[str, str | None]]:
    found = {name: shutil.which(name) for name in names}
    return all(found.values()), found


def python_header_probe() -> tuple[bool, dict[str, Any]]:
    include = Path(sysconfig.get_paths().get("include", ""))
    header = include / "Python.h"
    return header.is_file(), {"include": str(include), "header": str(header)}


def python_venv_probe() -> tuple[bool, dict[str, Any]]:
    try:
        import ensurepip  # noqa: F401
        import venv  # noqa: F401
    except Exception as exc:
        return False, {"error": repr(exc)}
    result = run_command([sys.executable, "-m", "venv", "--help"], timeout=30)
    return result["returncode"] == 0, result


def host_package_probes() -> list[dict[str, Any]]:
    py_ok = sys.version_info[:2] == (3, 12)
    header_ok, header_detail = python_header_probe()
    venv_ok, venv_detail = python_venv_probe()
    build_ok, build_detail = command_all(("gcc", "g++", "make"))
    command_map: dict[str, tuple[str, ...]] = {
        "cmake": ("cmake",),
        "ninja-build": ("ninja",),
        "git": ("git",),
        "pkg-config": ("pkg-config",),
        "curl": ("curl",),
        "tar": ("tar",),
        "gzip": ("gzip",),
        "unzip": ("unzip",),
    }
    rows: list[dict[str, Any]] = [
        {"package": "build-essential", "passed": build_ok, "detail": build_detail},
    ]
    for package, commands in command_map.items():
        ok, detail = command_all(commands)
        rows.append({"package": package, "passed": ok, "detail": detail})
    rows.extend([
        {
            "package": "python3.12",
            "passed": py_ok,
            "detail": {"executable": sys.executable, "version": platform.python_version()},
        },
        {"package": "python3.12-venv", "passed": venv_ok, "detail": venv_detail},
        {"package": "python3.12-dev", "passed": header_ok, "detail": header_detail},
        {
            "package": "ca-certificates",
            "passed": Path("/etc/ssl/certs/ca-certificates.crt").is_file(),
            "detail": {"path": "/etc/ssl/certs/ca-certificates.crt"},
        },
    ])
    by_name = {row["package"]: row for row in rows}
    return [by_name[name] for name in APT_PACKAGE_ORDER]


def apt_availability(package: str) -> str:
    if shutil.which("apt-cache") is None:
        return "UNKNOWN_APT_CACHE_MISSING"
    result = run_command(["apt-cache", "show", "--no-all-versions", package], timeout=30)
    if result["returncode"] == 0 and result["stdout"].strip():
        return "AVAILABLE"
    return "UNAVAILABLE"


def rocm_probe(rocm_path: Path) -> dict[str, Any]:
    requested = absolute(rocm_path)
    resolved = requested.resolve(strict=False)
    required = {
        "root": requested,
        "hipcc": requested / "bin" / "hipcc",
        "clangxx": requested / "lib" / "llvm" / "bin" / "clang++",
        "roc_obj_ls": requested / "bin" / "roc-obj-ls",
        "hip_runtime_header": requested / "include" / "hip" / "hip_runtime.h",
    }
    checks = {
        "root": required["root"].is_dir(),
        "hipcc": os.access(required["hipcc"], os.X_OK),
        "clangxx": os.access(required["clangxx"], os.X_OK),
        "roc_obj_ls": os.access(required["roc_obj_ls"], os.X_OK),
        "hip_runtime_header": required["hip_runtime_header"].is_file(),
        "amdhip64_library": any(
            path.is_file()
            for path in (
                requested / "lib" / "libamdhip64.so",
                requested / "lib64" / "libamdhip64.so",
            )
        ),
    }
    versions: dict[str, Any] = {}
    if checks["hipcc"]:
        versions["hipcc"] = run_command([str(required["hipcc"]), "--version"], timeout=30)
    if checks["clangxx"]:
        versions["clangxx"] = run_command([str(required["clangxx"]), "--version"], timeout=30)
    return {
        "passed": all(checks.values()),
        "requested": str(requested),
        "resolved": str(resolved),
        "checks": checks,
        "paths": {name: str(path) for name, path in required.items()},
        "versions": versions,
    }


def build_package_requirements() -> list[str]:
    return [f"{name}=={version}" for name, version in BUILD_PACKAGE_PINS.items()]


def probe_python_packages(python: Path) -> dict[str, Any]:
    names = list(BUILD_PACKAGE_PINS)
    code = r'''import json
import shutil
import sys
from importlib import metadata

names = json.loads(sys.argv[1])
versions = {}
for name in names:
    try:
        versions[name] = metadata.version(name)
    except metadata.PackageNotFoundError:
        versions[name] = None
print(json.dumps({
    "versions": versions,
    "python": sys.executable,
    "cmake": shutil.which("cmake"),
    "ninja": shutil.which("ninja"),
}, sort_keys=True))
'''
    env = dict(os.environ)
    env["PATH"] = str(python.parent) + os.pathsep + env.get("PATH", "")
    result = run_command(
        [str(python), "-c", code, json.dumps(names)],
        timeout=60,
        env=env,
    )
    payload: dict[str, Any] = {}
    if result.get("returncode") == 0:
        try:
            payload = json.loads(result.get("stdout", "").splitlines()[-1])
        except Exception:
            payload = {}
    versions = payload.get("versions", {})
    checks = {
        name: versions.get(name) == version
        for name, version in BUILD_PACKAGE_PINS.items()
    }
    checks["cmake_command_from_env"] = bool(payload.get("cmake")) and Path(payload["cmake"]).parent == python.parent
    checks["ninja_command_from_env"] = bool(payload.get("ninja")) and Path(payload["ninja"]).parent == python.parent
    return {
        "passed": result.get("returncode") == 0 and all(checks.values()),
        "checks": checks,
        "payload": payload,
        "process": result,
    }


def managed_marker_payload(env_path: Path, state: str, package_probe: dict[str, Any] | None = None) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "schema": "amd-nerfstudio-managed-env-v1",
        "installer_version": VERSION,
        "state": state,
        "environment": str(env_path),
        "python_major_minor": "3.12",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "build_package_pins": BUILD_PACKAGE_PINS,
    }
    if package_probe is not None:
        payload["package_probe"] = package_probe
    return payload


def write_managed_marker(env_path: Path, state: str, package_probe: dict[str, Any] | None = None) -> Path:
    marker = env_path / MANAGED_ENV_MARKER
    marker.write_text(
        json.dumps(managed_marker_payload(env_path, state, package_probe), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return marker


def verify_managed_marker(env_path: Path) -> dict[str, Any]:
    marker = env_path / MANAGED_ENV_MARKER
    if not marker.is_file():
        return {"passed": False, "path": str(marker), "reason": "MANAGED_ENV_MARKER_MISSING"}
    try:
        payload = json.loads(marker.read_text(encoding="utf-8"))
    except Exception as exc:
        return {"passed": False, "path": str(marker), "reason": "MANAGED_ENV_MARKER_INVALID", "error": repr(exc)}
    checks = {
        "schema": payload.get("schema") == "amd-nerfstudio-managed-env-v1",
        "environment": payload.get("environment") == str(env_path),
        "state": payload.get("state") == "READY",
    }
    return {
        "passed": all(checks.values()),
        "path": str(marker),
        "checks": checks,
        "payload": payload,
    }


def install_build_packages(python: Path) -> dict[str, Any]:
    argv = [
        str(python), "-m", "pip", "install",
        "--disable-pip-version-check",
        "--no-input",
        "--index-url", PYPI_INDEX,
        *build_package_requirements(),
    ]
    env = dict(os.environ)
    env["PATH"] = str(python.parent) + os.pathsep + env.get("PATH", "")
    install = run_command(argv, timeout=1800, env=env)
    pip_check = run_command([str(python), "-m", "pip", "check"], timeout=300, env=env)
    probe = probe_python_packages(python)
    return {
        "passed": install.get("returncode") == 0 and pip_check.get("returncode") == 0 and probe["passed"],
        "requirements": build_package_requirements(),
        "index": PYPI_INDEX,
        "install": install,
        "pip_check": pip_check,
        "probe": probe,
    }


def prepare_environment(report: dict[str, Any]) -> dict[str, Any]:
    if not report.get("passed"):
        return {"passed": False, "status": "BLOCKED", "reason": "PREFLIGHT_NOT_PASSED"}

    selection = report["environment_selection"]
    env_path = Path(selection["path"])
    python = env_path / "bin" / "python"
    action = selection["action"]

    if action == "CREATE_NEW_ENV":
        if env_path.exists():
            return {"passed": False, "status": "BLOCKED", "reason": "ENV_PATH_APPEARED_DURING_RUN"}
        env_path.parent.mkdir(parents=True, exist_ok=True)
        created = False
        try:
            creation = run_command([sys.executable, "-m", "venv", str(env_path)], timeout=600)
            if creation.get("returncode") != 0:
                return {
                    "passed": False,
                    "status": "FAIL",
                    "reason": "VENV_CREATION_FAILED",
                    "creation": creation,
                }
            created = True
            write_managed_marker(env_path, "CREATING")
            packages = install_build_packages(python)
            if not packages["passed"]:
                return {
                    "passed": False,
                    "status": "FAIL",
                    "reason": "BUILD_PACKAGE_INSTALL_FAILED",
                    "creation": creation,
                    "packages": packages,
                }
            marker = write_managed_marker(env_path, "READY", packages["probe"])
            return {
                "passed": True,
                "status": "READY",
                "reason": "MANAGED_ENV_CREATED",
                "created": True,
                "reused": False,
                "path": str(env_path),
                "python": str(python),
                "marker": str(marker),
                "creation": creation,
                "packages": packages,
            }
        finally:
            if created and env_path.exists():
                marker = env_path / MANAGED_ENV_MARKER
                ready = False
                if marker.is_file():
                    try:
                        ready = json.loads(marker.read_text(encoding="utf-8")).get("state") == "READY"
                    except Exception:
                        ready = False
                if not ready:
                    shutil.rmtree(env_path, ignore_errors=True)

    if action == "REUSE_MANAGED_ENV":
        marker = verify_managed_marker(env_path)
        if not marker["passed"]:
            return {
                "passed": False,
                "status": "BLOCKED",
                "reason": "MANAGED_ENV_OWNERSHIP_UNVERIFIED",
                "marker": marker,
            }
        packages = install_build_packages(python)
        if not packages["passed"]:
            return {
                "passed": False,
                "status": "FAIL",
                "reason": "MANAGED_ENV_BUILD_PACKAGE_REFRESH_FAILED",
                "created": False,
                "reused": True,
                "packages": packages,
            }
        marker_path = write_managed_marker(env_path, "READY", packages["probe"])
        return {
            "passed": True,
            "status": "READY",
            "reason": "MANAGED_ENV_REUSED",
            "created": False,
            "reused": True,
            "path": str(env_path),
            "python": str(python),
            "marker": str(marker_path),
            "packages": packages,
        }

    if action == "REUSE_EXISTING_ENV":
        packages = probe_python_packages(python)
        return {
            "passed": packages["passed"],
            "status": "READY" if packages["passed"] else "BLOCKED",
            "reason": "EXPLICIT_ENV_BUILD_BASE_VERIFIED" if packages["passed"] else "EXPLICIT_ENV_BUILD_BASE_INCOMPLETE",
            "created": False,
            "reused": True,
            "modified": False,
            "path": str(env_path),
            "python": str(python),
            "packages": packages,
        }

    return {"passed": False, "status": "BLOCKED", "reason": f"UNSUPPORTED_ENV_ACTION:{action}"}


def format_apt_command(packages: list[str]) -> str:
    ordered = [name for name in APT_PACKAGE_ORDER if name in packages]
    if not ordered:
        return ""
    lines = ["sudo apt update", "", "sudo apt install --no-install-recommends \\"]
    for index, package in enumerate(ordered):
        suffix = " \\" if index < len(ordered) - 1 else ""
        lines.append(f"  {package}{suffix}")
    return "\n".join(lines)


def print_paths(paths: dict[str, Path], rocm_path: Path, arch: str, validation: str) -> None:
    print(f"=== AMD Nerfstudio ROCm 7.2 / gfx1201 setup v{VERSION} ===")
    print()
    labels = [
        ("installer", "installer"),
        ("workdir", "workdir"),
        ("project repo", "project_repo"),
        ("nerfstudio", "nerfstudio_source"),
        ("tiny source", "tiny_source"),
        ("tiny build", "tiny_build"),
        ("tiny runtime", "tiny_runtime"),
        ("env", "env"),
        ("dataset", "dataset"),
        ("cache", "cache"),
        ("reports", "reports"),
        ("logs", "logs"),
    ]
    width = max(len(label) for label, _ in labels) + 1
    for label, key in labels:
        print(f"{label + ':':<{width}} {paths[key]}")
    print(f"{'rocm path:':<{width}} {absolute(rocm_path)}")
    print(f"{'arch:':<{width}} {arch}")
    print(f"{'validation:':<{width}} {validation}")
    print()


def build_report(args: argparse.Namespace, script_path: Path) -> dict[str, Any]:
    workdir = absolute(args.workdir) if args.workdir is not None else infer_default_workdir(script_path)
    paths = derive_paths(script_path, workdir, args.env)
    env_selection = select_environment(paths, args.env is not None)
    packages = host_package_probes()
    missing = [row["package"] for row in packages if not row["passed"]]
    availability = {package: apt_availability(package) for package in missing}
    rocm = rocm_probe(args.rocm_path)
    root_blocked = hasattr(os, "geteuid") and os.geteuid() == 0
    platform_checks = {
        "linux": sys.platform.startswith("linux"),
        "x86_64": platform.machine() in {"x86_64", "AMD64"},
        "arch_supported": args.arch == SUPPORTED_ARCH,
        "not_root": not root_blocked,
    }
    passed = (
        all(platform_checks.values())
        and not missing
        and rocm["passed"]
        and env_selection["passed"]
    )
    return {
        "schema": SCHEMA,
        "version": VERSION,
        "passed": passed,
        "mode": "ENV_PREPARATION" if args.prepare_env else "PREFLIGHT_ONLY",
        "paths": {key: str(value) for key, value in paths.items()},
        "platform_checks": platform_checks,
        "host_packages": packages,
        "missing_host_packages": missing,
        "apt_availability": availability,
        "rocm": rocm,
        "environment_selection": env_selection,
        "installation": {
            "status": "PREFLIGHT_PASS" if passed else "BLOCKED",
            "env_created": False,
            "system_modified": False,
            "automatic_sudo": False,
            "automatic_apt": False,
        },
    }


def print_report_summary(report: dict[str, Any]) -> None:
    env = report["environment_selection"]
    print("environment selection:")
    print(f"  path:      {env['path']}")
    print(f"  ownership: {env['ownership']}")
    print(f"  action:    {env['action']}")
    print(f"  reason:    {env['reason']}")
    print()
    print("host package installation:")
    print("  automatic sudo: DISABLED")
    print("  automatic apt:  DISABLED")
    print()

    missing = report["missing_host_packages"]
    if missing:
        available = [name for name in missing if report["apt_availability"].get(name) == "AVAILABLE"]
        unavailable = [name for name in missing if report["apt_availability"].get(name) == "UNAVAILABLE"]
        unknown = [name for name in missing if name not in available and name not in unavailable]
        print("FEHLENDE HOST-VORAUSSETZUNGEN")
        print()
        print("Die Installation kann nicht fortgesetzt werden, weil erforderliche")
        print("Systempakete oder Entwicklungswerkzeuge fehlen.")
        print()
        print("Der Installer fordert keine Administratorrechte an und führt keine")
        print("Systemänderungen automatisch durch.")
        print()
        if available:
            print("Bitte installieren Sie die verfügbaren fehlenden Pakete manuell:")
            print()
            print(format_apt_command(available))
            print()
        if unavailable:
            print("NICHT IN DEN AKTIVEN APT-QUELLEN GEFUNDEN:")
            for name in unavailable:
                print(f"  - {name}")
            print()
            if any(name.startswith("python3.12") for name in unavailable):
                print("PYTHON 3.12 NICHT ÜBER DIE AKTIVEN APT-QUELLEN VERFÜGBAR")
                print("Der Installer fügt keine Paketquellen hinzu.")
                print()
        if unknown:
            print("APT-VERFÜGBARKEIT NICHT BESTIMMBAR:")
            for name in unknown:
                print(f"  - {name}")
            print()

    if not report["rocm"]["passed"]:
        print("ROCM DEVELOPMENT STACK: BLOCKED")
        print(f"Geprüfter Pfad: {report['rocm']['requested']}")
        for name, passed in report["rocm"]["checks"].items():
            if not passed:
                print(f"  - fehlt oder unbrauchbar: {name}")
        print("Der Installer installiert ROCm nicht automatisch.")
        print()

    if not report["platform_checks"]["not_root"]:
        print("RUNNING_AS_ROOT: FAIL")
        print("Root-Ausführung wird nicht unterstützt.")
        print()

    if not env["passed"]:
        print("ENVIRONMENT_SELECTION: BLOCKED")
        print(f"REASON: {env['reason']}")
        print()

    print(f"HOST_BUILD_PREREQUISITES: {'PASS' if not report['missing_host_packages'] and report['rocm']['passed'] else 'FAIL'}")
    print(f"INSTALLATION_STATUS: {report['installation']['status']}")
    print("ENV_CREATED: NO")
    print("SYSTEM_MODIFIED: NO")


def print_environment_preparation(result: dict[str, Any]) -> None:
    print()
    print("environment preparation:")
    print(f"  status:  {result.get('status')}")
    print(f"  reason:  {result.get('reason')}")
    if result.get("path"):
        print(f"  path:    {result.get('path')}")
    if result.get("python"):
        print(f"  python:  {result.get('python')}")
    print(f"ENV_CREATED: {'YES' if result.get('created') else 'NO'}")
    print(f"ENV_REUSED: {'YES' if result.get('reused') else 'NO'}")
    print(f"PYTHON_BUILD_BASE: {'PASS' if result.get('passed') else 'FAIL'}")
    print("SYSTEM_MODIFIED: NO")
    workdir_modified = bool(result.get("created") or (result.get("reused") and result.get("modified", True)))
    print(f"WORKDIR_MODIFIED: {'YES' if workdir_modified else 'NO'}")


def self_test() -> int:
    failures: list[str] = []
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        script = root / "amd_nerfstudio_setup.py"
        script.write_text("#!/usr/bin/env python3\n", encoding="utf-8")
        workdir = root / "install-root"
        paths = derive_paths(script, workdir, None)
        if paths["env"] != workdir / "venv":
            failures.append("managed env path")
        if paths["tiny_source"] != workdir / "sources" / "tiny-rdna4-nn":
            failures.append("tiny source path")
        selection = select_environment(paths, explicit_env=False)
        if selection["action"] != "CREATE_NEW_ENV" or not selection["passed"]:
            failures.append("new env selection")
        external = root / "external-env"
        explicit_paths = derive_paths(script, workdir, external)
        selection = select_environment(explicit_paths, explicit_env=True)
        if selection["reason"] != "EXPLICIT_ENV_NOT_FOUND":
            failures.append("missing explicit env")
        command = format_apt_command(["git", "python3.12-dev", "cmake"])
        expected_order = [command.find("cmake"), command.find("git"), command.find("python3.12-dev")]
        if expected_order != sorted(expected_order):
            failures.append("apt order")
        if "sudo apt install --no-install-recommends" not in command:
            failures.append("apt command")
        requirements = build_package_requirements()
        if requirements != [f"{name}=={version}" for name, version in BUILD_PACKAGE_PINS.items()]:
            failures.append("build package requirements")
        marker = managed_marker_payload(workdir / "venv", "READY")
        if marker.get("schema") != "amd-nerfstudio-managed-env-v1":
            failures.append("managed marker schema")
    payload = {
        "schema": SCHEMA,
        "passed": not failures,
        "failures": failures,
        "tests": 7,
    }
    print(json.dumps(payload, indent=2, sort_keys=True))
    print(f"PUBLIC_INSTALLER_V1_5_SELF_TEST: {'PASS' if not failures else 'FAIL'}")
    return 0 if not failures else 2


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="AMD Nerfstudio v1.5 host preflight and managed environment preparation")
    p.add_argument("--workdir", type=Path)
    p.add_argument("--env", "--venv", dest="env", type=Path)
    p.add_argument("--rocm-path", type=Path, default=DEFAULT_ROCM_PATH)
    p.add_argument("--arch", default=SUPPORTED_ARCH)
    p.add_argument("--validation", choices=("none", "quick", "full"), default="quick")
    p.add_argument("--json-report", type=Path)
    p.add_argument("--prepare-env", action="store_true", help="Create or refresh the installer-managed environment after a passing preflight")
    p.add_argument("--self-test", action="store_true")
    return p


def main() -> int:
    args = parser().parse_args()
    if args.self_test:
        return self_test()
    script_path = Path(__file__).resolve()
    workdir = absolute(args.workdir) if args.workdir is not None else infer_default_workdir(script_path)
    paths = derive_paths(script_path, workdir, args.env)
    print_paths(paths, args.rocm_path, args.arch, args.validation)
    report = build_report(args, script_path)
    print_report_summary(report)

    preparation: dict[str, Any] | None = None
    if args.prepare_env:
        print()
        print("ENV_PREPARATION: START")
        print("Pinned Python build base:")
        for requirement in build_package_requirements():
            print(f"  - {requirement}")
        print("No sudo or apt commands will be executed.")
        print()
        preparation = prepare_environment(report)
        report["environment_preparation"] = preparation
        report["installation"] = {
            "status": "ENV_READY" if preparation.get("passed") else preparation.get("status", "BLOCKED"),
            "env_created": bool(preparation.get("created")),
            "env_reused": bool(preparation.get("reused")),
            "system_modified": False,
            "workdir_modified": bool(preparation.get("created") or (preparation.get("reused") and preparation.get("modified", True))),
            "automatic_sudo": False,
            "automatic_apt": False,
        }
        print_environment_preparation(preparation)

    output: Path | None = None
    if args.json_report is not None:
        output = absolute(args.json_report)
    elif args.prepare_env:
        output = Path(report["paths"]["reports"]) / "installer-v1.5-env-preparation.json"
    if output is not None:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(f"REPORT_JSON: {output}")

    if preparation is not None:
        return 0 if preparation.get("passed") else 2
    return 0 if report["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
