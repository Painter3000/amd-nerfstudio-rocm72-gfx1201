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

VERSION = "1.5.0-dev3"
SCHEMA = "amd-nerfstudio-public-installer-v1-5-torch-tiny-native"
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
PYTORCH_INDEX = "https://download.pytorch.org/whl/rocm7.2"
TORCH_PINS = {
    "torch": "2.13.0+rocm7.2",
    "torchvision": "0.28.0+rocm7.2",
}
EXPECTED_TORCH_HIP = "7.2.53211"
TINY_REPOSITORY = "https://github.com/Painter3000/tiny-rdna4-nn.git"
TINY_TAG = "phase4a2-model-b-public-gfx1201-pass"
TINY_COMMIT = "b98bdcc6b2878f6cb6c10a2141e50867cec6d96a"
TINY_RUNTIME_MARKER = ".amd-nerfstudio-tiny-runtime-v1.json"


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


def run_command_logged(
    argv: list[str],
    log_path: Path,
    timeout: int,
    *,
    cwd: Path | None = None,
    env: dict[str, str] | None = None,
) -> dict[str, Any]:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    started = datetime.now(timezone.utc).isoformat()
    try:
        with log_path.open("w", encoding="utf-8") as log:
            log.write("COMMAND: " + " ".join(argv) + "\n")
            log.write(f"STARTED_AT_UTC: {started}\n\n")
            log.flush()
            proc = subprocess.Popen(
                argv,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                cwd=str(cwd) if cwd is not None else None,
                env=env,
                bufsize=1,
            )
            assert proc.stdout is not None
            lines: list[str] = []
            total_chars = 0
            try:
                for line in proc.stdout:
                    print(line, end="", flush=True)
                    log.write(line)
                    lines.append(line)
                    total_chars += len(line)
                    if total_chars > 1_000_000 and len(lines) > 2000:
                        lines = lines[-2000:]
                        total_chars = sum(len(item) for item in lines)
                returncode = proc.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait()
                returncode = 124
                line = f"\nTIMEOUT_AFTER_SECONDS={timeout}\n"
                print(line, end="", flush=True)
                log.write(line)
                lines.append(line)
            finished = datetime.now(timezone.utc).isoformat()
            log.write(f"\nFINISHED_AT_UTC: {finished}\nRETURN_CODE: {returncode}\n")
        return {
            "argv": argv,
            "returncode": returncode,
            "stdout_tail": "".join(lines[-2000:]),
            "stderr": "",
            "log": str(log_path),
            "started_at_utc": started,
            "finished_at_utc": finished,
        }
    except Exception as exc:
        return {
            "argv": argv,
            "returncode": None,
            "stdout_tail": "",
            "stderr": repr(exc),
            "log": str(log_path),
            "started_at_utc": started,
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


def torch_requirements() -> list[str]:
    return [f"{name}=={version}" for name, version in TORCH_PINS.items()]


def probe_torch_stack(python: Path, arch: str = SUPPORTED_ARCH) -> dict[str, Any]:
    code = r'''
from importlib import metadata
import json
import sys

payload = {
    "python": sys.executable,
    "prefix": sys.prefix,
    "base_prefix": getattr(sys, "base_prefix", sys.prefix),
    "packages": {},
}
for name in ("torch", "torchvision"):
    try:
        payload["packages"][name] = metadata.version(name)
    except metadata.PackageNotFoundError:
        payload["packages"][name] = None
try:
    import torch
    import torchvision
    payload["torch"] = {
        "version": torch.__version__,
        "hip": torch.version.hip,
        "cuda_available": bool(torch.cuda.is_available()),
        "device_count": int(torch.cuda.device_count()),
        "torchvision_module_version": torchvision.__version__,
    }
    if payload["torch"]["cuda_available"] and payload["torch"]["device_count"] > 0:
        props = torch.cuda.get_device_properties(0)
        payload["torch"].update(
            device_name=props.name,
            gcn_arch=getattr(props, "gcnArchName", None),
        )
except Exception as exc:
    payload["import_error"] = repr(exc)
print(json.dumps(payload, sort_keys=True))
'''
    result = run_command([str(python), "-c", code], timeout=300)
    payload: dict[str, Any] = {}
    if result.get("returncode") == 0:
        try:
            payload = json.loads(result.get("stdout", "").splitlines()[-1])
        except Exception:
            payload = {}
    torch_payload = payload.get("torch", {})
    packages = payload.get("packages", {})
    checks = {
        "process": result.get("returncode") == 0,
        "isolated_env": payload.get("prefix") not in (None, payload.get("base_prefix")),
        "torch_version": packages.get("torch") == TORCH_PINS["torch"] and torch_payload.get("version") == TORCH_PINS["torch"],
        "torchvision_version": packages.get("torchvision") == TORCH_PINS["torchvision"],
        "torch_rocm_build": torch_payload.get("hip") is not None,
        "hip_version": torch_payload.get("hip") == EXPECTED_TORCH_HIP,
        "cuda_available": torch_payload.get("cuda_available") is True,
        "device_count": int(torch_payload.get("device_count") or 0) >= 1,
        "gcn_arch": torch_payload.get("gcn_arch") == arch,
    }
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "payload": payload,
        "process": result,
    }


def torch_gpu_smoke(python: Path) -> dict[str, Any]:
    code = r'''
import json
import torch

torch.manual_seed(1234)
a = torch.randn((256, 256), device="cuda", dtype=torch.float32)
b = torch.randn((256, 256), device="cuda", dtype=torch.float32)
c = a @ b
torch.cuda.synchronize()
payload = {
    "device": str(c.device),
    "shape": list(c.shape),
    "finite": bool(torch.isfinite(c).all().item()),
    "checksum": float(c.abs().sum().item()),
}
payload["passed"] = (
    payload["device"] == "cuda:0"
    and payload["shape"] == [256, 256]
    and payload["finite"]
    and payload["checksum"] > 0.0
)
print(json.dumps(payload, sort_keys=True))
raise SystemExit(0 if payload["passed"] else 2)
'''
    result = run_command([str(python), "-c", code], timeout=300)
    payload: dict[str, Any] = {}
    if result.get("stdout"):
        try:
            payload = json.loads(result["stdout"].splitlines()[-1])
        except Exception:
            payload = {}
    return {
        "passed": result.get("returncode") == 0 and payload.get("passed") is True,
        "payload": payload,
        "process": result,
    }


def install_torch_stack(report: dict[str, Any]) -> dict[str, Any]:
    if not report.get("passed"):
        return {"passed": False, "status": "BLOCKED", "reason": "PREFLIGHT_NOT_PASSED"}
    selection = report["environment_selection"]
    if selection.get("ownership") == "EXTERNAL_EXPLICIT":
        return {
            "passed": False,
            "status": "BLOCKED",
            "reason": "EXPLICIT_EXTERNAL_ENV_IS_VERIFY_ONLY",
            "modified": False,
        }
    env_path = Path(selection["path"])
    python = env_path / "bin" / "python"
    marker = verify_managed_marker(env_path)
    if not marker.get("passed"):
        return {
            "passed": False,
            "status": "BLOCKED",
            "reason": "MANAGED_ENV_OWNERSHIP_UNVERIFIED",
            "marker": marker,
        }
    before = probe_torch_stack(python)
    install: dict[str, Any] | None = None
    reused = before.get("passed") is True
    if not reused:
        cache = Path(report["paths"]["cache"]) / "pip"
        cache.mkdir(parents=True, exist_ok=True)
        log = Path(report["paths"]["logs"]) / "installer-v1.5-torch-install.log"
        argv = [
            str(python), "-m", "pip", "install",
            "--disable-pip-version-check",
            "--no-input",
            "--index-url", PYTORCH_INDEX,
            "--cache-dir", str(cache),
            "--upgrade",
            *torch_requirements(),
        ]
        env = dict(os.environ)
        env["PYTHONNOUSERSITE"] = "1"
        env["PATH"] = str(python.parent) + os.pathsep + env.get("PATH", "")
        install = run_command_logged(argv, log, timeout=14400, env=env)
        if install.get("returncode") != 0:
            return {
                "passed": False,
                "status": "FAIL",
                "reason": "ROCM_PYTORCH_INSTALL_FAILED",
                "before": before,
                "install": install,
            }
    after = probe_torch_stack(python)
    smoke = torch_gpu_smoke(python) if after.get("passed") else {"passed": False, "reason": "STACK_PROBE_FAILED"}
    pip_check = run_command([str(python), "-m", "pip", "check"], timeout=300)
    freeze_path = Path(report["paths"]["reports"]) / "installer-v1.5-torch-pip-freeze.txt"
    freeze = run_command([str(python), "-m", "pip", "freeze", "--all"], timeout=300)
    freeze_path.parent.mkdir(parents=True, exist_ok=True)
    freeze_path.write_text(freeze.get("stdout", "") + freeze.get("stderr", ""), encoding="utf-8")
    passed = (
        after.get("passed") is True
        and smoke.get("passed") is True
        and pip_check.get("returncode") == 0
        and freeze.get("returncode") == 0
    )
    if passed:
        marker_payload = marker["payload"]
        marker_payload["installer_version"] = VERSION
        marker_payload["torch_stack"] = after
        marker_payload["updated_at_utc"] = datetime.now(timezone.utc).isoformat()
        Path(marker["path"]).write_text(
            json.dumps(marker_payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    return {
        "passed": passed,
        "status": "READY" if passed else "FAIL",
        "reason": "QUALIFIED_TORCH_STACK_REUSED" if passed and reused else ("QUALIFIED_TORCH_STACK_INSTALLED" if passed else "ROCM_PYTORCH_QUALIFICATION_FAILED"),
        "reused": reused,
        "modified": not reused,
        "requirements": torch_requirements(),
        "index": PYTORCH_INDEX,
        "before": before,
        "install": install,
        "after": after,
        "gpu_smoke": smoke,
        "pip_check": pip_check,
        "freeze": {"path": str(freeze_path), "process": freeze},
    }


def sha256_file(path: Path) -> str:
    import hashlib
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_output(repo: Path, *args: str) -> str:
    result = run_command(["git", "-C", str(repo), *args], timeout=300)
    if result.get("returncode") != 0:
        raise RuntimeError(f"git {' '.join(args)} failed: {result.get('stderr')}")
    return result.get("stdout", "").rstrip()


def verify_tiny_source(source: Path) -> dict[str, Any]:
    if not (source / ".git").exists():
        return {"passed": False, "reason": "TINY_SOURCE_NOT_A_GIT_CHECKOUT", "path": str(source)}
    try:
        commit = git_output(source, "rev-parse", "HEAD")
        tree = git_output(source, "rev-parse", "HEAD^{tree}")
        status = git_output(source, "status", "--porcelain=v1", "--untracked-files=all")
        tag_commit = git_output(source, "rev-list", "-n", "1", TINY_TAG)
        submodule_raw = git_output(source, "submodule", "status", "--recursive")
    except Exception as exc:
        return {"passed": False, "reason": "TINY_SOURCE_GIT_PROBE_FAILED", "error": repr(exc), "path": str(source)}
    submodule_lines = [line for line in submodule_raw.splitlines() if line.strip()]
    submodules_clean = all(line[0] == " " for line in submodule_lines)
    required = {
        "bindings_setup": (source / "bindings" / "torch" / "setup.py").is_file(),
        "hipcc_shim": os.access(source / "scripts" / "phase4a_hipcc_compat.sh", os.X_OK),
        "rocwmma_source": (source / "src" / "rocwmma_width64_mlp.cu").is_file(),
        "cutlass": (source / "dependencies" / "cutlass" / ".git").exists(),
        "fmt": (source / "dependencies" / "fmt" / ".git").exists(),
        "cmrc": (source / "dependencies" / "cmrc" / ".git").exists(),
    }
    checks = {
        "commit": commit == TINY_COMMIT,
        "tag_commit": tag_commit == TINY_COMMIT,
        "clean": status == "",
        "submodules_clean": submodules_clean,
        **required,
    }
    return {
        "passed": all(checks.values()),
        "reason": "QUALIFIED_SOURCE" if all(checks.values()) else "TINY_SOURCE_CONTRACT_MISMATCH",
        "path": str(source),
        "commit": commit,
        "tree": tree,
        "tag": TINY_TAG,
        "tag_commit": tag_commit,
        "status": status,
        "submodules": submodule_lines,
        "checks": checks,
    }


def acquire_tiny_source(report: dict[str, Any]) -> dict[str, Any]:
    source = Path(report["paths"]["tiny_source"])
    if source.exists():
        verification = verify_tiny_source(source)
        return {
            "passed": verification.get("passed") is True,
            "status": "READY" if verification.get("passed") else "BLOCKED",
            "reason": "QUALIFIED_SOURCE_REUSED" if verification.get("passed") else verification.get("reason"),
            "created": False,
            "verification": verification,
        }
    source.parent.mkdir(parents=True, exist_ok=True)
    staging = source.parent / f".{source.name}.clone-{os.getpid()}"
    if staging.exists():
        return {"passed": False, "status": "BLOCKED", "reason": "CLONE_STAGING_ALREADY_EXISTS", "path": str(staging)}
    log = Path(report["paths"]["logs"]) / "installer-v1.5-tiny-clone.log"
    clone = run_command_logged(
        [
            "git", "clone", "--recursive", "--branch", TINY_TAG,
            "--single-branch", TINY_REPOSITORY, str(staging),
        ],
        log,
        timeout=7200,
    )
    try:
        if clone.get("returncode") != 0:
            return {"passed": False, "status": "FAIL", "reason": "TINY_SOURCE_CLONE_FAILED", "clone": clone}
        verification = verify_tiny_source(staging)
        if not verification.get("passed"):
            return {"passed": False, "status": "FAIL", "reason": "CLONED_SOURCE_FAILED_CONTRACT", "clone": clone, "verification": verification}
        staging.rename(source)
        verification = verify_tiny_source(source)
        if not verification.get("passed"):
            shutil.rmtree(source, ignore_errors=True)
            return {
                "passed": False,
                "status": "FAIL",
                "reason": "POST_RENAME_SOURCE_VERIFICATION_FAILED",
                "created": False,
                "clone": clone,
                "verification": verification,
            }
        return {
            "passed": True,
            "status": "READY",
            "reason": "QUALIFIED_SOURCE_CLONED",
            "created": True,
            "clone": clone,
            "verification": verification,
        }
    finally:
        if staging.exists():
            shutil.rmtree(staging, ignore_errors=True)


def verify_tiny_runtime(runtime: Path, python: Path, rocm_path: Path, arch: str) -> dict[str, Any]:
    marker = runtime / TINY_RUNTIME_MARKER
    binary = runtime / "tinycudann_bindings" / "_120_C.cpython-312-x86_64-linux-gnu.so"
    marker_payload: dict[str, Any] = {}
    if marker.is_file():
        try:
            marker_payload = json.loads(marker.read_text(encoding="utf-8"))
        except Exception:
            marker_payload = {}
    roc_obj = run_command([str(rocm_path / "bin" / "roc-obj-ls"), str(binary)], timeout=300) if binary.is_file() else {"returncode": None, "stdout": "", "stderr": "binary missing"}
    ldd = run_command(["ldd", str(binary)], timeout=300) if binary.is_file() else {"returncode": None, "stdout": "", "stderr": "binary missing"}
    import_code = r'''
import json
import pathlib
import torch
import tinycudann as tcnn
import tinycudann.modules as modules

config = {
    "otype": "RocWMMAWidth64MLP",
    "n_neurons": 64,
    "n_hidden_layers": 2,
    "activation": "ReLU",
    "output_activation": "None",
    "precision": "Fp16",
    "bias": True,
}
net = tcnn.Network(64, 64, config, seed=1)
x = torch.randn((128, 64), device="cuda", dtype=torch.float16, requires_grad=True)
y = net(x)
loss = y.float().square().mean()
loss.backward()
torch.cuda.synchronize()
payload = {
    "module": str(pathlib.Path(modules.__file__).resolve()),
    "native": str(pathlib.Path(modules._C.__file__).resolve()),
    "shape": list(y.shape),
    "finite": bool(torch.isfinite(y).all().item()),
    "loss": float(loss.item()),
    "grad_finite": bool(torch.isfinite(x.grad).all().item()),
}
payload["passed"] = payload["shape"] == [128, 64] and payload["finite"] and payload["grad_finite"] and payload["loss"] >= 0.0
print(json.dumps(payload, sort_keys=True))
raise SystemExit(0 if payload["passed"] else 2)
'''
    env = dict(os.environ)
    env["PYTHONNOUSERSITE"] = "1"
    env["PYTHONPATH"] = str(runtime)
    env["TCNN_RDNA4_ENABLE_PORTABLE_MLP_SHIM"] = "1"
    imported = run_command([str(python), "-c", import_code], timeout=1200, env=env) if binary.is_file() else {"returncode": None, "stdout": "", "stderr": "binary missing"}
    import_payload: dict[str, Any] = {}
    if imported.get("returncode") == 0:
        try:
            import_payload = json.loads(imported.get("stdout", "").splitlines()[-1])
        except Exception:
            import_payload = {}
    binary_hash = sha256_file(binary) if binary.is_file() else None
    checks = {
        "marker": marker.is_file(),
        "marker_schema": marker_payload.get("schema") == "amd-nerfstudio-tiny-runtime-v1",
        "source_commit": marker_payload.get("source_commit") == TINY_COMMIT,
        "binary": binary.is_file() and binary.stat().st_size > 0,
        "binary_hash": marker_payload.get("native_sha256") == binary_hash,
        "binary_contains_arch": binary.is_file() and arch.encode() in binary.read_bytes(),
        "roc_obj_ls": roc_obj.get("returncode") == 0 and f"--{arch}" in roc_obj.get("stdout", ""),
        "ldd": ldd.get("returncode") == 0 and "not found" not in (ldd.get("stdout", "") + ldd.get("stderr", "")),
        "runtime_smoke": imported.get("returncode") == 0 and import_payload.get("passed") is True,
        "runtime_origin": Path(import_payload.get("native", "/__missing__")).is_file() and runtime in Path(import_payload.get("native", "/__missing__")).resolve().parents,
    }
    return {
        "passed": all(checks.values()),
        "path": str(runtime),
        "binary": str(binary),
        "native_sha256": binary_hash,
        "checks": checks,
        "marker": marker_payload,
        "roc_obj_ls": roc_obj,
        "ldd": ldd,
        "runtime_smoke": {"process": imported, "payload": import_payload},
    }


def build_tiny_runtime(report: dict[str, Any], max_jobs: int = 8) -> dict[str, Any]:
    if not report.get("passed"):
        return {"passed": False, "status": "BLOCKED", "reason": "PREFLIGHT_NOT_PASSED"}
    env_path = Path(report["environment_selection"]["path"])
    python = env_path / "bin" / "python"
    torch_probe = probe_torch_stack(python)
    if not torch_probe.get("passed"):
        return {"passed": False, "status": "BLOCKED", "reason": "QUALIFIED_TORCH_STACK_REQUIRED", "torch": torch_probe}
    source_result = acquire_tiny_source(report)
    if not source_result.get("passed"):
        return {"passed": False, "status": source_result.get("status", "FAIL"), "reason": source_result.get("reason"), "source": source_result}
    source = Path(report["paths"]["tiny_source"])
    build = Path(report["paths"]["tiny_build"])
    runtime = Path(report["paths"]["tiny_runtime"])
    rocm = Path(report["rocm"]["requested"])
    if runtime.exists():
        existing = verify_tiny_runtime(runtime, python, rocm, SUPPORTED_ARCH)
        if existing.get("passed"):
            return {
                "passed": True,
                "status": "READY",
                "reason": "QUALIFIED_TINY_RUNTIME_REUSED",
                "source": source_result,
                "runtime": existing,
                "modified": False,
            }
        return {"passed": False, "status": "BLOCKED", "reason": "EXISTING_TINY_RUNTIME_UNQUALIFIED", "source": source_result, "runtime": existing}
    if build.exists():
        return {"passed": False, "status": "BLOCKED", "reason": "EXISTING_TINY_BUILD_ROOT_UNMANAGED", "path": str(build)}
    build.parent.mkdir(parents=True, exist_ok=True)
    runtime.parent.mkdir(parents=True, exist_ok=True)
    staging_runtime = runtime.parent / f".{runtime.name}.stage-{os.getpid()}"
    if staging_runtime.exists():
        return {"passed": False, "status": "BLOCKED", "reason": "RUNTIME_STAGING_ALREADY_EXISTS", "path": str(staging_runtime)}
    created_build = False
    runtime_ready = False
    try:
        (build / "rocm" / "bin").mkdir(parents=True)
        (build / "tooling").mkdir(parents=True)
        (build / "tcnn" / "temp").mkdir(parents=True)
        (build / "tcnn" / "lib").mkdir(parents=True)
        (build / "torch_extensions").mkdir(parents=True)
        created_build = True
        shutil.copytree(rocm / "include", build / "rocm" / "include", symlinks=True)
        os.symlink(rocm / "lib", build / "rocm" / "lib", target_is_directory=True)
        shim = source / "scripts" / "phase4a_hipcc_compat.sh"
        staged_shim = build / "tooling" / "phase4a_hipcc_compat.sh"
        shutil.copy2(shim, staged_shim)
        staged_shim.chmod(0o755)
        os.symlink(staged_shim, build / "rocm" / "bin" / "hipcc")
        env = dict(os.environ)
        env.update({
            "PATH": str(python.parent) + os.pathsep + env.get("PATH", ""),
            "PYTHONNOUSERSITE": "1",
            "PYTHONDONTWRITEBYTECODE": "1",
            "PHASE4A_ROCM_REAL": str(rocm),
            "PHASE4A_ROCM_CLANGXX": str(rocm / "lib" / "llvm" / "bin" / "clang++"),
            "ROCM_HOME": str(build / "rocm"),
            "ROCM_PATH": str(build / "rocm"),
            "PYTORCH_ROCM_ARCH": SUPPORTED_ARCH,
            "MAX_JOBS": str(max_jobs),
            "TCNN_DEPENDENCY_ROOT": str(source / "dependencies"),
            "TCNN_HALF_PRECISION": "1",
            "TCNN_ENABLE_ROCWMMA_WIDTH64_MLP": "1",
            "TORCH_EXTENSIONS_DIR": str(build / "torch_extensions"),
        })
        log = Path(report["paths"]["logs"]) / "installer-v1.5-tiny-build.log"
        command = [
            str(python), "setup.py", "build_ext",
            "--build-temp", str(build / "tcnn" / "temp"),
            "--build-lib", str(build / "tcnn" / "lib"),
        ]
        compiled = run_command_logged(
            command,
            log,
            timeout=14400,
            cwd=source / "bindings" / "torch",
            env=env,
        )
        if compiled.get("returncode") != 0:
            return {"passed": False, "status": "FAIL", "reason": "TINY_NATIVE_BUILD_FAILED", "source": source_result, "build": compiled}
        post_source = verify_tiny_source(source)
        if not post_source.get("passed"):
            return {
                "passed": False,
                "status": "FAIL",
                "reason": "TINY_SOURCE_CHANGED_DURING_BUILD",
                "source": source_result,
                "post_source": post_source,
                "build": compiled,
            }
        matches = sorted((build / "tcnn" / "lib").glob("tinycudann_bindings/_120_C.cpython-312-x86_64-linux-gnu.so"))
        if len(matches) != 1:
            return {"passed": False, "status": "FAIL", "reason": "TINY_NATIVE_OUTPUT_COUNT_INVALID", "matches": [str(path) for path in matches], "build": compiled}
        shutil.copytree(build / "tcnn" / "lib", staging_runtime, symlinks=True)
        shutil.copytree(source / "bindings" / "torch" / "tinycudann", staging_runtime / "tinycudann", symlinks=True)
        binary = staging_runtime / "tinycudann_bindings" / "_120_C.cpython-312-x86_64-linux-gnu.so"
        source_verification = source_result["verification"]
        marker_payload = {
            "schema": "amd-nerfstudio-tiny-runtime-v1",
            "installer_version": VERSION,
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "source_url": TINY_REPOSITORY,
            "source_tag": TINY_TAG,
            "source_commit": TINY_COMMIT,
            "source_tree": source_verification.get("tree"),
            "submodules": source_verification.get("submodules"),
            "python": str(python),
            "torch": TORCH_PINS["torch"],
            "torchvision": TORCH_PINS["torchvision"],
            "hip": EXPECTED_TORCH_HIP,
            "arch": SUPPORTED_ARCH,
            "native_relative_path": "tinycudann_bindings/_120_C.cpython-312-x86_64-linux-gnu.so",
            "native_sha256": sha256_file(binary),
            "build_log": str(log),
        }
        (staging_runtime / TINY_RUNTIME_MARKER).write_text(
            json.dumps(marker_payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        staging_runtime.rename(runtime)
        verification = verify_tiny_runtime(runtime, python, rocm, SUPPORTED_ARCH)
        if not verification.get("passed"):
            return {"passed": False, "status": "FAIL", "reason": "BUILT_TINY_RUNTIME_FAILED_ATTESTATION", "source": source_result, "build": compiled, "runtime": verification}
        runtime_ready = True
        return {
            "passed": True,
            "status": "READY",
            "reason": "TINY_RUNTIME_BUILT_AND_ATTESTED",
            "source": source_result,
            "post_source": post_source,
            "build": compiled,
            "runtime": verification,
            "modified": True,
        }
    finally:
        if staging_runtime.exists():
            shutil.rmtree(staging_runtime, ignore_errors=True)
        if runtime.exists() and not runtime_ready:
            shutil.rmtree(runtime, ignore_errors=True)
        if created_build and build.exists() and not runtime_ready:
            shutil.rmtree(build, ignore_errors=True)


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
        "mode": (
            "TINY_NATIVE_BUILD" if args.build_tiny else
            "TORCH_INSTALL" if args.install_torch else
            "ENV_PREPARATION" if args.prepare_env else
            "PREFLIGHT_ONLY"
        ),
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
        if torch_requirements() != [f"{name}=={version}" for name, version in TORCH_PINS.items()]:
            failures.append("torch requirements")
        if TINY_COMMIT != "b98bdcc6b2878f6cb6c10a2141e50867cec6d96a":
            failures.append("tiny commit lock")
        if TINY_TAG != "phase4a2-model-b-public-gfx1201-pass":
            failures.append("tiny tag lock")
    payload = {
        "schema": SCHEMA,
        "passed": not failures,
        "failures": failures,
        "tests": 10,
    }
    print(json.dumps(payload, indent=2, sort_keys=True))
    print(f"PUBLIC_INSTALLER_V1_5_SELF_TEST: {'PASS' if not failures else 'FAIL'}")
    return 0 if not failures else 2


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="AMD Nerfstudio v1.5 host, environment, ROCm PyTorch, and tiny-rdna4-nn installer")
    p.add_argument("--workdir", type=Path)
    p.add_argument("--env", "--venv", dest="env", type=Path)
    p.add_argument("--rocm-path", type=Path, default=DEFAULT_ROCM_PATH)
    p.add_argument("--arch", default=SUPPORTED_ARCH)
    p.add_argument("--validation", choices=("none", "quick", "full"), default="quick")
    p.add_argument("--json-report", type=Path)
    p.add_argument("--prepare-env", action="store_true", help="Create or refresh the installer-managed environment")
    p.add_argument("--install-torch", action="store_true", help="Install or verify the pinned ROCm PyTorch stack in the managed environment")
    p.add_argument("--build-tiny", action="store_true", help="Clone, compile, and attest tiny-rdna4-nn for gfx1201")
    p.add_argument("--max-jobs", type=int, default=8, help="Maximum parallel native build jobs")
    p.add_argument("--self-test", action="store_true")
    return p

def main() -> int:
    args = parser().parse_args()
    if args.self_test:
        return self_test()
    if args.max_jobs < 1 or args.max_jobs > 64:
        print("MAX_JOBS_OUT_OF_RANGE: expected 1..64", file=sys.stderr)
        return 2
    script_path = Path(__file__).resolve()
    workdir = absolute(args.workdir) if args.workdir is not None else infer_default_workdir(script_path)
    paths = derive_paths(script_path, workdir, args.env)
    print_paths(paths, args.rocm_path, args.arch, args.validation)
    report = build_report(args, script_path)
    print_report_summary(report)

    preparation: dict[str, Any] | None = None
    torch_result: dict[str, Any] | None = None
    tiny_result: dict[str, Any] | None = None

    needs_env = args.prepare_env or args.install_torch or args.build_tiny
    if needs_env:
        print()
        print("ENV_PREPARATION: START")
        print("Pinned Python build base:")
        for requirement in build_package_requirements():
            print(f"  - {requirement}")
        print("No sudo or apt commands will be executed.")
        print()
        preparation = prepare_environment(report)
        report["environment_preparation"] = preparation
        print_environment_preparation(preparation)

    if args.install_torch or args.build_tiny:
        if preparation is not None and preparation.get("passed"):
            print()
            print("ROCM_PYTORCH_STAGE: START")
            print(f"index: {PYTORCH_INDEX}")
            for requirement in torch_requirements():
                print(f"  - {requirement}")
            torch_result = install_torch_stack(report)
        else:
            torch_result = {"passed": False, "status": "BLOCKED", "reason": "ENV_PREPARATION_FAILED"}
        report["torch_installation"] = torch_result
        print()
        print("torch installation:")
        print(f"  status: {torch_result.get('status')}")
        print(f"  reason: {torch_result.get('reason')}")
        print(f"TORCH_INSTALL: {'PASS' if torch_result.get('passed') else 'FAIL'}")
        print(f"TORCH_REUSED: {'YES' if torch_result.get('reused') else 'NO'}")

    if args.build_tiny:
        if torch_result is not None and torch_result.get("passed"):
            print()
            print("TINY_RDNA4_NATIVE_BUILD: START")
            print(f"source: {paths['tiny_source']}")
            print(f"build: {paths['tiny_build']}")
            print(f"runtime: {paths['tiny_runtime']}")
            print(f"tag: {TINY_TAG}")
            print(f"commit: {TINY_COMMIT}")
            print(f"max jobs: {args.max_jobs}")
            tiny_result = build_tiny_runtime(report, max_jobs=args.max_jobs)
        else:
            tiny_result = {"passed": False, "status": "BLOCKED", "reason": "QUALIFIED_TORCH_STAGE_REQUIRED"}
        report["tiny_rdna4_native_build"] = tiny_result
        print()
        print("tiny-rdna4-nn native build:")
        print(f"  status: {tiny_result.get('status')}")
        print(f"  reason: {tiny_result.get('reason')}")
        runtime = tiny_result.get("runtime", {})
        if runtime.get("binary"):
            print(f"  native: {runtime.get('binary')}")
            print(f"  sha256: {runtime.get('native_sha256')}")
        print(f"TINYRDN4_NATIVE_BUILD: {'PASS' if tiny_result.get('passed') else 'FAIL'}")

    passed = report.get("passed") is True
    if preparation is not None:
        passed = passed and preparation.get("passed") is True
    if torch_result is not None:
        passed = passed and torch_result.get("passed") is True
    if tiny_result is not None:
        passed = passed and tiny_result.get("passed") is True

    report["installation"] = {
        "status": "READY" if passed else "FAIL",
        "env_created": bool(preparation and preparation.get("created")),
        "env_reused": bool(preparation and preparation.get("reused")),
        "torch_ready": bool(torch_result and torch_result.get("passed")),
        "tiny_runtime_ready": bool(tiny_result and tiny_result.get("passed")),
        "system_modified": False,
        "workdir_modified": bool(
            (preparation and (preparation.get("created") or preparation.get("modified")))
            or (torch_result and torch_result.get("modified"))
            or (tiny_result and tiny_result.get("modified"))
        ),
        "automatic_sudo": False,
        "automatic_apt": False,
    }

    output: Path | None = absolute(args.json_report) if args.json_report is not None else None
    if output is None and needs_env:
        output = Path(report["paths"]["reports"]) / "installer-v1.5-dev3.json"
    if output is not None:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(f"REPORT_JSON: {output}")

    print("SYSTEM_MODIFIED: NO")
    print(f"WORKDIR_MODIFIED: {'YES' if report['installation']['workdir_modified'] else 'NO'}")
    return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
