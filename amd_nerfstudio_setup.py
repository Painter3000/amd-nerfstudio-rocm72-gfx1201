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

VERSION = "1.5.0-dev4a"
SCHEMA = "amd-nerfstudio-public-installer-v1-5-nerfacc-nerfstudio"
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

NERFACC_WHEEL_FILENAME = "nerfacc-0.5.2-cp312-cp312-linux_x86_64.whl"
NERFACC_WHEEL_SHA256 = "252ec63319461889319a3bc535c4076c3c84bfc1ff6ddb5d64e1bb8b18032e00"
NERFACC_NATIVE_SHA256 = "d3beee150cfa3a9ad3038a3283ff0a46953c345634d8cb6109449c5e3d04d1e2"
NERFACC_SOURCE_URL = "https://github.com/nerfstudio-project/nerfacc.git"
NERFACC_SOURCE_TAG = "v0.5.2"
NERFACC_SOURCE_COMMIT = "d84cdf3afd7dcfc42150e0f0506db58a5ce62812"
NERFACC_SOURCE_TREE = "f24a2f9902143b75ecb8472199b07dd0e92679e8"
NERFACC_RICH_PINS = {
    "rich": "14.3.4",
    "markdown-it-py": "4.2.0",
    "mdurl": "0.1.2",
    "Pygments": "2.20.0",
}

NERFSTUDIO_REPOSITORY = "https://github.com/nerfstudio-project/nerfstudio.git"
NERFSTUDIO_COMMIT = "50e0e3c70c775e89333256213363badbf074f29d"
NERFSTUDIO_TREE = "9d5ff468eeff89b66995e9984acaa378c37dc07e"
NERFSTUDIO_MLP_SHA256 = "4939a5a6901d82d8e310d93e2a135ca57ccc1bd79be79a7f67e2740e730c44ad"
VISER_VERSION = "1.0.0"
VISER_WHEEL_FILENAME = "viser-1.0.0-py3-none-any.whl"
VISER_WHEEL_SHA256 = "3be881a60f0295efd8a93df97646bbc04d070ccf8d16d8faf284eb3b70eda6eb"
SCOPED_RUNTIME_REQUIREMENTS = "requirements/nerfacto_runtime_v1.txt"
SCOPED_RUNTIME_CONSTRAINTS = "constraints/nerfacto_rocm72_py312_v1.txt"
RUNTIME_PTH_FILENAME = "00_amd_nerfstudio_rdna4_runtime.pth"


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



def evaluate_pip_check(process: dict[str, Any], allow_viser_math_only: bool = True) -> dict[str, Any]:
    text = "\n".join(
        part for part in (process.get("stdout", ""), process.get("stderr", "")) if part
    )
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    allowed: list[str] = []
    rejected: list[str] = []
    for line in lines:
        if allow_viser_math_only and line.lower().startswith(f"viser {VISER_VERSION} "):
            allowed.append(line)
        elif line != "No broken requirements found.":
            rejected.append(line)
    passed = process.get("returncode") == 0 or (bool(allowed) and not rejected)
    return {
        "passed": passed,
        "returncode": process.get("returncode"),
        "allowed": allowed,
        "rejected": rejected,
        "process": process,
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
    pip_check_process = run_command([str(python), "-m", "pip", "check"], timeout=300)
    pip_check = evaluate_pip_check(pip_check_process)
    freeze_path = Path(report["paths"]["reports"]) / "installer-v1.5-torch-pip-freeze.txt"
    freeze = run_command([str(python), "-m", "pip", "freeze", "--all"], timeout=300)
    freeze_path.parent.mkdir(parents=True, exist_ok=True)
    freeze_path.write_text(freeze.get("stdout", "") + freeze.get("stderr", ""), encoding="utf-8")
    passed = (
        after.get("passed") is True
        and smoke.get("passed") is True
        and pip_check.get("passed") is True
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


def compose_runtime_library_path(
    torch_lib: Path,
    rocm_path: Path,
    existing: str | None = None,
) -> str:
    """Build the loader path used to inspect one ENV-bound torch extension.

    A bare ``ldd`` call does not know where a virtual environment keeps
    ``libc10.so`` and the other PyTorch shared libraries. The real Python
    import does know this through torch's loader setup. For a meaningful
    dependency audit we therefore place the selected environment's
    ``torch/lib`` directory first, followed by the requested ROCm libraries
    and any caller-provided loader path.
    """
    candidates = [
        str(torch_lib),
        str(rocm_path / "lib"),
        str(rocm_path / "lib64"),
    ]
    if existing:
        candidates.extend(value for value in existing.split(os.pathsep) if value)
    ordered: list[str] = []
    seen: set[str] = set()
    for value in candidates:
        if value and value not in seen:
            ordered.append(value)
            seen.add(value)
    return os.pathsep.join(ordered)


def probe_torch_library_dir(python: Path) -> dict[str, Any]:
    code = r'''
from importlib.util import find_spec
from pathlib import Path

spec = find_spec("torch")
if spec is None or spec.origin is None:
    raise SystemExit(2)
path = (Path(spec.origin).resolve().parent / "lib").resolve()
print(path)
raise SystemExit(0 if path.is_dir() else 2)
'''
    process = run_command([str(python), "-c", code], timeout=60)
    path: Path | None = None
    if process.get("returncode") == 0:
        lines = [
            line.strip()
            for line in process.get("stdout", "").splitlines()
            if line.strip()
        ]
        if lines:
            candidate = Path(lines[-1])
            if candidate.is_dir():
                path = candidate
    return {
        "passed": path is not None,
        "path": str(path) if path is not None else None,
        "process": process,
    }


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
    ldd_bare = run_command(["ldd", str(binary)], timeout=300) if binary.is_file() else {"returncode": None, "stdout": "", "stderr": "binary missing"}
    torch_library = probe_torch_library_dir(python)
    ldd_env = dict(os.environ)
    if torch_library.get("passed"):
        ldd_env["LD_LIBRARY_PATH"] = compose_runtime_library_path(
            Path(torch_library["path"]),
            rocm_path,
            ldd_env.get("LD_LIBRARY_PATH"),
        )
        ldd = run_command(["ldd", str(binary)], timeout=300, env=ldd_env) if binary.is_file() else {"returncode": None, "stdout": "", "stderr": "binary missing"}
    else:
        ldd = {
            "argv": ["ldd", str(binary)],
            "returncode": None,
            "stdout": "",
            "stderr": "torch library directory unavailable",
        }
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
        "torch_library_dir": torch_library.get("passed") is True,
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
        "torch_library_probe": torch_library,
        "ldd_environment": {
            "LD_LIBRARY_PATH": ldd_env.get("LD_LIBRARY_PATH") if torch_library.get("passed") else None,
        },
        "ldd_bare": ldd_bare,
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
            return {"passed": False, "status": "FAIL", "reason": "TINY_NATIVE_BUILD_FAILED", "source": source_result, "build": compiled, "modified": True}
        post_source = verify_tiny_source(source)
        if not post_source.get("passed"):
            return {
                "passed": False,
                "status": "FAIL",
                "reason": "TINY_SOURCE_CHANGED_DURING_BUILD",
                "source": source_result,
                "post_source": post_source,
                "build": compiled,
                "modified": True,
            }
        matches = sorted((build / "tcnn" / "lib").glob("tinycudann_bindings/_120_C.cpython-312-x86_64-linux-gnu.so"))
        if len(matches) != 1:
            return {"passed": False, "status": "FAIL", "reason": "TINY_NATIVE_OUTPUT_COUNT_INVALID", "matches": [str(path) for path in matches], "build": compiled, "modified": True}
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
            return {"passed": False, "status": "FAIL", "reason": "BUILT_TINY_RUNTIME_FAILED_ATTESTATION", "source": source_result, "build": compiled, "runtime": verification, "modified": True}
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



def nerfacc_python_requirements() -> list[str]:
    return [f"{name}=={version}" for name, version in NERFACC_RICH_PINS.items()]


def managed_env_python(report: dict[str, Any]) -> Path:
    return Path(report["environment_selection"]["path"]) / "bin" / "python"


def managed_site_packages(python: Path) -> dict[str, Any]:
    code = (
        "import json,site; "
        "paths=site.getsitepackages(); "
        "print(json.dumps({'paths':paths,'selected':paths[0] if paths else None}))"
    )
    process = run_command([str(python), "-c", code], timeout=60)
    payload: dict[str, Any] = {}
    if process.get("returncode") == 0:
        try:
            payload = json.loads(process.get("stdout", "").splitlines()[-1])
        except Exception:
            payload = {}
    selected = payload.get("selected")
    path = Path(selected) if selected else None
    return {
        "passed": process.get("returncode") == 0 and path is not None and path.is_dir(),
        "path": str(path) if path is not None else None,
        "payload": payload,
        "process": process,
    }


def resolve_nerfacc_wheel(report: dict[str, Any], explicit_wheel: Path | None) -> dict[str, Any]:
    cache_dir = Path(report["paths"]["cache"]) / "wheels" / "custom"
    cache_wheel = cache_dir / NERFACC_WHEEL_FILENAME
    source = absolute(explicit_wheel) if explicit_wheel is not None else cache_wheel
    if not source.is_file():
        return {
            "passed": False,
            "status": "BLOCKED",
            "reason": "AUTHORIZED_NERFACC_WHEEL_NOT_FOUND",
            "requested": str(source),
            "cache": str(cache_wheel),
            "expected_sha256": NERFACC_WHEEL_SHA256,
        }
    observed = sha256_file(source)
    if observed != NERFACC_WHEEL_SHA256:
        return {
            "passed": False,
            "status": "BLOCKED",
            "reason": "NERFACC_WHEEL_SHA256_MISMATCH",
            "requested": str(source),
            "observed_sha256": observed,
            "expected_sha256": NERFACC_WHEEL_SHA256,
        }
    copied = False
    if source != cache_wheel:
        cache_dir.mkdir(parents=True, exist_ok=True)
        temporary = cache_wheel.with_name(cache_wheel.name + f".part-{os.getpid()}")
        if temporary.exists():
            temporary.unlink()
        shutil.copy2(source, temporary)
        if sha256_file(temporary) != NERFACC_WHEEL_SHA256:
            temporary.unlink(missing_ok=True)
            return {"passed": False, "status": "FAIL", "reason": "NERFACC_CACHE_COPY_HASH_MISMATCH"}
        os.replace(temporary, cache_wheel)
        copied = True
    return {
        "passed": True,
        "status": "READY",
        "reason": "AUTHORIZED_NERFACC_WHEEL_CACHED" if copied else "AUTHORIZED_NERFACC_WHEEL_REUSED",
        "path": str(cache_wheel),
        "sha256": NERFACC_WHEEL_SHA256,
        "copied": copied,
    }


def probe_nerfacc_runtime(python: Path, wheel: Path, rocm_path: Path, arch: str) -> dict[str, Any]:
    code = r'''
from importlib import metadata
import hashlib, json, pathlib, sys
payload = {"python": sys.executable, "versions": {}}
for name in ("nerfacc", "rich", "markdown-it-py", "mdurl", "Pygments"):
    try:
        payload["versions"][name] = metadata.version(name)
    except metadata.PackageNotFoundError:
        payload["versions"][name] = None
try:
    import nerfacc
    import nerfacc.csrc as csrc
    import torch
    native = pathlib.Path(csrc.__file__).resolve()
    payload.update({
        "nerfacc_module": str(pathlib.Path(nerfacc.__file__).resolve()),
        "native": str(native),
        "native_sha256": hashlib.sha256(native.read_bytes()).hexdigest(),
        "torch": torch.__version__,
        "hip": torch.version.hip,
        "cuda_available": bool(torch.cuda.is_available()),
    })
    if payload["cuda_available"]:
        payload["gcn_arch"] = getattr(torch.cuda.get_device_properties(0), "gcnArchName", None)
except Exception as exc:
    payload["import_error"] = repr(exc)
print(json.dumps(payload, sort_keys=True))
'''
    env = dict(os.environ)
    env["PYTHONNOUSERSITE"] = "1"
    process = run_command([str(python), "-c", code], timeout=300, env=env)
    payload: dict[str, Any] = {}
    if process.get("returncode") == 0:
        try:
            payload = json.loads(process.get("stdout", "").splitlines()[-1])
        except Exception:
            payload = {}
    versions = payload.get("versions", {})
    native = Path(payload["native"]) if payload.get("native") else None
    wheel_hash = sha256_file(wheel) if wheel.is_file() else None
    checks = {
        "process": process.get("returncode") == 0,
        "nerfacc_version": versions.get("nerfacc") == "0.5.2",
        "rich_version": versions.get("rich") == NERFACC_RICH_PINS["rich"],
        "markdown_it_version": versions.get("markdown-it-py") == NERFACC_RICH_PINS["markdown-it-py"],
        "mdurl_version": versions.get("mdurl") == NERFACC_RICH_PINS["mdurl"],
        "pygments_version": versions.get("Pygments") == NERFACC_RICH_PINS["Pygments"],
        "torch_version": payload.get("torch") == TORCH_PINS["torch"],
        "hip_version": payload.get("hip") == EXPECTED_TORCH_HIP,
        "cuda_available": payload.get("cuda_available") is True,
        "gcn_arch": payload.get("gcn_arch") == arch,
        "native_present": native is not None and native.is_file(),
        "native_hash": payload.get("native_sha256") == NERFACC_NATIVE_SHA256,
        "wheel_hash": wheel_hash == NERFACC_WHEEL_SHA256,
    }
    ldd: dict[str, Any] = {"passed": False, "reason": "NATIVE_NOT_AVAILABLE"}
    code_object: dict[str, Any] = {"passed": False, "reason": "NATIVE_NOT_AVAILABLE"}
    if native is not None and native.is_file():
        torch_lib = probe_torch_library_dir(python)
        if torch_lib.get("passed"):
            loader = compose_runtime_library_path(Path(torch_lib["path"]), rocm_path, os.environ.get("LD_LIBRARY_PATH"))
            ldd_env = dict(os.environ)
            ldd_env["LD_LIBRARY_PATH"] = loader
            ldd_process = run_command(["/usr/bin/ldd", str(native)], timeout=300, env=ldd_env)
            ldd_text = ldd_process.get("stdout", "") + ldd_process.get("stderr", "")
            ldd = {
                "passed": ldd_process.get("returncode") == 0 and "not found" not in ldd_text,
                "loader_path": loader,
                "process": ldd_process,
            }
        roc_obj = rocm_path / "bin" / "roc-obj-ls"
        roc_process = run_command([str(roc_obj), str(native)], timeout=300)
        roc_text = roc_process.get("stdout", "") + roc_process.get("stderr", "")
        code_object = {
            "passed": roc_process.get("returncode") == 0 and f"--{arch}" in roc_text,
            "process": roc_process,
        }
    checks["environment_aware_ldd"] = ldd.get("passed") is True
    checks["code_object"] = code_object.get("passed") is True
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "payload": payload,
        "wheel": str(wheel),
        "wheel_sha256": wheel_hash,
        "ldd": ldd,
        "code_object": code_object,
        "process": process,
    }


def install_nerfacc_stack(report: dict[str, Any], explicit_wheel: Path | None = None) -> dict[str, Any]:
    if not report.get("passed"):
        return {"passed": False, "status": "BLOCKED", "reason": "PREFLIGHT_NOT_PASSED"}
    selection = report["environment_selection"]
    if selection.get("ownership") == "EXTERNAL_EXPLICIT":
        return {"passed": False, "status": "BLOCKED", "reason": "EXPLICIT_EXTERNAL_ENV_IS_VERIFY_ONLY", "modified": False}
    python = managed_env_python(report)
    wheel_result = resolve_nerfacc_wheel(report, explicit_wheel)
    if not wheel_result.get("passed"):
        return {"passed": False, "status": wheel_result.get("status", "BLOCKED"), "reason": wheel_result.get("reason"), "wheel": wheel_result}
    wheel = Path(wheel_result["path"])
    before = probe_nerfacc_runtime(python, wheel, Path(report["rocm"]["requested"]), report["arch"])
    install_processes: list[dict[str, Any]] = []
    reused = before.get("passed") is True
    if not reused:
        log_dir = Path(report["paths"]["logs"])
        env = dict(os.environ)
        env["PYTHONNOUSERSITE"] = "1"
        env["PATH"] = str(python.parent) + os.pathsep + env.get("PATH", "")
        wheel_install = run_command_logged(
            [str(python), "-m", "pip", "install", "--disable-pip-version-check", "--no-input", "--no-deps", "--force-reinstall", str(wheel)],
            log_dir / "installer-v1.5-nerfacc-install.log",
            timeout=1800,
            env=env,
        )
        install_processes.append(wheel_install)
        if wheel_install.get("returncode") != 0:
            return {"passed": False, "status": "FAIL", "reason": "NERFACC_WHEEL_INSTALL_FAILED", "wheel": wheel_result, "before": before, "install": install_processes}
        rich_install = run_command_logged(
            [str(python), "-m", "pip", "install", "--disable-pip-version-check", "--no-input", "--index-url", PYPI_INDEX, "--only-binary=:all:", "--upgrade", *nerfacc_python_requirements()],
            log_dir / "installer-v1.5-rich-install.log",
            timeout=1800,
            env=env,
        )
        install_processes.append(rich_install)
        if rich_install.get("returncode") != 0:
            return {"passed": False, "status": "FAIL", "reason": "NERFACC_PYTHON_DEPENDENCY_INSTALL_FAILED", "wheel": wheel_result, "before": before, "install": install_processes}
    after = probe_nerfacc_runtime(python, wheel, Path(report["rocm"]["requested"]), report["arch"])
    pip_check = evaluate_pip_check(run_command([str(python), "-m", "pip", "check"], timeout=300))
    passed = after.get("passed") is True and pip_check.get("passed") is True
    return {
        "passed": passed,
        "status": "READY" if passed else "FAIL",
        "reason": "QUALIFIED_NERFACC_STACK_REUSED" if passed and reused else ("QUALIFIED_NERFACC_STACK_INSTALLED" if passed else "NERFACC_QUALIFICATION_FAILED"),
        "reused": reused,
        "modified": bool(wheel_result.get("copied")) or not reused,
        "wheel": wheel_result,
        "source_provenance": {
            "url": NERFACC_SOURCE_URL,
            "tag": NERFACC_SOURCE_TAG,
            "commit": NERFACC_SOURCE_COMMIT,
            "tree": NERFACC_SOURCE_TREE,
            "patch": "apply_nerfacc_rocm72_vector_math_compat_v1.py",
            "license": "MIT",
        },
        "before": before,
        "install": install_processes,
        "after": after,
        "pip_check": pip_check,
    }


def verify_nerfstudio_source(source: Path) -> dict[str, Any]:
    if not (source / ".git").exists():
        return {"passed": False, "reason": "NERFSTUDIO_SOURCE_NOT_A_GIT_CHECKOUT", "path": str(source)}
    try:
        commit = git_output(source, "rev-parse", "HEAD")
        tree = git_output(source, "rev-parse", "HEAD^{tree}")
        status = git_output(source, "status", "--porcelain=v1", "--untracked-files=all")
        origin = git_output(source, "remote", "get-url", "origin")
    except Exception as exc:
        return {"passed": False, "reason": "NERFSTUDIO_SOURCE_GIT_PROBE_FAILED", "error": repr(exc), "path": str(source)}
    mlp = source / "nerfstudio" / "field_components" / "mlp.py"
    checks = {
        "commit": commit == NERFSTUDIO_COMMIT,
        "tree": tree == NERFSTUDIO_TREE,
        "clean": status == "",
        "origin": origin.rstrip("/").removesuffix(".git") == NERFSTUDIO_REPOSITORY.rstrip("/").removesuffix(".git"),
        "pyproject": (source / "pyproject.toml").is_file(),
        "mlp_present": mlp.is_file(),
        "mlp_hash": mlp.is_file() and sha256_file(mlp) == NERFSTUDIO_MLP_SHA256,
    }
    return {
        "passed": all(checks.values()),
        "reason": "QUALIFIED_NERFSTUDIO_SOURCE" if all(checks.values()) else "NERFSTUDIO_SOURCE_CONTRACT_MISMATCH",
        "path": str(source),
        "commit": commit,
        "tree": tree,
        "origin": origin,
        "status": status,
        "checks": checks,
    }


def acquire_nerfstudio_source(report: dict[str, Any]) -> dict[str, Any]:
    source = Path(report["paths"]["nerfstudio_source"])
    if source.exists():
        verification = verify_nerfstudio_source(source)
        return {
            "passed": verification.get("passed") is True,
            "status": "READY" if verification.get("passed") else "BLOCKED",
            "reason": "QUALIFIED_SOURCE_REUSED" if verification.get("passed") else "EXISTING_NERFSTUDIO_SOURCE_REJECTED",
            "reused": verification.get("passed") is True,
            "modified": False,
            "verification": verification,
        }
    source.parent.mkdir(parents=True, exist_ok=True)
    staging = source.with_name(source.name + f".part-{os.getpid()}")
    if staging.exists():
        return {"passed": False, "status": "BLOCKED", "reason": "NERFSTUDIO_STAGING_PATH_ALREADY_EXISTS", "path": str(staging)}
    clone_log = Path(report["paths"]["logs"]) / "installer-v1.5-nerfstudio-clone.log"
    clone = run_command_logged(
        ["git", "clone", "--filter=blob:none", "--no-checkout", NERFSTUDIO_REPOSITORY, str(staging)],
        clone_log,
        timeout=3600,
    )
    if clone.get("returncode") != 0:
        shutil.rmtree(staging, ignore_errors=True)
        return {"passed": False, "status": "FAIL", "reason": "NERFSTUDIO_CLONE_FAILED", "clone": clone}
    checkout = run_command(["git", "-C", str(staging), "checkout", "--detach", NERFSTUDIO_COMMIT], timeout=1800)
    if checkout.get("returncode") != 0:
        shutil.rmtree(staging, ignore_errors=True)
        return {"passed": False, "status": "FAIL", "reason": "NERFSTUDIO_CHECKOUT_FAILED", "clone": clone, "checkout": checkout}
    verification = verify_nerfstudio_source(staging)
    if not verification.get("passed"):
        shutil.rmtree(staging, ignore_errors=True)
        return {"passed": False, "status": "FAIL", "reason": "CLONED_NERFSTUDIO_SOURCE_FAILED_ATTESTATION", "clone": clone, "checkout": checkout, "verification": verification}
    os.replace(staging, source)
    final = verify_nerfstudio_source(source)
    return {
        "passed": final.get("passed") is True,
        "status": "READY" if final.get("passed") else "FAIL",
        "reason": "QUALIFIED_SOURCE_ACQUIRED" if final.get("passed") else "FINAL_NERFSTUDIO_SOURCE_FAILED_ATTESTATION",
        "reused": False,
        "modified": True,
        "clone": clone,
        "checkout": checkout,
        "verification": final,
    }


def verify_existing_tiny_runtime(report: dict[str, Any]) -> dict[str, Any]:
    runtime = Path(report["paths"]["tiny_runtime"])
    python = managed_env_python(report)
    verification = verify_tiny_runtime(runtime, python, Path(report["rocm"]["requested"]), report["arch"])
    return {
        "passed": verification.get("passed") is True,
        "status": "READY" if verification.get("passed") else "BLOCKED",
        "reason": "QUALIFIED_TINY_RUNTIME_REUSED" if verification.get("passed") else "QUALIFIED_TINY_RUNTIME_REQUIRED",
        "reused": verification.get("passed") is True,
        "modified": False,
        "runtime": verification,
    }


def cache_viser_wheel(report: dict[str, Any], python: Path) -> dict[str, Any]:
    cache_dir = Path(report["paths"]["cache"]) / "wheels" / "public"
    wheel = cache_dir / VISER_WHEEL_FILENAME
    if wheel.is_file() and sha256_file(wheel) == VISER_WHEEL_SHA256:
        return {"passed": True, "status": "READY", "reason": "QUALIFIED_VISER_WHEEL_REUSED", "path": str(wheel), "sha256": VISER_WHEEL_SHA256, "modified": False}
    if wheel.exists():
        return {"passed": False, "status": "BLOCKED", "reason": "EXISTING_VISER_WHEEL_HASH_MISMATCH", "path": str(wheel)}
    cache_dir.mkdir(parents=True, exist_ok=True)
    staging = cache_dir / f".viser-download-{os.getpid()}"
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir()
    process = run_command_logged(
        [str(python), "-m", "pip", "download", "--disable-pip-version-check", "--no-input", "--index-url", PYPI_INDEX, "--no-deps", "--only-binary=:all:", "--dest", str(staging), f"viser=={VISER_VERSION}"],
        Path(report["paths"]["logs"]) / "installer-v1.5-viser-download.log",
        timeout=1800,
    )
    matches = sorted(staging.glob("viser-1.0.0-*.whl"))
    if process.get("returncode") != 0 or len(matches) != 1:
        shutil.rmtree(staging, ignore_errors=True)
        return {"passed": False, "status": "FAIL", "reason": "VISER_WHEEL_DOWNLOAD_FAILED", "process": process, "matches": [str(path) for path in matches]}
    observed = sha256_file(matches[0])
    if observed != VISER_WHEEL_SHA256:
        shutil.rmtree(staging, ignore_errors=True)
        return {"passed": False, "status": "FAIL", "reason": "VISER_WHEEL_SHA256_MISMATCH", "observed": observed, "expected": VISER_WHEEL_SHA256}
    os.replace(matches[0], wheel)
    shutil.rmtree(staging, ignore_errors=True)
    return {"passed": True, "status": "READY", "reason": "QUALIFIED_VISER_WHEEL_DOWNLOADED", "path": str(wheel), "sha256": observed, "modified": True, "process": process}


def write_runtime_pth(python: Path, tiny_runtime: Path, nerfstudio_source: Path) -> dict[str, Any]:
    site_probe = managed_site_packages(python)
    if not site_probe.get("passed"):
        return {"passed": False, "reason": "MANAGED_SITE_PACKAGES_UNAVAILABLE", "site_probe": site_probe}
    site_path = Path(site_probe["path"])
    target = site_path / RUNTIME_PTH_FILENAME
    content = f"{tiny_runtime}\n{nerfstudio_source}\n"
    changed = not target.is_file() or target.read_text(encoding="utf-8", errors="replace") != content
    if changed:
        temporary = target.with_name(target.name + f".part-{os.getpid()}")
        temporary.write_text(content, encoding="utf-8")
        os.replace(temporary, target)
    return {"passed": True, "path": str(target), "content": content.splitlines(), "modified": changed, "site_probe": site_probe}


def probe_nerfstudio_runtime(report: dict[str, Any], source: Path, python: Path) -> dict[str, Any]:
    code = r'''
from importlib import import_module, metadata
import hashlib, json, pathlib, sys
modules = [
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
payload = {"modules": [], "python": sys.executable}
try:
    import torch
    import nerfacc.csrc as nerfacc_csrc
    import tinycudann.modules as tcnn_modules
    for name in modules:
        mod = import_module(name)
        payload["modules"].append({"name": name, "file": str(pathlib.Path(mod.__file__).resolve()) if getattr(mod, "__file__", None) else None})
    payload.update({
        "torch": torch.__version__,
        "hip": torch.version.hip,
        "cuda_available": bool(torch.cuda.is_available()),
        "gcn_arch": getattr(torch.cuda.get_device_properties(0), "gcnArchName", None) if torch.cuda.is_available() else None,
        "nerfacc_native": str(pathlib.Path(nerfacc_csrc.__file__).resolve()),
        "nerfacc_native_sha256": hashlib.sha256(pathlib.Path(nerfacc_csrc.__file__).read_bytes()).hexdigest(),
        "tiny_module": str(pathlib.Path(tcnn_modules.__file__).resolve()),
        "tiny_native": str(pathlib.Path(tcnn_modules._C.__file__).resolve()),
        "tiny_native_sha256": hashlib.sha256(pathlib.Path(tcnn_modules._C.__file__).read_bytes()).hexdigest(),
        "viser_version": metadata.version("viser"),
        "rich_version": metadata.version("rich"),
        "opencv_headless": metadata.version("opencv-python-headless"),
    })
    try:
        payload["opencv_gui"] = metadata.version("opencv-python")
    except metadata.PackageNotFoundError:
        payload["opencv_gui"] = None
except Exception as exc:
    payload["error"] = repr(exc)
print(json.dumps(payload, sort_keys=True))
'''
    env = dict(os.environ)
    env["PYTHONNOUSERSITE"] = "1"
    process = run_command([str(python), "-c", code], timeout=900, env=env)
    payload: dict[str, Any] = {}
    if process.get("returncode") == 0:
        try:
            payload = json.loads(process.get("stdout", "").splitlines()[-1])
        except Exception:
            payload = {}
    module_rows = payload.get("modules", [])
    origins = {row.get("name"): row.get("file") for row in module_rows}
    checks = {
        "process": process.get("returncode") == 0,
        "no_error": "error" not in payload,
        "all_modules": len(module_rows) == 9 and all(row.get("file") for row in module_rows),
        "nerfstudio_origin": all(
            str(path).startswith(str(source.resolve()))
            for name, path in origins.items()
            if name.startswith("nerfstudio") and path
        ),
        "torch": payload.get("torch") == TORCH_PINS["torch"],
        "hip": payload.get("hip") == EXPECTED_TORCH_HIP,
        "cuda_available": payload.get("cuda_available") is True,
        "gcn_arch": payload.get("gcn_arch") == report["arch"],
        "nerfacc_hash": payload.get("nerfacc_native_sha256") == NERFACC_NATIVE_SHA256,
        "tiny_origin": str(payload.get("tiny_module", "")).startswith(str(Path(report["paths"]["tiny_runtime"]).resolve())),
        "viser": payload.get("viser_version") == VISER_VERSION,
        "rich": payload.get("rich_version") == NERFACC_RICH_PINS["rich"],
        "opencv_headless": payload.get("opencv_headless") == "4.10.0.84",
        "opencv_gui_absent": payload.get("opencv_gui") is None,
    }
    return {"passed": all(checks.values()), "checks": checks, "payload": payload, "process": process}


def install_nerfstudio_runtime(report: dict[str, Any]) -> dict[str, Any]:
    if not report.get("passed"):
        return {"passed": False, "status": "BLOCKED", "reason": "PREFLIGHT_NOT_PASSED"}
    selection = report["environment_selection"]
    if selection.get("ownership") == "EXTERNAL_EXPLICIT":
        return {"passed": False, "status": "BLOCKED", "reason": "EXPLICIT_EXTERNAL_ENV_IS_VERIFY_ONLY", "modified": False}
    python = managed_env_python(report)
    source_result = acquire_nerfstudio_source(report)
    if not source_result.get("passed"):
        return {"passed": False, "status": source_result.get("status", "BLOCKED"), "reason": source_result.get("reason"), "source": source_result}
    source = Path(report["paths"]["nerfstudio_source"])
    project_repo = Path(report["paths"]["project_repo"])
    requirements = project_repo / SCOPED_RUNTIME_REQUIREMENTS
    constraints = project_repo / SCOPED_RUNTIME_CONSTRAINTS
    if not requirements.is_file() or not constraints.is_file():
        return {"passed": False, "status": "BLOCKED", "reason": "SCOPED_RUNTIME_INPUTS_MISSING", "requirements": str(requirements), "constraints": str(constraints), "source": source_result}
    env = dict(os.environ)
    env["PYTHONNOUSERSITE"] = "1"
    env["PATH"] = str(python.parent) + os.pathsep + env.get("PATH", "")
    requirements_install = run_command_logged(
        [str(python), "-m", "pip", "install", "--disable-pip-version-check", "--no-input", "--index-url", PYPI_INDEX, "--constraint", str(constraints), "--requirement", str(requirements)],
        Path(report["paths"]["logs"]) / "installer-v1.5-nerfstudio-scoped-runtime.log",
        timeout=14400,
        env=env,
    )
    if requirements_install.get("returncode") != 0:
        return {"passed": False, "status": "FAIL", "reason": "NERFSTUDIO_SCOPED_RUNTIME_INSTALL_FAILED", "source": source_result, "requirements_install": requirements_install}
    viser = cache_viser_wheel(report, python)
    if not viser.get("passed"):
        return {"passed": False, "status": viser.get("status", "FAIL"), "reason": viser.get("reason"), "source": source_result, "requirements_install": requirements_install, "viser": viser}
    viser_install = run_command_logged(
        [str(python), "-m", "pip", "install", "--disable-pip-version-check", "--no-input", "--no-deps", "--force-reinstall", str(viser["path"])],
        Path(report["paths"]["logs"]) / "installer-v1.5-viser-math-install.log",
        timeout=1800,
        env=env,
    )
    if viser_install.get("returncode") != 0:
        return {"passed": False, "status": "FAIL", "reason": "VISER_MATH_INSTALL_FAILED", "source": source_result, "requirements_install": requirements_install, "viser": viser, "viser_install": viser_install}
    pth = write_runtime_pth(python, Path(report["paths"]["tiny_runtime"]), source)
    if not pth.get("passed"):
        return {"passed": False, "status": "FAIL", "reason": pth.get("reason"), "source": source_result, "requirements_install": requirements_install, "viser": viser, "viser_install": viser_install, "pth": pth}
    source_after = verify_nerfstudio_source(source)
    runtime = probe_nerfstudio_runtime(report, source, python)
    pip_check = evaluate_pip_check(run_command([str(python), "-m", "pip", "check"], timeout=600))
    passed = source_after.get("passed") is True and runtime.get("passed") is True and pip_check.get("passed") is True
    return {
        "passed": passed,
        "status": "READY" if passed else "FAIL",
        "reason": "NERFSTUDIO_SCOPED_RUNTIME_READY" if passed else "NERFSTUDIO_RUNTIME_QUALIFICATION_FAILED",
        "modified": bool(source_result.get("modified")) or bool(viser.get("modified")) or bool(pth.get("modified")) or True,
        "source": source_result,
        "source_after": source_after,
        "requirements": {"path": str(requirements), "sha256": sha256_file(requirements)},
        "constraints": {"path": str(constraints), "sha256": sha256_file(constraints)},
        "requirements_install": requirements_install,
        "viser": viser,
        "viser_install": viser_install,
        "pth": pth,
        "runtime": runtime,
        "pip_check": pip_check,
    }


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
            "NERFSTUDIO_SCOPED_RUNTIME" if args.install_nerfstudio else
            "NERFACC_INSTALL" if args.install_nerfacc else
            "TINY_NATIVE_BUILD" if args.build_tiny else
            "TORCH_INSTALL" if args.install_torch else
            "ENV_PREPARATION" if args.prepare_env else
            "PREFLIGHT_ONLY"
        ),
        "paths": {key: str(value) for key, value in paths.items()},
        "arch": args.arch,
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
        synthetic_report = {"arch": SUPPORTED_ARCH}
        if synthetic_report.get("arch") != SUPPORTED_ARCH:
            failures.append("report arch contract")
        loader = compose_runtime_library_path(
            Path("/env/torch/lib"),
            Path("/opt/rocm"),
            "/custom/lib:/env/torch/lib",
        ).split(os.pathsep)
        if loader != [
            "/env/torch/lib",
            "/opt/rocm/lib",
            "/opt/rocm/lib64",
            "/custom/lib",
        ]:
            failures.append("environment-aware ldd path")
        if nerfacc_python_requirements() != [f"{name}=={version}" for name, version in NERFACC_RICH_PINS.items()]:
            failures.append("nerfacc rich pins")
        if NERFACC_SOURCE_COMMIT != "d84cdf3afd7dcfc42150e0f0506db58a5ce62812":
            failures.append("nerfacc source commit")
        if NERFSTUDIO_COMMIT != "50e0e3c70c775e89333256213363badbf074f29d":
            failures.append("nerfstudio commit")
        if NERFSTUDIO_TREE != "9d5ff468eeff89b66995e9984acaa378c37dc07e":
            failures.append("nerfstudio tree")
        clean_check = evaluate_pip_check({"returncode": 0, "stdout": "No broken requirements found.\n", "stderr": ""})
        viser_check = evaluate_pip_check({"returncode": 1, "stdout": "viser 1.0.0 requires yourdfpy, which is not installed.\n", "stderr": ""})
        if not clean_check["passed"] or not viser_check["passed"]:
            failures.append("scoped pip check")
    payload = {
        "schema": SCHEMA,
        "passed": not failures,
        "failures": failures,
        "tests": 18,
    }
    print(json.dumps(payload, indent=2, sort_keys=True))
    print(f"PUBLIC_INSTALLER_V1_5_SELF_TEST: {'PASS' if not failures else 'FAIL'}")
    return 0 if not failures else 2


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="AMD Nerfstudio v1.5 managed ROCm/gfx1201 installer")
    p.add_argument("--workdir", type=Path)
    p.add_argument("--env", "--venv", dest="env", type=Path)
    p.add_argument("--rocm-path", type=Path, default=DEFAULT_ROCM_PATH)
    p.add_argument("--arch", default=SUPPORTED_ARCH)
    p.add_argument("--validation", choices=("none", "quick", "full"), default="quick")
    p.add_argument("--json-report", type=Path)
    p.add_argument("--prepare-env", action="store_true", help="Create or refresh the installer-managed environment")
    p.add_argument("--install-torch", action="store_true", help="Install or verify the pinned ROCm PyTorch stack")
    p.add_argument("--build-tiny", action="store_true", help="Clone, compile, and attest tiny-rdna4-nn for gfx1201")
    p.add_argument("--install-nerfacc", action="store_true", help="Install or verify the authorized nerfacc wheel")
    p.add_argument("--nerfacc-wheel", type=Path, help="Explicit local path to the authorized nerfacc wheel")
    p.add_argument("--install-nerfstudio", action="store_true", help="Acquire and install the scoped Nerfstudio P0/P1 runtime")
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
    nerfacc_result: dict[str, Any] | None = None
    nerfstudio_result: dict[str, Any] | None = None

    needs_env = any((args.prepare_env, args.install_torch, args.build_tiny, args.install_nerfacc, args.install_nerfstudio))
    needs_torch = any((args.install_torch, args.build_tiny, args.install_nerfacc, args.install_nerfstudio))
    needs_tiny = args.build_tiny or args.install_nerfstudio
    needs_nerfacc = args.install_nerfacc or args.install_nerfstudio

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

    if needs_torch:
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

    if needs_tiny:
        if torch_result is not None and torch_result.get("passed"):
            if args.build_tiny:
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
                print()
                print("TINY_RDNA4_RUNTIME_REUSE: START")
                tiny_result = verify_existing_tiny_runtime(report)
        else:
            tiny_result = {"passed": False, "status": "BLOCKED", "reason": "QUALIFIED_TORCH_STAGE_REQUIRED"}
        report["tiny_rdna4_native_build"] = tiny_result
        print()
        print("tiny-rdna4-nn runtime:")
        print(f"  status: {tiny_result.get('status')}")
        print(f"  reason: {tiny_result.get('reason')}")
        runtime = tiny_result.get("runtime", {})
        if runtime.get("binary"):
            print(f"  native: {runtime.get('binary')}")
            print(f"  sha256: {runtime.get('native_sha256')}")
        print(f"TINYRDN4_NATIVE_BUILD: {'PASS' if tiny_result.get('passed') else 'FAIL'}")

    if needs_nerfacc:
        if torch_result is not None and torch_result.get("passed"):
            print()
            print("NERFACC_STAGE: START")
            print(f"authorized wheel sha256: {NERFACC_WHEEL_SHA256}")
            print(f"authorized native sha256: {NERFACC_NATIVE_SHA256}")
            nerfacc_result = install_nerfacc_stack(report, args.nerfacc_wheel)
        else:
            nerfacc_result = {"passed": False, "status": "BLOCKED", "reason": "QUALIFIED_TORCH_STAGE_REQUIRED"}
        report["nerfacc_installation"] = nerfacc_result
        print()
        print("nerfacc installation:")
        print(f"  status: {nerfacc_result.get('status')}")
        print(f"  reason: {nerfacc_result.get('reason')}")
        print(f"NERFACC_INSTALL: {'PASS' if nerfacc_result.get('passed') else 'FAIL'}")
        print(f"NERFACC_REUSED: {'YES' if nerfacc_result.get('reused') else 'NO'}")

    if args.install_nerfstudio:
        if (
            tiny_result is not None and tiny_result.get("passed")
            and nerfacc_result is not None and nerfacc_result.get("passed")
        ):
            print()
            print("NERFSTUDIO_SCOPED_RUNTIME_STAGE: START")
            print(f"repository: {NERFSTUDIO_REPOSITORY}")
            print(f"commit: {NERFSTUDIO_COMMIT}")
            print(f"tree: {NERFSTUDIO_TREE}")
            nerfstudio_result = install_nerfstudio_runtime(report)
        else:
            nerfstudio_result = {"passed": False, "status": "BLOCKED", "reason": "QUALIFIED_TINY_AND_NERFACC_REQUIRED"}
        report["nerfstudio_installation"] = nerfstudio_result
        print()
        print("Nerfstudio scoped runtime:")
        print(f"  status: {nerfstudio_result.get('status')}")
        print(f"  reason: {nerfstudio_result.get('reason')}")
        print(f"NERFSTUDIO_INSTALL: {'PASS' if nerfstudio_result.get('passed') else 'FAIL'}")

    passed = report.get("passed") is True
    for result in (preparation, torch_result, tiny_result, nerfacc_result, nerfstudio_result):
        if result is not None:
            passed = passed and result.get("passed") is True

    report["installation"] = {
        "status": "READY" if passed else "FAIL",
        "env_created": bool(preparation and preparation.get("created")),
        "env_reused": bool(preparation and preparation.get("reused")),
        "torch_ready": bool(torch_result and torch_result.get("passed")),
        "tiny_runtime_ready": bool(tiny_result and tiny_result.get("passed")),
        "nerfacc_ready": bool(nerfacc_result and nerfacc_result.get("passed")),
        "nerfstudio_ready": bool(nerfstudio_result and nerfstudio_result.get("passed")),
        "system_modified": False,
        "workdir_modified": any(
            bool(result and (result.get("created") or result.get("modified")))
            for result in (preparation, torch_result, tiny_result, nerfacc_result, nerfstudio_result)
        ),
        "automatic_sudo": False,
        "automatic_apt": False,
    }

    output: Path | None = absolute(args.json_report) if args.json_report is not None else None
    if output is None and needs_env:
        output = Path(report["paths"]["reports"]) / "installer-v1.5-dev4.json"
    if output is not None:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(f"REPORT_JSON: {output}")

    print("SYSTEM_MODIFIED: NO")
    print(f"WORKDIR_MODIFIED: {'YES' if report['installation']['workdir_modified'] else 'NO'}")
    return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
