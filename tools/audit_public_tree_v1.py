#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import stat
import sys
sys.dont_write_bytecode = True
from typing import Any

from public_toolchain_common import inventory_tree, json_dump, sha256

SCHEMA = "amd-nerfstudio-public-tree-audit-v1"
TEXT_SUFFIXES = {".md", ".txt", ".py", ".sh", ".json", ".toml", ".yml", ".yaml", ".ini", ".cfg", ".gitignore"}
BINARY_DENY = {".so", ".o", ".a", ".whl", ".ckpt", ".pt", ".pth", ".zip", ".tar", ".gz", ".zst", ".pyc", ".pyo"}
PATH_PATTERNS = [
    ("ABSOLUTE_HOME_PATH", re.compile(r"/home/[A-Za-z0-9._-]+/")),
    ("TILDE_HOME_PATH", re.compile(r"(?<![A-Za-z0-9])~/")),
    ("PRIVATE_WORKSPACE_NAME", re.compile(r"therock_test|Dokumente|Downloads")),
    ("HOST_PROMPT", re.compile(r"\boem@|oem-System-Product-Name")),
]
SECRET_PATTERNS = [
    ("GITHUB_TOKEN", re.compile(r"\b(?:ghp|github_pat)_[A-Za-z0-9_]{20,}\b")),
    ("HUGGINGFACE_TOKEN", re.compile(r"\bhf_[A-Za-z0-9]{20,}\b")),
    ("PRIVATE_KEY", re.compile(r"-----BEGIN (?:RSA |OPENSSH |EC )?PRIVATE KEY-----")),
    ("ASSIGNMENT_SECRET", re.compile(r"(?i)\b(?:api[_-]?key|access[_-]?token|secret|password)\s*[:=]\s*['\"][^'\"]{8,}['\"]")),
]
ALLOWED_PRIVATE_WORD_FILES = {"docs/PUBLIC_TOOLCHAIN_V1.md", "tools/audit_public_tree_v1.py"}


def audit(root: Path, max_file_bytes: int) -> dict[str, Any]:
    findings: list[dict[str, Any]] = []
    files = 0
    for path in sorted(root.rglob("*")):
        rel = str(path.relative_to(root))
        if ".git" in path.parts:
            if path.name == ".git" and path != root / ".git":
                findings.append({"kind": "NESTED_GIT", "path": rel})
            continue
        if path.is_symlink():
            findings.append({"kind": "SYMLINK", "path": rel, "target": str(path.readlink())})
            continue
        if not path.is_file():
            continue
        files += 1
        size = path.stat().st_size
        if size > max_file_bytes:
            findings.append({"kind": "FILE_TOO_LARGE", "path": rel, "size_bytes": size, "limit": max_file_bytes})
        if path.suffix.lower() in BINARY_DENY:
            findings.append({"kind": "BINARY_OR_ARCHIVE_DENIED", "path": rel, "suffix": path.suffix.lower()})
        is_text = path.suffix.lower() in TEXT_SUFFIXES or path.name in {"LICENSE", "NOTICE", "SHA256SUMS.txt", ".gitignore"}
        if not is_text:
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            findings.append({"kind": "NON_UTF8_TEXT", "path": rel})
            continue
        for kind, pattern in SECRET_PATTERNS:
            for match in pattern.finditer(text):
                findings.append({"kind": kind, "path": rel, "line": text.count("\n", 0, match.start()) + 1})
        if rel not in ALLOWED_PRIVATE_WORD_FILES:
            for kind, pattern in PATH_PATTERNS:
                for match in pattern.finditer(text):
                    findings.append({"kind": kind, "path": rel, "line": text.count("\n", 0, match.start()) + 1, "match": match.group(0)})
        if path.suffix == ".json":
            try:
                json.loads(text)
            except Exception as exc:
                findings.append({"kind": "INVALID_JSON", "path": rel, "error": repr(exc)})
        if path.suffix == ".py":
            if not text.startswith("#!/usr/bin/env python3"):
                findings.append({"kind": "PYTHON_SHEBANG_MISSING", "path": rel})
            if rel.startswith(("tools/", "tests/")) and not (path.stat().st_mode & stat.S_IXUSR):
                findings.append({"kind": "PYTHON_NOT_EXECUTABLE", "path": rel})
        if path.suffix == ".sh":
            if not text.startswith("#!/usr/bin/env bash"):
                findings.append({"kind": "SHELL_SHEBANG_MISSING", "path": rel})
            if "set -euo pipefail" not in text:
                findings.append({"kind": "SHELL_STRICT_MODE_MISSING", "path": rel})
            if not (path.stat().st_mode & stat.S_IXUSR):
                findings.append({"kind": "SHELL_NOT_EXECUTABLE", "path": rel})

    sums_path = root / "SHA256SUMS.txt"
    if sums_path.is_file():
        listed: dict[str, str] = {}
        for number, line in enumerate(sums_path.read_text(encoding="utf-8").splitlines(), 1):
            if not line.strip() or line.lstrip().startswith("#"):
                continue
            parts = line.split(None, 1)
            if len(parts) != 2:
                findings.append({"kind": "SHA256SUMS_INVALID_LINE", "path": "SHA256SUMS.txt", "line": number})
                continue
            digest, relname = parts
            relname = relname.removeprefix("*").removeprefix("./")
            listed[relname] = digest
        expected_files = {
            str(path.relative_to(root)) for path in root.rglob("*")
            if path.is_file()
            and ".git" not in path.parts
            and "__pycache__" not in path.parts
            and path.name != "SHA256SUMS.txt"
        }
        missing = sorted(expected_files - set(listed))
        extra = sorted(set(listed) - expected_files)
        if missing:
            findings.append({"kind": "SHA256SUMS_MISSING_FILES", "path": "SHA256SUMS.txt", "files": missing})
        if extra:
            findings.append({"kind": "SHA256SUMS_EXTRA_FILES", "path": "SHA256SUMS.txt", "files": extra})
        for relname, digest in sorted(listed.items()):
            target = root / relname
            if target.is_file() and sha256(target) != digest:
                findings.append({"kind": "SHA256SUMS_MISMATCH", "path": relname, "expected": digest, "observed": sha256(target)})
    else:
        findings.append({"kind": "SHA256SUMS_MISSING", "path": "SHA256SUMS.txt"})
    return {
        "schema": SCHEMA,
        "root": str(root),
        "file_count": files,
        "finding_count": len(findings),
        "findings": findings,
        "passed": not findings,
    }


def self_test() -> int:
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        ok_file = root / "ok.py"
        ok_file.write_text("#!/usr/bin/env python3\nprint('ok')\n")
        ok_file.chmod(0o755)
        (root / "SHA256SUMS.txt").write_text(f"{sha256(ok_file)}  ok.py\n")
        clean = audit(root, 1024)
        bad = root / "bad.txt"
        bad.write_text("/home/alice/private\n")
        dirty = audit(root, 1024)
    ok = clean["passed"] and not dirty["passed"]
    print(json.dumps({"schema": SCHEMA, "passed": ok, "clean": clean, "dirty_finding_count": dirty["finding_count"]}, indent=2))
    return 0 if ok else 2


def main() -> int:
    parser = argparse.ArgumentParser(description="Fail-closed public repository tree audit")
    parser.add_argument("--mode", choices=["run", "self-test"], default="run")
    parser.add_argument("--repo", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--max-file-bytes", type=int, default=2 * 1024 * 1024)
    args = parser.parse_args()
    if args.mode == "self-test":
        return self_test()
    if args.repo is None:
        parser.error("run mode requires --repo")
    root = args.repo.expanduser().resolve()
    report = audit(root, args.max_file_bytes)
    if args.output:
        json_dump(args.output.expanduser().resolve(), report)
    print(json.dumps(report, indent=2, sort_keys=True))
    print(f"PUBLIC_TREE_AUDIT: {'PASS' if report['passed'] else 'FAIL'}")
    return 0 if report["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
