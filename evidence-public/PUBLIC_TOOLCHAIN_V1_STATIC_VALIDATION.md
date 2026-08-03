# Public Toolchain v1 — static validation

**Date:** 2026-08-03
**Scope:** static conversion and fail-closed fixture validation only
**GPU execution:** **NOT RUN**
**Neutral fresh-clone execution:** **NOT RUN**

## Result

```text
PUBLIC_TOOLCHAIN_V1_PYTHON_COMPILE: PASS
PUBLIC_TOOLCHAIN_V1_SHELL_PARSE: PASS
PUBLIC_TOOLCHAIN_V1_SELF_TESTS: PASS (18/18)
PUBLIC_TOOLCHAIN_V1_P0_FAIL_CLOSED_FIXTURE: PASS
PUBLIC_TOOLCHAIN_V1_PUBLIC_FREEZE_FIXTURE: PASS
PUBLIC_TOOLCHAIN_V1_PUBLIC_TREE_AUDIT: PASS
PUBLIC_TOOLCHAIN_V1_GPU_QUALIFICATION: NOT_RUN
PUBLIC_TOOLCHAIN_V1_FRESH_CLONE_GPU_QUALIFICATION: NOT_RUN
```

## Validated properties

- all public runtime, source, dataset, and output paths are explicit;
- no home-directory discovery or project-specific local default path remains;
- P0 blocks on incomplete source/runtime/dataset fixtures;
- P1 and P2 numeric/helper self-tests pass;
- the normal-user quick wrapper self-test proves that P2 is never launched;
- successful P1 checkpoint files are deleted only after recorded size and SHA-256 verification;
- P2 shell and Python entry points require explicit maintainer confirmation;
- the public freeze verifies P0/P1/P2 manifests and native decision semantics;
- P0's dataset-object representation and P1/P2's dataset-string representation resolve to the same canonical path;
- the optional freeze archive does not mutate files already covered by its manifest;
- shell wrappers parse under `bash -n` and use `set -euo pipefail`;
- Python sources compile successfully;
- the repository audit rejects private host paths, credential patterns, nested Git trees, symlinks, native binaries, checkpoints, archives, bytecode, oversized files, invalid JSON, missing executable bits, and incomplete or mismatched repository hashes.

## Commands

```bash
python3 -m py_compile tools/*.py tests/*.py
for file in scripts/*.sh; do bash -n "$file"; done
python3 -m unittest -v tests/test_public_toolchain_v1.py
python3 tools/audit_public_tree_v1.py --repo .
```

## Important boundary

This validation proves that the public conversion is structurally self-consistent and fail-closed under its static fixtures. It does **not** prove that a fresh public clone has already reproduced the real R9700 P0/P1/P2 GPU chain. That remains a separate execution gate.

The public requalification tools do not modify or supersede the canonical private A5 freeze `20260803T102615Z_65645`.

## Public Toolchain v1.1 correction

```text
VIRTUAL_ENV_PYTHON_SYMLINK_PRESERVATION: PASS
P0_P1_P2_PYTHON_PATH_POLICY: ABSOLUTE_WITH_FINAL_SYMLINK_PRESERVED
```

The regression fixture supplies a synthetic `venv/bin/python` symlink and verifies that the public runner retains the launcher path instead of resolving it to the system interpreter.

## Public Toolchain v1.2 quick-user policy

```text
PUBLIC_QUICK_VALIDATION_SELF_TEST: PASS
P1_DEFAULT_CHECKPOINT_POLICY: DELETE_AFTER_VERIFICATION
P1_FAILURE_CHECKPOINT_POLICY: RETAIN_ON_FAILURE
P2_POLICY: MAINTAINER_ONLY
P2_CONFIRMATION_GUARD: PASS
```

The static fixture creates synthetic producer and reload checkpoint files, records their size and SHA-256, applies the default retention policy, and verifies that both files are removed only after exact verification.

The neutral-directory GPU replay of the reference runtime is documented separately in `PUBLIC_QUICK_VALIDATION_REFERENCE_REPLAY.md`. The v1.3 installer candidate is now present, but a real environment created by it remains unqualified until the fresh GPU run passes.

## Public Toolchain v1.3 fresh-environment candidate

```text
PUBLIC_FRESH_ENV_RESOURCE_MANAGER_SELF_TEST: PASS
PUBLIC_FRESH_ENV_INSTALLER_SELF_TEST: PASS
SCOPED_NERFACTO_CONFIG_POLICY: PASS
OFFLINE_MISSING_RESOURCE_FAIL_CLOSED: PASS
FRESH_NATIVE_PROFILE_REJECTION: PASS
WHEELHOUSE_HASH_MUTATION_REJECTION: PASS
PINNED_VISER_0_2_7_POLICY: PASS
DUPLICATE_CV2_PROVIDER_REJECTION: PASS
PUBLIC_TOOLCHAIN_V1_SELF_TESTS: PASS (18/18)
PUBLIC_FRESH_ENV_GPU_EXECUTION: NOT_RUN
```

The v1.3 static fixtures verify exact custom-resource anchors, non-mutating cache
verification, wheelhouse SHA-256 locking, modified-wheel rejection, activation
policy generation, offline failure behavior, and rejection of the unqualified
`fresh-native-build` profile.

P0, P1, and P2 now construct only the pinned Nerfacto configuration through the
scoped public builder. The static policy rejects a return to Nerfstudio's global
method registry, which eagerly imports unrelated model stacks.

This static result does not yet qualify a real environment created by the new
installer. The next gate is an online cache preparation followed by a clean
Python 3.12 environment creation and the real R9700 P0+P1 quick validation.

## Public Toolchain v1.3.1 dependency correction

The first real v1.3 resource fetch was stopped before installation after the
SHA-256-clean lock exposed both `opencv-python` and `opencv-python-headless`.
The correction pins Viser 0.2.7, matching the pinned Nerfstudio 1.1.5 line,
and adds fail-closed wheelhouse policy checks for duplicate `cv2` providers.
The previous external lock is invalidated by the changed requirements hash.
