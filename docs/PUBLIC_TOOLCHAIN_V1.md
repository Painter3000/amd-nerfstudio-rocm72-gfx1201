# Public Toolchain v1

## Purpose

Public Toolchain v1 converts the private A5 methodology into a path-independent, fail-closed public requalification workflow. It does **not** rewrite or replace the canonical private A5 freeze.

The maintainer chain is:

```text
public P0 preflight
→ public P1 real Nerfacto smoke + exact fresh-process reload
→ public P2 sustained A/C/B split-resume qualification
→ public requalification freeze
```

All runtime paths are explicit. The tools do not search user home directories, `Downloads`, project-specific workspaces, or ambient Python installations.

The normal user path stops after P1:

```text
public P0 preflight
→ public P1 real Nerfacto smoke + exact fresh-process reload
→ PUBLIC_QUICK_VALIDATION_QUALIFIED
```

The quick wrapper has no code path that launches P2.

### v1.1 Python-launcher correction

Public Toolchain v1.1 preserves the explicitly supplied `venv/bin/python` path. It intentionally does not resolve the final Python symlink, because doing so changes Python prefix discovery and can replace the virtual-environment launcher with `/usr/bin/python3.12`. Repository, runtime, dataset, and evidence paths remain canonically resolved.

### v1.2 quick-user policy

Public Toolchain v1.2 adds a one-command P0+P1 path, deletes the two temporary P1 checkpoints after successful hash and reload verification by default, and requires an explicit maintainer confirmation before P2 can start. Failed P1 runs retain any checkpoint that exists so recovery evidence is not destroyed.

## Components

| File | Purpose |
|---|---|
| `config/reference_gfx1201_rocm72.json` | pinned target versions, source identities, runtime hashes, loader policy, and nonclaims |
| `tools/run_public_a5p0_preflight_v1.py` | source/runtime/dataset validation and runtime-policy generation |
| `tools/run_public_a5p1_nerfacto_smoke_v1.py` | real DataManager, forward, backward, optimizer, checkpoint, exact fresh reload, and one resumed step |
| `tools/run_public_quick_validation_v1.py` | normal-user P0+P1 orchestration, quick gate, manifest chain, and checkpoint-retention policy |
| `tools/run_public_a5p2_sustained_v1.py` | A/C/B trajectory design, memory trends, checkpoint chain, data-stream replay, and resume envelope |
| `tools/run_public_a5_freeze_v1.py` | manifest-verifies and freezes a successful public P0/P1/P2 chain |
| `tools/audit_public_tree_v1.py` | rejects private host paths, credentials, nested Git trees, symlinks, archives, checkpoints, native binaries, and oversized files |

## Runtime contract

The validated reference remains:

```text
GPU:        AMD Radeon AI PRO R9700
Architecture: gfx1201 / RDNA4
ROCm:       7.2
PyTorch:    2.13.0+rocm7.2
Python:     3.12
Nerfstudio: 50e0e3c70c775e89333256213363badbf074f29d
```

The public tools intentionally use the PyTorch-facing `cuda` device namespace on ROCm where Nerfstudio and PyTorch expose their HIP backend through that compatibility surface.

## P0

P0 is read-only. It requires explicit paths for:

- Python launcher;
- pinned Nerfstudio worktree;
- `tiny-rdna4-nn` runtime root;
- Nerfstudio-format dataset;
- output root.

P0 verifies source commit/tree, a clean tracked worktree, `mlp.py`, PyTorch/HIP/architecture, `tinycudann` origins and hashes, ROCm `nerfacc`, selected Nerfacto imports, DataLoader configuration fields, and all dataset frame paths. It emits a reviewed runtime policy instead of silently modifying the shell.

## P1

P1 launches two fresh child processes:

1. producer at step 0;
2. fresh checkpoint reload at step 1.

It sets `dataloader_num_workers=1` and `prefetch_factor=2` through the fields present in the pinned Nerfstudio source. Missing fields are hard failures. The runtime-created DataLoader must report one worker.

P1 does not claim sustained stability.

On success, P1 verifies the SHA-256 and size of both locally generated checkpoints. The default policy then deletes them before the final P1 manifest is written. The child JSON reports retain their original paths, sizes, and hashes as evidence. `--keep-checkpoints` retains both files explicitly. A failed P1 run uses `RETAIN_ON_FAILURE` regardless of the requested default.

## Quick user validation

With the explicit environment variables shown below:

```bash
scripts/run_public_quick_validation_v1.sh
```

The quick wrapper creates fresh P0 and P1 run IDs, verifies both manifests, reports a single `PUBLIC_QUICK_VALIDATION_QUALIFIED` decision, and records:

```text
p2_execution=NOT_RUN
p2_policy=MAINTAINER_ONLY
```

To retain the temporary checkpoints:

```bash
scripts/run_public_quick_validation_v1.sh --keep-checkpoints
```

## P2 runtime warning

The reference P2 design runs **576 total training steps**:

```text
A_continuous:             192
C_continuous_reference:   192
B_split:                   96
B_resume:                  96
Total:                    576
```

On the original R9700 qualification system this took roughly 85–95 minutes. Runtime varies with hardware, clocks, compilation caches, and storage. P2 is not a normal-user test and cannot be launched through the quick wrapper.

The separate P2 entry point requires:

```bash
scripts/run_public_a5p2_sustained_v1.sh --maintainer-confirm
```

P2 keeps the original predeclared tolerance and memory policies. Its result is a public requalification result, not a retroactive modification of the canonical private A5 freeze.

## Quick static validation

No GPU work:

```bash
python3 -m unittest -v tests/test_public_toolchain_v1.py
scripts/audit_public_tree_v1.sh
```

## Example environment

```bash
export NERFSTUDIO_RDNA4_PUBLIC_PYTHON=/absolute/path/to/venv/bin/python
export NERFSTUDIO_RDNA4_PUBLIC_NERFSTUDIO_WORKTREE=/absolute/path/to/nerfstudio
export NERFSTUDIO_RDNA4_PUBLIC_TCNN_RUNTIME=/absolute/path/to/tiny-rdna4-runtime
export NERFSTUDIO_RDNA4_PUBLIC_DATASET=/absolute/path/to/dataset
export NERFSTUDIO_RDNA4_PUBLIC_OUTPUT_ROOT=/absolute/path/to/evidence
```

Normal users should run the quick wrapper. Maintainers may run P0 and P1 separately when investigating or preserving individual stage evidence. Do not guess or reuse a failed run directory.

## Scope

```text
NERFACTO_TRAINING_CHAIN_NOT_FULL_NERFSTUDIO
```

Viewer, export, Splatfacto, multi-GPU, unlimited-horizon leak freedom, VMM performance parity, and CUDA performance superiority remain outside the claim.

## v1.3 scoped configuration and fresh environment

Public Toolchain v1.3 adds `tools/public_nerfacto_config_v1.py`. P0, P1, and P2
construct the pinned Nerfacto configuration through this scoped builder instead
of importing Nerfstudio's global method registry. The global registry eagerly
imports unrelated models such as Splatfacto and would impose dependencies that
are outside the qualified Nerfacto scope.

The configuration builder preserves the pinned Nerfacto DataManager, model,
optimizer, scheduler, viewer configuration object, and numeric defaults from the
qualified upstream commit.

The `reference-binary-fresh-env` installer is documented in
`PUBLIC_FRESH_ENV_V1.md`. It creates a new Python environment but deliberately
reuses the exact qualified native artifacts. P2 remains maintainer-only and is
never called by setup or quick validation.
