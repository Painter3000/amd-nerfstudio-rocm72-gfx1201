# Public Toolchain v1

## Purpose

Public Toolchain v1 converts the private A5 methodology into a path-independent, fail-closed public requalification workflow. It does **not** rewrite or replace the canonical private A5 freeze.

The public chain is:

```text
public P0 preflight
→ public P1 real Nerfacto smoke + exact fresh-process reload
→ public P2 sustained A/C/B split-resume qualification
→ public requalification freeze
```

All runtime paths are explicit. The tools do not search user home directories, `Downloads`, project-specific workspaces, or ambient Python installations.

## Components

| File | Purpose |
|---|---|
| `config/reference_gfx1201_rocm72.json` | pinned target versions, source identities, runtime hashes, loader policy, and nonclaims |
| `tools/run_public_a5p0_preflight_v1.py` | source/runtime/dataset validation and runtime-policy generation |
| `tools/run_public_a5p1_nerfacto_smoke_v1.py` | real DataManager, forward, backward, optimizer, checkpoint, exact fresh reload, and one resumed step |
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

## P2 runtime warning

The reference P2 design runs **576 total training steps**:

```text
A_continuous:             192
C_continuous_reference:   192
B_split:                   96
B_resume:                  96
Total:                    576
```

On the original R9700 qualification system this took roughly 85–95 minutes. Runtime varies with hardware, clocks, compilation caches, and storage. P2 should not be started as a quick smoke test.

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

Then run P0 and use the printed run directory for P1. Do not guess or reuse a failed run directory.

## Scope

```text
NERFACTO_TRAINING_CHAIN_NOT_FULL_NERFSTUDIO
```

Viewer, export, Splatfacto, multi-GPU, unlimited-horizon leak freedom, VMM performance parity, and CUDA performance superiority remain outside the claim.
