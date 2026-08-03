# AMD Nerfstudio ROCm 7.2 / RDNA4 gfx1201

Community integration, qualification and reproducibility project for running the **Nerfacto training chain from Nerfstudio** on AMD RDNA4 / `gfx1201` with ROCm 7.2, PyTorch ROCm, `tiny-rdna4-nn`, and a ROCm-compatible `nerfacc` runtime.

> [!IMPORTANT]
> This repository is **not** a standalone fork of Nerfstudio and does **not** claim support for every Nerfstudio model or feature.
>
> Qualified scope:
>
> ```text
> NERFACTO_TRAINING_CHAIN_NOT_FULL_NERFSTUDIO
> ```

## Status

The internal A0–A5 correctness chain is complete and frozen:

```text
A0 / nerfacc gfx1201 / 4A2:        PASS
A1 Torch baseline:                 PASS / FROZEN
A3-P, A3-S and generic A3:         PASS
A4:                                PASS_SCOPED / FROZEN
A5-P0:                             PASS
A5-P1:                             PASS
A5-P2:                             PASS
A5 correctness-chain freeze:       PASS
```

Canonical A5 freeze:

```text
Freeze ID:  20260803T102615Z_65645
Gate:       NERFSTUDIO_RDNA4_A5_NERFACTO_CORRECTNESS_CHAIN_FREEZE: PASS
Decision:   A5_FROZEN_PROCEED_TO_SEPARATE_VMM_BENCHMARK
Scope:      NERFACTO_TRAINING_CHAIN_NOT_FULL_NERFSTUDIO
```

The separate VMM fallback performance benchmark has **not** been run yet, and no VMM performance claim is made.

## What was qualified

- pinned Nerfstudio source and runtime provenance;
- a real Nerfstudio dataset, dataparser, and `ParallelDataManager` path;
- real Nerfacto forward and loss computation;
- backward with finite, non-zero gradients;
- real optimizer parameter updates;
- checkpoint writing and SHA-256 verification;
- exact fresh-process checkpoint state reload;
- a real resumed training step;
- sustained training over a predefined multi-process trajectory design;
- GPU allocated/reserved memory and host RSS trend checks after warm-up;
- zero observed OOMs and zero allocation retries in the qualified window;
- resume trajectory comparison against the natural A-vs-C process/GPU variation envelope;
- one attested `tiny-rdna4-nn` / SH runtime path without mixed `tinycudann` origins;
- preservation of blocked predecessor runs as recovery evidence;
- a final fail-closed manifest freeze of the complete A5 chain.

### A5-P2 trajectory design

A5-P2 executed **576 training steps in total**, distributed across four fresh processes:

```text
A_continuous:             192 steps
C_continuous_reference:   192 steps
B_split:                   96 steps
B_resume:                  96 steps
Total:                    576 steps
```

`A_continuous` and `C_continuous_reference` measured the natural process/GPU variation for the same seed and configuration. The split/resume B trajectory was required to remain within a predefined envelope derived from fixed numeric floors and that A-vs-C reference variation.

Checkpoint state integrity and resumed trajectory behavior were separate gates:

- loaded pipeline, optimizer, scheduler, and scaler state: exact;
- multi-step floating-point trajectory: tolerance-based, with thresholds fixed before the run.

## Validated environment

| Component | Validated value |
|---|---|
| GPU | AMD Radeon AI PRO R9700 |
| Architecture | RDNA4 / `gfx1201` |
| Operating system | Ubuntu 24.04 |
| Python | 3.12.3 |
| PyTorch | `2.13.0+rocm7.2` |
| HIP reported by PyTorch | `7.2.53211` |
| Nerfstudio commit | `50e0e3c70c775e89333256213363badbf074f29d` |
| Nerfstudio Git tree | `9d5ff468eeff89b66995e9984acaa378c37dc07e` |
| Training model | Nerfacto |
| Training rays per batch | 1024 |
| Qualification dataset | 6 images at 128×128 |

Runtime anchors:

```text
Nerfstudio mlp.py
4939a5a6901d82d8e310d93e2a135ca57ccc1bd79be79a7f67e2740e730c44ad

tiny-rdna4-nn native extension
883f89efdad7bb909a4a3899ab79b2defe9713fdb5c7cf22cf4882c626b3efc4

tinycudann/modules.py
b4df43b54f64fe2b31272a997aafd50137aecac411d59b05251acedcd5512d12

ROCm-compatible nerfacc native extension
d3beee150cfa3a9ad3038a3283ff0a46953c345634d8cb6109449c5e3d04d1e2
```

## Claim boundary

This repository supports the following scoped statement:

> The qualified Nerfacto training chain from Nerfstudio was demonstrated on AMD RDNA4 / `gfx1201` through `tiny-rdna4-nn`, PyTorch ROCm, and a ROCm-compatible `nerfacc` path. The frozen evidence covers the real data path, forward, backward, optimizer updates, checkpointing, exact fresh-process state reload, resumed trajectory behavior within the predefined reference envelope, and stable memory behavior in the qualified 576-step window.

Not claimed:

- support for all Nerfstudio models and features;
- Viewer or export qualification;
- Splatfacto qualification;
- multi-GPU or distributed training;
- unlimited-horizon memory-leak freedom;
- universal support for other AMD architectures or ROCm versions;
- VMM fallback performance parity;
- performance superiority over CUDA/NVIDIA;
- cross-host or cross-checkout binary identity unless separately demonstrated.

## Documentation

- [Qualification scope](docs/QUALIFICATION_SCOPE.md)
- [A5 validation chain](docs/A5_VALIDATION_CHAIN.md)
- [Recovery history](docs/RECOVERY_HISTORY.md)
- [Checkpoint trust policy](docs/CHECKPOINT_TRUST_POLICY.md)
- [Public release roadmap](docs/PUBLICATION_ROADMAP.md)
- [Public Toolchain v1](docs/PUBLIC_TOOLCHAIN_V1.md)
- [Public Toolchain v1 static validation](evidence-public/PUBLIC_TOOLCHAIN_V1_STATIC_VALIDATION.md)
- [Public Toolchain v1 tool hashes](evidence-public/PUBLIC_TOOLCHAIN_V1_TOOL_SHA256SUMS.txt)
- [Public quick-validation reference replay](evidence-public/PUBLIC_QUICK_VALIDATION_REFERENCE_REPLAY.md)
- [Public fresh environment v1](docs/PUBLIC_FRESH_ENV_V1.md)
- [Resource-cache policy](resources/README.md)
- [Frozen A5 hashes](evidence-public/A5_FREEZE_SHA256SUMS.txt)

## Public Toolchain v1.2

The repository now includes Public Toolchain v1.2, with a short normal-user validation path and a separately guarded maintainer qualification path:

- fail-closed P0 source/runtime/dataset preflight;
- real P1 Nerfacto DataManager, forward, backward, optimizer, checkpoint, exact fresh-process reload, and resumed step;
- one-command P0+P1 quick validation that never launches P2;
- verified deletion of temporary P1 checkpoints by default, with `--keep-checkpoints` as an explicit opt-in;
- sustained P2 A/C/B split-resume qualification using the original 576-step design, guarded as maintainer-only;
- a separate public requalification freeze;
- a public-tree audit that rejects host-specific paths, secrets, nested Git trees, archives, checkpoints, and native binaries.

The public runners do **not** rewrite or supersede the canonical private A5 freeze. They produce a new public requalification chain on each host.

See [Public Toolchain v1](docs/PUBLIC_TOOLCHAIN_V1.md).

### Normal user validation

After setting the five explicit `NERFSTUDIO_RDNA4_PUBLIC_*` path variables described in the toolchain documentation:

```bash
scripts/run_public_quick_validation_v1.sh
```

This executes only P0 and the two-step P1 producer/reload smoke. Temporary checkpoints are hash-verified and removed after a successful run. Use `--keep-checkpoints` only when the files are needed for debugging or evidence retention.

P2 is never started by the quick wrapper. The P2 entry point requires an explicit `--maintainer-confirm` acknowledgement.

### Public Toolchain v1.3 fresh environment

Public Toolchain v1.3 adds the `reference-binary-fresh-env` installer:

```bash
scripts/setup_public_fresh_env_v1.sh \
  --resource-dir "$PWD/resource-cache-v1" \
  --install-root "$PWD/fresh-env-v1" \
  --nerfacc-wheel /path/to/qualified-nerfacc.whl \
  --tcnn-runtime /path/to/qualified-tiny-rdna4-nn-runtime \
  --dataset /path/to/qualified-quick-dataset
```

The default mode asks before public network access. `--auto` approves only pinned public operations; `--offline` forbids network access; `--download-only` prepares the cache; and `--verify-resources` performs non-mutating verification.

The installer creates a new Python 3.12 virtual environment, downloads a scoped wheelhouse from PyPI plus the official PyTorch ROCm 7.2 index, writes a SHA-256 lock for every fetched wheel, installs only from that local wheelhouse, copies the exact qualified custom runtime inputs, records pip provenance, and runs the normal-user P0+P1 quick validation.

The pinned Nerfstudio 1.1.5 line uses `viser==0.2.7`. Wheelhouse verification requires `opencv-python-headless==4.10.0.84` and rejects a simultaneous `opencv-python` wheel because both distributions install the same `cv2` package tree.

The three custom resources currently have no download URL in the manifest. They must be supplied from an explicit local path or an existing verified cache. A fresh native rebuild is not claimed and is rejected fail-closed.

See [Public fresh environment v1](docs/PUBLIC_FRESH_ENV_V1.md).

### Publication state

Public Toolchain v1.3 has passed static self-tests and repository-tree audit as an installer candidate. The v1.2 neutral-directory P0+P1 reference-runtime replay has passed. A real fresh-environment GPU execution of v1.3 remains the next qualification gate and is not claimed until that run succeeds.

## Related projects

- [Painter3000/tiny-rdna4-nn](https://github.com/Painter3000/tiny-rdna4-nn)
- [Painter3000/amd-nvdiffrast-rocm72-gfx1201](https://github.com/Painter3000/amd-nvdiffrast-rocm72-gfx1201)
- [Painter3000/amd-gsplat-rocm72-gfx1201](https://github.com/Painter3000/amd-gsplat-rocm72-gfx1201)
- [nerfstudio-project/nerfstudio](https://github.com/nerfstudio-project/nerfstudio)
- [nerfstudio-project/nerfacc](https://github.com/nerfstudio-project/nerfacc)

## License and attribution

Original helper scripts and documentation added by this repository are covered by the repository license. Third-party projects retain their own licenses and are not relicensed here.

This is an independent community project and is not an official AMD, Nerfstudio, PyTorch, or NVIDIA repository.
