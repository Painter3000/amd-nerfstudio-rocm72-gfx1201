# AMD Nerfstudio ROCm 7.2 / RDNA4 gfx1201

Community integration, qualification and reproducibility project for running the **Nerfacto training chain from Nerfstudio** on AMD RDNA4 / `gfx1201` with ROCm 7.2, PyTorch ROCm, `tiny-rdna4-nn`, and a ROCm-compatible `nerfacc` runtime.

<!-- BEGIN PUBLIC_INSTALLER_V1_5_0_RELEASE -->
## Public Installer v1.5.0 — qualified P0/P1 release

> [!NOTE]
> This is the current public release path for the qualified Nerfacto training
> chain on AMD Radeon AI PRO R9700 / RDNA4 (`gfx1201`) with ROCm 7.2.

```text
PUBLIC_INSTALLER_V1_5_0: QUALIFIED
TARGET: AMD Radeon AI PRO R9700 / gfx1201
PYTHON: 3.12.3
PYTORCH: 2.13.0+rocm7.2
HIP: 7.2.53211
PUBLIC_P0_P1_GATES: 28/28 PASS
FAILED_GATES: 0
REPLACEMENT_RUNS: NONE
P2_EXECUTION: NOT_RUN
P2_POLICY: MAINTAINER_ONLY
```

Qualified anchors:

```text
Implementation commit:
d73e72615bba404dc8e6b105674890f4abcb6311

Implementation tree:
787e8e296e55cccc2b0df21106ecb951a9d9f343

Qualification freeze commit:
d9555c83183c89bf9e9a79fc77cc1c0c6b9d6c16

Main merge commit:
117d625d8925c90f99222f18a0cb22eec90b31fa

Qualification run:
20260805T092040Z_43673

Evidence archive SHA-256:
9f104f5fac3b434852e0f31483cd2b421964654641b8454666c8f6911c53556c
```

The qualified run demonstrated a real DataLoader batch, Nerfacto forward and
finite loss, backward with finite nonzero gradients, an optimizer parameter
update, checkpoint write and hashing, exact reload in a fresh process, and a
fresh-process resume step. The spawned DataLoader worker used the attested
`public_nerfacto_config_v1.public_spawn_worker_init` hook.

Release documentation:

- [`docs/PUBLIC_INSTALLER_V1_5.md`](docs/PUBLIC_INSTALLER_V1_5.md)
- [`docs/PUBLIC_INSTALLER_V1_5_DEV5.md`](docs/PUBLIC_INSTALLER_V1_5_DEV5.md)
- [`docs/PUBLIC_INSTALLER_V1_5_DEV5F_QUALIFICATION.md`](docs/PUBLIC_INSTALLER_V1_5_DEV5F_QUALIFICATION.md)
- [`config/public_installer_v1_5_dev5f_qualification.json`](config/public_installer_v1_5_dev5f_qualification.json)

The qualification is scoped to the pinned `gfx1201` P0/P1 mechanics. It does
not claim full Nerfstudio feature coverage, Viewer or Splatfacto support,
multi-GPU support, production reconstruction quality, sustained P2 stability,
or performance superiority over CUDA/NVIDIA.
<!-- END PUBLIC_INSTALLER_V1_5_0_RELEASE -->

<!-- BEGIN NERFSTUDIO_EXPLAINER_AND_VALIDATION_STAGES -->
## What Is Nerfstudio — and Why Does This Repository Exist?

Nerfstudio is not a traditional photogrammetry application. Instead of
primarily measuring hard 3D geometry, it uses AI methods such as **Neural
Radiance Fields (NeRFs)** and graphics techniques such as **Gaussian
Splatting** to reconstruct how light, color, and viewing direction interact
within a scene.

The result is not primarily a precise CAD model or a mesh intended for 3D
printing. It is a photorealistic spatial representation that can be explored
interactively and viewed from new camera positions in real time.

Nerfstudio is especially strong in scenes and on surfaces that often cause
problems for traditional photogrammetry:

- reflections and polished metal;
- fine hair and complex surface detail;
- smoke, fog, and partially transparent regions;
- scenes where visual appearance matters more than millimeter-accurate
  geometry.

Traditional photogrammetry tools such as RealityCapture or Meshroom remain the
better choice when the goal is a geometrically precise and editable
reconstruction, for example:

- 3D printing;
- CAD and BIM;
- surveying;
- technical inspection;
- dimensionally accurate meshes.

In simple terms:

> **Nerfstudio is particularly well suited for video, VFX, virtual tours, and
> immersive scenes. Photogrammetry is better suited for precise, measurable,
> and editable 3D models.**

The two approaches are not true competitors. Nerfstudio itself commonly uses
Structure-from-Motion and photogrammetry tools during preprocessing to estimate
the camera positions of the input images.

### Why this repository exists

The practical limitation has traditionally been GPU support. Nerfstudio's
high-performance Nerfacto training path depends, among other components, on
`tiny-cuda-nn`, which made it strongly tied to NVIDIA GPUs and CUDA in
real-world use.

**This repository addresses that limitation.**

It brings the qualified **Nerfacto training path to AMD RDNA4 (`gfx1201`) using
ROCm 7.2 — without CUDA and without an NVIDIA GPU.**

The current public Installer v1.5.0 qualification covers:

```text
Managed AMD RDNA4 / gfx1201 runtime: PASS
Pinned ROCm PyTorch stack: PASS
Pinned tiny-rdna4-nn runtime: PASS
Hash-attested nerfacc runtime: PASS
Synthetic dataset deployment: PASS
PortableMLP policy: PASS
Pillow encoder-extents compatibility: PASS
Spawn-worker compatibility hook: PASS
P0/P1 gates: 28/28 PASS
Checkpoint write/reload/resume: PASS
P2 sustained validation: NOT_RUN / MAINTAINER_ONLY
```

### What do P0, P1, and P2 mean?

The validation process is deliberately divided into separate stages.

#### P0 — Runtime and provenance validation

P0 verifies the required environment before training begins, including the AMD
GPU and `gfx1201` architecture, ROCm and PyTorch, the native `nerfacc`
extension, the qualified `tiny-rdna4-nn` runtime, the pinned Nerfstudio source,
the validation dataset and its hashes, and the disabled fail-closed Viewer
path.

```text
PUBLIC_RDNA4_QUICK_P0_PREFLIGHT: PASS
```

#### P1 — Short real Nerfacto execution

P1 is not merely an import test. It runs the real Nerfacto mechanics on the
GPU, including training execution, checkpoint creation, checkpoint reload, and
the associated evidence chain.

```text
PUBLIC_RDNA4_QUICK_P1_REAL_MECHANICS: PASS
PUBLIC_RDNA4_QUICK_VALIDATION: PASS
```

#### P2 — Sustained stability validation

P2 is a separate, substantially longer maintainer validation. It evaluates
sustained training, longer execution times, and stability problems that may not
appear during a short qualification run. The public quick-validation wrapper
never launches P2 automatically.

### Why does the README show both `P2: PASS` and `P2: NOT_RUN`?

They refer to **two different qualification runs**:

1. The canonical internal A5 correctness chain completed A5-P2 successfully.
   That frozen run executed 576 training steps across the defined continuous
   and split/resume trajectories, so the top-level project status correctly
   reports `A5-P2: PASS`.
2. The public Installer v1.5.0 qualification executed the real short P0+P1
   path with the managed runtime, dataset deployment, PortableMLP policy,
   Pillow compatibility, and the spawned DataLoader worker hook. It did not
   rerun the maintainer-only P2 stage or a separate long-duration campaign.

The v1.5.0 qualification evidence therefore states:

```text
P2_EXECUTION: NOT_RUN
P2_POLICY: MAINTAINER_ONLY
LONG_DURATION_TRAINING: NOT_RUN
```

`NOT_RUN` does not mean `FAIL`. It means that this specific public release
qualification did not execute that stage and therefore does not claim it as a
new P2 pass.

The two statements are not contradictory: the historical canonical A5-P2
freeze remains `PASS`, while the separate public Installer v1.5.0
qualification is scoped to P0+P1. The earlier v1.4.3 adaptive qualification is
retained below as historical public evidence.

The following are not claimed by the public Installer v1.5.0 qualification:

- real GPU qualification of a separately created Fresh-ENV;
- a new P2 sustained-training qualification;
- a new multi-hour training-stability claim;
- complete support for every Nerfstudio method;
- Viewer support;
- Splatfacto support;
- general performance superiority over CUDA.
<!-- END NERFSTUDIO_EXPLAINER_AND_VALIDATION_STAGES -->

> [!IMPORTANT]
> This repository is **not** a standalone fork of Nerfstudio and does **not** claim support for every Nerfstudio model or feature.
>
> Qualified scope:
>
> ```text
> NERFACTO_TRAINING_CHAIN_NOT_FULL_NERFSTUDIO
> ```

<details>
<summary><strong>ROCm / gfx1201 port note</strong></summary>

- AMD ROCm / RDNA4 integration and qualification project for the **Nerfacto training chain from Nerfstudio**.
- Target: AMD Radeon AI PRO R9700 / RDNA4 / `gfx1201`.
- Stack: ROCm 7.2, Python 3.12, PyTorch `2.13.0+rocm7.2`.
- Runtime path: `tiny-rdna4-nn`, PyTorch ROCm, and a ROCm-compatible `nerfacc` extension.
- Qualified scope: Nerfacto training, checkpointing, reload, resume, and validation — **not full Nerfstudio support**.
- Canonical A5-P2 qualification: `PASS` with 576 frozen training steps.
- Public Installer v1.5.0 qualification: managed runtime plus real 28/28-gate P0+P1 validation.
- Viewer, Splatfacto, multi-GPU, Fresh-ENV GPU execution, and CUDA performance superiority are not claimed.
- License and attribution: see [`LICENSE`](./LICENSE) and [`NOTICE.md`](./NOTICE.md). Third-party components retain their original licenses.

</details>

---

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

Public Installer v1.5.0 runtime anchors:

```text
Nerfstudio mlp.py
4939a5a6901d82d8e310d93e2a135ca57ccc1bd79be79a7f67e2740e730c44ad

tiny-rdna4-nn native extension
4a561cc605bb7a6353d0eca1f9effc5ac9fcdfa3a9cb605a8cf36e1ae25b1917

tinycudann/modules.py
6555845d9483f672feefeef3b7ca5a264737ffe0e43ead1bbdebb661d6a3663a

ROCm-compatible nerfacc native extension
d3beee150cfa3a9ad3038a3283ff0a46953c345634d8cb6109449c5e3d04d1e2
```

Historical canonical A5 runtime anchors:

```text
tiny-rdna4-nn native extension
883f89efdad7bb909a4a3899ab79b2defe9713fdb5c7cf22cf4882c626b3efc4

tinycudann/modules.py
b4df43b54f64fe2b31272a997aafd50137aecac411d59b05251acedcd5512d12
```

The historical A5 hashes remain part of the earlier frozen correctness chain;
they are not rewritten as v1.5.0 runtime identities.

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

## Public Installer v1.5.0

The v1.5.0 workflow separates runtime installation from the final real P0/P1
qualification. The installer operates below an explicit user-controlled
`--workdir`, never executes `sudo` or `apt`, and modifies only its managed
environment and project-owned runtime directories.

### Managed runtime installation

A complete managed installation with an explicit authorized nerfacc wheel uses:

```bash
scripts/setup_public_installer_v1_5.sh \
  --workdir "$HOME/amd-nerfstudio-workdir" \
  --rocm-path /opt/rocm \
  --arch gfx1201 \
  --validation quick \
  --prepare-env \
  --install-torch \
  --build-tiny \
  --install-nerfacc \
  --nerfacc-wheel /path/to/nerfacc-0.5.2-cp312-cp312-linux_x86_64.whl \
  --install-nerfstudio
```

The authorized nerfacc wheel must match SHA-256
`252ec63319461889319a3bc535c4076c3c84bfc1ff6ddb5d64e1bb8b18032e00`.
The installer can reuse an already qualified managed runtime, but an explicitly
selected external environment remains verify-only and is never repaired in
place.

### Real public P0/P1 qualification

After the runtime and the hash-locked dataset archive are available, set the
six explicit paths and run the fixed 28-gate protocol:

```bash
export NERFSTUDIO_RDNA4_PUBLIC_PYTHON=/path/to/venv/bin/python
export NERFSTUDIO_RDNA4_PUBLIC_NERFSTUDIO_WORKTREE=/path/to/nerfstudio
export NERFSTUDIO_RDNA4_PUBLIC_TCNN_RUNTIME=/path/to/tiny-rdna4-nn-runtime
export NERFSTUDIO_RDNA4_PUBLIC_DATASET_ARCHIVE=/path/to/quick-validation-dataset-v2.tar.gz
export NERFSTUDIO_RDNA4_PUBLIC_DATASET=/path/to/quick-validation-dataset-v2
export NERFSTUDIO_RDNA4_PUBLIC_OUTPUT_ROOT=/path/to/evidence
scripts/run_public_dev5_p0_p1_v1.sh
```

The quick-validation dataset archive must match SHA-256
`0a968da041884f1f815bc9176aef1a13dc72beb7531e25c5c98cf24db1db25ac`.
P2 is never launched automatically.

### Previous public workflows — Toolchain v1.4.3

The v1.4.3 adaptive existing-environment path and strict Fresh-ENV fallback
remain available as earlier public workflows. Older v1.2 and v1.3.2 headings
are retained in [`CHANGELOG.md`](CHANGELOG.md) and the detailed documentation.

### Recommended path — adaptive existing environment

The normal entry point can reuse an already compatible Python 3.12 virtual
environment, Conda environment, or explicitly selected interpreter without
mutating it:

```bash
scripts/setup_public_adaptive_env_v1.sh \
  --env /path/to/selected/environment \
  --resource-dir "$PWD/resource-cache-v1" \
  --install-root "$PWD/rdna4-nerfacto-env" \
  --quick
```

The environment is always selected explicitly. The installer does not search
the disk and does not silently fall back to system Python. Existing
environments use `ENV_ROOT/bin/python`; isolated fallback still requires an
explicit `--install-root`.

For an existing shared environment, unrelated extra packages are recorded as
advisories. Compatibility is decided by the qualified runtime identities,
fail-closed Viewer quarantine, unchanged package state, and the real P0+P1
quick validation.

`pip check` is advisory for an existing shared environment. `--repair` creates
an isolated replacement and never modifies the candidate environment in place.
If reuse is not possible, `auto` can delegate to the pinned Fresh-ENV installer.

See [Public adaptive environment v1](docs/PUBLIC_ADAPTIVE_ENV_V1.md).

### Strict fallback — fresh environment

The strict `reference-binary-fresh-env` path creates a new isolated Python 3.12
environment:

```bash
scripts/setup_public_fresh_env_v1.sh \
  --resource-dir "$PWD/resource-cache-v1" \
  --install-root "$PWD/fresh-env-v1" \
  --nerfacc-wheel /path/to/qualified-nerfacc.whl \
  --tcnn-runtime /path/to/qualified-tiny-rdna4-nn-runtime \
  --dataset /path/to/qualified-quick-dataset
```

The default mode asks before public network access. `--auto` approves only
pinned public operations; `--offline` forbids network access;
`--download-only` prepares the cache; and `--verify-resources` performs
non-mutating verification.

The installer downloads a scoped wheelhouse from PyPI plus the official
PyTorch ROCm 7.2 index, writes a SHA-256 lock for every fetched wheel, installs
only from the local wheelhouse, copies the exact qualified custom runtime
inputs, records pip provenance, and runs the normal-user P0+P1 quick
validation.

The Fresh-ENV profile remains strict and viewer-free. Its wheelhouse must
contain `opencv-python-headless==4.10.0.84`, must not contain
`opencv-python`, and must not contain `viser`, `pyliblzfse`, or `yourdfpy`.
A scoped import quarantine prevents Nerfstudio 1.1.5's eager Trainer-time
Viewer imports while `vis="tensorboard"`; attempted Viewer construction fails
closed.

The three custom resources currently have no download URL in the manifest.
They must be supplied through explicit local paths or an existing verified
cache. A fresh native rebuild is not claimed and is rejected fail-closed.

See [Public fresh environment v1](docs/PUBLIC_FRESH_ENV_V1.md).

### Validation paths

The normal-user quick validation executes P0 and the two-step P1
producer/reload smoke:

```bash
scripts/run_public_quick_validation_v1.sh
```

Before running it directly, set the five explicit
`NERFSTUDIO_RDNA4_PUBLIC_*` path variables described in the toolchain
documentation.

Temporary P1 checkpoints are hash-verified and removed after a successful run.
Use `--keep-checkpoints` only when the files are needed for debugging or
evidence retention.

P2 is never started by the quick wrapper. Its entry point is maintainer-only
and requires explicit `--maintainer-confirm` acknowledgement.

The public runners do not rewrite or supersede the canonical A5 freeze. They
produce a separate public requalification chain on each host.

See [Public Toolchain v1](docs/PUBLIC_TOOLCHAIN_V1.md).

<!-- BEGIN PUBLIC_INSTALLER_V1_5_0_PUBLIC_QUALIFICATION -->
### Current public qualification — v1.5.0

Public Installer v1.5.0 completed the fixed real P0/P1 qualification on AMD
Radeon AI PRO R9700 / `gfx1201` against implementation commit
`d73e72615bba404dc8e6b105674890f4abcb6311` and tree
`787e8e296e55cccc2b0df21106ecb951a9d9f343`.

The run deployed and verified the public synthetic dataset, attested the pinned
sources and native runtimes, exercised the real spawned DataLoader worker,
completed forward, backward, optimizer update, checkpoint write, exact
fresh-process reload, and fresh-process resume, and passed all 28 declared
gates.

```text
PUBLIC_INSTALLER_V1_5_0_P0_P1_QUALIFICATION: PASS
PUBLIC_RDNA4_DEV5_P0_P1: PASS
DEV5F_28_GATE_CONTRACT: PASS
PASSED_GATES: 28
FAILED_GATES: 0
REPLACEMENT_RUNS: NONE
P2_EXECUTION: NOT_RUN
```

Qualification run: `20260805T092040Z_43673`
Evidence SHA-256:
`9f104f5fac3b434852e0f31483cd2b421964654641b8454666c8f6911c53556c`

See the [final qualification report](docs/PUBLIC_INSTALLER_V1_5_DEV5F_QUALIFICATION.md)
and the [machine-readable qualification contract](config/public_installer_v1_5_dev5f_qualification.json).
<!-- END PUBLIC_INSTALLER_V1_5_0_PUBLIC_QUALIFICATION -->

<!-- BEGIN ADAPTIVE_INSTALLER_V1_4_3_PUBLIC_QUALIFICATION -->
### Previous public qualification — v1.4.3

Public Toolchain v1.4.3 completed a real adaptive-reuse qualification on the
AMD Radeon AI PRO R9700 against implementation commit
`8104a4c6cce4b45cc7fd92d50cd9a8b2699e8a0f` and tree
`9cd8f5e389899fca64b6c1e65d3c81b2ce825178`.

The explicitly selected environment was reused unchanged, `pip freeze`
remained identical, the runtime and provenance gates passed, and the short
public P0+P1 Nerfacto GPU validation passed.

```text
ADAPTIVE_INSTALLER_V1_4_3_P0_P1_QUALIFICATION_FREEZE: PASS
EXISTING_ENV_REUSED_AND_QUALIFIED: PASS
PUBLIC_RDNA4_QUICK_VALIDATION: PASS
P2_EXECUTION: NOT_RUN
FRESH_ENV_GPU_EXECUTION: NOT_RUN
```

`P2_EXECUTION: NOT_RUN` refers only to this later public adaptive-installer
qualification. The canonical frozen A5-P2 run remains `PASS`.

A real GPU execution of the separately created Fresh-ENV remains unqualified
and is not claimed.

See [the sanitized qualification evidence](evidence-public/ADAPTIVE_INSTALLER_V1_4_3_P0_P1_QUALIFICATION.md).
<!-- END ADAPTIVE_INSTALLER_V1_4_3_PUBLIC_QUALIFICATION -->

## Documentation

- [Public Installer v1.5](docs/PUBLIC_INSTALLER_V1_5.md)
- [Public Installer v1.5 Dev5 protocol](docs/PUBLIC_INSTALLER_V1_5_DEV5.md)
- [Public Installer v1.5 final P0/P1 qualification](docs/PUBLIC_INSTALLER_V1_5_DEV5F_QUALIFICATION.md)
- [Public Installer v1.5 machine-readable qualification](config/public_installer_v1_5_dev5f_qualification.json)
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
- [Public adaptive environment v1](docs/PUBLIC_ADAPTIVE_ENV_V1.md)
- [Adaptive installer v1.4.3 P0+P1 qualification](evidence-public/ADAPTIVE_INSTALLER_V1_4_3_P0_P1_QUALIFICATION.md)
- [Adaptive installer v1.4.3 evidence hashes](evidence-public/ADAPTIVE_INSTALLER_V1_4_3_P0_P1_EVIDENCE_SHA256SUMS.txt)
- [Resource-cache policy](resources/README.md)
- [Frozen A5 hashes](evidence-public/A5_FREEZE_SHA256SUMS.txt)

## Related projects

- [Painter3000/tiny-rdna4-nn](https://github.com/Painter3000/tiny-rdna4-nn)
- [Painter3000/amd-nvdiffrast-rocm72-gfx1201](https://github.com/Painter3000/amd-nvdiffrast-rocm72-gfx1201)
- [Painter3000/amd-gsplat-rocm72-gfx1201](https://github.com/Painter3000/amd-gsplat-rocm72-gfx1201)
- [nerfstudio-project/nerfstudio](https://github.com/nerfstudio-project/nerfstudio)
- [nerfstudio-project/nerfacc](https://github.com/nerfstudio-project/nerfacc)

## License and attribution

Original helper scripts and documentation added by this repository are covered by the repository license. Third-party projects retain their own licenses and are not relicensed here.

This is an independent community project and is not an official AMD, Nerfstudio, PyTorch, or NVIDIA repository.
