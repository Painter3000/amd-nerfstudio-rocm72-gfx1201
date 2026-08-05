# Public Installer v1.5 — Final P0/P1 Qualification

## Qualification status

**Result:** `PASS`

**Classification:** `PUBLIC_DEV5_DATASET_DEPLOYMENT_PLUS_REAL_NERFACTO_P0_P1`

**Decision:** `DEV5_P0_P1_QUALIFIED`

This report freezes the first fully qualified public P0/P1 state of the
AMD Nerfstudio installer for Radeon AI PRO R9700 / RDNA4 (`gfx1201`) with
ROCm 7.2.

## Qualified repository state

| Field | Value |
|---|---|
| Branch | `feature/installer-v1.5.0` |
| Qualified implementation commit | `d73e72615bba404dc8e6b105674890f4abcb6311` |
| Qualified implementation tree | `787e8e296e55cccc2b0df21106ecb951a9d9f343` |
| Qualification patch level | `F` |
| Remote push during qualification | `NOT_RUN` |

The freeze tag is created by the accompanying fail-closed freeze script after
this report and its machine-readable contract are committed.

## Qualified platform

| Component | Qualified identity |
|---|---|
| GPU architecture | `gfx1201` |
| GPU | AMD Radeon AI PRO R9700 class, runtime name `AMD Radeon Graphics` |
| Python | `3.12.3` |
| PyTorch | `2.13.0+rocm7.2` |
| HIP runtime | `7.2.53211` |
| Nerfstudio commit | `50e0e3c70c775e89333256213363badbf074f29d` |
| Nerfstudio tree | `9d5ff468eeff89b66995e9984acaa378c37dc07e` |
| tiny-rdna4-nn commit | `b98bdcc6b2878f6cb6c10a2141e50867cec6d96a` |
| tiny-rdna4-nn tree | `a8ffaaa3f509400c40f6de58e8a74fb047f8e16e` |
| Pillow | `12.2.0` |
| DataLoader topology | one `spawn` worker |

## Runtime anchors

| Artifact | SHA-256 |
|---|---|
| active `tinycudann/modules.py` | `6555845d9483f672feefeef3b7ca5a264737ffe0e43ead1bbdebb661d6a3663a` |
| active tiny-rdna4-nn native module | `4a561cc605bb7a6353d0eca1f9effc5ac9fcdfa3a9cb605a8cf36e1ae25b1917` |
| active nerfacc native module | `d3beee150cfa3a9ad3038a3283ff0a46953c345634d8cb6109449c5e3d04d1e2` |
| quick-validation dataset archive | `0a968da041884f1f815bc9176aef1a13dc72beb7531e25c5c98cf24db1db25ac` |

## Qualification evidence

| Field | Value |
|---|---|
| Run ID | `20260805T092040Z_43673` |
| Evidence run directory | `evidence/dev5f/public_dev5_p0_p1_v1/20260805T092040Z_43673` |
| Evidence archive | `evidence/dev5f/public_dev5_p0_p1_archives/20260805T092040Z_43673.tar.gz` |
| Evidence archive SHA-256 | `9f104f5fac3b434852e0f31483cd2b421964654641b8454666c8f6911c53556c` |
| Declared gates | `28` |
| Observed gates | `28` |
| Passed gates | `28` |
| Failed gates | `0` |
| Blockers | `NONE` |
| Replacement runs | `NONE` |
| P2 execution | `NOT_RUN` |
| P2 policy | `MAINTAINER_ONLY` |

## Real mechanics demonstrated

The qualified run demonstrated the following real, non-fixture mechanics:

1. The public synthetic Nerfstudio dataset was deployed and validated.
2. P0 verified pinned sources, runtime identities, `gfx1201`, the viewer-free
   TensorBoard policy, and single-origin tiny-rdna4-nn loading.
3. Nerfacto used `tcnn` with the AMD-qualified `PortableMLP` configuration.
4. Pillow 12.2.0 image conversion used the scoped encoder-extents compatibility
   call `encoder.setimage(im.im, extents)`.
5. The compatibility hook was installed inside the real `spawn` DataLoader
   worker through
   `public_nerfacto_config_v1.public_spawn_worker_init`.
6. The real DataLoader produced a finite GPU batch of 1,024 rays.
7. Nerfacto forward produced a finite loss of
   `0.09959939122200012`.
8. Backward produced finite, nonzero gradients.
9. The optimizer changed real model parameters.
10. VRAM/VMM telemetry completed without an out-of-memory event.
11. A checkpoint was written and hashed.
12. A fresh process performed an exact checkpoint reload.
13. A fresh process resumed training for the next step.
14. Checkpoint retention policy passed.
15. No mixed tiny-cuda-nn/tiny-rdna4-nn runtime origins were observed.

## Captured batch summary

| Tensor | Shape | Device | Dtype |
|---|---:|---|---|
| training image samples | `[1024, 3]` | `cuda:0` | `torch.float32` |
| sampled indices | `[1024, 3]` | `cuda:0` | `torch.int64` |
| ray origins | `[1024, 3]` | `cuda:0` | `torch.float32` |
| ray directions | `[1024, 3]` | `cuda:0` | `torch.float32` |
| ray pixel area | `[1024, 1]` | `cuda:0` | `torch.float32` |
| camera indices | `[1024, 1]` | `cuda:0` | `torch.int64` |

All floating-point tensors reported as finite.

## Scoped compatibility policies

### PortableMLP policy

Known Nerfstudio TCNN MLP backend names are rewritten to the qualified
`PortableMLP` backend. Unknown backend names fail closed. Nerfstudio source and
the tiny-rdna4-nn native runtime remain unchanged.

### Pillow encoder-extents policy

Only the imported Nerfstudio `pil_to_numpy` aliases are replaced per process.
The Pillow distribution and pinned Nerfstudio source remain unchanged.

### Spawn-worker policy

The pinned `ParallelDataManager` continues to use one real multiprocessing
worker with start method `spawn`. A top-level, picklable `worker_init_fn`
installs the Pillow compatibility policy inside that worker. Existing foreign
worker hooks fail closed.

## Upstream and environment modification status

| Component | Modified by qualification patches |
|---|---|
| Pinned Nerfstudio source tree | `NO` |
| Pillow distribution | `NO` |
| tiny-rdna4-nn native runtime | `NO` |
| nerfacc native runtime | `NO` |
| PyTorch/ROCm installation | `NO` |

## Nonclaims

This qualification proves the public P0/P1 mechanics for the pinned software
and hardware contract. It does **not** claim:

- production image quality;
- geometric reconstruction quality;
- long-duration stability;
- sustained throughput;
- multi-GPU behavior;
- support for GPUs other than the pinned `gfx1201` target;
- automatic P2 qualification.

P2 remains a separate maintainer-only phase and was not executed.

## Final gate

```text
PUBLIC_RDNA4_DEV5_P0_P1: PASS
DEV5F_28_GATE_CONTRACT: PASS
P2_EXECUTION: NOT_RUN
REMOTE_PUSH: NOT_RUN
```
