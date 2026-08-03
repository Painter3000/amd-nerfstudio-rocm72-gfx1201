# Public quick-validation reference-runtime replay

**Date:** 2026-08-03
**Hardware:** AMD Radeon AI PRO R9700 / `gfx1201`
**Runtime:** PyTorch `2.13.0+rocm7.2`, HIP `7.2.53211`
**Scope:** neutral-directory public-repository replay using the already qualified reference runtime

## Result

```text
PUBLIC_A5_P0_REFERENCE_RUNTIME_REPLAY: PASS
PUBLIC_A5_P1_REFERENCE_RUNTIME_REPLAY: PASS
PUBLIC_QUICK_VALIDATION_REFERENCE_RUNTIME_REPLAY: PASS
PUBLIC_FRESH_ENVIRONMENT_BUILD: NOT_RUN
PUBLIC_A5_P2_EXTENDED_MAINTAINER_RUN: NOT_RUN
```

Run identifiers:

```text
P0: 20260803T123239Z_70540
P1: 20260803T123501Z_70683
```

P1 executed exactly two real training iterations in distinct fresh processes:

```text
producer step: 0
reload/resume step: 1
producer process duration: 16.495 seconds
reload process duration:   16.684 seconds
```

Observed qualification properties:

- real six-image Nerfstudio dataset and `ParallelDataManager`;
- 1,024 rays per batch;
- finite positive forward losses;
- finite non-zero gradients;
- real optimizer parameter changes;
- producer checkpoint write and SHA-256;
- exact fresh-process pipeline, optimizer, scheduler, and scaler reload;
- one real resumed training step;
- `gfx1201`, HIP, PyTorch, `tinycudann`, and `nerfacc` runtime anchors exact;
- zero GPU OOMs and zero allocation retries.

Peak memory observed in either child process was approximately 506 MB allocated and 659 MB reserved.

The two checkpoints were locally generated smoke-test artifacts, not pretrained or downloadable model resources. This replay predates the v1.2 default cleanup policy; v1.2 deletes equivalent verified checkpoints unless `--keep-checkpoints` is supplied.
