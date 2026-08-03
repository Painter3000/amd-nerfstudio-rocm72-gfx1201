# Recovery history

Blocked attempts are retained because they document defects in the qualification machinery and prove that the final PASS was not produced by rewriting prior evidence.

## A5-P2 v1 — DataLoader topology failure

The first P2 runner tried obsolete configuration names. The pinned Nerfstudio source actually used:

```text
dataloader_num_workers
prefetch_factor
```

The intended worker reduction was therefore not applied and the default of four workers remained active. With a six-image dataset, the fourth worker received an empty slice. `variable_res_collate()` then failed at `batch[0]`.

The corrected P2 runner:

- set `dataloader_num_workers=1`;
- set `prefetch_factor=2`;
- attested the `num_workers` value on the created runtime DataLoader;
- ran as a new evidence generation without modifying the failed v1 evidence.

## A5 freeze v1 — native P0 schema mismatch

The first freeze adapter expected a top-level `passed` field in the P0 JSON. The native P0-v3 artifact intentionally expressed success through:

- exact schema and classification;
- `decision=PROCEED_TO_A5_P1`;
- an empty blocker list;
- read-only audit semantics;
- pinned file hashes.

All five P0 hashes were already exact. Freeze v2 corrected the adapter to evaluate the native P0-v3 contract instead of inventing a uniform field.

## Public A5-P0 v1 — virtual-environment launcher was symlink-resolved

The first public P0 replay received the correct explicit launcher:

```text
/absolute/path/to/venv/bin/python
```

but converted every input path with `Path.resolve()`. Because the venv launcher is a symlink, the executed path became `/usr/bin/python3.12`; Python then lost the virtual-environment prefix and the child probe correctly blocked with `ModuleNotFoundError: torch`.

Public Toolchain v1.1 keeps the Python launcher absolute while preserving its final symlink. The same correction is applied to P0, P1, and P2. A regression test now supplies a synthetic `venv/bin/python` symlink and verifies that the report retains that launcher path. The blocked run remains recovery evidence and must not be reused for P1.

## General rule

A fail-closed gate can expose a defect in the adapter or orchestrator rather than in the object being tested. The blocked run must remain intact, the measurement bug must be versioned, and the corrected attempt must run as a new evidence generation.

## Public P1 checkpoint classification clarification

The two approximately 176 MB files produced by the public P1 replay were initially discussed as possible external resources. Inspection of their run paths and P1 reports confirmed that they were created locally by the producer and fresh reload processes at steps 0 and 1. They are not official Nerfstudio downloads or pretrained model assets.

Public Toolchain v1.2 therefore verifies and deletes both successful smoke-test checkpoints by default. Failed runs retain available checkpoints for recovery, and `--keep-checkpoints` is the explicit opt-in for successful runs.

## 2026-08-03 — Fresh-ENV wheelhouse dependency correction

The first real v1.3 resource download produced a hash-clean 113-wheel cache but exposed a package-contract defect: `viser==1.0.0` pulled `opencv-python` while the scoped Nerfacto seed also required `opencv-python-headless==4.10.0.84`. Both distributions provide the same `cv2` package tree. Installation was stopped before creating the Fresh-ENV.

Recovery: pin Viser 0.2.7, matching the pinned Nerfstudio 1.1.5 release, and reject any wheelhouse containing both OpenCV distribution providers. The previous external cache remains diagnostic evidence and is invalidated automatically by the changed requirements hash.
