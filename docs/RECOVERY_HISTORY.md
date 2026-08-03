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

## General rule

A fail-closed gate can expose a defect in the adapter or orchestrator rather than in the object being tested. The blocked run must remain intact, the measurement bug must be versioned, and the corrected attempt must run as a new evidence generation.
