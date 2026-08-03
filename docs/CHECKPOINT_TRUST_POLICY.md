# Checkpoint trust policy

The qualified checkpoints were generated locally by the pinned training process and were SHA-256-anchored in the evidence chain.

The pinned framework call site did not explicitly pass PyTorch's `weights_only` argument. For these trusted local checkpoints, the qualification wrapper used:

```text
TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
TORCH_FORCE_WEIGHTS_ONLY_LOAD unset
```

This policy is narrow:

- it applies only to trusted, locally generated, hash-verified qualification checkpoints;
- it is not a recommendation to load unknown or downloaded pickle-based checkpoints unsafely;
- the policy and its trust scope are part of the reproducible runtime record.

Public tooling must preserve this distinction and must not silently disable restricted loading for arbitrary external files.

## Public quick-validation retention

The P1 producer and reload checkpoints are temporary local smoke-test artifacts, not downloadable resources or pretrained models. Public Toolchain v1.2 records their paths, sizes, and SHA-256 values in the child reports, verifies those values again after the fresh-process reload chain, and deletes both files before writing the final P1 manifest by default.

Retention rules:

- successful P1, default: `DELETE_AFTER_VERIFICATION`;
- successful P1 with `--keep-checkpoints`: `KEEP`;
- failed P1: `RETAIN_ON_FAILURE` so recovery evidence is not destroyed.
