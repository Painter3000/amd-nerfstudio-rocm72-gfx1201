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
