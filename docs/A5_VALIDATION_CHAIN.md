# A5 validation chain

## A5-P0 — runtime adjudication

P0 pinned the Nerfstudio source, Git tree, runtime origins, native extension hashes, environment policy, and nonclaims without running training.

Decision:

```text
PROCEED_TO_A5_P1
```

## A5-P1 — real mechanics

P1 exercised the real Nerfstudio path:

- six-image dataset;
- `ParallelDataManager`;
- 1024-ray batch;
- Nerfacto forward and loss;
- backward with finite, non-zero gradients;
- optimizer parameter update;
- checkpoint write and SHA-256;
- fresh-process checkpoint load;
- exact loaded pipeline, optimizer, scheduler, and scaler state;
- one real resumed training step.

P1 proved checkpoint integrity and resume mechanics. It did not claim long-run stability or multi-step trajectory equivalence.

## A5-P2 — sustained and trajectory qualification

P2 executed 576 training steps in total:

```text
A_continuous:             192
C_continuous_reference:   192
B_split:                   96
B_resume:                  96
Total:                    576
```

A and C were uninterrupted reference processes with the same seed and configuration. B was split at step 96, checkpointed, loaded in a fresh process, data-stream-aligned by qualification replay, and resumed to step 192.

The test separated:

1. exact loaded state integrity;
2. tolerant multi-step floating-point trajectory behavior.

Tolerance limits were fixed before execution and separated by FP16/BF16, FP32, FP64, and scalar trajectory classes. The allowed resume envelope was the maximum of fixed numeric floors and four times the observed A-vs-C reference variation.

Hard stability gates included:

- all scheduled steps completed;
- finite losses and metrics;
- finite, non-zero gradients;
- finite parameters;
- no OOMs;
- no allocation retries;
- no positive GPU allocated/reserved or host RSS ramp in the evaluated post-warm-up window;
- complete checkpoint chain and hashes;
- exact split fresh reload;
- aligned qualification data replay;
- resume trajectory within the predefined reference envelope.

## Final freeze

```text
Freeze ID:  20260803T102615Z_65645
Decision:   A5_FROZEN_PROCEED_TO_SEPARATE_VMM_BENCHMARK
Result:     PASS
```

The VMM benchmark remains a separate performance experiment and is not part of the correctness freeze.
