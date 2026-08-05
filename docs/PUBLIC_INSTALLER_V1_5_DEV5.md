# Public installer v1.5 dev5

Dev5 is the first real GPU execution stage after the dev4e runtime closure.
It does not broaden the claim to full Nerfstudio support.

## Qualified scope

```text
NERFACTO_TRAINING_CHAIN_NOT_FULL_NERFSTUDIO
P0_PLUS_P1_ONLY
P2_NOT_RUN
```

Dev5 performs four operations:

1. verify and atomically deploy `quick-validation-dataset-v2`;
2. verify the clean pinned Nerfstudio source and the qualified native runtimes;
3. execute the existing public P0 preflight and real two-process P1 Nerfacto smoke;
4. freeze the reports, gate files, manifests and process logs into a deterministic evidence archive.

## Dataset contract

The bundle carries the CC0 synthetic six-view dataset as an external resource.
The archive is not committed to the public repository. The repository stores
only its contract:

```text
archive SHA256:
0a968da041884f1f815bc9176aef1a13dc72beb7531e25c5c98cf24db1db25ac

images:       6
resolution:   128 x 128
camera model: OPENCV
```

Every archive member and every extracted file is hash-verified. Links, device
members, path traversal, extra files and a wrong root directory fail closed.
An existing dataset is reused only when it satisfies the complete contract.
Replacement is atomic and rollback-safe.

The dataset proves installation and short training mechanics only. It does not
claim reconstruction quality, geometric accuracy or photorealism.

## Runtime anchors

```text
Nerfstudio commit:
50e0e3c70c775e89333256213363badbf074f29d

Nerfstudio tree:
9d5ff468eeff89b66995e9984acaa378c37dc07e

tiny-rdna4-nn native SHA256:
4a561cc605bb7a6353d0eca1f9effc5ac9fcdfa3a9cb605a8cf36e1ae25b1917

nerfacc native SHA256:
d3beee150cfa3a9ad3038a3283ff0a46953c345634d8cb6109449c5e3d04d1e2
```

Historical evidence that used an earlier qualified tiny native binary remains
historical and is not rewritten. Dev5 updates only the active public
requalification reference and the active P1 runner.

## Fixed 28-gate protocol

The gate count and possible outcome are fixed before execution. There are no
replacement runs inside the same dev5 series.

- 5 dataset identity and deployment gates;
- 4 repository, source and native identity gates;
- 4 quick-wrapper chain gates;
- 15 detailed P1 mechanics gates.

Outcomes:

```text
PASS
- all 28 gates pass

FAIL
- one or more gates fail
- blockers remain named in the report
- no automatic rerun or P2 transition

INFRASTRUCTURE_FAIL
- represented as the corresponding failed process, archive, source or runtime gate
- not reclassified as a correctness result
```

P2 is maintainer-only and is never launched by dev5.

## Evidence freeze

After the run, dev5 writes:

- the dataset deployment report and process log;
- the complete quick-validation process log;
- `final_aggregate.json`;
- `final_gate.txt`;
- `MANIFEST.json`;
- `DEV5_EVIDENCE_SHA256SUMS.txt`;
- a deterministic `.tar.gz` evidence archive;
- an archive-attestation JSON with SHA256 and source-manifest hash.

Checkpoint files are hash-verified and deleted after successful P1 by default.
`--keep-checkpoints` is an explicit diagnostic opt-in.

## Normal invocation

Set the six explicit paths and run:

```bash
export NERFSTUDIO_RDNA4_PUBLIC_PYTHON=/path/to/venv/bin/python
export NERFSTUDIO_RDNA4_PUBLIC_NERFSTUDIO_WORKTREE=/path/to/nerfstudio
export NERFSTUDIO_RDNA4_PUBLIC_TCNN_RUNTIME=/path/to/tiny-runtime
export NERFSTUDIO_RDNA4_PUBLIC_DATASET_ARCHIVE=/path/to/quick-validation-dataset-v2.tar.gz
export NERFSTUDIO_RDNA4_PUBLIC_DATASET=/path/to/deployed/quick-validation-dataset-v2
export NERFSTUDIO_RDNA4_PUBLIC_OUTPUT_ROOT=/path/to/evidence
scripts/run_public_dev5_p0_p1_v1.sh
```

## Nonclaims

A dev5 PASS does not establish:

- full Nerfstudio feature coverage;
- Viewer or export support;
- Splatfacto support;
- multi-GPU or distributed training;
- sustained P2 stability;
- reconstruction-quality performance;
- superiority over CUDA or NVIDIA hardware.
