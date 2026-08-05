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

## dev5a nerfacc loader correction

The original dev5 identity child loaded ``nerfacc.csrc`` before PyTorch and
without explicitly exposing the PyTorch and ROCm shared-library directories.
That differed from the already-qualified dev4e load contract and could reject
the correct native before P0 was allowed to start.

Dev5a preserves all 28 gates and all existing hashes. It changes only the
nerfacc identity probe:

- import ``torch`` before ``nerfacc.csrc``;
- prepend ``torch/lib``, ``/opt/rocm/lib`` and ``/opt/rocm/lib64``;
- preserve existing ``PYTHONPATH`` and ``LD_LIBRARY_PATH`` values;
- emit the exact import error, runtime versions, architecture and native hash;
- require every probe check before P0 may proceed.

The failed dev5 evidence archive remains immutable historical evidence. A
dev5a execution creates a new run ID and evidence archive.

## dev5b managed-Python path correction

Dev5a corrected the native load order, but the orchestrator still resolved the
managed virtual-environment Python symlink to ``/usr/bin/python3.12``.  The
nerfacc child then derived ``torch/lib`` below ``/usr`` instead of below the
managed environment and failed before P0.

Dev5b preserves the absolute lexical path ``.../venv/bin/python`` while keeping
the real path as diagnostics only.  It derives the Python ABI directory from
the target interpreter, requires ``python3.12``, records both paths, and keeps
all 28 fail-closed gates unchanged.

## dev5c active `tinycudann/modules.py` anchor correction

The dev5b P0 child successfully imported the pinned runtime, the viewer-free
Nerfacto configuration, `nerfacc`, and the RDNA4 native extension. P0 was
blocked only because the active v1.5 reference still carried the earlier
`tinycudann/modules.py` hash while the pinned source and deployed runtime both
carry the current public Model-B wrapper.

Dev5c verifies the pinned tiny-rdna4-nn source commit and tree, requires the
source and deployed runtime copies of `tinycudann/modules.py` to be byte-exact,
and then aligns all active v1.5 consumers: the P0 reference, fresh-environment
resource contract, dev5 pre-gate, and P1 runtime identity. Historical frozen
A5 evidence and the top-level historical runtime table are intentionally not
rewritten.

## dev5d explicit PortableMLP configuration policy

The first real dev5c P1 producer reached Nerfacto model construction and then
failed before dataset iteration because the pinned Nerfstudio MLP helper emits
`FullyFusedMLP` for its standard layer widths.  The qualified AMD portable
runtime intentionally rejects that CUDA-specific backend and requires
`PortableMLP`.

Dev5d leaves both pinned source trees and the native runtime unchanged.  The
scoped public Nerfacto configuration installs a fail-closed class-level policy
that preserves every activation, width, layer-count, and output setting while
rewriting only the TCNN network `otype` from the known upstream values
`FullyFusedMLP` or `CutlassMLP` to `PortableMLP`.  Unknown backend names are
rejected.  P0 records and verifies the policy before emitting its runtime
policy, and P1 independently checks the same policy before trainer setup.

## dev5e Pillow encoder-extents compatibility

The dev5d producer passed P0, constructed the PortableMLP Nerfacto model, and
entered the real training iteration. The first DataLoader image then exposed
an API mismatch between the pinned Nerfstudio image fast path and the active
Pillow runtime: Nerfstudio called `encoder.setimage(im.im)` while the active
encoder requires explicit tile extents.

Dev5e does not modify the pinned Nerfstudio worktree and does not downgrade or
replace Pillow. The scoped public configuration installs a process-local
`pil_to_numpy` compatibility function before the DataLoader worker is created.
It preserves Nerfstudio's raw-encoder path and supplies `(0, 0, width, height)`
to `setimage`. Both imported Nerfstudio aliases are patched together and a
real writable 2x2 RGB conversion must pass before P0 can emit its policy and
before P1 can call `trainer.setup()`.
