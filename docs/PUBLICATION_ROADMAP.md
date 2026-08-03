# Public release roadmap

## Stage 0 — documentation bootstrap

- exact claim scope;
- validated environment;
- A5 design;
- frozen hashes;
- recovery history;
- explicit nonclaims.

## Stage 1 — public-tree conversion

- remove local default paths and user-directory searches;
- make repository root, Python interpreter, dataset, and evidence paths explicit;
- reject missing configuration fail-closed;
- remove host-specific evidence references;
- preserve canonical numeric contracts and hashes where applicable;
- add public conversion tests.

## Stage 2 — sanitized public evidence

- publish claim scope, gates, summaries, and hash manifests;
- exclude large private checkpoints and unnecessary host-specific raw logs;
- document which artifacts are public derivatives and which private originals remain hash-anchored.

## Stage 3 — reproducible setup

- install pinned upstream Nerfstudio;
- install or build the pinned ROCm-compatible `nerfacc` path;
- install the pinned `tiny-rdna4-nn` runtime surface;
- apply the Nerfstudio integration changes;
- run neutral import and runtime-origin checks.

## Stage 4 — public functional validation

- run from a neutral working directory;
- verify loaded module paths and hashes;
- execute the public P0/P1 quick path;
- delete successfully verified temporary checkpoints by default;
- require explicit maintainer acknowledgement for the extended P2 path;
- use a bounded public sustained test rather than silently reproducing the private qualification claim;
- keep performance explicitly separate.

## Stage 5 — fresh clone and release

- independent clean clone;
- recursive dependency materialization;
- clean build;
- public functional validation;
- public-tree audit;
- annotated tag and release only after every public gate passes.


## Public Toolchain v1 milestone

Implemented in the repository after the documentation bootstrap:

```text
PUBLIC_TOOLCHAIN_V1_STATIC_SELF_TESTS: PASS
PUBLIC_TOOLCHAIN_V1_PUBLIC_TREE_AUDIT: PASS
PUBLIC_TOOLCHAIN_V1_REFERENCE_RUNTIME_P0_P1_REPLAY: PASS
PUBLIC_TOOLCHAIN_V1_REFERENCE_BINARY_FRESH_ENV_INSTALLER: STATIC_PASS
PUBLIC_TOOLCHAIN_V1_FRESH_ENV_GPU_RUN: NOT_YET_CLAIMED
PUBLIC_TOOLCHAIN_V1_FRESH_NATIVE_BUILD: NOT_CLAIMED
```

The toolchain uses explicit absolute paths supplied by the operator and performs no home-directory discovery. Public Toolchain v1.2 provides a P0+P1 normal-user gate; the long P2 run is maintainer-only and requires an explicit confirmation flag after its 576-step design and expected runtime are displayed.

Public Toolchain v1.3 implements Stage 3 for a new Python environment using the exact qualified native binary inputs. It also introduces a locked external resource cache and a scoped Nerfacto configuration loader. A real v1.3 fresh-environment GPU run is still required before Stage 5 can be claimed. Fresh native source builds remain a separate future stage.
