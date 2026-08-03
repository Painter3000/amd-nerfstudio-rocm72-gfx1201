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
- execute the public P0/P1 path;
- use a bounded public sustained test rather than silently reproducing the private qualification claim;
- keep performance explicitly separate.

## Stage 5 — fresh clone and release

- independent clean clone;
- recursive dependency materialization;
- clean build;
- public functional validation;
- public-tree audit;
- annotated tag and release only after every public gate passes.
