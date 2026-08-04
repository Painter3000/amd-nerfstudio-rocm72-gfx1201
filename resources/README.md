# Resource-cache policy

This repository directory contains only the public resource policy and
manifests. Binary resources are stored in an external directory selected with
`--resource-dir`; they are never committed to this repository.

Public Toolchain v1.3.2 defines the `reference-binary-fresh-env` cache contract in
`config/public_fresh_env_resources_v1.json`.

The external cache contains:

- pinned Nerfstudio source at an exact commit and Git tree;
- the exact qualified `nerfacc` CPython 3.12 wheel;
- the exact qualified `tiny-rdna4-nn` runtime directory;
- the exact six-image quick-validation dataset;
- a viewer-free Python wheelhouse for the scoped Nerfacto TensorBoard runtime;
- `WHEELHOUSE_LOCK.json`, containing size and SHA-256 for every fetched wheel.

The three qualified custom resources currently have no public download URL in
the manifest. They must be supplied through explicit local paths or copied from
an already prepared cache. `--auto` does not weaken this requirement.

Network behavior is fail-closed:

- interactive mode asks before cloning or downloading a missing public resource;
- `--auto` permits only manifest-pinned public operations;
- `--offline` forbids network access;
- `--verify-resources` performs a non-mutating cache verification;
- online wheel downloads go to a temporary directory and are moved into place
  only after the pip download command succeeds;
- every completed wheelhouse is immediately SHA-256 locked;
- wheelhouses containing `opencv-python`, `viser`, `pyliblzfse`, or `yourdfpy` are rejected;
- installation itself reads only from the locked local wheelhouse.

Generated P1 checkpoints do **not** belong in the cache. They are produced
inside the evidence run, used for fresh-process reload verification, and deleted
by default after successful verification. Use `--keep-checkpoints` only for
explicit debugging or evidence retention.


## v1.4.3 adaptive environment note

The adaptive installer may reuse a compatible existing environment unchanged or
create a new isolated Fresh-ENV. `viser==1.0.0` is now the qualified math-only
dependency for `viser.transforms.SO3`; Viewer construction remains quarantined
fail-closed, and `pyliblzfse` / `yourdfpy` remain outside the scoped contract.
See `docs/PUBLIC_ADAPTIVE_ENV_V1.md`.
