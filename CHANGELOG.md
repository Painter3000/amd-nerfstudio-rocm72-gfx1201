# Changelog

## Unreleased

### Fixed

- corrected the Fresh-ENV Viser pin from 1.0.0 to the pinned Nerfstudio 1.1.5 dependency, Viser 0.2.7;
- reject wheelhouses containing both `opencv-python` and `opencv-python-headless`, because both provide the same `cv2` package tree.

### Fixed

- Public Toolchain v1.1 preserves virtual-environment Python launcher paths instead of resolving `venv/bin/python` to the system interpreter;
- the fix applies consistently to P0, P1, and P2 and includes a symlink regression test.

### Added

- Public Toolchain v1.3 `reference-binary-fresh-env` installer with interactive, `--auto`, `--offline`, `--download-only`, and non-mutating `--verify-resources` modes;
- external resource manifest and exact SHA-256 verification for the qualified `nerfacc` wheel, `tiny-rdna4-nn` runtime, and quick-validation dataset;
- first-fetch Python wheelhouse locking followed by network-free installation from the local cache;
- scoped Nerfacto configuration construction that avoids eager imports of unrelated Nerfstudio models and their dependency stacks;
- fresh-environment pip provenance, activation policy, and automatic P0+P1 quick validation;
- Public Toolchain v1.2 one-command P0+P1 quick validation;
- default hash-verified deletion of the two locally generated P1 checkpoints, plus `--keep-checkpoints`;
- explicit P2 maintainer confirmation in both the shell and Python entry points;
- sanitized reference-runtime replay summary and resource-cache policy;
- Public Toolchain v1 with path-independent P0/P1/P2 requalification runners;
- explicit `gfx1201` / ROCm 7.2 reference manifest;
- fail-closed public-tree audit and public requalification freeze;
- static self-test suite for all public tools;
- public repository bootstrap;
- exact Nerfacto training-chain scope and nonclaims;
- A5-P0/P1/P2 qualification summary;
- canonical A5 freeze identifiers and hashes;
- DataLoader and freeze-adapter recovery history;
- checkpoint trust policy;
- staged public release roadmap.

### Not yet included

- qualified fresh native builds of `nerfacc` and `tiny-rdna4-nn`;
- successful real GPU execution of the v1.3 fresh-environment installer;
- sanitized full public evidence bundle from an independent host;
- VMM performance benchmark.
