# Changelog

## Unreleased

### Fixed

- Public Toolchain v1.1 preserves virtual-environment Python launcher paths instead of resolving `venv/bin/python` to the system interpreter;
- the fix applies consistently to P0, P1, and P2 and includes a symlink regression test.

### Added

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

- one-command dependency installer;
- sanitized full public evidence bundle from an independent host;
- neutral fresh-clone GPU validation of Public Toolchain v1;
- VMM performance benchmark.
