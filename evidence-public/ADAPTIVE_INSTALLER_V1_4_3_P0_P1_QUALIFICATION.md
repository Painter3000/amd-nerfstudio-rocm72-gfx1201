# Adaptive installer v1.4.3 — public P0+P1 qualification

**Date:** 2026-08-04
**Freeze ID:** `20260804T073046Z_adaptive_installer_v1.4.3_p0_p1`
**Gate:** `ADAPTIVE_INSTALLER_V1_4_3_P0_P1_QUALIFICATION_FREEZE: PASS`
**Scope:** `NERFACTO_TRAINING_CHAIN_ADAPTIVE_REUSE_P0_PLUS_P1_ONLY`

## Frozen implementation

```text
Base commit:           32ee27816cb9cda82052c49e73e2e80fdfabe9c6
Implementation commit: 8104a4c6cce4b45cc7fd92d50cd9a8b2699e8a0f
Implementation tree:   9cd8f5e389899fca64b6c1e65d3c81b2ce825178
Installer SHA256:      f8e690553ff1ac942e74b784c9181fdf9d2666b6dc3e19ffd1b376e3b032a121
Freeze archive SHA256: 1ce04c4594abd9ffa781bb7d7d0d65eb3e6cbcf0d91262ff4fb2789aa975ce26
```

## Qualified execution

```text
EXPLICIT_EXISTING_ENV_SELECTION: PASS
EXISTING_ENV_REUSE: PASS
ENVIRONMENT_MUTATED: FALSE
PIP_FREEZE_UNCHANGED: TRUE
RUNTIME_AND_PROVENANCE: PASS
PUBLIC_P0_PREFLIGHT: PASS
PUBLIC_P1_REAL_MECHANICS: PASS
CHECKPOINT_POLICY: PASS
MANIFEST_CHAIN: PASS
P2_EXECUTION: NOT_RUN
LONG_DURATION_TRAINING: NOT_RUN
```

The installer was launched outside the selected environment and used the exact
user-supplied environment interpreter. No disk-wide environment search, silent
system-Python fallback, package repair in place, or P2 transition occurred.

## Validated runtime

| Component | Frozen value |
|---|---|
| Python | `3.12.3` |
| PyTorch | `2.13.0+rocm7.2` |
| HIP reported by PyTorch | `7.2.53211` |
| GPU architecture | `gfx1201` |
| PyTorch device name | `AMD Radeon Graphics` |
| Reported GPU memory | `29.86 GiB` |
| Reported compute units | `32` |
| Viewer policy | `TENSORBOARD_ONLY_VIEWER_IMPORT_QUARANTINE` |
| Viser use | `viser.transforms.SO3` math bridge only |

## Shared-environment advisories

- `SHARED_ENV_EXTRA_PRESENT:yourdfpy==0.0.60`

These observations were recorded but were not package-presence blockers for the
explicitly selected shared environment. Qualification still required exact
runtime identities, fail-closed Viewer construction, an unchanged `pip freeze`,
and the real P0+P1 execution.

## Evidence anchors

The full qualification freeze is distributed outside the Git tree. Its archive
and the installer bundle are identified by SHA-256 in
`ADAPTIVE_INSTALLER_V1_4_3_P0_P1_EVIDENCE_SHA256SUMS.txt`. The repository stores
only this sanitized summary and public hashes; raw host paths, binary artifacts,
checkpoints, and private logs are not committed.

## Claim boundary

This evidence qualifies the adaptive **existing-environment reuse** path and the
short real Nerfacto P0+P1 GPU validation on the pinned RDNA4 / `gfx1201` stack.
It does not qualify the separate Fresh-ENV creation path, P2 sustained training,
long-duration stability, Viewer operation, Splatfacto, full Nerfstudio support,
VMM performance parity, or performance superiority over CUDA/NVIDIA.
