# Public fresh environment v1

## Scope

Public Toolchain v1.3.2 provides a viewer-free fresh Python-environment installer for the qualified
Nerfacto training-chain scope:

```text
reference-binary-fresh-env
NERFACTO_TRAINING_CHAIN_NOT_FULL_NERFSTUDIO
```

The profile creates a new Python 3.12 virtual environment and a self-contained
installation tree while reusing the exact qualified native `nerfacc` and
`tiny-rdna4-nn` artifacts. It does not claim a fresh native rebuild of either
component.

A requested `fresh-native-build` profile is rejected fail-closed. That profile
requires a separately qualified source-build and provenance workflow.

## Installation tree

The installer creates this structure below `--install-root`:

```text
install-root/
├── venv/
├── worktrees/nerfstudio/
├── runtime/tiny-rdna4-nn/
├── data/quick-validation/
├── evidence/
├── provenance/
└── activate_rdna4_nerfacto.sh
```

The cached inputs remain separate below `--resource-dir`:

```text
resource-dir/
├── datasets/quick-validation/
├── runtime/tiny-rdna4-nn/
├── sources/nerfstudio/
├── wheels/custom/
├── wheelhouse/py312-rocm72/
└── locks/WHEELHOUSE_LOCK.json
```

Generated checkpoints and run evidence never belong in the resource cache.

## Required custom resources

The following resources currently have no public download URL in the manifest.
They must already be present in the resource cache or be supplied through an
explicit local path:

- qualified `nerfacc` 0.5.2 CPython 3.12 wheel;
- qualified `tiny-rdna4-nn` runtime directory;
- qualified six-image quick-validation dataset.

The manager verifies all three against the exact SHA-256 anchors in
`config/public_fresh_env_resources_v1.json`. `--auto` never substitutes an
unqualified artifact and never invents a URL for a missing custom resource.

Pinned Nerfstudio source may be cloned from its official repository after an
interactive confirmation, or without a prompt under `--auto`.

## First online preparation

```bash
RESOURCE_DIR="$PWD/resource-cache-v1"
INSTALL_ROOT="$PWD/fresh-env-v1"

scripts/setup_public_fresh_env_v1.sh \
  --resource-dir "$RESOURCE_DIR" \
  --install-root "$INSTALL_ROOT" \
  --nerfacc-wheel /path/to/nerfacc-0.5.2-cp312-cp312-linux_x86_64.whl \
  --tcnn-runtime /path/to/tiny-rdna4-nn-runtime \
  --dataset /path/to/quick-validation-dataset \
  --nerfstudio-source /path/to/pinned-nerfstudio-worktree
```

In interactive mode the script asks before the Python wheelhouse is downloaded.
The wheelhouse uses pinned top-level requirements, PyPI, and the official
PyTorch ROCm 7.2 index. If the selected Python 3.12 launcher has no pip module,
the installer creates a temporary bootstrap virtual environment for the download
step. Every downloaded file is then SHA-256 locked in
`WHEELHOUSE_LOCK.json`. Installation is performed from that local wheelhouse
with networking disabled for the pip install step.

Use `--auto` to approve only the pinned network operations without prompts:

```bash
scripts/setup_public_fresh_env_v1.sh \
  --auto \
  --resource-dir "$RESOURCE_DIR" \
  --install-root "$INSTALL_ROOT" \
  --nerfacc-wheel /path/to/qualified-nerfacc.whl \
  --tcnn-runtime /path/to/qualified-tiny-rdna4-nn-runtime \
  --dataset /path/to/qualified-quick-dataset
```

## Download-only and offline preparation

Prepare the complete cache without creating an environment:

```bash
scripts/setup_public_fresh_env_v1.sh \
  --download-only \
  --auto \
  --resource-dir "$RESOURCE_DIR" \
  --nerfacc-wheel /path/to/qualified-nerfacc.whl \
  --tcnn-runtime /path/to/qualified-tiny-rdna4-nn-runtime \
  --dataset /path/to/qualified-quick-dataset
```

Verify an existing cache without modifying it or using the network:

```bash
scripts/setup_public_fresh_env_v1.sh \
  --verify-resources \
  --resource-dir "$RESOURCE_DIR"
```

Create the environment strictly offline from a verified cache:

```bash
scripts/setup_public_fresh_env_v1.sh \
  --offline \
  --resource-dir "$RESOURCE_DIR" \
  --install-root "$INSTALL_ROOT"
```

`--offline` and `--auto` are mutually exclusive. Missing or modified resources
block the run and are listed explicitly.

## Existing invalid paths

The manager does not silently replace an invalid cache entry. Use
`--replace-invalid` only after reviewing the reported hash mismatch.

The installer similarly refuses an existing installation root. Use
`--force-recreate` only when replacing that complete installation tree is
intentional.

## Validation and evidence

Unless `--no-validate` is specified, setup finishes by running the normal-user
P0+P1 quick validation. P2 is not called.

The successful gate is:

```text
decision=FRESH_ENV_QUALIFIED
quick_validation=PASS
fresh_native_build=NOT_CLAIMED
p2_execution=NOT_RUN
blockers=NONE
PUBLIC_RDNA4_FRESH_ENV: PASS
```

The provenance directory records:

- `pip freeze --all`;
- `pip list --format=freeze`;
- `pip check`;
- `pip inspect --local`;
- the install report and final gate;
- a SHA-256 manifest of the provenance output.

A successful install requires `pip check` to pass. The Nerfstudio worktree is
used directly through the explicit runtime policy rather than installed as a
full metadata dependency set, so unrelated Viewer, Splatfacto, notebook,
telemetry, and exporter dependencies are not falsely required by this scoped
profile.

## Scoped Nerfacto configuration

The public runners use `tools/public_nerfacto_config_v1.py`, which reproduces
the pinned upstream Nerfacto configuration directly. This avoids importing
Nerfstudio's global method registry, which eagerly imports unrelated models and
would otherwise force Splatfacto/`gsplat` dependencies into a Nerfacto-only
environment.

Nerfstudio 1.1.5 imports `viser` and both Viewer classes while importing
`TrainerConfig`, even when TensorBoard is selected. The scoped builder therefore
installs fail-closed import stubs for those Viewer modules before importing the
upstream Trainer. The stubs are never instantiated under `vis="tensorboard"`; an
accidental Viewer request raises `VISER_VIEWER_DISABLED_BY_PUBLIC_P0_P1_CONTRACT`.

This changes dependency loading, not the qualified Trainer implementation, model,
DataManager, optimizer, scheduler, dataset, forward/backward path, checkpoint
format, or fresh-process reload mechanics.

## Claim boundary

Public Toolchain v1.3.2 does not yet claim:

- a fresh source build of the native `nerfacc` extension;
- a fresh source build of the `tiny-rdna4-nn` native extension;
- the full upstream Nerfstudio dependency set;
- Viewer, export, Splatfacto, notebook, telemetry, or multi-GPU qualification;
- cross-time identity of an online dependency resolution before a published
  wheelhouse lock exists;
- VMM fallback performance parity.


The wheelhouse is locked by SHA-256 after the first successful download. Verification rejects multiple Python distributions that install the same `cv2` package tree: `opencv-python-headless` is required and `opencv-python` is forbidden. The viewer-free contract also forbids `viser`, `pyliblzfse`, and `yourdfpy`; the installed-runtime probe confirms that none of those distributions is present and that the scoped config resolves to `vis="tensorboard"`.


## v1.4.2 adaptive environment note

The adaptive installer may reuse a compatible existing environment unchanged or
create a new isolated Fresh-ENV. `viser==1.0.0` is now the qualified math-only
dependency for `viser.transforms.SO3`; Viewer construction remains quarantined
fail-closed, and `pyliblzfse` / `yourdfpy` remain outside the scoped contract.
See `docs/PUBLIC_ADAPTIVE_ENV_V1.md`.
