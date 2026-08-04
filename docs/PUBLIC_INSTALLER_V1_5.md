# Public installer v1.5

The v1.5 installer is designed for a user-controlled installation root and does
not request administrator privileges.

## Path policy

The installer accepts:

```bash
python3.12 ./amd_nerfstudio_setup.py \
  --workdir "$HOME/nerfstudio" \
  --env "$HOME/nerfstudio/venv" \
  --rocm-path /opt/rocm \
  --arch gfx1201 \
  --validation quick
```

When `--workdir` is omitted, a standalone installer uses its own directory. If
the script is executed from the project repository, the parent directory is the
default installation root.

Without `--env`, only `$WORKDIR/venv` is inspected. Other virtual-environment
names and unrelated directories are not scanned.

## Privilege policy

The installer never invokes `sudo`, `apt`, or another operating-system package
manager. Missing host prerequisites produce a clear command suggestion and a
blocked exit. The environment is not created before the host preflight passes.

## Host package contract

The preflight checks the capabilities provided by:

- `build-essential`
- `cmake`
- `ninja-build`
- `git`
- `pkg-config`
- `python3.12`
- `python3.12-venv`
- `python3.12-dev`
- `ca-certificates`
- `curl`
- `tar`
- `gzip`
- `unzip`

It also checks the selected ROCm root for `hipcc`, the ROCm Clang compiler,
`roc-obj-ls`, HIP headers, and `libamdhip64`.

## Managed environment preparation

The second v1.5 stage adds an explicit `--prepare-env` action. A passing host
preflight may create `$WORKDIR/venv` and install the pinned Python build base.
The installer does not require shell activation; every command uses
`$WORKDIR/venv/bin/python` directly.

```bash
python3.12 ./amd_nerfstudio_setup.py \
  --workdir "$HOME/nerfstudio" \
  --rocm-path /opt/rocm \
  --arch gfx1201 \
  --validation quick \
  --prepare-env
```

The managed environment receives a marker named
`.amd-nerfstudio-managed-v1.json`. An incomplete environment created by the
current invocation is removed on failure. Existing unmarked environments are
not repaired or deleted. An explicitly selected external environment is only
verified and is never modified by this stage.

The pinned build base is: `pip`, `setuptools`, `wheel`, `packaging`, `ninja`,
and `cmake`. The exact versions are recorded in
`config/public_installer_resources_v1_5.json`.

## Current implementation stage

The current v1.5 development stage implements path resolution, host and ROCm
preflight, managed environment creation, pinned Python build-base installation,
managed ownership markers, cleanup of incomplete newly-created environments,
JSON reporting, and self-tests. It does not yet install ROCm PyTorch, clone or
compile `tiny-rdna4-nn`, install Nerfstudio, or run P0/P1.
