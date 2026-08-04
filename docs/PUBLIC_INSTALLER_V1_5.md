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

## Current implementation stage

The first v1.5 commit implements path resolution, environment selection, host
package diagnostics, ROCm development-stack diagnostics, root rejection, JSON
reporting, and self-tests. It does not yet create an environment, download
resources, compile `tiny-rdna4-nn`, or run P0/P1.
