# Public installer v1.5

The v1.5 installer is designed for a user-controlled installation root and does
not request administrator privileges.

## Path policy

```bash
python3.12 ./amd_nerfstudio_setup.py \
  --workdir "$HOME/nerfstudio" \
  --rocm-path /opt/rocm \
  --arch gfx1201 \
  --validation quick
```

When `--workdir` is omitted, a standalone installer uses its own directory. If
the script is executed from the project repository, the parent directory is the
default installation root.

Without `--env`, only `$WORKDIR/venv` is inspected. Other virtual-environment
names and unrelated directories are not scanned. An explicitly selected
external environment is verify-only and is not modified by the installer.

Managed layout:

```text
$WORKDIR/
├── amd-nerfstudio-rocm72-gfx1201/
├── venv/
├── sources/
│   ├── nerfstudio/
│   └── tiny-rdna4-nn/
├── build/tiny-rdna4-nn/
├── runtime/tiny-rdna4-nn/
├── datasets/quick-validation-dataset-v2/
├── cache/
├── reports/
└── logs/
```

## Privilege and host-package policy

The installer never invokes `sudo`, `apt`, or another operating-system package
manager. Missing host prerequisites produce a clear command suggestion and a
blocked exit. The environment is not created before the host preflight passes.

The preflight checks capabilities supplied by:

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

## Managed environment

`--prepare-env` creates or refreshes `$WORKDIR/venv` and installs the pinned
Python build base. Shell activation is not required; every operation uses
`$WORKDIR/venv/bin/python` explicitly.

```bash
python3.12 ./amd_nerfstudio_setup.py \
  --workdir "$HOME/nerfstudio" \
  --rocm-path /opt/rocm \
  --arch gfx1201 \
  --prepare-env
```

The environment receives `.amd-nerfstudio-managed-v1.json`. An incomplete
environment created by the current invocation is removed on failure. Existing
unmarked environments are not repaired or deleted.

## Qualified ROCm PyTorch stage

`--install-torch` installs or verifies:

```text
torch==2.13.0+rocm7.2
torchvision==0.28.0+rocm7.2
index=https://download.pytorch.org/whl/rocm7.2
```

Qualification requires Python 3.12, HIP `7.2.53211`, a visible GPU reporting
`gfx1201`, a real GPU matrix multiplication, and a clean `pip check`. When the
exact stack is already present, it is reused rather than downloaded again.

## tiny-rdna4-nn source build

`--build-tiny` implies environment preparation and ROCm PyTorch qualification.
It then:

1. clones `Painter3000/tiny-rdna4-nn` recursively at tag
   `phase4a2-model-b-public-gfx1201-pass`;
2. verifies exact commit
   `b98bdcc6b2878f6cb6c10a2141e50867cec6d96a`;
3. verifies the clean source and recursive submodule state;
4. compiles only the tiny-rdna4-nn Python extension with the qualified
   Phase-4A HIPCC compatibility shim;
5. assembles `$WORKDIR/runtime/tiny-rdna4-nn`;
6. verifies `_120_C.cpython-312-x86_64-linux-gnu.so` with `roc-obj-ls`, an environment-aware `ldd`,
   local SHA256 recording, Python import, and a real `RocWMMAWidth64MLP`
   forward/backward GPU smoke test.

```bash
python3.12 ./amd_nerfstudio_setup.py \
  --workdir "$HOME/nerfstudio" \
  --rocm-path /opt/rocm \
  --arch gfx1201 \
  --validation quick \
  --install-torch \
  --build-tiny \
  --max-jobs 8
```

A pre-existing source checkout, build tree, or runtime that does not satisfy the
recorded contract causes a blocked exit. The installer does not delete or repair
foreign paths. It removes only incomplete staging data created by its current
invocation.

## Current implementation stage

The dev3 stage implements path and host preflight, managed environment creation,
pinned Python build tooling, qualified ROCm PyTorch installation/reuse, locked
recursive tiny-rdna4-nn acquisition, local gfx1201 compilation, runtime
assembly, and native attestation. Nerfacc installation, Nerfstudio installation,
dataset deployment, and P0/P1 remain later v1.5 stages.

### Environment-aware native dependency audit

A bare `ldd` invocation is retained as diagnostic evidence, but it is not the
compatibility gate for a Python extension linked against PyTorch. Libraries
such as `libc10.so`, `libtorch.so`, `libtorch_cpu.so`, and
`libtorch_python.so` live below the selected environment's `torch/lib`
directory and are not system libraries. The gating audit therefore executes
`ldd` with `LD_LIBRARY_PATH` composed from:

1. the selected environment's `torch/lib`,
2. the requested ROCm `lib` and `lib64` directories, and
3. any pre-existing loader path.

The real GPU forward/backward import smoke remains mandatory. A bare `ldd`
showing environment-local Torch libraries as `not found` is recorded but is
not treated as a build failure when the environment-aware audit and runtime
smoke both pass.
