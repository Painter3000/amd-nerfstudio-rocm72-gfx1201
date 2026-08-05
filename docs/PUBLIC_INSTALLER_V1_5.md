# Public installer v1.5

The v1.5 installer operates entirely below a user-controlled installation root.
It does not invoke `sudo`, `apt`, or another operating-system package manager.

## Managed layout

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

Without `--env`, only `$WORKDIR/venv` is considered. An explicitly selected
external environment is verify-only and is never repaired or modified.

## Implemented stages

### Host and environment preflight

The installer validates the Python 3.12 development environment, CMake, Ninja,
Git, required host headers, the selected ROCm root, and the `gfx1201` target
before creating or modifying the managed environment.

### Qualified ROCm PyTorch

The managed environment uses:

```text
torch==2.13.0+rocm7.2
torchvision==0.28.0+rocm7.2
HIP==7.2.53211
architecture==gfx1201
```

Qualification includes a real GPU matrix multiplication. An already qualified
stack is reused.

### tiny-rdna4-nn

`--build-tiny` acquires the recursive public checkout at:

```text
tag:    phase4a2-model-b-public-gfx1201-pass
commit: b98bdcc6b2878f6cb6c10a2141e50867cec6d96a
```

The local build is accepted only after source and submodule checks, an
environment-aware dynamic-link audit, `gfx1201` code-object inspection, Python
origin validation, and a real `RocWMMAWidth64MLP` forward/backward GPU smoke.

### Authorized nerfacc runtime

The authorized wheel is:

```text
nerfacc-0.5.2-cp312-cp312-linux_x86_64.whl
wheel SHA256: 252ec63319461889319a3bc535c4076c3c84bfc1ff6ddb5d64e1bb8b18032e00
csrc.so SHA256: d3beee150cfa3a9ad3038a3283ff0a46953c345634d8cb6109449c5e3d04d1e2
```

Its documented source provenance is:

```text
repository: https://github.com/nerfstudio-project/nerfacc.git
tag:        v0.5.2
commit:     d84cdf3afd7dcfc42150e0f0506db58a5ce62812
tree:       f24a2f9902143b75ecb8472199b07dd0e92679e8
license:    MIT
```

Until a public release URL is recorded, the installer accepts only the exact
wheel already present in `$WORKDIR/cache/wheels/custom` or an explicit local
`--nerfacc-wheel` path. A filename match is insufficient; the SHA256 must match.

The associated Python stack is pinned to:

```text
rich==14.3.4
markdown-it-py==4.2.0
mdurl==0.1.2
Pygments==2.20.0
```

### Locked Nerfstudio source runtime

`--install-nerfstudio` acquires:

```text
repository: https://github.com/nerfstudio-project/nerfstudio.git
commit:     50e0e3c70c775e89333256213363badbf074f29d
tree:       9d5ff468eeff89b66995e9984acaa378c37dc07e
```

The stage installs the repository's scoped Nerfacto P0/P1 requirement set, not
the complete optional Nerfstudio feature surface. It then registers the locked
Nerfstudio source and the qualified tiny-rdna4-nn runtime through a managed
`.pth` file in the selected environment.

`viser==1.0.0` is downloaded as a wheel, verified against SHA256
`3be881a60f0295efd8a93df97646bbc04d070ccf8d16d8faf284eb3b70eda6eb`,
and installed with `--no-deps`. Only `viser.transforms` belongs to the qualified
surface. Viewer construction remains fail-closed.

The scoped runtime requires `opencv-python-headless==4.10.0.84` and rejects the
GUI `opencv-python` distribution.

## dev4 invocation

For an installation where the earlier stages are already qualified:

```bash
python3.12 ./amd_nerfstudio_setup.py \
  --workdir "$HOME/nerfstudio" \
  --rocm-path /opt/rocm \
  --arch gfx1201 \
  --validation quick \
  --install-nerfacc \
  --install-nerfstudio
```

For a new managed installation with an explicit authorized nerfacc wheel:

```bash
python3.12 ./amd_nerfstudio_setup.py \
  --workdir "$HOME/nerfstudio" \
  --rocm-path /opt/rocm \
  --arch gfx1201 \
  --validation quick \
  --install-torch \
  --build-tiny \
  --install-nerfacc \
  --nerfacc-wheel /path/to/nerfacc-0.5.2-cp312-cp312-linux_x86_64.whl \
  --install-nerfstudio
```

## Scope boundary

Dev4 does not claim the full Nerfstudio dependency set, viewer-server support,
Open3D, COLMAP, data-processing utilities, or P0/P1 execution. Dataset deployment
and the actual P0/P1 gates are later v1.5 stages.
