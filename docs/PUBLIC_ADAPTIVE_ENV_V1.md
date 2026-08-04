# Public adaptive environment v1

`setup_public_adaptive_env_v1.py` is the normal entry point for the scoped
RDNA4 / `gfx1201` Nerfacto environment. It does not replace the pinned Fresh-ENV
installer. It decides whether an existing environment can be reused unchanged
or whether a new isolated environment must be created by the Fresh-ENV path.

## Safety model

- The candidate environment is selected explicitly with `--env ENV_ROOT` or, for advanced use, `--python PYTHON`.
- No disk-wide environment search is performed and there is no implicit system-Python fallback.
- Existing virtual environments, Conda environments, and system Python installs are classified only after explicit selection.
- A compatible existing environment is reused without package mutation.
- `pip freeze --all` is captured before and after reuse; any unexpected change
  blocks qualification.
- `pip check` is recorded as advisory evidence and is not the scoped P0/P1
  compatibility gate for a shared environment.
- An incompatible environment is never repaired in place. `--repair` means a
  safe replacement in a new isolated install root.
- System Python is never modified implicitly.
- P2 is never launched.

## Viewer policy

Nerfstudio geometry requires `viser.transforms.SO3`. The scoped contract pins
`viser==1.0.0` as a math-only runtime dependency while viewer construction is
quarantined fail-closed. `pyliblzfse` and `yourdfpy` remain excluded. The claim
is therefore **viewer-runtime-disabled**, not “the viser distribution is absent”.

## Typical reuse

```bash
scripts/setup_public_adaptive_env_v1.sh \
  --env /absolute/path/to/venv \
  --resource-dir /absolute/path/to/resource-cache-v1 \
  --output-root /absolute/path/to/evidence \
  --quick
```

The three runtime inputs may also be supplied explicitly:

```bash
scripts/setup_public_adaptive_env_v1.sh \
  --env /absolute/path/to/venv \
  --nerfstudio-worktree /absolute/path/to/nerfstudio \
  --tcnn-runtime /absolute/path/to/tiny-rdna4-nn \
  --data /absolute/path/to/quick-validation \
  --output-root /absolute/path/to/evidence
```


`--env` means exactly one environment root. If it exists, the candidate is
`ENV_ROOT/bin/python`. If it is missing or incompatible, `auto` may create a new
isolated installation only at the separately supplied `--install-root`. The
installer never substitutes another environment found elsewhere on the host.

## Automatic fallback to a new environment

```bash
scripts/setup_public_adaptive_env_v1.sh \
  --env-policy auto \
  --resource-dir "$PWD/resource-cache-v1" \
  --install-root "$PWD/rdna4-nerfacto-env" \
  --bootstrap-python python3.12 \
  --auto
```

The existing candidate is probed first. If it is incompatible, the tool invokes
`setup_public_fresh_env_v1.py` for a new isolated environment. Custom native
resources still require an explicit local path or a verified resource cache.

## Modes

- `--dry-run` or `--mode plan`: emit a non-mutating decision report.
- `--env-policy auto|current|reuse|new`: select the environment strategy.
- `--repair`: create an isolated replacement instead of mutating the candidate.
- `--no-build`: reuse or fail closed; never create a new environment.
- `--quick`: P0 plus the real P1 producer/reload test. This is the default.
- `--verify`: P0 only.
- `--no-test`: runtime probe only.
- `--full-test`: quick validation plus the public unit-test suite.
- `--keep-work` or `--no-cleanup`: retain installer-owned temporary work.
- `--mode cleanup-only`: remove only `work/` directories carrying the adaptive
  ownership marker; final reports and evidence remain intact.
- `--keep-built-wheels`: accepted for forward compatibility. The current
  reference-binary profile performs no native wheel build.

## Decisions

Successful reuse emits:

```text
EXISTING_ENV_REUSED_AND_QUALIFIED
PUBLIC_RDNA4_ADAPTIVE_ENV: PASS
```

Successful isolated fallback emits:

```text
NEW_ISOLATED_ENV_CREATED_AND_QUALIFIED
PUBLIC_RDNA4_ADAPTIVE_ENV: PASS
```

Every run writes `final_aggregate.json`, `final_gate.txt`, and `MANIFEST.json`.
