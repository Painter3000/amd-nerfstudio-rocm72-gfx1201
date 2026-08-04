# Qualification scope

## Qualified claim

The qualified object is the **Nerfacto training chain from Nerfstudio** on the pinned AMD RDNA4 / `gfx1201` environment.

```text
NERFACTO_TRAINING_CHAIN_NOT_FULL_NERFSTUDIO
```

The evidence covers:

- pinned source and runtime identity;
- real Nerfstudio data loading;
- Nerfacto forward, loss, backward, and optimizer updates;
- checkpoint creation and hashing;
- exact fresh-process state restoration;
- real resume execution;
- sustained training in the predefined 576-step window;
- memory-trend gates after warm-up;
- resume trajectory behavior relative to a natural A-vs-C variation envelope.

## Explicit nonclaims

The qualification does not establish:

- support for every Nerfstudio model;
- Viewer or export support;
- Splatfacto support;
- multi-GPU or distributed support;
- infinite-horizon leak freedom;
- VMM performance parity;
- performance superiority over CUDA/NVIDIA;
- support outside the pinned ROCm, PyTorch, GPU, and source revisions;
- cross-host or cross-checkout binary identity.

## Interpretation rule

A scoped PASS must not be broadened into a platform-wide claim. The project demonstrates technical feasibility and a qualified integration path, not official upstream support for all configurations.


## v1.4.3 adaptive environment note

The adaptive installer may reuse a compatible existing environment unchanged or
create a new isolated Fresh-ENV. `viser==1.0.0` is now the qualified math-only
dependency for `viser.transforms.SO3`; Viewer construction remains quarantined
fail-closed, and `pyliblzfse` / `yourdfpy` remain outside the scoped contract.
See `docs/PUBLIC_ADAPTIVE_ENV_V1.md`.
