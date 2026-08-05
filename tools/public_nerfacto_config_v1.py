#!/usr/bin/env python3
from __future__ import annotations

"""Scoped Nerfacto configuration for the public P0+P1 contract.

Importing ``nerfstudio.configs.method_configs`` eagerly imports configurations
for unrelated models.  Nerfstudio 1.1.5 also imports its Viser viewer at module
load time from ``nerfstudio.engine.trainer`` even when TensorBoard is selected.
The public qualified scope is Nerfacto P0+P1 with TensorBoard only, so this
module installs fail-closed viewer import stubs before importing TrainerConfig.
"""

import argparse
import importlib
import json
import sys
import types
from typing import Any

VIEWER_FREE_STUB_MARKER = "amd_nerfstudio_public_viewer_free_stub_v1"


class ViewerDisabledError(RuntimeError):
    """Raised if the viewer-free P0+P1 contract tries to construct a viewer."""


class _ViewerUnavailable:
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        del args, kwargs
        raise ViewerDisabledError(
            "VISER_VIEWER_DISABLED_BY_PUBLIC_P0_P1_CONTRACT: use vis='tensorboard'"
        )


class _ViserServerUnavailable:
    pass


def _mark_stub(module: types.ModuleType) -> types.ModuleType:
    setattr(module, VIEWER_FREE_STUB_MARKER, True)
    return module


def _require_absent_or_stub(name: str) -> None:
    existing = sys.modules.get(name)
    if existing is not None and not getattr(existing, VIEWER_FREE_STUB_MARKER, False):
        raise ViewerDisabledError(f"VIEWER_MODULE_ALREADY_IMPORTED_BEFORE_QUARANTINE: {name}")


def _install_package_stub(name: str) -> types.ModuleType:
    _require_absent_or_stub(name)
    module = sys.modules.get(name)
    if module is None:
        module = _mark_stub(types.ModuleType(name))
        module.__path__ = []  # type: ignore[attr-defined]
        module.__package__ = name
        sys.modules[name] = module
    parent_name, _, child_name = name.rpartition(".")
    if parent_name:
        parent = sys.modules[parent_name]
        setattr(parent, child_name, module)
    return module


def _install_module_stub(name: str, **attributes: Any) -> types.ModuleType:
    _require_absent_or_stub(name)
    module = sys.modules.get(name)
    if module is None:
        module = _mark_stub(types.ModuleType(name))
        module.__package__ = name.rpartition(".")[0]
        sys.modules[name] = module
    for key, value in attributes.items():
        setattr(module, key, value)
    parent_name, _, child_name = name.rpartition(".")
    if parent_name:
        parent = sys.modules[parent_name]
        setattr(parent, child_name, module)
    return module


def install_viewer_free_import_quarantine() -> dict[str, Any]:
    """Prevent eager Viewer imports while preserving the upstream Trainer.

    The actual viewer packages are intentionally not installed in this profile.
    Stubs are sufficient because ``vis='tensorboard'`` keeps every viewer branch
    disabled.  Any accidental viewer construction raises immediately.
    """

    importlib.import_module("nerfstudio")
    _require_absent_or_stub("viser")

    # NERFSTUDIO_RDNA4_VIEWER_TRANSFORMS_BRIDGE_V2_BEGIN
    import importlib as _viewer_importlib
    import sys as _viewer_sys

    # All fail-closed pre-import guards have passed. Load only the
    # mathematical transforms subtree, retain it, then remove the real
    # viser package before the existing quarantine installs its stub.
    try:
        _viewer_free_vtf = _viewer_importlib.import_module("viser.transforms")
    except Exception as _viewer_transforms_exc:
        raise RuntimeError(
            "Viewer-free Nerfacto requires viser.transforms for "
            "OrientedBox rotation mathematics"
        ) from _viewer_transforms_exc

    _viewer_transform_modules = {
        name: module
        for name, module in list(_viewer_sys.modules.items())
        if name == "viser.transforms"
        or name.startswith("viser.transforms.")
    }

    for _viewer_name in list(_viewer_sys.modules):
        if _viewer_name == "viser" or _viewer_name.startswith("viser."):
            _viewer_sys.modules.pop(_viewer_name, None)
    # NERFSTUDIO_RDNA4_VIEWER_TRANSFORMS_BRIDGE_V2_END
    _install_module_stub("viser", ViserServer=_ViserServerUnavailable)

    _install_package_stub("nerfstudio.viewer")
    _install_module_stub("nerfstudio.viewer.viewer", Viewer=_ViewerUnavailable)

    _install_package_stub("nerfstudio.viewer_legacy")
    _install_package_stub("nerfstudio.viewer_legacy.server")
    _install_module_stub(
        "nerfstudio.viewer_legacy.server.viewer_state",
        ViewerLegacyState=_ViewerUnavailable,
    )


    # NERFSTUDIO_RDNA4_VIEWER_TRANSFORMS_BRIDGE_V2_BEGIN
    _viewer_viser_stub = _viewer_sys.modules.get("viser")
    if _viewer_viser_stub is None:
        raise RuntimeError(
            "Viewer quarantine did not create the expected viser stub"
        )

    # Make the fail-closed stub package-like and expose only the retained
    # transforms subtree required by Nerfstudio geometry code.
    _viewer_viser_stub.__path__ = []
    _viewer_viser_stub.__package__ = "viser"

    for _viewer_name, _viewer_module in _viewer_transform_modules.items():
        _viewer_sys.modules[_viewer_name] = _viewer_module

    _viewer_viser_stub.transforms = _viewer_free_vtf
    # NERFSTUDIO_RDNA4_VIEWER_TRANSFORMS_BRIDGE_V2_END

    return {
        "policy": "TENSORBOARD_ONLY_VIEWER_IMPORT_QUARANTINE",
        "viewer_modules_stubbed": [
            "viser",
            "nerfstudio.viewer.viewer",
            "nerfstudio.viewer_legacy.server.viewer_state",
        ],
        "viewer_construction": "FAIL_CLOSED",
    }


PORTABLE_MLP_POLICY_MARKER = "amd_nerfstudio_public_portable_mlp_policy_v1"
PORTABLE_MLP_OTYPE = "PortableMLP"
PORTABLE_MLP_SOURCE_OTYPES = frozenset({"FullyFusedMLP", "CutlassMLP", "PortableMLP"})


def install_rdna4_portable_mlp_policy() -> dict[str, Any]:
    """Rewrite Nerfstudio TCNN MLP configs to the qualified AMD backend.

    The pinned Nerfstudio revision emits ``FullyFusedMLP`` for the layer widths
    used by Nerfacto.  The qualified tiny-rdna4-nn portable runtime deliberately
    rejects that CUDA-specific backend and accepts ``PortableMLP`` instead.
    This scoped policy keeps Nerfstudio and tiny-rdna4-nn source trees pristine
    while changing only the configuration dictionary supplied at construction.
    """

    from nerfstudio.field_components.mlp import MLP

    existing = getattr(MLP, PORTABLE_MLP_POLICY_MARKER, None)
    if existing is not None:
        if not isinstance(existing, dict):
            raise RuntimeError("portable MLP policy marker has invalid type")
        if existing.get("effective_otype") != PORTABLE_MLP_OTYPE:
            raise RuntimeError("portable MLP policy marker has invalid effective backend")
        if existing.get("fail_closed_unknown_otype") is not True:
            raise RuntimeError("portable MLP policy marker is not fail closed")
        return dict(existing)

    original = MLP.get_tcnn_network_config
    if not callable(original):
        raise RuntimeError("pinned Nerfstudio MLP config factory is not callable")

    def portable_get_tcnn_network_config(
        cls: type,
        activation: Any,
        out_activation: Any,
        layer_width: int,
        num_layers: int,
    ) -> dict[str, Any]:
        del cls
        config = original(
            activation=activation,
            out_activation=out_activation,
            layer_width=layer_width,
            num_layers=num_layers,
        )
        if not isinstance(config, dict):
            raise RuntimeError("pinned Nerfstudio MLP config factory returned a non-dict")
        source_otype = config.get("otype")
        if source_otype not in PORTABLE_MLP_SOURCE_OTYPES:
            raise RuntimeError(
                "unsupported TCNN MLP backend in public RDNA4 scope: "
                f"{source_otype!r}"
            )
        rewritten = dict(config)
        rewritten["otype"] = PORTABLE_MLP_OTYPE
        return rewritten

    MLP.get_tcnn_network_config = classmethod(portable_get_tcnn_network_config)
    policy = {
        "policy": "RDNA4_PORTABLE_MLP_CONFIG_REWRITE",
        "scope": "NERFACTO_TCNN_MLP_CONFIG_ONLY",
        "source_otypes": sorted(PORTABLE_MLP_SOURCE_OTYPES),
        "effective_otype": PORTABLE_MLP_OTYPE,
        "native_runtime_modified": False,
        "nerfstudio_source_modified": False,
        "fail_closed_unknown_otype": True,
    }
    setattr(MLP, PORTABLE_MLP_POLICY_MARKER, dict(policy))
    return dict(policy)


PILLOW_ENCODER_COMPAT_MARKER = "amd_nerfstudio_public_pillow_encoder_extents_compat_v1"


def _public_pil_to_numpy_with_extents(im: Any) -> Any:
    """Convert a loaded Pillow image using the encoder extents API."""

    import numpy as np
    from PIL import Image

    if not hasattr(Image, "_getencoder") or not hasattr(Image, "_conv_type_shape"):
        raise RuntimeError("required Pillow raw encoder API is unavailable")

    im.load()
    encoder = Image._getencoder(im.mode, "raw", im.mode)
    extents = (0, 0) + tuple(im.size)
    encoder.setimage(im.im, extents)

    shape, typestr = Image._conv_type_shape(im)
    data = np.empty(shape, dtype=np.dtype(typestr))
    memory = data.data.cast("B", (data.data.nbytes,))

    buffer_size, status, offset = 65536, 0, 0
    while not status:
        _, status, encoded = encoder.encode(buffer_size)
        memory[offset : offset + len(encoded)] = encoded
        offset += len(encoded)
    if status < 0:
        raise RuntimeError(f"Pillow raw encoder error {status}")
    if offset != data.data.nbytes:
        raise RuntimeError(
            "Pillow raw encoder byte count mismatch: "
            f"encoded={offset} expected={data.data.nbytes}"
        )
    return data


def install_pillow_encoder_extents_compatibility() -> dict[str, Any]:
    """Install a scoped Pillow compatibility function before DataLoader fork."""

    import importlib.metadata
    import numpy as np
    from PIL import Image
    import nerfstudio.data.datasets.base_dataset as base_dataset
    import nerfstudio.data.utils.data_utils as data_utils

    existing = getattr(data_utils, PILLOW_ENCODER_COMPAT_MARKER, None)
    if existing is not None:
        if not isinstance(existing, dict):
            raise RuntimeError("Pillow compatibility marker has invalid type")
        if existing.get("effective_call") != "encoder.setimage(im.im, extents)":
            raise RuntimeError("Pillow compatibility marker has invalid call contract")
        if existing.get("smoke_test_passed") is not True:
            raise RuntimeError("Pillow compatibility marker lacks passing smoke test")
        if data_utils.pil_to_numpy is not _public_pil_to_numpy_with_extents:
            raise RuntimeError("data_utils pil_to_numpy compatibility function was replaced")
        if base_dataset.pil_to_numpy is not _public_pil_to_numpy_with_extents:
            raise RuntimeError("base_dataset pil_to_numpy compatibility function was replaced")
        return dict(existing)

    original_data_utils = data_utils.pil_to_numpy
    original_base_dataset = base_dataset.pil_to_numpy
    if original_data_utils is not original_base_dataset:
        raise RuntimeError("pinned Nerfstudio pil_to_numpy aliases are not identical")

    data_utils.pil_to_numpy = _public_pil_to_numpy_with_extents
    base_dataset.pil_to_numpy = _public_pil_to_numpy_with_extents

    try:
        sample = Image.new("RGB", (2, 2), color=(17, 34, 51))
        observed = base_dataset.pil_to_numpy(sample)
        smoke_ok = bool(
            observed.shape == (2, 2, 3)
            and observed.dtype == np.uint8
            and observed.flags.writeable
            and np.all(observed == np.array([17, 34, 51], dtype=np.uint8))
        )
    except Exception:
        data_utils.pil_to_numpy = original_data_utils
        base_dataset.pil_to_numpy = original_base_dataset
        raise

    if not smoke_ok:
        data_utils.pil_to_numpy = original_data_utils
        base_dataset.pil_to_numpy = original_base_dataset
        raise RuntimeError("Pillow encoder-extents compatibility smoke test failed")

    try:
        pillow_version = importlib.metadata.version("Pillow")
    except importlib.metadata.PackageNotFoundError:
        pillow_version = "UNKNOWN"

    policy = {
        "policy": "PINNED_NERFSTUDIO_PILLOW_ENCODER_EXTENTS_COMPATIBILITY",
        "scope": "PIL_TO_NUMPY_FUNCTION_ALIAS_ONLY",
        "pillow_version": pillow_version,
        "effective_call": "encoder.setimage(im.im, extents)",
        "extents_contract": "(0, 0, width, height)",
        "smoke_test_passed": True,
        "data_utils_alias_patched": True,
        "base_dataset_alias_patched": True,
        "nerfstudio_source_modified": False,
        "pillow_distribution_modified": False,
        "fail_closed_missing_encoder_api": True,
    }
    setattr(data_utils, PILLOW_ENCODER_COMPAT_MARKER, dict(policy))
    setattr(base_dataset, PILLOW_ENCODER_COMPAT_MARKER, dict(policy))
    return dict(policy)


SPAWN_WORKER_COMPAT_MARKER = "amd_nerfstudio_public_spawn_worker_compat_v1"


def public_spawn_worker_init(worker_id: int) -> None:
    """Install the Pillow compatibility inside a spawned DataLoader worker.

    This function is intentionally top-level so Python multiprocessing can
    pickle it by module and qualified name under the ``spawn`` start method.
    """

    from torch.utils.data import get_worker_info

    info = get_worker_info()
    if info is None:
        raise RuntimeError("public spawn worker init executed outside a DataLoader worker")
    if int(info.id) != int(worker_id):
        raise RuntimeError(
            "public spawn worker id mismatch: "
            f"argument={worker_id}, runtime={info.id}"
        )

    policy = install_pillow_encoder_extents_compatibility()
    if policy.get("smoke_test_passed") is not True:
        raise RuntimeError("spawn worker Pillow compatibility smoke test failed")
    if policy.get("effective_call") != "encoder.setimage(im.im, extents)":
        raise RuntimeError("spawn worker Pillow compatibility call contract is invalid")
    if policy.get("nerfstudio_source_modified") is not False:
        raise RuntimeError("spawn worker compatibility modified Nerfstudio source")
    if policy.get("pillow_distribution_modified") is not False:
        raise RuntimeError("spawn worker compatibility modified Pillow distribution")


def install_spawn_worker_compatibility() -> dict[str, Any]:
    """Inject the qualified worker hook into ParallelDataManager DataLoaders."""

    import nerfstudio.data.datamanagers.parallel_datamanager as parallel_datamanager

    existing = getattr(parallel_datamanager, SPAWN_WORKER_COMPAT_MARKER, None)
    if existing is not None:
        if not isinstance(existing, dict):
            raise RuntimeError("spawn worker compatibility marker has invalid type")
        if existing.get("worker_init_fn") != "public_nerfacto_config_v1.public_spawn_worker_init":
            raise RuntimeError("spawn worker compatibility marker has invalid worker hook")
        if existing.get("spawn_safe_top_level") is not True:
            raise RuntimeError("spawn worker compatibility marker is not spawn safe")
        active_loader = parallel_datamanager.DataLoader
        if getattr(active_loader, SPAWN_WORKER_COMPAT_MARKER, False) is not True:
            raise RuntimeError("ParallelDataManager DataLoader compatibility wrapper was replaced")
        return dict(existing)

    original_loader = parallel_datamanager.DataLoader
    if not callable(original_loader):
        raise RuntimeError("pinned ParallelDataManager DataLoader alias is not callable")

    def public_parallel_data_loader(*args: Any, **kwargs: Any) -> Any:
        if "num_workers" in kwargs:
            num_workers = kwargs["num_workers"]
        elif len(args) > 5:
            num_workers = args[5]
        else:
            num_workers = 0

        if not isinstance(num_workers, int) or num_workers < 0:
            raise RuntimeError(f"invalid DataLoader num_workers: {num_workers!r}")

        if num_workers > 0:
            if len(args) > 10:
                existing_hook = args[10]
                if existing_hook is not None and existing_hook is not public_spawn_worker_init:
                    raise RuntimeError(
                        "public ParallelDataManager scope refuses an unqualified positional worker_init_fn"
                    )
                mutable_args = list(args)
                mutable_args[10] = public_spawn_worker_init
                args = tuple(mutable_args)
            else:
                existing_hook = kwargs.get("worker_init_fn")
                if existing_hook is not None and existing_hook is not public_spawn_worker_init:
                    raise RuntimeError(
                        "public ParallelDataManager scope refuses an unqualified worker_init_fn"
                    )
                kwargs["worker_init_fn"] = public_spawn_worker_init

        return original_loader(*args, **kwargs)

    public_parallel_data_loader.__name__ = "PublicSpawnCompatibleDataLoaderFactory"
    public_parallel_data_loader.__qualname__ = "PublicSpawnCompatibleDataLoaderFactory"
    public_parallel_data_loader.__module__ = __name__
    setattr(public_parallel_data_loader, SPAWN_WORKER_COMPAT_MARKER, True)
    parallel_datamanager.DataLoader = public_parallel_data_loader

    worker_qualname = public_spawn_worker_init.__qualname__
    policy = {
        "policy": "PARALLEL_DATAMANAGER_SPAWN_WORKER_COMPATIBILITY",
        "scope": "PARALLEL_DATAMANAGER_DATALOADER_ALIAS_ONLY",
        "multiprocessing_start_method": "spawn",
        "dataloader_num_workers_required": 1,
        "worker_init_fn": f"{public_spawn_worker_init.__module__}.{worker_qualname}",
        "spawn_safe_top_level": "<locals>" not in worker_qualname,
        "parallel_datamanager_alias_patched": True,
        "pillow_compatibility_installed_in_worker": True,
        "nerfstudio_source_modified": False,
        "pillow_distribution_modified": False,
        "fail_closed_existing_worker_hook": True,
    }
    if policy["worker_init_fn"] != "public_nerfacto_config_v1.public_spawn_worker_init":
        raise RuntimeError("spawn worker hook module identity is not stable")
    if policy["spawn_safe_top_level"] is not True:
        raise RuntimeError("spawn worker hook is not a top-level function")

    setattr(parallel_datamanager, SPAWN_WORKER_COMPAT_MARKER, dict(policy))
    return dict(policy)


def build_public_nerfacto_config() -> Any:
    install_viewer_free_import_quarantine()
    portable_mlp_policy = install_rdna4_portable_mlp_policy()
    pillow_image_compatibility = install_pillow_encoder_extents_compatibility()
    spawn_worker_compatibility = install_spawn_worker_compatibility()

    from nerfstudio.cameras.camera_optimizers import CameraOptimizerConfig
    from nerfstudio.configs.base_config import ViewerConfig
    from nerfstudio.data.datamanagers.parallel_datamanager import ParallelDataManagerConfig
    from nerfstudio.data.dataparsers.nerfstudio_dataparser import NerfstudioDataParserConfig
    from nerfstudio.engine.optimizers import AdamOptimizerConfig
    from nerfstudio.engine.schedulers import ExponentialDecaySchedulerConfig
    from nerfstudio.engine.trainer import TrainerConfig
    from nerfstudio.models.nerfacto import NerfactoModelConfig
    from nerfstudio.pipelines.base_pipeline import VanillaPipelineConfig

    cfg = TrainerConfig(
        method_name="nerfacto",
        steps_per_eval_batch=500,
        steps_per_save=2000,
        max_num_iterations=30000,
        mixed_precision=True,
        pipeline=VanillaPipelineConfig(
            datamanager=ParallelDataManagerConfig(
                dataparser=NerfstudioDataParserConfig(),
                train_num_rays_per_batch=4096,
                eval_num_rays_per_batch=4096,
            ),
            model=NerfactoModelConfig(
                eval_num_rays_per_chunk=1 << 15,
                average_init_density=0.01,
                camera_optimizer=CameraOptimizerConfig(mode="SO3xR3"),
            ),
        ),
        optimizers={
            "proposal_networks": {
                "optimizer": AdamOptimizerConfig(lr=1e-2, eps=1e-15),
                "scheduler": ExponentialDecaySchedulerConfig(lr_final=0.0001, max_steps=200000),
            },
            "fields": {
                "optimizer": AdamOptimizerConfig(lr=1e-2, eps=1e-15),
                "scheduler": ExponentialDecaySchedulerConfig(lr_final=0.0001, max_steps=200000),
            },
            "camera_opt": {
                "optimizer": AdamOptimizerConfig(lr=1e-3, eps=1e-15),
                "scheduler": ExponentialDecaySchedulerConfig(lr_final=1e-4, max_steps=5000),
            },
        },
        viewer=ViewerConfig(num_rays_per_chunk=1 << 15),
        vis="tensorboard",
    )
    setattr(cfg, "amd_rdna4_portable_mlp_policy", portable_mlp_policy)
    setattr(cfg, "amd_nerfstudio_pillow_image_compatibility", pillow_image_compatibility)
    setattr(cfg, "amd_nerfstudio_spawn_worker_compatibility", spawn_worker_compatibility)
    return cfg


def self_test() -> int:
    source = __file__
    passed = bool(source and build_public_nerfacto_config.__name__ == "build_public_nerfacto_config")
    print(json.dumps({
        "schema": "amd-nerfstudio-public-nerfacto-config-v1",
        "passed": passed,
        "scope": "NERFACTO_ONLY_TENSORBOARD",
        "global_method_configs_imported": False,
        "viewer_dependency_installed": False,
        "viewer_import_policy": "FAIL_CLOSED_QUARANTINE",
    }, indent=2, sort_keys=True))
    return 0 if passed else 2


def main() -> int:
    parser = argparse.ArgumentParser(description="Scoped public Nerfacto configuration builder")
    parser.add_argument("--mode", choices=["self-test"], default="self-test")
    parser.parse_args()
    return self_test()


if __name__ == "__main__":
    raise SystemExit(main())
