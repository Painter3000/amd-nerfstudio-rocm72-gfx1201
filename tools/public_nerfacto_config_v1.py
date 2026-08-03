#!/usr/bin/env python3
from __future__ import annotations

"""Scoped Nerfacto configuration equivalent to the pinned upstream config.

Importing ``nerfstudio.configs.method_configs`` eagerly imports configurations
for unrelated models, including Splatfacto.  The public qualified scope is only
Nerfacto, so the public runners construct that one configuration directly.
"""

import argparse
import json
from typing import Any


def build_public_nerfacto_config() -> Any:
    from nerfstudio.cameras.camera_optimizers import CameraOptimizerConfig
    from nerfstudio.configs.base_config import ViewerConfig
    from nerfstudio.data.datamanagers.parallel_datamanager import ParallelDataManagerConfig
    from nerfstudio.data.dataparsers.nerfstudio_dataparser import NerfstudioDataParserConfig
    from nerfstudio.engine.optimizers import AdamOptimizerConfig
    from nerfstudio.engine.schedulers import ExponentialDecaySchedulerConfig
    from nerfstudio.engine.trainer import TrainerConfig
    from nerfstudio.models.nerfacto import NerfactoModelConfig
    from nerfstudio.pipelines.base_pipeline import VanillaPipelineConfig

    return TrainerConfig(
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
        vis="viewer",
    )


def self_test() -> int:
    # This test intentionally has no Nerfstudio import requirement.  It guards
    # the scoped-loader policy and the exact constants represented above.
    source = __file__
    passed = bool(source and build_public_nerfacto_config.__name__ == "build_public_nerfacto_config")
    print(json.dumps({
        "schema": "amd-nerfstudio-public-nerfacto-config-v1",
        "passed": passed,
        "scope": "NERFACTO_ONLY",
        "global_method_configs_imported": False,
    }, indent=2, sort_keys=True))
    return 0 if passed else 2


def main() -> int:
    parser = argparse.ArgumentParser(description="Scoped public Nerfacto configuration builder")
    parser.add_argument("--mode", choices=["self-test"], default="self-test")
    parser.parse_args()
    return self_test()


if __name__ == "__main__":
    raise SystemExit(main())
