#!/usr/bin/env python3
"""Load both frozen checkpoints and print compatibility/freeze diagnostics."""

import argparse
import inspect
import json

from vistream.config import load_yaml, model_config
from vistream.model.system import VIStream


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/vistream_tartanair.yaml")
    args = parser.parse_args()
    config = model_config(load_yaml(args.config))
    model = VIStream.from_pretrained(config)
    model.arm_backbone_guard()
    result = {
        "airimu_checkpoint": config.airimu_checkpoint,
        "airimu_loaded_tensors": model.airimu._loaded_keys,
        "lingbot_checkpoint": config.lingbot_checkpoint,
        "lingbot_module_path": inspect.getfile(model.lingbot.__class__),
        "lingbot_trainable_parameters": sum(p.requires_grad for p in model.lingbot.parameters()),
        "airimu_trainable_parameters": sum(p.requires_grad for p in model.airimu.parameters()),
        "adapter_trainable_parameter_count": sum(p.numel() for p in model.trainable_parameters()),
        "trainable_parameter_prefixes": sorted({name.split(".")[0] for name, p in model.named_parameters() if p.requires_grad}),
    }
    model.assert_backbones_intact()
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
