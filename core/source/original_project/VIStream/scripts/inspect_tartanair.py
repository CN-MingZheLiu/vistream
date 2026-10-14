#!/usr/bin/env python3
"""Print the deterministic environment-level TartanAir split and one sample."""

import argparse

from vistream.config import load_yaml
from vistream.data import (
    TartanAirWindowDataset,
    discover_tartanair_sequences,
    split_tartanair_by_environment,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/vistream_tartanair.yaml")
    args = parser.parse_args()
    config = load_yaml(args.config)
    root = config["data"]["tartanair_root"]
    sequences = discover_tartanair_sequences(root)
    train, validation = split_tartanair_by_environment(
        sequences, config["data"]["validation_environments"], config["data"]["split_seed"]
    )
    print(f"sequences={len(sequences)} train={len(train)} validation={len(validation)}")
    print(f"train_envs={len({seq.split('/')[0] for seq in train})}")
    print(f"validation_envs={len({seq.split('/')[0] for seq in validation})}")
    dataset = TartanAirWindowDataset(
        root,
        train[:1],
        window_frames=config["data"]["window_frames"],
        stride_frames=config["data"]["stride_frames"],
        image_size=config["model"]["image_size"],
        max_imu_samples=config["data"]["max_imu_samples"],
        acc_source=config["data"].get("acc_source", "acc"),
    )
    sample = dataset[0]
    print(
        f"sample images={tuple(sample['images'].shape)} imu={tuple(sample['imu'].shape)} "
        f"depth={tuple(sample['depth'].shape)}"
    )


if __name__ == "__main__":
    main()
