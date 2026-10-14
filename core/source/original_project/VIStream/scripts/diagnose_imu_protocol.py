#!/usr/bin/env python3
"""Train/validation-only IMU input protocol diagnostic.

This never loads external test data and does not update any model weights.
"""

import argparse
from pathlib import Path

import numpy as np

from vistream.config import load_yaml
from vistream.data import discover_tartanair_sequences, split_tartanair_by_environment


def stats(path: Path, source: str) -> dict[str, float]:
    values = np.loadtxt(path / "imu" / f"{source}.txt", dtype=np.float64).reshape(-1, 3)
    return {
        "mean_norm": float(np.linalg.norm(values, axis=-1).mean()),
        "std_norm": float(np.linalg.norm(values, axis=-1).std()),
        "mean_x": float(values[:, 0].mean()),
        "mean_y": float(values[:, 1].mean()),
        "mean_z": float(values[:, 2].mean()),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/vistream_tartanair.yaml")
    parser.add_argument("--sequences", type=int, default=4)
    args = parser.parse_args()
    config = load_yaml(args.config)
    root = Path(config["data"]["tartanair_root"])
    all_sequences = discover_tartanair_sequences(root)
    train, validation = split_tartanair_by_environment(
        all_sequences, config["data"]["validation_environments"], config["data"]["split_seed"]
    )
    selected = train[: args.sequences] + validation[: args.sequences]
    for sequence in selected:
        environment, difficulty, trajectory = sequence.split("/")
        path = root / environment / f"Data_{difficulty}" / trajectory
        print(sequence)
        for source in ("acc", "acc_nograv", "acc_nograv_body"):
            print(f"  {source}: {stats(path, source)}")
        print("  gyro:", stats(path, "gyro"))
    print("The default model protocol uses raw 'acc' and lets the frozen AirIMU correction plus SE(3) preintegrator handle gravity.")


if __name__ == "__main__":
    main()

