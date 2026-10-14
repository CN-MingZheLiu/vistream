#!/usr/bin/env python3
"""Single entry point for the frozen hierarchical Lingbot-ORB MoE pipelines."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import subprocess
import sys


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
CONFIG_DIR = PROJECT_ROOT / "configs"
if not CONFIG_DIR.is_dir():
    # The local delivery mirror keeps scripts and configs in one directory.
    CONFIG_DIR = SCRIPT_DIR


@dataclass(frozen=True)
class Pipeline:
    script: str
    policy: str
    policy_commands: frozenset[str]


PIPELINES = {
    "kitti": Pipeline(
        script="training_free_orb_consensus_moe.py",
        policy="training_free_orb_consensus_moe_fixed.json",
        policy_commands=frozenset({"generate", "evaluate"}),
    ),
    "inertial": Pipeline(
        script="training_free_orbslam3_inertial_moe.py",
        policy="training_free_orbslam3_inertial_moe_fixed.json",
        policy_commands=frozenset({"generate"}),
    ),
    "oxford": Pipeline(
        script="training_free_orbslam3_oxford_suite.py",
        policy="training_free_orbslam3_inertial_moe_fixed.json",
        policy_commands=frozenset({"generate"}),
    ),
}

OXFORD_SHELL_COMMANDS = {
    "prepare": "run_orbslam3_oxford_suite.sh",
    "retry": "run_orbslam3_oxford_health_retries.sh",
}

HELP = f"""\
Frozen hierarchical Lingbot-ORB MoE

Usage:
  {Path(sys.argv[0]).name} kitti <generate|evaluate> [arguments...]
  {Path(sys.argv[0]).name} inertial <generate|evaluate> [arguments...]
  {Path(sys.argv[0]).name} oxford <prepare|retry|generate|evaluate> [arguments...]

The wrapper delegates to the frozen implementation and automatically supplies
the matching policy when --policy is not provided. Ground truth remains
available only to the delegated evaluate command.

Examples:
  {Path(sys.argv[0]).name} kitti generate --base-root BASE --orb-root ORB --output-dir OUT
  {Path(sys.argv[0]).name} inertial generate --dataset tumvi --sequence room2 \\
      --cache CACHE --timestamps TIMES --run-dir ORB_RUN --output-dir OUT
  {Path(sys.argv[0]).name} oxford prepare
  {Path(sys.argv[0]).name} oxford generate --input-root INPUTS --run-root RUNS \\
      --binary BINARY --vocabulary VOCAB --source-diff DIFF --output-dir OUT
"""


def has_option(arguments: list[str], name: str) -> bool:
    return any(value == name or value.startswith(f"{name}=") for value in arguments)


def existing(path: Path, label: str) -> Path:
    if not path.is_file():
        raise FileNotFoundError(f"missing {label}: {path}")
    return path


def build_command(arguments: list[str]) -> list[str] | None:
    if not arguments or arguments[0] in {"-h", "--help"}:
        return None

    pipeline_name = arguments[0]
    if pipeline_name not in PIPELINES:
        raise ValueError(f"unknown pipeline {pipeline_name!r}; expected {', '.join(PIPELINES)}")

    pipeline = PIPELINES[pipeline_name]
    if len(arguments) == 1 or arguments[1] in {"-h", "--help"}:
        script = existing(SCRIPT_DIR / pipeline.script, f"{pipeline_name} entry")
        return [sys.executable, str(script), "--help"]

    command = arguments[1]
    forwarded = list(arguments[2:])
    if pipeline_name == "oxford" and command in OXFORD_SHELL_COMMANDS:
        if forwarded:
            raise ValueError(f"oxford {command} does not accept positional arguments")
        script = existing(SCRIPT_DIR / OXFORD_SHELL_COMMANDS[command], f"Oxford {command} script")
        return ["bash", str(script)]

    if command not in {"generate", "evaluate"}:
        raise ValueError(f"unknown {pipeline_name} command {command!r}")

    script = existing(SCRIPT_DIR / pipeline.script, f"{pipeline_name} entry")
    if command in pipeline.policy_commands and not has_option(forwarded, "--policy"):
        policy = existing(CONFIG_DIR / pipeline.policy, f"{pipeline_name} policy")
        forwarded.extend(["--policy", str(policy)])
    return [sys.executable, str(script), command, *forwarded]


def main(arguments: list[str] | None = None) -> int:
    try:
        command = build_command(list(sys.argv[1:] if arguments is None else arguments))
    except (FileNotFoundError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        print(HELP, file=sys.stderr)
        return 2
    if command is None:
        print(HELP)
        return 0
    subprocess.run(command, check=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
