#!/usr/bin/env python3
"""Generate, seal, and evaluate the ten-sequence Oxford inertial MoE suite."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import numpy as np


SCRIPT_ROOT = Path(__file__).resolve().parent
EXPERT_SCRIPT = SCRIPT_ROOT / "training_free_orbslam3_inertial_moe.py"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def run(arguments: list[str]) -> None:
    subprocess.run([sys.executable, str(EXPERT_SCRIPT), *arguments], check=True)


def input_entries(
    input_root: Path, run_root: Path, replica_root: Path | None = None
) -> list[dict]:
    entries = []
    for manifest_path in sorted(input_root.glob("*/manifest_before_gt.json")):
        slug = manifest_path.parent.name
        manifest = json.loads(manifest_path.read_text())
        run_dir = run_root / slug
        if len(list(run_dir.glob("f_*.txt"))) != 1:
            raise FileNotFoundError(f"Oxford ORB run is incomplete: {run_dir}")
        run_dirs = [str(run_dir.resolve())]
        if replica_root is not None:
            for replica in sorted((replica_root / slug).glob("replica_*")):
                if len(list(replica.glob("f_*.txt"))) != 1:
                    raise FileNotFoundError(f"Oxford ORB replica is incomplete: {replica}")
                run_dirs.append(str(replica.resolve()))
        entries.append(
            {
                "slug": slug,
                "sequence": Path(manifest["source_imu"]).parents[1].name,
                "cache": manifest["cache"],
                "input_manifest": str(manifest_path.resolve()),
                "timestamps": manifest["timestamps"],
                "settings": manifest["settings"],
                "imu": manifest["converted_imu"],
                "run_dirs": run_dirs,
            }
        )
    if len(entries) != 10:
        raise RuntimeError(f"expected 10 prepared Oxford sequences, got {len(entries)}")
    return entries


def generate(args: argparse.Namespace) -> None:
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError(f"output directory is not empty: {args.output_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    entries = input_entries(args.input_root, args.run_root, args.replica_root)
    sealed_entries = []
    for entry in entries:
        output = args.output_dir / entry["slug"]
        command = [
                "generate",
                "--dataset", "oxford",
                "--sequence", entry["sequence"],
                "--cache", entry["cache"],
                "--timestamps", entry["timestamps"],
                "--policy", str(args.policy),
                "--output-dir", str(output),
                "--visual-convention", "w2c",
                "--sealed-artifact", f"input_manifest={entry['input_manifest']}",
                "--sealed-artifact", f"settings={entry['settings']}",
                "--sealed-artifact", f"imu={entry['imu']}",
                "--sealed-artifact", f"binary={args.binary}",
                "--sealed-artifact", f"vocabulary={args.vocabulary}",
                "--sealed-artifact", f"orbslam3_source_diff={args.source_diff}",
                "--protocol",
                "exploratory Oxford suite sealed before GT; development results observed",
            ]
        for run_dir in entry["run_dirs"]:
            command.extend(["--run-dir", run_dir])
        run(command)
        child_manifest = output / "manifest_before_gt.json"
        sealed_entries.append(
            {
                **entry,
                "output_dir": str(output.resolve()),
                "manifest": str(child_manifest.resolve()),
                "manifest_sha256": sha256(child_manifest),
            }
        )
    suite_manifest = {
        "protocol": (
            "ten Oxford candidates and zero-parameter health gates sealed before "
            "any suite GT evaluation"
        ),
        "policy": {"path": str(args.policy.resolve()), "sha256": sha256(args.policy)},
        "binary": {"path": str(args.binary.resolve()), "sha256": sha256(args.binary)},
        "vocabulary": {
            "path": str(args.vocabulary.resolve()),
            "sha256": sha256(args.vocabulary),
        },
        "source_diff": {
            "path": str(args.source_diff.resolve()),
            "sha256": sha256(args.source_diff),
        },
        "expert_script": {
            "path": str(EXPERT_SCRIPT.resolve()),
            "sha256": sha256(EXPERT_SCRIPT),
        },
        "suite_script": {
            "path": str(Path(__file__).resolve()),
            "sha256": sha256(Path(__file__).resolve()),
        },
        "entries": sealed_entries,
    }
    manifest_path = args.output_dir / "manifest_before_gt.json"
    manifest_path.write_text(json.dumps(suite_manifest, indent=2) + "\n")
    print(f"suite_manifest_sha256={sha256(manifest_path)}")


def verify(path: Path, expected: str, label: str) -> None:
    if sha256(path) != expected:
        raise ValueError(f"sealed {label} changed: {path}")


def evaluate(args: argparse.Namespace) -> None:
    manifest_path = args.output_dir / "manifest_before_gt.json"
    manifest = json.loads(manifest_path.read_text())
    verify(Path(manifest["policy"]["path"]), manifest["policy"]["sha256"], "policy")
    verify(Path(manifest["binary"]["path"]), manifest["binary"]["sha256"], "binary")
    verify(
        Path(manifest["vocabulary"]["path"]),
        manifest["vocabulary"]["sha256"],
        "vocabulary",
    )
    for name in ("source_diff", "expert_script", "suite_script"):
        verify(Path(manifest[name]["path"]), manifest[name]["sha256"], name)
    for entry in manifest["entries"]:
        child_manifest = Path(entry["manifest"])
        verify(child_manifest, entry["manifest_sha256"], f"{entry['slug']} manifest")
    rows = []
    for entry in manifest["entries"]:
        output = Path(entry["output_dir"])
        run(["evaluate", "--output-dir", str(output), "--ground-truth-key", "ground_truth_c2w"])
        summary = json.loads((output / "summary_after_gt.json").read_text())
        diagnostics = json.loads((output / "inertial_gate_before_gt.json").read_text())
        rows.append(
            {
                "sequence": entry["sequence"],
                "environment": diagnostics["gate"]["environment"],
                "orb_weight": diagnostics["gate"]["orb_weight"],
                "tracking_completeness": diagnostics["selected_features"][
                    "cache_tracking_completeness"
                ],
                "visual_ate_rmse_m": summary["visual"]["ate_rmse_m"],
                "candidate_ate_rmse_m": summary["candidate"]["ate_rmse_m"],
                "candidate_overlap_ate_rmse_m": summary["candidate_overlap"][
                    "ate_rmse_m"
                ],
            }
        )
    visual = np.asarray([row["visual_ate_rmse_m"] for row in rows])
    candidate = np.asarray([row["candidate_ate_rmse_m"] for row in rows])
    summary = {
        "protocol": manifest["protocol"],
        "manifest_sha256": sha256(manifest_path),
        "rows": rows,
        "macro": {
            "sequence_count": len(rows),
            "visual_ate_rmse_mean_m": float(visual.mean()),
            "candidate_ate_rmse_mean_m": float(candidate.mean()),
            "improved_sequences": int(np.sum(candidate < visual)),
        },
    }
    output = args.output_dir / "summary_after_gt.json"
    output.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    generate_parser = subparsers.add_parser("generate")
    generate_parser.add_argument("--input-root", type=Path, required=True)
    generate_parser.add_argument("--run-root", type=Path, required=True)
    generate_parser.add_argument("--replica-root", type=Path)
    generate_parser.add_argument("--policy", type=Path, required=True)
    generate_parser.add_argument("--binary", type=Path, required=True)
    generate_parser.add_argument("--vocabulary", type=Path, required=True)
    generate_parser.add_argument("--source-diff", type=Path, required=True)
    generate_parser.add_argument("--output-dir", type=Path, required=True)
    evaluate_parser = subparsers.add_parser("evaluate")
    evaluate_parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_args()
    generate(arguments) if arguments.command == "generate" else evaluate(arguments)
