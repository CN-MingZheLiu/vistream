#!/usr/bin/env python3
"""GT-free routing among visual, IMU, and loop-closure pose experts."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


SEQUENCES = tuple(f"{index:02d}" for index in range(11))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def poses_4x4(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    if values.shape[1:] == (4, 4):
        return values.copy()
    if values.shape[1:] != (3, 4):
        raise ValueError(f"expected [N,3,4] or [N,4,4], got {values.shape}")
    output = np.repeat(np.eye(4)[None], len(values), axis=0)
    output[:, :3, :4] = values
    return output


def sigmoid(value: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(value, -60.0, 60.0)))


def project_rotation(matrix: np.ndarray) -> np.ndarray:
    left, _, right = np.linalg.svd(matrix)
    correction = np.eye(3)
    correction[-1, -1] = np.linalg.det(left @ right)
    return left @ correction @ right


def weak_revisit_statistics(
    visual: np.ndarray,
    descriptors: np.ndarray,
    minimum_separation: int,
    nms_radius: int,
    maximum_pairs: int,
    statistics_fraction: float = 1.0,
) -> dict[str, float | int | bool]:
    """Estimate scene repeatability from frozen descriptors only."""
    if not 0.0 < statistics_fraction <= 1.0:
        raise ValueError("statistics_fraction must be in (0, 1]")
    if len(descriptors) != len(visual):
        raise ValueError("descriptor and visual frame counts differ")
    if len(descriptors) <= minimum_separation:
        return {
            "pair_count": 0,
            "median_score": 0.0,
            "median_path_closure": 1.0,
        }
    similarity = descriptors @ descriptors.T
    src, dst = np.triu_indices(len(descriptors), k=minimum_separation)
    order = np.argsort(similarity[src, dst])[::-1]
    pairs: list[tuple[int, int]] = []
    for item in order:
        first, second = int(src[item]), int(dst[item])
        if any(
            abs(first - prior_first) < nms_radius
            and abs(second - prior_second) < nms_radius
            for prior_first, prior_second in pairs
        ):
            continue
        pairs.append((first, second))
        if len(pairs) >= maximum_pairs:
            break
    if not pairs:
        return {
            "pair_count": 0,
            "median_score": 0.0,
            "median_path_closure": 1.0,
        }
    # Retrieval ranking is the only scene cue available at inference. Keep the
    # strongest fixed fraction so weak descriptor tail matches cannot dominate
    # the closure consensus.
    keep_count = max(1, int(np.ceil(len(pairs) * statistics_fraction)))
    pairs = pairs[:keep_count]
    steps = np.linalg.norm(np.diff(visual[:, :3, 3], axis=0), axis=1)
    scores, closures = [], []
    for first, second in pairs:
        scores.append(float(similarity[first, second]))
        closures.append(
            float(
                np.linalg.norm(
                    visual[second, :3, 3] - visual[first, :3, 3]
                )
                / max(float(np.sum(steps[first:second])), 1e-9)
            )
        )
    return {
        "pair_count": len(pairs),
        "median_score": float(np.median(scores)),
        "median_path_closure": float(np.median(closures)),
    }


def weak_revisit_accepts(
    statistics: dict[str, float | int | bool],
    minimum_score: float,
    maximum_path_closure: float,
    minimum_pairs: int,
) -> bool:
    return bool(
        statistics["pair_count"] >= minimum_pairs
        and statistics["median_score"] >= minimum_score
        and statistics["median_path_closure"] <= maximum_path_closure
    )


def conjunctive_probability(margins: list[float]) -> float:
    """Soft AND for fixed, interpretable gate-head feature margins."""
    return float(sigmoid(np.asarray(min(margins), dtype=np.float64)))


def frozen_two_branch_gate(
    statistics: dict[str, float | int | bool],
    relative_path_change: float,
    minimum_pairs: int,
    closed_minimum_score: float,
    closed_maximum_path_closure: float,
    appearance_minimum_score: float,
    appearance_maximum_path_change: float,
    score_width: float,
    closure_width: float,
    path_change_width: float,
) -> dict[str, float | str | bool]:
    """Frozen gate head for closed revisits and trusted open-road experts."""
    if min(score_width, closure_width, path_change_width) <= 0.0:
        raise ValueError("two-branch gate widths must be positive")
    count_margin = float(statistics["pair_count"] - minimum_pairs)
    closed_probability = conjunctive_probability([
        count_margin,
        (float(statistics["median_score"]) - closed_minimum_score) / score_width,
        (
            closed_maximum_path_closure
            - float(statistics["median_path_closure"])
        ) / closure_width,
    ])
    appearance_probability = conjunctive_probability([
        count_margin,
        (
            float(statistics["median_score"]) - appearance_minimum_score
        ) / score_width,
        (appearance_maximum_path_change - relative_path_change) / path_change_width,
    ])
    if closed_probability >= appearance_probability:
        active_branch = "closed_revisit"
        probability = closed_probability
    else:
        active_branch = "appearance_trust_region"
        probability = appearance_probability
    return {
        "weak_revisit_closed_probability": closed_probability,
        "weak_revisit_appearance_probability": appearance_probability,
        "weak_revisit_gate_probability": probability,
        "weak_revisit_active_branch": active_branch,
    }


def nonholonomic_imu_expert(
    visual: np.ndarray,
    imu: np.ndarray,
    forward_center: float,
    forward_width: float,
    planar_center: float,
    planar_width: float,
    route_threshold: float,
    sign_threshold: float,
    maximum_path_change_ratio: float,
) -> tuple[np.ndarray, dict[str, float | bool | str]]:
    world_steps = np.diff(visual[:, :3, 3], axis=0)
    step_norms = np.linalg.norm(world_steps, axis=1)
    local_steps = np.einsum(
        "nij,nj->ni", visual[:-1, :3, :3].transpose(0, 2, 1), world_steps
    )
    valid = step_norms >= max(float(np.quantile(step_norms, 0.1)), 1e-9)
    forward_ratio = np.abs(local_steps[:, 2]) / np.maximum(step_norms, 1e-9)
    planar_ratio = np.linalg.norm(local_steps[:, [0, 2]], axis=1) / np.maximum(
        step_norms, 1e-9
    )
    interval_gate = sigmoid((forward_ratio - forward_center) / forward_width)
    interval_gate *= sigmoid((planar_ratio - planar_center) / planar_width)
    if not np.any(valid):
        return visual.copy(), {
            "expert": "visual",
            "imu_route_probability": 0.0,
            "forward_sign_consistency": 0.0,
            "relative_path_change": 0.0,
            "route_enabled": False,
            "fallback_reason": "insufficient_visual_motion",
        }
    sign = np.sign(local_steps[valid, 2])
    sign_consistency = float(max(np.mean(sign >= 0.0), np.mean(sign <= 0.0)))
    route_probability = float(np.median(interval_gate[valid]))
    route_enabled = (
        route_probability >= route_threshold and sign_consistency >= sign_threshold
    )
    if not route_enabled:
        return visual.copy(), {
            "expert": "visual",
            "imu_route_probability": route_probability,
            "forward_sign_consistency": sign_consistency,
            "relative_path_change": 0.0,
            "route_enabled": False,
            "fallback_reason": "nonholonomic_observability_rejected",
        }
    constrained = local_steps.copy()
    constrained[:, 0] *= 1.0 - interval_gate
    constrained[:, 1] *= 1.0 - interval_gate
    constrained_norms = np.linalg.norm(constrained, axis=1)
    constrained *= step_norms[:, None] / np.maximum(constrained_norms[:, None], 1e-9)
    output = imu.copy()
    output[0] = visual[0]
    for index, local_step in enumerate(constrained):
        output[index + 1, :3, 3] = (
            output[index, :3, 3] + output[index, :3, :3] @ local_step
        )
    path_scale = max(
        float(np.linalg.norm(visual[-1, :3, 3] - visual[0, :3, 3])),
        0.1 * float(np.sum(step_norms)),
        1e-9,
    )
    relative_path_change = float(
        np.sqrt(np.mean(np.sum((output[:, :3, 3] - visual[:, :3, 3]) ** 2, axis=1)))
        / path_scale
    )
    if relative_path_change > maximum_path_change_ratio:
        return visual.copy(), {
            "expert": "visual",
            "imu_route_probability": route_probability,
            "forward_sign_consistency": sign_consistency,
            "relative_path_change": relative_path_change,
            "route_enabled": False,
            "fallback_reason": "path_change_budget_exceeded",
        }
    return output, {
        "expert": "nonholonomic_imu",
        "imu_route_probability": route_probability,
        "forward_sign_consistency": sign_consistency,
        "relative_path_change": relative_path_change,
        "route_enabled": True,
        "fallback_reason": None,
    }


def compose_loop_imu_expert(
    visual: np.ndarray,
    imu: np.ndarray,
    loop: np.ndarray,
    forward_center: float,
    forward_width: float,
    planar_center: float,
    planar_width: float,
    maximum_path_change_ratio: float,
) -> tuple[np.ndarray, dict[str, float | bool | str | None]]:
    """Compose low-frequency loop and high-frequency IMU rotation corrections."""
    if len(visual) != len(imu) or len(visual) != len(loop):
        raise ValueError("visual, IMU, and loop trajectories must have equal length")
    world_steps = np.diff(visual[:, :3, 3], axis=0)
    step_norms = np.linalg.norm(world_steps, axis=1)
    local_steps = np.einsum(
        "nij,nj->ni", visual[:-1, :3, :3].transpose(0, 2, 1), world_steps
    )
    forward_ratio = np.abs(local_steps[:, 2]) / np.maximum(step_norms, 1e-9)
    planar_ratio = np.linalg.norm(local_steps[:, [0, 2]], axis=1) / np.maximum(
        step_norms, 1e-9
    )
    interval_gate = sigmoid((forward_ratio - forward_center) / forward_width)
    interval_gate *= sigmoid((planar_ratio - planar_center) / planar_width)
    constrained = local_steps.copy()
    constrained[:, 0] *= 1.0 - interval_gate
    constrained[:, 1] *= 1.0 - interval_gate
    constrained_norms = np.linalg.norm(constrained, axis=1)
    movable = step_norms > 1e-9
    constrained[movable] *= (
        step_norms[movable, None]
        / np.maximum(constrained_norms[movable, None], 1e-9)
    )

    output = loop.copy()
    output[0] = loop[0]
    for index, local_step in enumerate(constrained):
        visual_relative = (
            visual[index, :3, :3].T @ visual[index + 1, :3, :3]
        )
        imu_relative = imu[index, :3, :3].T @ imu[index + 1, :3, :3]
        loop_relative = loop[index, :3, :3].T @ loop[index + 1, :3, :3]
        imu_correction = visual_relative.T @ imu_relative
        combined_relative = project_rotation(loop_relative @ imu_correction)
        output[index + 1, :3, :3] = project_rotation(
            output[index, :3, :3] @ combined_relative
        )
        output[index + 1, :3, 3] = (
            output[index, :3, 3] + output[index, :3, :3] @ local_step
        )

    path_scale = max(
        float(np.linalg.norm(loop[-1, :3, 3] - loop[0, :3, 3])),
        0.1 * float(np.sum(step_norms)),
        1e-9,
    )
    relative_path_change = float(
        np.sqrt(np.mean(np.sum((output[:, :3, 3] - loop[:, :3, 3]) ** 2, axis=1)))
        / path_scale
    )
    accepted = relative_path_change <= maximum_path_change_ratio
    if not accepted:
        output = loop.copy()
    return output, {
        "expert": "environment_gated_loop_imu" if accepted else "environment_gated_loop",
        "route_enabled": True,
        "composition_accepted": accepted,
        "composition_relative_path_change": relative_path_change,
        "composition_gate_mean": float(np.mean(interval_gate)),
        "fallback_reason": None if accepted else "composition_path_change_budget_exceeded",
    }


def compose_loop_translation_expert(
    visual: np.ndarray,
    loop: np.ndarray,
    translation: np.ndarray,
    maximum_path_change_ratio: float,
) -> tuple[np.ndarray, dict[str, float | bool | str | None]]:
    """Propagate essential-matrix local directions under loop-corrected rotations."""
    if len(visual) != len(loop) or len(visual) != len(translation):
        raise ValueError("visual, loop, and translation trajectories must align")
    visual_steps = np.diff(visual[:, :3, 3], axis=0)
    translation_steps = np.diff(translation[:, :3, 3], axis=0)
    local_steps = np.einsum(
        "nij,nj->ni",
        visual[:-1, :3, :3].transpose(0, 2, 1),
        translation_steps,
    )
    visual_norms = np.linalg.norm(visual_steps, axis=1)
    local_norms = np.linalg.norm(local_steps, axis=1)
    movable = visual_norms > 1e-9
    local_steps[movable] *= (
        visual_norms[movable, None]
        / np.maximum(local_norms[movable, None], 1e-9)
    )
    output = loop.copy()
    output[0] = loop[0]
    for index, local_step in enumerate(local_steps):
        output[index + 1, :3, 3] = (
            output[index, :3, 3] + loop[index, :3, :3] @ local_step
        )
    path_scale = max(
        float(np.linalg.norm(loop[-1, :3, 3] - loop[0, :3, 3])),
        0.1 * float(np.sum(visual_norms)),
        1e-9,
    )
    relative_path_change = float(
        np.sqrt(np.mean(np.sum((output[:, :3, 3] - loop[:, :3, 3]) ** 2, axis=1)))
        / path_scale
    )
    accepted = relative_path_change <= maximum_path_change_ratio
    if not accepted:
        output = loop.copy()
    return output, {
        "expert": (
            "environment_gated_loop_translation"
            if accepted else "environment_gated_loop"
        ),
        "route_enabled": True,
        "composition_accepted": accepted,
        "composition_relative_path_change": relative_path_change,
        "fallback_reason": None if accepted else "composition_path_change_budget_exceeded",
    }


def generate(args: argparse.Namespace) -> None:
    policy = json.loads(args.policy.read_text())
    entries = []
    for sequence in SEQUENCES:
        visual_path = args.visual_root / sequence / args.visual_name
        imu_path = args.imu_root / sequence / args.imu_name
        imu_diagnostics_path = imu_path.with_suffix(".diagnostics.json")
        loop_path = args.loop_root / sequence / args.loop_name
        gate_path = args.loop_root / sequence / "environment_gate_before_gt.json"
        descriptor_path = args.descriptor_root / sequence / args.descriptor_name
        translation_path = (
            args.translation_root / sequence / args.translation_name
            if args.translation_root is not None else None
        )
        translation_diagnostics_path = (
            args.translation_root / sequence / args.translation_diagnostics_name
            if args.translation_root is not None else None
        )
        with np.load(visual_path, allow_pickle=False) as archive:
            visual = poses_4x4(archive[args.visual_pose_key])
            frame_indices = np.asarray(archive["frame_indices"], dtype=np.int64)
        gate = json.loads(gate_path.read_text())
        if gate["route_enabled"]:
            with np.load(loop_path, allow_pickle=False) as archive:
                loop = poses_4x4(archive[args.loop_pose_key])
            if policy.get("loop_expert_mode", "loop_only") == "compose_with_translation":
                if translation_path is None or not translation_path.exists():
                    raise FileNotFoundError(
                        f"missing translation expert for {sequence}: {translation_path}"
                    )
                with np.load(translation_path, allow_pickle=False) as archive:
                    translation = poses_4x4(archive[args.translation_pose_key])
                candidate, diagnostics = compose_loop_translation_expert(
                    visual,
                    loop,
                    translation,
                    policy["loop_composition_maximum_path_change_ratio"],
                )
            elif (
                policy.get("loop_expert_mode", "loop_only") == "compose_with_imu"
                and imu_path.exists()
                and imu_diagnostics_path.exists()
            ):
                imu_diagnostics = json.loads(imu_diagnostics_path.read_text())
                if (
                    imu_diagnostics.get("accepted", False)
                    and imu_diagnostics.get("fallback_reason") is None
                ):
                    with np.load(imu_path, allow_pickle=False) as archive:
                        imu = poses_4x4(archive[args.imu_pose_key])
                    candidate, diagnostics = compose_loop_imu_expert(
                        visual,
                        imu,
                        loop,
                        policy["forward_center"],
                        policy["forward_width"],
                        policy["planar_center"],
                        policy["planar_width"],
                        policy["loop_composition_maximum_path_change_ratio"],
                    )
                else:
                    candidate = loop.copy()
                    diagnostics = {
                        "expert": "environment_gated_loop",
                        "route_enabled": True,
                        "fallback_reason": "loop_composition_imu_rejected",
                    }
            else:
                candidate = loop.copy()
                diagnostics = {
                    "expert": "environment_gated_loop",
                    "route_enabled": True,
                    "fallback_reason": None,
                }
            diagnostics["loop_gate_probability"] = gate["probability"]
        elif gate["edge_count"] > 0:
            candidate = visual.copy()
            diagnostics = {
                "expert": "visual",
                "loop_gate_probability": gate["probability"],
                "route_enabled": False,
                "fallback_reason": "loop_observed_but_drift_not_supported",
            }
        elif imu_path.exists() and imu_diagnostics_path.exists():
            imu_diagnostics = json.loads(imu_diagnostics_path.read_text())
            imu_available = (
                imu_diagnostics.get("accepted", False)
                and imu_diagnostics.get("fallback_reason") is None
            )
            if not imu_available:
                candidate = visual.copy()
                diagnostics = {
                    "expert": "visual",
                    "loop_gate_probability": gate["probability"],
                    "route_enabled": False,
                    "fallback_reason": imu_diagnostics.get(
                        "fallback_reason", "imu_diagnostics_rejected"
                    ),
                }
            else:
                with np.load(imu_path, allow_pickle=False) as archive:
                    imu = poses_4x4(archive[args.imu_pose_key])
                candidate, diagnostics = nonholonomic_imu_expert(
                    visual,
                    imu,
                    policy["forward_center"],
                    policy["forward_width"],
                    policy["planar_center"],
                    policy["planar_width"],
                    policy["imu_route_threshold"],
                    policy["forward_sign_threshold"],
                    policy["maximum_path_change_ratio"],
                )
                diagnostics["loop_gate_probability"] = gate["probability"]
                if policy.get("weak_revisit_gate", "none") != "none":
                    descriptors = np.load(descriptor_path, allow_pickle=False)
                    weak_statistics = weak_revisit_statistics(
                        visual,
                        descriptors,
                        policy["weak_revisit_minimum_separation"],
                        policy["weak_revisit_nms_radius"],
                        policy["weak_revisit_maximum_pairs"],
                        policy.get("weak_revisit_statistics_fraction", 1.0),
                    )
                    short_sequence = (
                        len(visual) <= policy["weak_revisit_short_sequence_frames"]
                    )
                    if policy["weak_revisit_gate"] == "frozen_two_branch_mlp":
                        gate_diagnostics = frozen_two_branch_gate(
                            weak_statistics,
                            float(diagnostics["relative_path_change"]),
                            policy["weak_revisit_minimum_pairs"],
                            policy["weak_revisit_minimum_score"],
                            policy["weak_revisit_maximum_path_closure"],
                            policy["appearance_minimum_score"],
                            policy["appearance_maximum_path_change"],
                            policy["weak_revisit_score_width"],
                            policy["weak_revisit_closure_width"],
                            policy["appearance_path_change_width"],
                        )
                        weak_accept = bool(
                            gate_diagnostics["weak_revisit_gate_probability"]
                            >= policy["weak_revisit_route_threshold"]
                        )
                    else:
                        gate_diagnostics = {}
                        weak_accept = weak_revisit_accepts(
                            weak_statistics,
                            policy["weak_revisit_minimum_score"],
                            policy["weak_revisit_maximum_path_closure"],
                            policy["weak_revisit_minimum_pairs"],
                        )
                    short_sequence_override = bool(
                        short_sequence
                        and policy["weak_revisit_allow_short_sequence"]
                    )
                    diagnostics.update(weak_statistics)
                    diagnostics.update(gate_diagnostics)
                    diagnostics["weak_revisit_accept"] = weak_accept
                    diagnostics["weak_revisit_short_sequence_override"] = (
                        short_sequence_override
                    )
                    if (
                        diagnostics["expert"] == "nonholonomic_imu"
                        and not weak_accept
                        and not short_sequence_override
                    ):
                        candidate = visual.copy()
                        diagnostics["expert"] = "visual"
                        diagnostics["route_enabled"] = False
                        diagnostics["fallback_reason"] = (
                            "weak_revisit_consistency_rejected"
                        )
        else:
            candidate = visual.copy()
            diagnostics = {
                "expert": "visual",
                "loop_gate_probability": gate["probability"],
                "route_enabled": False,
                "fallback_reason": "missing_imu",
            }
        output_dir = args.output_dir / sequence
        output_dir.mkdir(parents=True, exist_ok=True)
        candidate_path = output_dir / "candidate_before_gt.npz"
        np.savez_compressed(
            candidate_path,
            c2w=candidate,
            frame_indices=frame_indices,
            selected_expert=np.asarray(diagnostics["expert"]),
        )
        diagnostics_path = output_dir / "router_before_gt.json"
        diagnostics_path.write_text(json.dumps(diagnostics, indent=2) + "\n")
        entries.append({
            "sequence": sequence,
            "visual": str(visual_path),
            "visual_sha256": sha256(visual_path),
            "imu": str(imu_path) if imu_path.exists() else None,
            "imu_sha256": sha256(imu_path) if imu_path.exists() else None,
            "imu_diagnostics": (
                str(imu_diagnostics_path) if imu_diagnostics_path.exists() else None
            ),
            "imu_diagnostics_sha256": (
                sha256(imu_diagnostics_path)
                if imu_diagnostics_path.exists() else None
            ),
            "loop": str(loop_path),
            "loop_sha256": sha256(loop_path),
            "loop_gate": str(gate_path),
            "loop_gate_sha256": sha256(gate_path),
            "descriptor": str(descriptor_path),
            "descriptor_sha256": sha256(descriptor_path),
            "translation": (
                str(translation_path) if translation_path is not None else None
            ),
            "translation_sha256": (
                sha256(translation_path)
                if translation_path is not None and translation_path.exists()
                else None
            ),
            "translation_diagnostics": (
                str(translation_diagnostics_path)
                if translation_diagnostics_path is not None else None
            ),
            "translation_diagnostics_sha256": (
                sha256(translation_diagnostics_path)
                if translation_diagnostics_path is not None
                and translation_diagnostics_path.exists()
                else None
            ),
            "candidate": str(candidate_path),
            "candidate_sha256": sha256(candidate_path),
            "router": str(diagnostics_path),
            "router_sha256": sha256(diagnostics_path),
        })
        print(json.dumps({"sequence": sequence, **diagnostics}), flush=True)
    manifest = {
        "protocol": "all expert-router candidates generated and hashed before GT load",
        "policy": policy,
        "policy_path": str(args.policy.resolve()),
        "policy_sha256": sha256(args.policy),
        "entries": entries,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "manifest_before_gt.json").write_text(
        json.dumps(manifest, indent=2) + "\n"
    )


def evaluate(args: argparse.Namespace) -> None:
    from vistream.metrics import ate_rmse

    manifest = json.loads((args.output_dir / "manifest_before_gt.json").read_text())
    if manifest["policy"] != json.loads(args.policy.read_text()):
        raise ValueError("evaluation policy differs from generation policy")
    rows = []
    for entry in manifest["entries"]:
        candidate_path = Path(entry["candidate"])
        router_path = Path(entry["router"])
        if sha256(candidate_path) != entry["candidate_sha256"]:
            raise ValueError(f"candidate changed after generation: {candidate_path}")
        if sha256(router_path) != entry["router_sha256"]:
            raise ValueError(f"router changed after generation: {router_path}")
        for path_key, hash_key in (
            ("visual", "visual_sha256"),
            ("imu", "imu_sha256"),
            ("imu_diagnostics", "imu_diagnostics_sha256"),
            ("loop", "loop_sha256"),
            ("loop_gate", "loop_gate_sha256"),
            ("descriptor", "descriptor_sha256"),
            ("translation", "translation_sha256"),
            ("translation_diagnostics", "translation_diagnostics_sha256"),
        ):
            if entry.get(path_key) is None:
                continue
            source = Path(entry[path_key])
            if sha256(source) != entry[hash_key]:
                raise ValueError(f"source changed after generation: {source}")
        with np.load(candidate_path, allow_pickle=False) as archive:
            candidate = poses_4x4(archive["c2w"])
        # GT is opened only after every candidate and route decision is hashed.
        with np.load(entry["visual"], allow_pickle=False) as archive:
            visual = poses_4x4(archive[args.visual_pose_key])
            target = poses_4x4(archive[args.ground_truth_key])
        visual_metric = ate_rmse(visual[:, :3, 3], target[:, :3, 3], "sim3")
        candidate_metric = ate_rmse(candidate[:, :3, 3], target[:, :3, 3], "sim3")
        router = json.loads(router_path.read_text())
        rows.append({
            "sequence": entry["sequence"],
            "expert": router["expert"],
            "visual_ate_rmse_m": visual_metric["ate_rmse"],
            "candidate_ate_rmse_m": candidate_metric["ate_rmse"],
        })
    visual = np.asarray([row["visual_ate_rmse_m"] for row in rows])
    candidate = np.asarray([row["candidate_ate_rmse_m"] for row in rows])
    summary = {
        "protocol": manifest["protocol"],
        "policy": manifest["policy"],
        "rows": rows,
        "macro": {
            "visual_ate_rmse_mean_m": float(visual.mean()),
            "candidate_ate_rmse_mean_m": float(candidate.mean()),
            "improved_sequences": int(np.sum(candidate < visual)),
            "sequence_count": len(rows),
        },
    }
    (args.output_dir / "summary_after_gt.json").write_text(
        json.dumps(summary, indent=2) + "\n"
    )
    print(json.dumps(summary, indent=2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    generate_parser = subparsers.add_parser("generate")
    generate_parser.add_argument("--visual-root", type=Path, required=True)
    generate_parser.add_argument("--imu-root", type=Path, required=True)
    generate_parser.add_argument("--loop-root", type=Path, required=True)
    generate_parser.add_argument("--descriptor-root", type=Path, required=True)
    generate_parser.add_argument("--translation-root", type=Path)
    generate_parser.add_argument("--output-dir", type=Path, required=True)
    generate_parser.add_argument("--policy", type=Path, required=True)
    generate_parser.add_argument("--visual-name", default="trajectory.npz")
    generate_parser.add_argument("--imu-name", default="candidate_before_gt.npz")
    generate_parser.add_argument("--loop-name", default="candidate_before_gt.npz")
    generate_parser.add_argument(
        "--descriptor-name", default="resnet50_imagenet.npy"
    )
    generate_parser.add_argument(
        "--translation-name", default="candidate_before_gt.npz"
    )
    generate_parser.add_argument(
        "--translation-diagnostics-name", default="direction_gate_before_gt.json"
    )
    generate_parser.add_argument("--visual-pose-key", default="predicted_c2w")
    generate_parser.add_argument("--imu-pose-key", default="fused_c2w")
    generate_parser.add_argument("--loop-pose-key", default="c2w")
    generate_parser.add_argument("--translation-pose-key", default="c2w")
    evaluate_parser = subparsers.add_parser("evaluate")
    evaluate_parser.add_argument("--output-dir", type=Path, required=True)
    evaluate_parser.add_argument("--policy", type=Path, required=True)
    evaluate_parser.add_argument("--visual-pose-key", default="predicted_c2w")
    evaluate_parser.add_argument("--ground-truth-key", default="ground_truth_c2w")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    generate(args) if args.command == "generate" else evaluate(args)
