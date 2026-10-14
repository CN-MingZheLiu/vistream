# Portable evaluator interface, v1

`verify_results.py seal --run-root RUN` validates the entire run and writes
`PREDICTIONS_SEALED_BEFORE_GT.json` using exclusive creation. No GT is read.
`verify_results.py evaluate --run-root RUN --gt-root /external/poses --report-dir NEW`
scores all 33 predictions, only after checking that seal. `verify` checks the
score report and reports numerical agreement separately from execution.

Root runner writes `GENERATION_CONTRACT.json` after all generation finishes:

```json
{
  "schema": "vistream-complete-generation-v1",
  "mode": "fresh_from_sensors",
  "run_id": "unique-run-id",
  "sequences": ["00","01","02","03","04","05","06","07","08","09","10"],
  "gt_evaluation_performed": false,
  "historical_prediction_dependencies": [],
  "stage_receipts": {"path":"stage_receipts.json","sha256":"..."},
  "source_manifest": {"path":"SOURCE_MANIFEST.json","sha256":"..."},
  "access_audit": {"path":"GENERATION_ACCESS_AUDIT.json","sha256":"..."},
  "predictions": [
    {"sequence":"00","method":"visual","stage":"visual","path":"visual/00/trajectory.npz","pose_key":"predicted_c2w","frame_indices_key":"frame_indices","sha256":"..."},
    {"sequence":"00","method":"loop_off","stage":"fusion_off","path":"upstream/00/fusion_off/00/candidate_before_gt.npz","pose_key":"c2w","frame_indices_key":"frame_indices","sha256":"..."}
  ]
}
```

Predictions must contain exactly visual/loop_off/loop_on for all 11 sequences.
All paths above and in artifacts are relative to RUN, without symlinks or `..`.
Every prediction must also occur in its producing stage's artifacts. Each NPZ
must contain finite [N,4,4] poses (or [N,3,4], expanded without other changes)
and exact int64-compatible stride6+last frame indices. No interpolation/trim.

`stage_receipts.json` is an array of exactly 132 final successful/admitted
receipts: each sequence crossed with visual, raw_imu, loop_geometry,
translation, router, dpvo, droid, base, orb_off, fusion_off, orb_on, fusion_on.
Preserve unsuccessful attempts separately; do not substitute or overwrite them.
Each final receipt has this exact normalized interface (extra fields retained):

```json
{
  "sequence":"00", "stage":"visual", "status":"SUCCEEDED", "returncode":0,
  "run_id":"unique-run-id", "started_unix":1.0, "finished_unix":2.0,
  "argv":["..."],
  "inputs":[{"kind":"sensor_manifest","path":"provenance/sensors.json","sha256":"..."}],
  "artifacts":{"visual/00/trajectory.npz":"..."},
  "log":{"path":"logs/00_visual.log","sha256":"..."},
  "isolation":{"gt_visible":false,"historical_predictions_visible":false,"network_disabled":true}
}
```

Allowed input kinds: sensor_manifest, checkpoint_manifest, code_manifest,
runtime_manifest, policy, generated_artifact. All inputs refer to sealed files
under RUN, including manifests of external read-only resources. Any generated
artifact input must be an output of an earlier stage in this same run. Resource
manifests must not include historical_prediction, cached_prediction, ground_truth
or precomputed_trajectory roles. Keep original external paths and hashes in the
manifests for provenance; the evaluator does not re-read raw sensor files.

Only orb_off/orb_on may have `status: "UNAVAILABLE"`, wrapper returncode 0 or 1,
and the additional `unavailable_evidence` object below. Setup, timeout, signal,
missing raw exports or native nonzero exit cannot be admitted. Outputs from a
configured expert's normal no-use branch still count as SUCCEEDED if generated
in this run and the original stage command exits 0.

```json
{
  "native_returncode":0,
  "native_completed":true,
  "export_completed":true,
  "validation_status":"REJECTED",
  "reason":"timestamps_not_strictly_increasing",
  "receipt":{"path":"upstream/00/orb_on/00/native_stage_receipt.json","sha256":"..."}
}
```

Only the above timestamp-rejection reason is currently authorized; add any new
reason only after documenting and reviewing its real evidence, never to hide
setup/inference failure. The native receipt uses the same six evidence fields
(native_returncode, native_completed, export_completed, validation_status,
validation_reason, status="UNAVAILABLE") or a normalized evidence file preserving the
original native receipt and its SHA. Empty keyframe export alone is not failure
if the original parser does not require it; preserve zero-byte files.
The native receipt's raw_artifacts must record run.log, FrameTrajectory.txt and
KeyFrameTrajectory.txt as objects with path/sha256/bytes. All three exact files
must also occur in this stage's artifacts and are sealed; only keyframes may be
empty. Absolute container paths in the native receipt are matched by unique
basename plus SHA within that same stage, without reading those external paths.

`SOURCE_MANIFEST.json`:

```json
{"schema":"vistream-source-manifest-v1","mode":"fresh_from_sensors",
 "historical_prediction_dependencies":[],
 "resources":[{"role":"checkpoint","path":"external/or/bundle/path","sha256":"..."}]}
```

Fresh resource roles are sensor, checkpoint, code, runtime, policy. Do not list
precomputed visual/base trajectories as checkpoints or sensors. A reference
recompute uses mode="reference_recompute", requires no generation receipts,
and may list reference_prediction resources. It is labeled reference-only and
cannot return a fresh reproduction success.

`GENERATION_ACCESS_AUDIT.json`:

```json
{"schema":"vistream-generation-access-audit-v1","run_id":"unique-run-id",
 "status":"PASS","evidence_mode":"namespace_and_trace",
 "stage_coverage":[{"sequence":"00","stage":"visual"}],
 "namespace_checks_complete":true,"forbidden_inputs":[],"issues":[],
 "unresolved_records":[],"evidence_files":{"logs/00_visual.strace":"..."}}
```

stage_coverage must equal all 132 stages; evidence_files must be nonempty and
each file nonempty. Accepted evidence modes: namespace_and_trace or
namespace_mounts. State the actual mode; do not call mount-only evidence a syscall
audit. Missing coverage or observed forbidden inputs fails. Unresolved records
require REVIEW_REQUIRED; scoring then needs explicit
`--allow-audit-review-required`, and verification cannot call it fully reproduced.

All source/receipt/audit/input/artifact/log files are hashed into the seal. The
evaluator validates all 33 supports and source metric SHA before the first GT
open, reads external KITTI poses/00.txt ... poses/10.txt, and checks every seal
hash again after scoring. GT file hashes must equal the original recorded KITTI
dataset hashes in target_results.json; no GT arrays are distributed here.
Numerical comparison uses fixed absolute 1e-8 m only;
two-decimal agreement is additionally shown and never substitutes for full
precision agreement. Any mismatch yields NOT_REPRODUCED (nonzero exit), even if
the fresh result has lower ATE. Runtime nondeterminism is an acknowledged limit.
