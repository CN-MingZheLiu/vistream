# VIStream — core source release

This release provides the VIStream implementation, fixed experiment configuration, inference orchestration, evaluation code, and relevant third-party build source. Selected implementation files are unchanged from the archived KITTI reproduction package.

**This is a source-only release.** It excludes model checkpoints, the offline environment, compiled CUDA/native libraries, datasets and predicted trajectories. It is not a standalone inference bundle. The unchanged strict runner requires separately provisioned, verified runtime assets before it can run.

## Start here

1. Read [setup and dependencies](docs/SETUP_EN.md).
2. Prepare the original images and IMU according to [the dataset layout](docs/DATASET_LAYOUT_EN.md).
3. Inspect the core modules and fixed configuration below. Use the existing runner only after restoring its required runtime contract.

| Directory | Contents |
|---|---|
| `source/original_project/VIStream/vistream/` | Geometry, IMU processing, model interfaces and motion integration |
| `source/original_project/VIStream/configs/` | Original fixed experiment and model configurations |
| `source/original_project/VIStream/scripts/` | Expert inference, fusion, training and analysis entry points |
| `source/original_project/lingbot-map/` | Frozen visual backbone source and original package metadata |
| `deps/DPVO/` | DPVO source, native-extension setup and original dependency specifications |
| `source/third_party/DROID-SLAM/` | DROID-SLAM, LieTorch and supporting build source |
| `source/third_party/pypose/` | PyPose source and package metadata |
| `deps/calibration/`, `deps/orb_settings/` | Calibration assets and loop-off/on camera settings |
| `native_stage/` | Native export repair, admission, fusion and build scripts |
| `native_stage/vendor/native_sources.tar.gz` | Archived ORB-SLAM3, OpenCV 4.5.5 and Pangolin build source; no prebuilt binaries |
| `scripts/` | Original full-bundle bootstrap, isolated KITTI runner and stage orchestration |
| `evaluation/` | Original metric implementation, generation contract and result verifier |
| `provenance/` | Selection hashes, omitted asset requirements, dependency provenance and prior-run score evidence |
| `licenses/` | Retained notices for included third-party source |

Original source may contain historical comments, filenames, absolute development paths and upstream documentation. These are preserved for source fidelity; current release instructions are in English. The root `ASSET_MANIFEST.json` is the original full-bundle manifest, **not a claim that every runtime asset is included here**. The source release has its own `SOURCE_RELEASE_MANIFEST.json`.

## Runtime entry points

`scripts/run_kitti.py` orchestrates fresh KITTI generation. `scripts/bootstrap.py` restores the former complete package's environment archive. Their source is retained unchanged, but **neither assembles the omitted assets from this ZIP**. In particular, bootstrap cannot restore an archive that is not present, and the runner deliberately rejects missing or mismatched assets rather than weakening its integrity checks.

The core Python package can be studied or used after installing its dependencies by adding `source/original_project/VIStream` to `PYTHONPATH`. Original per-stage scripts accept their own paths and configuration; inspect their `--help` output in a prepared Linux environment. This release does not add a new deployment API or promise one-command source installation.

## Experiment scope and evidence

The source reflects the **12 October 2026 repaired KITTI configuration**. Its final admission includes an offline whole-sequence consistency check. It differs from the paper's original frozen configuration.

The earlier complete binary/environment bundle ran all KITTI 00–10 sequences from original sensor inputs: 132 successful stages, 22 native ORB jobs and 33 predictions. Its independent mean Sim(3) ATE was:

| Method | Mean ATE (m) |
|---|---:|
| LingBot-MAP | 32.49 |
| VIStream, loop off | 15.25 |
| VIStream, loop on | 9.10 |

That full-bundle run did not exactly reproduce the earlier repaired reference of **15.08 / 10.41 m**; its verification status was `NOT_REPRODUCED`. These are prior execution records, not a new GPU validation of this source release. Custom CUDA operations and native thread scheduling can yield different results. Do not substitute these numbers for the paper's original table or assert exact reproduction.

`provenance/FULL_BUNDLE_VALIDATION.json` and `provenance/FULL_BUNDLE_EVALUATION.json` retain the recorded scope and full-precision scores. No historical trajectories are included as inference inputs.

## Validation of this release

Run the CPU-only static check:

```bash
python3 validate_source.py
```

It verifies source hashes, parses selected Python source and reports imported module names without importing model code or running inference. See `validation/SOURCE_STATIC_VALIDATION.json` for the performed checks and their scope. ZIP CRC and per-file SHA256 checks are reported separately with the release archive.

Source and native compilation are distinct checks. A clean rebuild of all native projects and a fresh GPU run were not performed for this source-only packaging step.

Retain each component's original license and attribution. No blanket license grant is added for differently licensed upstream components. The paper and original full archive were not changed.
