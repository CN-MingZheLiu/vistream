# Setup and external dependencies

This source archive intentionally excludes weights, compiled binaries and the 6.54 GB offline environment. It is useful for inspecting and building the implementation; it is not sufficient by itself for independent end-to-end inference.

## Recorded baseline

The earlier full-bundle execution used Ubuntu 22.04 x86-64, glibc 2.35, Python 3.10.20, PyTorch 2.8.0+cu128, NumPy 2.2.6, SciPy 1.15.3 and an OpenCV Python build reporting 5.0.0. NVIDIA driver 550.127.08 and H20-3e GPUs were used. These are observed versions, not a claim that an arbitrary environment with the same version labels is binary-equivalent. See `provenance/VALIDATED_RUNTIME_VERSIONS.json`.

Install a compatible NVIDIA driver, CUDA-enabled PyTorch and CUDA build toolchain on Linux. Native extensions include DPVO's CUDA correlation/bundle-adjustment modules, DROID/LieTorch backends, and native ORB-SLAM3 with its OpenCV/Pangolin dependencies. A macOS source inspection does not validate these Linux CUDA components.

## Python source and package metadata

The release retains upstream dependency and installation files:

- `source/original_project/lingbot-map/pyproject.toml`
- `deps/DPVO/environment.yml` and `deps/DPVO/setup.py`
- `source/third_party/DROID-SLAM/requirements.txt`, `requirements_frozen.txt`, environment files and `setup.py`
- `source/third_party/DROID-SLAM/thirdparty/lietorch/setup.py`
- `source/third_party/pypose/setup.py` and its source metadata

These upstream environment specifications may describe older upstream environments, such as DROID's original PyTorch 1.10/CUDA 11.3 configuration; they are preserved verbatim and must not be confused with the validated runtime above. `deps/source_provenance.json`, the DPVO compatibility patch, and native source records identify the actual archived source states.

`requirements-core.txt` provides common pure-Python/array dependencies used by the core and backbone. It is a convenience list, not a complete environment lock. It does not install CUDA-enabled PyTorch, OpenCV, native extensions, model weights or system development headers.

After preparing an appropriate Python/CUDA environment, the following are source installation steps, **not a complete inference setup or validated clean-build recipe**:

```bash
python -m pip install -r requirements-core.txt
python -m pip install -e source/third_party/pypose
python -m pip install -e source/original_project/lingbot-map
```

Build DPVO, DROID and LieTorch using their included setup/build instructions for that environment. Review compiler/CUDA compatibility rather than assuming that the old upstream environment files are interchangeable. Add the core project path when invoking its scripts:

```bash
export PYTHONPATH="$PWD/source/original_project/VIStream${PYTHONPATH:+:$PYTHONPATH}"
```

The VIStream core tree had no top-level installation metadata in the source archive, so this release preserves its existing import layout rather than inventing a new package installer.

## Required external assets

The exact omitted paths, sizes and available recorded SHA256 values are in `provenance/EXTERNAL_ASSETS.json`. Major assets include:

| Required asset | Original package location |
|---|---|
| LingBot-Map long checkpoint | `runtime/weights/lingbot-map-long.pt` |
| DPVO checkpoint | `deps/weights/dpvo.pth` |
| DROID checkpoint and identical mirror | `deps/weights/droid.pth`, `deps/weights/droid.mirror.pth` |
| ResNet50 checkpoint | `deps/torch/hub/checkpoints/resnet50-11ad3fa6.pth` |
| ORB vocabulary and native binaries | `native_stage/prebuilt/ORB_SLAM3/` |
| OpenCV/Pangolin native libraries | `native_stage/prebuilt/` |
| Original offline environment, for the unchanged bootstrap route | `runtime/environment/lingbot-map-linux-x86_64.tar.gz` |
| Runtime shared-library closure and namespace utilities | `runtime/lib/`, `tools/` |

Obtain weights from the original experiment owner or their applicable upstream distributions and verify their recorded hashes when reproducing that configuration. A similarly named or newer checkpoint is not evidence of equivalence. Respect each asset's distribution terms. No implicit download, model retraining or replacement checkpoint is provided by this source package.

Calibration and loop-off/on camera settings are included, but the original images, IMU and per-sequence KITTI calibration/time files must be supplied separately. Ground truth belongs only to the scoring stage. See `DATASET_LAYOUT_EN.md`.

## Native source

`native_stage/vendor/native_sources.tar.gz` contains the recorded build source for ORB-SLAM3, OpenCV 4.5.5 and Pangolin, with the original exclusion/source manifests. Source verification and optional building are described in `native_stage/README.txt`.

The native builder requires a separately supplied vocabulary, CMake, patch, a compatible C/C++ compiler, make, Eigen3, Boost serialization, OpenSSL, OpenGL/GLEW, JPEG/PNG and related development headers. This source-only archive does not supply those system development packages. A source build may produce different binary hashes; it requires its own runtime manifest and validation and must not be relabelled as the original archived binaries.

## Original strict runner

The original `scripts/bootstrap.py`, `scripts/run_kitti.py`, `native_stage/runtime.json`, asset hashes and isolated namespace contract are preserved. They expect the full original resource tree. Merely installing Python packages does not satisfy that contract.

If the exact excluded assets are separately restored at their recorded paths, the original entry point is:

```bash
python3 scripts/bootstrap.py
python3 scripts/run_kitti.py \
  --kitti-root /path/to/kitti_odometry/dataset \
  --imu-root /path/to/KITTI_ODO_IMU \
  --run-root /path/to/new_run \
  --gpus 0 \
  --native-workers 1
```

These commands are intentionally conditional on separately supplying the runtime and checkpoints. Every run directory must be new. The runner does not consume old trajectories, and GT is not mounted during generation.

The full run seals all 132 stage receipts and 33 predictions before independent scoring. Only after that seal exists:

```bash
runtime/envs/lingbot-map/bin/python evaluation/verify_results.py evaluate \
  --run-root /path/to/new_run \
  --gt-root /path/to/kitti_odometry/dataset/poses \
  --report-dir /path/to/new_report
runtime/envs/lingbot-map/bin/python evaluation/verify_results.py verify \
  --run-root /path/to/new_run \
  --report-dir /path/to/new_report
```

Evaluation uses per-trajectory Sim(3) ATE on the fixed stride-six-plus-final-frame support. A successful evaluation does not imply exact numerical reproduction. `NOT_REPRODUCED` must remain reported when the actual run differs from the fixed target.
