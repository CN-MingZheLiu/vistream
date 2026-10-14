VIStream repaired KITTI native stage

Purpose
This module packages the exact shutdown-only native patch and the exact final
reference-aware geometry admission used by the completed side experiment.
It does not promise that fresh asynchronous ORB runs reproduce 15.08/10.41 m
exactly. Those are archived observations, not constants used by this code.
This target differs from the historical paper's 16.52/9.53 configuration.
No previous trajectory, dataset or ground truth is included in native_stage.

Runtime
Use the complete package's Linux Python environment (NumPy, SciPy and PyTorch;
the original vistream package imports torch even for metrics). Prebuilt native
binaries target the original Linux x86-64 environment. System dependency closure
and sandbox aliases are provided by the complete package runner. runtime.json
uses /bundle/native_stage as its execution prefix; make_runtime.py can create
a new hashed record for another location. Do not edit the recorded SHA values.

Generate native ORB, one sequence and one loop mode:
  python /bundle/native_stage/run_native.py \
    --project-root /bundle/source/original_project/VIStream \
    --runtime /bundle/native_stage/runtime.json \
    --sequence-dir /sensors/sequences/01 \
    --trajectory /work/01/visual/trajectory.npz \
    --settings <the original KITTI off or on YAML> \
    --output-dir /work/01/orb_off/01

Fuse that run with its newly generated base:
  python /bundle/native_stage/run_fusion.py \
    --project-root /bundle/source/original_project/VIStream \
    --sequence 01 --base-root /work/01/base \
    --orb-root /work/01/orb_off --output-dir /work/01/fusion_off

The base manifest must contain exactly the requested sequence, using the
unchanged original schema. All referenced visual/base/diagnostic SHA hashes
are verified. Support NPZ files must contain no GT members. The outer runner
must additionally prevent historical/reference/data-GT paths from entering the
generation namespace; a receipt by itself is not an access-control mechanism.
Both output roots must be new. Inputs remain unmodified.

Receipts and actual failures
native_stage_receipt.json includes status SUCCEEDED, UNAVAILABLE or FAILED;
native_returncode; native_completed/export_completed; validation_status and
validation_reason; actual raw files and SHA256; input/runtime/source hashes.
The original native export must finish with return code zero and a completed
shutdown barrier before frame export. Original mathematical validator failures
are preserved, without timestamp deduplication, pose interpolation or rewritten
raw files. A completed export with non-increasing timestamps is explicitly
UNAVAILABLE and the original fusion's fallback is allowed. Other recognized
validator errors are recorded as unavailable by generation, but the complete
strict package verifier may admit only the declared timestamp failure class.
Setup errors, library mismatches, native crashes and timeouts are FAILED and
cannot be converted into model fallback. Empty keyframe files are retained
byte-for-byte; the original validator decides admission from the frame export.
The external process-group timeout defaults to 7200 seconds and kills that
native process group if exceeded. It does not change a successful trajectory.

Unchanged mathematical code
admission_reference.py, geometry_admission_local.py and geometry_admission.py
are byte-for-byte copies of the final side policy, checked at runtime:
  global R2 >= 0.75; minimum spatial variance 1e-12;
  16-increment windows; at least 5 nondegenerate windows;
  local ratio q90/q10 <= 2.0.
These checks apply only when the current base router selected its native DPVO
motion head with route_enabled=true. No per-sequence rule, GT error or historical
result is consulted. Admitted experts retain the original mixture policy.
The target is an offline development safeguard after benchmark outcomes were
observed; it is not an unseen-test protocol or the original paper configuration.

Source and rebuilding
vendor/native_sources.tar.gz contains the working-tree build sources of the
exact ORB-SLAM3, OpenCV 4.5.5 and Pangolin repositories, including ORB's modified
monocular entry point and DBoW2/g2o/Sophus. Its per-file manifest records Git
commit, tracked patch SHA and explicit excluded example sensor tables/images,
example datasets/models and vocabulary. Vocabulary is separately included as
a required prebuilt runtime asset. The optional Pangolin pybind11 submodule is excluded because this target sets
BUILD_PANGOLIN_PYTHON=OFF. No original .o files are needed for the rebuild. target_source/ preserves the original shutdown patch, the exact patched
System.cc, original side build record and source scripts for provenance.

Source extraction and exact patch check, without compilation:
  python native_stage/build_from_sources.py --work-root <new directory> \
    --vocabulary <ORBvoc.txt> --plan-only

Full Linux build (new tree; conservative single-job default):
  python native_stage/build_from_sources.py --work-root <new directory> \
    --vocabulary <ORBvoc.txt> --jobs 1
Requires CMake, patch, GCC-compatible C/C++ compiler, make, Eigen3, Boost
serialization, OpenSSL, OpenGL/GLEW, JPEG/PNG and related system development
headers. The complete bundle runtime/build environment supplies the declared
system packages. Source rebuild writes a new NATIVE_BUILD.json and runtime.json.
The build uses the archived build choices; compiler, CPU architecture,
discovered optional system libraries and thread scheduling may change the
compiled bytes or trajectories. Source verification is distinct from a fully
executed native rebuild; see tests evidence for the performed validation scope.

Known target shutdown limitations (not silently changed)
The patch waits for LocalMapping/LoopClosing completion and isRunningGBA, then
joins the two named worker threads. It does not join every global-BA worker;
existing detached GBA behavior and a potential original lock-order issue remain.
The internal 600-second check is not a hard guarantee if a status getter blocks.
The external timeout limits hung native execution. Viewer use is disabled in the
KITTI CLI. This packaging intentionally does not redesign concurrency, tracking,
optimization, map reset or keyframe rules to obtain different scores.

Tests
  python native_stage/tests/test_portable_stage.py --project-root <VIStream> \
    --report <new synthetic report.json>
Uses generated synthetic poses and temporary files only. Covers uniform scale,
nonuniform progress, native-head applicability, unavailable fallback, rejection
of duplicate timestamps without modifying raw files, export order, and two
complete fusion CLI branches. It does not execute native inference or claim a
full source rebuild. Exact target hashes are checked before these tests.
