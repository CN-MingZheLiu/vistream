# Dataset layout


### Camera images, times and calibration

`--kitti-root` must point to the directory containing `sequences/`:

```text
/path/to/kitti_odometry/dataset/
  sequences/
    00/
      image_2/000000.png
      image_2/000001.png
      ...
      times.txt
      calib.txt
    01/...
    ...
    10/...
  poses/                  # Optional during generation; mounted only for scoring
    00.txt
    ...
    10.txt
```

Use complete `image_2` colour-left-camera sequences 00–10. PNG names must be consecutively numbered from `000000.png`, and the image count must equal the number of data rows in `times.txt`. **Do not subsample the images in advance.** The visual stage samples every sixth frame and retains the final frame internally; DPVO, DROID and ORB require the full frame stream.

`times.txt` must preserve the original camera timestamps in seconds, one row per image in the same order. Each sequence needs its own original `calib.txt`, including the required `P2` camera projection and `Tr` transform. The generation namespace exposes only the required image, time and calibration inputs; it does not expose the `poses/` directory, even when it is stored under the dataset root.

### Original IMU

`--imu-root` points to a directory with this structure, not to a single CSV file:

```text
/path/to/KITTI_ODO_IMU/
  kitti_odo_00/imu.csv
  kitti_odo_01/imu.csv
  kitti_odo_02/imu.csv
  # No kitti_odo_03/imu.csv in this target configuration
  kitti_odo_04/imu.csv
  ...
  kitti_odo_10/imu.csv
```

The CSV format is the KITTI/OXTS field layout used by this experiment: comma-separated numeric records with at least 24 columns and no uncommented text header. The implementation reads these **zero-based** columns:

| Array selection | Meaning | Units |
|---|---|---|
| `raw[:, 0]` | Timestamp | Seconds, on the camera `times.txt` time basis |
| `raw[:, 15:18]` | Three-axis acceleration | m/s² |
| `raw[:, 21:24]` | Three-axis angular velocity | rad/s |

Preserve the original KITTI axis convention. The loader does not automatically convert milliseconds, microseconds or nanoseconds, and it does not infer time offsets from ground truth. A seven-column `[time, ax, ay, az, gx, gy, gz]` CSV is **not** a compatible substitute.

The implementation sorts and deduplicates IMU times and checks camera-stream coverage. It applies only the small endpoint holds permitted by the original implementation; significant coverage gaps are errors. Use the matching IMU recording for each sequence. Do not copy another sequence, insert zeros, or alter the sampling to bypass validation.

Sequence 03 intentionally has no matching IMU in this target condition. Its `raw_imu` stage records `missing_matching_imu` and retains the visual candidate; the other stages still execute. The full runner rejects an added `kitti_odo_03/imu.csv`, because that would change the experiment. Missing IMU for any of the other ten sequences is an error.

The required IMU-to-Velodyne extrinsics for the two recording dates and ORB camera settings are bundled. Each sequence still requires its own `calib.txt`.

### Ground truth for later scoring

`--gt-root` points directly to the directory containing the original `poses/00.txt` through `poses/10.txt`. Each line has the 12 row-major entries of a KITTI 3×4 pose matrix. Preserve the complete frame support and original files: do not subsample, reorder, change precision, or resave them. Evaluation verifies their SHA256 hashes.

Ground truth is not required to generate and seal predictions. It is required to calculate ATE and compare the run with the numerical reference.

