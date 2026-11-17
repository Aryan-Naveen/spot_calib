# spot_calib

Extrinsic calibration tools for a Boston Dynamics Spot carrying extra sensors. Each tool produces
an entry you paste straight into a `calibration.yaml` file read by a static-transform broadcaster
(the format used by `dcist_launch_system/platforms/<platform>/calibration.yaml`).

| Tool | Solves | Input | Output frame pair |
|---|---|---|---|
| `spot-calib cam-zed` | ZED stereo camera ↔ Spot front body cameras | live checkerboard captures, approved one at a time in a web GUI | `<robot>/frontleft` → `<robot>_zed_camera_link` |
| `spot-calib lidar-handeye` | LiDAR (Livox Mid-360) ↔ Spot body | a ROS 2 bag of the robot holding several poses | `<robot>/body` → `<robot>/livox` |

`cam-zed` talks to the ZED SDK and the Spot SDK directly; it needs no ROS. `lidar-handeye` only
uses ROS 2 to read the bag.

---

## Install (on the robot computer)

```bash
git clone git@github.com:Aryan-Naveen/spot_calib.git
cd spot_calib

# Use an existing venv that already has bosdyn-client (e.g. the dcist spark_env) or make one:
python3 -m venv --system-site-packages ~/envs/spot_calib && source ~/envs/spot_calib/bin/activate

pip install -e ".[spot,lidar,dev]"          # numpy scipy opencv flask pyyaml bosdyn-client small_gicp pytest
python /usr/local/zed/get_python_api.py    # installs pyzed into the *active* venv (only needed for cam-zed)
```

`small_gicp` is optional. Without it, `lidar-handeye` falls back to a slower point-to-plane ICP
written in scipy, which gives the same results.

Spot credentials are read from `ADT4_BOSDYN_IP`, `ADT4_BOSDYN_USERNAME` and `ADT4_BOSDYN_PASSWORD`.
You can override them with `--spot-ip/--spot-user/--spot-password`. The tools only read images and
robot state and never take a lease, so the tablet keeps control of the robot.

Check the install without hardware:

```bash
source /opt/ros/jazzy/setup.bash   # only needed for the bag round-trip test
pytest -q
```

---

## 1. ZED ↔ Spot cameras (`cam-zed`)

### How it works
1. The ZED (rectified left image) and Spot's `frontleft_fisheye_image` / `frontright_fisheye_image`
   capture the same checkerboard.
2. Each capture grabs a ZED frame, then the Spot images, then another ZED frame. If the board moved
   more than 1 px between the two ZED frames, the capture is rejected, so the views are effectively
   simultaneous.
3. Corners are detected with OpenCV and put into a **canonical order** using the colour of the board's
   corner squares. Without this, two cameras can number the corners 180° apart and silently corrupt
   the result.
4. Each camera's board pose comes from PnP. A joint bundle adjustment then refines every Spot-camera ← ZED
   extrinsic together with all board poses, minimising pixel reprojection error in every camera.
   - **independent**: a separate extrinsic for each Spot camera. The frontleft and frontright
     estimates are then compared in the body frame as a consistency check.
   - **joint**: one `body_T_zed`, using Spot's factory `body_T_camera` taken from the image's
     transform snapshot.
5. On export, the optical-frame result is converted to the frames the launch uses:
   `frontleft_T_zed_camera_link = frontleft_T_frontleft_fisheye (Spot snapshot) · fisheye_T_zed_left_optical (solved) · zed_left_optical_T_zed_camera_link`.

### One-time check: ZED camera_link offset
The last factor above comes from the zed-ros2-wrapper URDF. This repo ships zed2i defaults for it,
but they were not checked against an installed wrapper. Run this once on the robot while the ZED
driver is up:

```bash
ros2 run tf2_ros tf2_echo <robot>_zed_camera_link <robot>_zed_left_camera_optical_frame
```

If the result differs from the defaults in `spot_calib/cam_zed/zed_frames.py`, pass it in
(translation first, then the quaternion as x y z w, which is tf2_echo's order):

```bash
--zed-link-T-left-optical "-0.01 0.06 0.015 -0.5 0.5 -0.5 0.5"   # these are the shipped defaults
```

### Run
Stop the ROS ZED driver first; only one process can open the camera.

```bash
spot-calib cam-zed --robot-name <robot> \
    --calib-yaml <ws>/src/awesome_dcist_t4/dcist_launch_system/platforms/<platform>/calibration.yaml
```

Then open `http://<robot-computer-ip>:8080` on any laptop on the same network.

Board options: `--cols 10 --rows 7` (inner corners) and `--square 0.036` (m). These are the defaults.

### Procedure
1. Hold the board 0.8–2 m away so it is **fully** visible in the ZED and at least one Spot camera.
   The live views show green when the board is detected. The red circle and arrow mark corner 0 and
   the board x axis, and they should point the same way in every camera.
2. Press **Capture** (or the space bar), then **Approve** (`a`) or **Reject** (`r`) after looking at
   the thumbnails and numbers:
   - **PnP RMS**: should be under about 0.5 px for the ZED and about 1 px for Spot.
   - **deviation vs estimate**: should be a few mm. A big jump means a bad detection.
   - **RMS if approved**: what the overall RMS would become with this capture added.
3. Vary the pose. Cover the corners and edges of each image (see the coverage heatmaps), tilt the
   board ±30–45° in both axes, and change the distance.
4. Keep going until every accuracy badge is green. The defaults are ≥15 samples, ZED RMS < 0.5 px,
   Spot RMS < 1 px, leave-one-out σ < 3 mm / 0.2°, and frontleft/frontright agreeing within
   1 cm / 0.5°. You can change the thresholds at the bottom of the page.
5. Use the sample table to **exclude** outliers; sort by `dev mm` or the RMS columns to find them.
6. Click **Export**, then **Copy**. The YAML entry, `report.json` and every approved image are saved
   in `results/cam_zed/<timestamp>/`.

To re-solve a saved session offline, for example after hand-editing `samples.json`:
```bash
spot-calib cam-zed --resolve results/cam_zed/<timestamp> [--mode joint]
```

---

## 2. LiDAR ↔ body hand-eye (`lidar-handeye`)

### How it works
1. Find **stations** in Spot's odometry: windows of at least 1.5 s where body speed is under
   0.02 m/s and 0.02 rad/s.
2. For each station, accumulate up to 20 LiDAR scans into one dense cloud. Points within 0.8 m are
   dropped because they hit the robot itself.
3. For pairs of stations:
   - `A = body_i⁻¹ body_j` comes from odometry.
   - `B = lidar_i⁻¹ lidar_j` comes from GICP scan-to-scan registration.
4. Solve `A X = X B` for `X = body_T_livox`. This uses a closed-form start followed by robust
   nonlinear refinement. Registration then runs a second time, seeded from the estimate, and the
   solve is repeated.
5. Report observability and accuracy: rotation excitation per axis, how well the translation is
   determined, per-pair residuals, and leave-one-station-out spread. Results go to the terminal and
   to an HTML report that shows every registration overlay.

Bag receive time is used as the clock, so the Livox and Spot clocks don't need to be synchronised.

### Record the bag
Topics needed (both are already in `dcist_launch_system/recording/spot_topics.txt`):

| Topic | Type | Notes |
|---|---|---|
| `/<robot>/odom` | `nav_msgs/Odometry` | published by the Spot sensor node: vision-frame body pose plus twist |
| `/<robot>/livox/lidar` | `sensor_msgs/PointCloud2` | livox_ros_driver2 with `xfer_format: 0` |

You can record with the usual recording setup, or just these two topics:

```bash
ros2 bag record -o handeye_$(date +%F_%H%M) /<robot>/odom /<robot>/livox/lidar
```

### Motion protocol
Drive Spot with the tablet and **pause for about 3 s at each pose** (the tool finds the pauses on
its own). Aim for 15–25 poses:

- **Body roll** left and right (about ±15°) and **body pitch** up and down (about ±12°), using
  stand-mode body adjustment. These are essential: with yaw-only motion the LiDAR height cannot be
  determined.
- **Turns in place** of about 30–90°, in both directions.
- **Short walks** of 0.3–1 m, combined with turns, plus a few roll/pitch tilts at the new spots.
- Body height up and down.

Calibrate somewhere structured, such as a room or corridor where walls, floor, ceiling and furniture
are all in view, not an open field. Don't move more than a few metres from the start, because every
station is registered against nearby stations and against station 0.

### Run
```bash
source /opt/ros/jazzy/setup.bash
spot-calib lidar-handeye --bag <bag_dir> --robot-name <robot> \
    --calib-yaml <ws>/src/awesome_dcist_t4/dcist_launch_system/platforms/<platform>/calibration.yaml
```

The existing `body → livox` entry is used as the starting guess and for comparison. The terminal
prints the accuracy badges, observability hints and the YAML entry. `results/lidar_handeye/<timestamp>/report.html`
shows the per-station and per-pair tables and a top and side view of every registration. A good
pair shows grey and blue overlapping as crisp, single walls.

Defaults: ≥12 stations, ≥10° excitation in roll, pitch and yaw, median residual < 0.2° / 10 mm, and
leave-one-out σ < 10 mm / 0.3°.

Iterate without re-reading the bag:
```bash
spot-calib lidar-handeye --from-session results/lidar_handeye/<timestamp> --exclude 4,9 [--pairs all] [--map-refine]
```
- `--exclude` drops stations: bad registrations, or the robot moving while odometry reported it
  still.
- `--pairs all` registers every pair of stations instead of the default chain (neighbours plus
  station 0). It is slower but gives more constraints.
- `--map-refine` adds a final point-to-plane refinement of X using all station clouds placed by
  odometry.

Topic names can be overridden with `--lidar-topic` and `--odom-topic`. The stillness thresholds are
`--still-lin`, `--still-ang` and `--min-still`.

---

## Updating the calibration in the dcist launch

The static transforms are read from
`awesome_dcist_t4/dcist_launch_system/platforms/<platform>/calibration.yaml`. Which file is used
depends on `ADT4_PLATFORM_ID`, so update the file for the platform the sensors are mounted on. The
file is published by `ianvs`' `transform_file_broadcaster` when `launch_calibration_publisher:=true`,
which the real-Spot tmux config sets.

1. **Back up** the current file:
   ```bash
   cd <ws>/src/awesome_dcist_t4/dcist_launch_system/platforms/<platform>
   cp calibration.yaml calibration.yaml.bak.$(date +%F)
   ```
2. **Replace the matching entry** under `frames:` with the exported block, and keep the other
   entries (GPS and so on):
   - `cam-zed` replaces the entry with `frame_id: $(var robot_name)/frontleft` and
     `child_frame_id: $(var robot_name)_zed_camera_link`.
   - `lidar-handeye` replaces the entry with `frame_id: $(var robot_name)/body` and
     `child_frame_id: $(var robot_name)/livox`.

   Each exported entry is also saved in the session folder as `calibration_entry.yaml`, and the
   indentation already matches the file's. The final file looks like:
   ```yaml
   ---
   frames:
     - frame_id: $(var robot_name)/frontleft
       child_frame_id: $(var robot_name)_zed_camera_link
       qw: ...
       qx: ...
       qy: ...
       qz: ...
       x: ...
       y: ...
       z: ...
     - frame_id: $(var robot_name)/body
       child_frame_id: $(var robot_name)/livox
       ...
   ```
   Values are `parent_T_child`: the pose of the child frame in the parent frame, with a unit
   quaternion and metres.
3. **Rebuild** only if the workspace wasn't built with `--symlink-install`. The launch reads the file
   from the package's share directory:
   ```bash
   colcon build --packages-select dcist_launch_system
   ```
4. **Verify** after relaunching:
   ```bash
   ros2 run tf2_ros tf2_echo <robot>/frontleft <robot>_zed_camera_link     # should match the exported entry
   ros2 run tf2_ros tf2_echo <robot>/body <robot>/livox
   ```
   In rviz, the accumulated LiDAR map should look crisp while the robot turns in place, and the ZED
   depth cloud should line up with the LiDAR points.
5. **Commit** the updated `calibration.yaml` in awesome_dcist_t4. Put the session folder name and the
   key accuracy numbers in the commit message so the calibration can be traced later.

> If the same physical mount is shared across platform folders, update each `calibration.yaml` that
> describes that robot.

---

## Trying it without hardware

```bash
spot-calib cam-zed --mock --port 8080            # synthetic ZED + Spot cameras showing a checkerboard
spot-calib mock-bag /tmp/mock_bag                # synthetic room, robot holding 15 poses
spot-calib lidar-handeye --bag /tmp/mock_bag --robot-name mock
```
Both mocks have known ground truth. The solvers recover it to within 0.3 mm and 0.01°.

## Layout
```
spot_calib/
  cli.py                     # spot-calib entry point
  common/                    # SE3 helpers, calibration.yaml export/compare, Flask helpers, bosdyn session
  cam_zed/                   # sources (pyzed, bosdyn), board detection, PnP + bundle adjustment, GUI
  lidar_handeye/             # bag reader, station detection, registration, AX=XB solver, HTML report
tests/                       # synthetic end-to-end tests
```
