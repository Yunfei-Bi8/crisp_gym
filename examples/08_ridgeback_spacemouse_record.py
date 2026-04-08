"""Stage 8 — SpaceMouse teleoperation data collection with LeRobot Dataset output.

Collects human demonstration episodes using a 3Dconnexion SpaceMouse and
saves them in two parallel formats:

  1. **LeRobot Dataset** (RGB video + proprioceptive state + action)
     Saved to: $HF_LEROBOT_HOME/<repo_id>/
     Directly usable with `lerobot train` for Diffusion Policy / ACT / SmolVLA.

  2. **Depth NPZ** (float32 metres, lossless, via DataCollector)
     Saved to: <depth_output_dir>/episode_NNNN/cam_camera.npz
     Loaded alongside LeRobot data at training time for RGBD policies.

The two stores are aligned by episode index: LeRobot episode 000003 corresponds
to depth episode_0003/.  The script prints both paths after each saved episode.

Architecture
------------
  SpaceMouse → TeleopCommand
    → apply_delta(target_pose)
    → robot.move_cartesian_async()     (CRISP CartesianController)
    → DataCollector.record_step()      (depth npz, every tick)
    → data_fn() returns (obs, action)
    → KeyboardRecordingManager         (LeRobot Dataset writer process)

Keyboard controls (episode management):
  r  →  start / stop recording
  s  →  save episode
  d  →  discard episode
  q  →  quit

SpaceMouse controls (robot motion):
  translate knob  →  EE +X/Y/Z
  rotate knob     →  EE roll/pitch/yaw (body-fixed)
  button 1        →  gripper close (hold)
  (no button = gripper holds last command)

Prerequisites
-------------
  1. Robot is powered on and bringup running.
  2. Orbbec driver running:
       pixi run ros2 launch tum09_custom orbbec.launch.py
  3. pyspacemouse installed:
       pip install pyspacemouse
  4. udev rule for SpaceMouse HID device (one-time setup):
       echo 'SUBSYSTEM=="hidraw", ATTRS{idVendor}=="256f", MODE="0666"' \\
           | sudo tee /etc/udev/rules.d/99-spacemouse.rules
       sudo udevadm control --reload-rules && sudo udevadm trigger

Usage
-----
  cd /home/yunfei/crisp_gym

  # Collect 10 episodes, real robot
  pixi run --environment jazzy-lerobot python3 examples/08_ridgeback_spacemouse_record.py \\
      --repo-id ridgeback_pick --num-episodes 10 --task "pick the red block"

  # Custom camera namespace and depth output
  pixi run --environment jazzy-lerobot python3 examples/08_ridgeback_spacemouse_record.py \\
      --repo-id ridgeback_pick --camera-namespace /camera_01 \\
      --depth-output-dir ~/data/ridgeback_depth

  # Simulation (no cameras, no SpaceMouse — keyboard only for testing)
  pixi run --environment jazzy-lerobot python3 examples/08_ridgeback_spacemouse_record.py \\
      --repo-id ridgeback_pick_sim --sim --num-episodes 2

Training after collection
-------------------------
  cd /home/yunfei/lerobot
  python lerobot/scripts/train.py \\
      dataset.repo_id=ridgeback_pick \\
      policy=diffusion \\
      env=ridgeback
"""

import argparse
import time
from pathlib import Path

import numpy as np
import yaml

# crisp_py — robot, camera, teleop, data
from crisp_py.camera import RgbdCameraConfig, make_rgbd_cameras
from crisp_py.data import DataCollector
from crisp_py.robot import RidgebackConfig, get_ridgeback_urdf_path, make_ridgeback_robot
from crisp_py.teleop import SpaceMouseTeleopInput, apply_delta

# crisp_gym — LeRobot dataset management
from crisp_gym.record.recording_manager import KeyboardRecordingManager
from crisp_gym.record.recording_manager_config import RecordingManagerConfig

# --------------------------------------------------------------------------- #
# Constants
# --------------------------------------------------------------------------- #

CLEARPATH_WS = "/home/yunfei/clearpath_remote_ws"
_ENV_CONFIG = Path(__file__).parent.parent / "crisp_gym/config/envs/ridgeback_env.yaml"

# --------------------------------------------------------------------------- #
# Feature definition (replaces crisp_gym get_features(), which needs ManipulatorEnv)
# --------------------------------------------------------------------------- #

def build_lerobot_features(
    camera_name: str,
    image_h: int,
    image_w: int,
    fps: float,
) -> dict:
    """Build the LeRobot Dataset feature schema for the Ridgeback setup.

    Observation state vector (7-dim):
        [x, y, z, roll, pitch, yaw]  — EE pose in base_link frame
        [gripper_norm]               — 0.0=open, 1.0=closed

    Action vector (7-dim, relative deltas for Diffusion Policy):
        [dx, dy, dz, d_roll, d_pitch, d_yaw, gripper_norm]

    Camera: RGB only (H, W, 3) uint8, stored as AV1 video by LeRobot.
    Depth:  stored separately as float32 npz (see DataCollector).
    """
    return {
        # RGB image from the Orbbec camera
        f"observation.images.{camera_name}": {
            "dtype": "video",
            "shape": (image_h, image_w, 3),
            "names": ["height", "width", "channels"],
            "video_info": {
                "video.fps": fps,
                "video.codec": "av1",
                "video.pix_fmt": "yuv420p",
                "video.is_depth_map": False,
                "has_audio": False,
            },
        },
        # EE Cartesian pose (6-dim)
        "observation.state.cartesian": {
            "dtype": "float32",
            "shape": (6,),
            "names": ["x", "y", "z", "roll", "pitch", "yaw"],
        },
        # Gripper normalised state (1-dim)
        "observation.state.gripper": {
            "dtype": "float32",
            "shape": (1,),
            "names": ["gripper"],
        },
        # Concatenated state vector (required by RecordingManager)
        "observation.state": {
            "dtype": "float32",
            "shape": (7,),
            "names": ["x", "y", "z", "roll", "pitch", "yaw", "gripper"],
        },
        # Relative delta action (7-dim) — suited for Diffusion Policy
        "action": {
            "dtype": "float32",
            "shape": (7,),
            "names": ["dx", "dy", "dz", "d_roll", "d_pitch", "d_yaw", "gripper"],
        },
    }


# --------------------------------------------------------------------------- #
# Argument parsing
# --------------------------------------------------------------------------- #

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Stage 8: SpaceMouse teleoperation → LeRobot Dataset + depth npz",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--repo-id",
        default="ridgeback_pick",
        help="LeRobot dataset repo ID (local name). Default: ridgeback_pick",
    )
    parser.add_argument(
        "--task",
        default="pick the object",
        help='Task language description stored per episode. Default: "pick the object"',
    )
    parser.add_argument(
        "--num-episodes",
        type=int,
        default=0,
        help="Number of episodes to collect. 0 = unlimited (press q to quit). Default: 0",
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=20.0,
        help="Control and recording frequency [Hz]. Default: 20",
    )
    parser.add_argument(
        "--sim",
        action="store_true",
        help="Use simulation robot config (no cameras, no SpaceMouse).",
    )
    parser.add_argument(
        "--camera-namespace",
        default="/camera",
        metavar="NS",
        help="ROS namespace of the Orbbec camera. Default: /camera",
    )
    parser.add_argument(
        "--depth-output-dir",
        default="~/data/ridgeback_depth",
        help="Root directory for depth npz files. Default: ~/data/ridgeback_depth",
    )
    parser.add_argument(
        "--camera-timeout",
        type=float,
        default=15.0,
        help="Seconds to wait for cameras. Default: 15",
    )
    parser.add_argument(
        "--pos-scale",
        type=float,
        default=0.005,
        help="SpaceMouse position increment per tick [m]. Default: 0.005",
    )
    parser.add_argument(
        "--rot-scale",
        type=float,
        default=0.02,
        help="SpaceMouse rotation increment per tick [rad]. Default: 0.02",
    )
    parser.add_argument(
        "--push-to-hub",
        action="store_true",
        help="Push the finished dataset to HuggingFace Hub.",
    )
    return parser.parse_args()


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #

def main() -> None:  # noqa: C901
    args = _parse_args()

    print("=" * 60)
    print(f"  Mode:      {'SIMULATION' if args.sim else 'REAL ROBOT'}")
    print(f"  Repo ID:   {args.repo_id}")
    print(f"  Task:      {args.task}")
    print(f"  Episodes:  {'unlimited (press q to quit)' if args.num_episodes <= 0 else args.num_episodes}")
    print(f"  Frequency: {args.fps:.0f} Hz")
    print(f"  Camera:    {args.camera_namespace if not args.sim else 'none (sim)'}")
    print(f"  Depth dir: {args.depth_output_dir}")
    print("=" * 60)

    # ------------------------------------------------------------------ #
    # Robot
    # ------------------------------------------------------------------ #
    if args.sim:
        cfg = RidgebackConfig.for_simulation()
        urdf = (
            f"{CLEARPATH_WS}/src/tum09_ridgeback/tum09_bringup/config/robot.urdf"
        )
    else:
        cfg = RidgebackConfig.for_real_robot()
        urdf = get_ridgeback_urdf_path(CLEARPATH_WS)

    print("\nConnecting to robot...")
    robot = make_ridgeback_robot(urdf_path=urdf, config=cfg)
    robot.wait_until_ready(timeout=15.0)
    print("Robot ready.")

    # ------------------------------------------------------------------ #
    # Cameras (skip in sim)
    # ------------------------------------------------------------------ #
    cameras = []
    camera_name = "camera"  # default LeRobot feature key suffix
    image_h, image_w = 480, 640

    if not args.sim:
        ns = args.camera_namespace
        camera_name = ns.strip("/").replace("/", "_") or "camera"
        cam_cfg = RgbdCameraConfig.for_orbbec(ns)
        print(f"\nConnecting to camera {ns}...")
        cameras = make_rgbd_cameras([cam_cfg])
        cameras[0].wait_until_ready(timeout=args.camera_timeout)
        intr = cameras[0].intrinsics
        image_h, image_w = intr.height, intr.width
        print(f"Camera ready — {image_w}×{image_h}")

    # ------------------------------------------------------------------ #
    # DataCollector — depth npz storage
    # ------------------------------------------------------------------ #
    collector = DataCollector(
        robot=robot,
        cameras=cameras,
        output_dir=args.depth_output_dir,
    )

    # ------------------------------------------------------------------ #
    # SpaceMouse (skip in sim)
    # ------------------------------------------------------------------ #
    if not args.sim:
        teleop_input = SpaceMouseTeleopInput(
            pos_scale=args.pos_scale,
            rot_scale=args.rot_scale,
            # button 0 = stop episode (ignored here — keyboard handles that)
            # button 1 = gripper close
            button_stop=-1,             # disable stop-episode button on SpaceMouse
            button_gripper_open=-1,     # no dedicated open button
            button_gripper_close=1,
        )
    else:
        teleop_input = None

    # ------------------------------------------------------------------ #
    # LeRobot features + RecordingManager
    # ------------------------------------------------------------------ #
    features = build_lerobot_features(
        camera_name=camera_name,
        image_h=image_h,
        image_w=image_w,
        fps=args.fps,
    )
    # In simulation there is no camera — remove the video feature
    if args.sim:
        features = {k: v for k, v in features.items() if not k.startswith("observation.images")}

    recording_config = RecordingManagerConfig(
        features=features,
        repo_id=args.repo_id,
        robot_type="ridgeback_ur10e",
        fps=int(args.fps),
        num_episodes=args.num_episodes,
        push_to_hub=args.push_to_hub,
        use_sound=True,
    )
    recording_manager = KeyboardRecordingManager(config=recording_config)
    recording_manager.wait_until_ready()
    print("LeRobot RecordingManager ready.")

    # ------------------------------------------------------------------ #
    # Home robot + switch to CartesianController
    # ------------------------------------------------------------------ #
    # Ensure JointTrajectoryController is active before homing
    # (CartesianController may be left active from a previous crashed run)
    robot.switch_to_joint_trajectory_controller()

    print("\nMoving to home...")
    robot.home(duration=8.0)
    if not robot.is_homed():
        raise RuntimeError("Failed to reach home. Aborting.")
    print("At home.")

    print("Activating CartesianController...")
    if not robot.switch_to_cartesian_controller():
        raise RuntimeError("Failed to switch to CartesianController.")
    print("CartesianController active.")

    # ------------------------------------------------------------------ #
    # Episode collection loop
    # ------------------------------------------------------------------ #

    # Mutable state shared between data_fn closure and the outer loop
    state = {
        "target_pose": robot.end_effector_pose,
        "gripper_norm": 0.0,   # 0.0 = open, 1.0 = closed
    }

    def on_start() -> None:
        """Reset target pose to current robot pose when a new episode begins."""
        state["target_pose"] = robot.end_effector_pose
        state["gripper_norm"] = 0.0

    def make_data_fn():
        """
        Return a data_fn closure for one episode.

        data_fn is called once per tick by RecordingManager.record_episode().
        It polls the SpaceMouse, commands the robot, records the depth step,
        and returns (obs, action) in LeRobot format.

        Action format (relative, 7-dim):
            [dx, dy, dz, d_roll, d_pitch, d_yaw, gripper_norm]
        Observation format:
            - observation.images.<name>    : (H,W,3) uint8 RGB
            - observation.state.cartesian  : (6,) float32 [x,y,z,roll,pitch,yaw]
            - observation.state.gripper    : (1,) float32
            - task                         : str
        """
        def _fn():
            # --- Poll SpaceMouse (or zeros in sim) ---
            if teleop_input is not None and teleop_input.is_open:
                cmd = teleop_input.poll()
            else:
                from crisp_py.teleop.teleop_input import TeleopCommand
                cmd = TeleopCommand()   # all-zero no-op

            # --- Save previous target for delta action computation ---
            prev_pose = state["target_pose"]

            # --- Apply SpaceMouse delta to target pose ---
            if np.any(cmd.pos_delta != 0.0) or np.any(cmd.rot_delta != 0.0):
                state["target_pose"] = apply_delta(state["target_pose"], cmd)

            # --- Gripper command ---
            # button 1 pressed → close; no button → hold last value
            if cmd.gripper > 0.5:
                robot.gripper_close()
                state["gripper_norm"] = 1.0
            elif cmd.gripper < -0.5:
                robot.gripper_open()
                state["gripper_norm"] = 0.0
            # else: hold — do not send a new gripper command

            # --- Send Cartesian target to robot ---
            robot.move_cartesian_async(state["target_pose"])

            # --- Record depth step (DataCollector) ---
            if collector.is_recording:
                collector.record_step(action=state["target_pose"])

            # --- Build relative delta action (for Diffusion Policy) ---
            delta_pos = state["target_pose"].position - prev_pose.position
            delta_rot = (
                state["target_pose"].orientation * prev_pose.orientation.inv()
            ).as_euler("xyz")
            action = np.array(
                [*delta_pos, *delta_rot, state["gripper_norm"]],
                dtype=np.float32,
            )

            # --- Build observation ---
            ee = robot.end_effector_pose
            cartesian = np.array(
                [*ee.position, *ee.orientation.as_euler("xyz")],
                dtype=np.float32,
            )
            gripper_arr = np.array([state["gripper_norm"]], dtype=np.float32)

            obs = {
                "observation.state.cartesian": cartesian,
                "observation.state.gripper":   gripper_arr,
                "task": args.task,
            }
            if cameras:
                try:
                    color, _, _ = cameras[0].current_rgbd(max_stamp_diff_ms=100.0)
                except RuntimeError:
                    color = cameras[0].current_color
                obs[f"observation.images.{camera_name}"] = color

            # Rate control is handled by RecordingManager's fps loop — no sleep here.
            return obs, action

        return _fn

    # ------------------------------------------------------------------ #
    # Main loop
    # ------------------------------------------------------------------ #
    ctx = teleop_input if teleop_input is not None else _NullContext()
    with ctx, recording_manager:
        print(
            "\n=== Ready to collect ==="
            "\n  Keyboard: r=start/stop  s=save  d=discard  q=quit"
            "\n  SpaceMouse: translate/rotate knob → EE motion | button 1 → gripper close"
            "\n"
        )

        while not recording_manager.done():
            episode_count_before = recording_manager.episode_count

            # Start depth collector for this episode
            episode_id = collector.start_episode()
            ep_total = "?" if args.num_episodes <= 0 else args.num_episodes
            print(f"\n--- Episode {episode_count_before + 1} / {ep_total} ---")
            print(f"    Depth episode ID: {episode_id}")
            print("    Press 'r' to start recording.")

            recording_manager.record_episode(
                data_fn=make_data_fn(),
                task=args.task,
                on_start=on_start,
            )

            # Sync DataCollector with RecordingManager's save/discard decision
            if recording_manager.episode_count > episode_count_before:
                saved_path = collector.stop_episode()
                n_steps = collector.last_episode_steps
                lerobot_ep = recording_manager.episode_count - 1
                print(
                    f"Saved — LeRobot episode {lerobot_ep:06d} | "
                    f"Depth: {saved_path} ({n_steps} steps)"
                )
            else:
                collector.discard_episode()
                print("Episode discarded (both LeRobot and depth).")

    # ------------------------------------------------------------------ #
    # Cleanup
    # ------------------------------------------------------------------ #
    print("\nRestoring JointTrajectoryController...")
    robot.switch_to_joint_trajectory_controller()
    print("Done.")
    print(
        f"\nCollection complete."
        f"\n  LeRobot dataset : $HF_LEROBOT_HOME/{args.repo_id}/"
        f"\n  Depth npz       : {Path(args.depth_output_dir).expanduser()}"
    )


# --------------------------------------------------------------------------- #
# Utility: null context manager for sim mode (no SpaceMouse)
# --------------------------------------------------------------------------- #

class _NullContext:
    """No-op context manager used in simulation mode (no SpaceMouse)."""
    def __enter__(self): return self
    def __exit__(self, *_): pass


if __name__ == "__main__":
    main()
