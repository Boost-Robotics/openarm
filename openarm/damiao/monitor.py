"""Simple motor angle monitor - disables motors and shows current angles.

This script disables all Damiao motors and displays their current angles.
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import re
import struct
import sys
from dataclasses import dataclass, field
from math import pi
import numpy as np

# Platform-specific imports for keyboard input
try:
    import select
    import termios
    import tty

    HAS_TERMIOS = True
except ImportError:
    HAS_TERMIOS = False

try:
    import msvcrt

    HAS_MSVCRT = True
except ImportError:
    HAS_MSVCRT = False

import can

from openarm.bus import Bus

from .config import MOTOR_CONFIGS
from .detect import detect_motors
from .encoding import (
    ControlMode,
    MitControlParams,
    PosVelControlParams,
    decode_motor_state_sync,
    encode_control_mit,
    encode_control_pos_vel,
)
from .gravity import GravityCompensator
from .motor import Motor
import mujoco
import serial
import threading
import time
import cv2
HAS_CV2 = True

try:
    import rclpy
    from rclpy.node import Node
    from geometry_msgs.msg import PoseStamped
    HAS_ROS2 = True
except ImportError:
    HAS_ROS2 = False

# ANSI color codes for terminal output
RED = "\033[91m"
GREEN = "\033[92m"
RESET = "\033[0m"

# Constants
FOLLOW_SPEC_PARTS = 4  # MASTER:POSITION:SLAVE:POSITION

# Set up logging
logger = logging.getLogger(__name__)


def _pose_to_T(pos: np.ndarray, quat_wxyz: np.ndarray) -> np.ndarray:
    """Build a 4x4 homogeneous transform from position + MuJoCo quaternion [w,x,y,z]."""
    T = np.eye(4)
    T[:3, 3] = pos
    R = np.zeros(9)
    import mujoco
    mujoco.mju_quat2Mat(R, quat_wxyz)
    T[:3, :3] = R.reshape(3, 3)
    return T


def _ros_pose_to_T(pose_msg) -> np.ndarray:
    """Build a 4x4 transform from a ROS geometry_msgs/Pose."""
    T = np.eye(4)
    p = pose_msg.position
    o = pose_msg.orientation
    T[:3, 3] = [p.x, p.y, p.z]
    # ROS quaternion: (x,y,z,w) → MuJoCo: (w,x,y,z)
    quat_wxyz = np.array([o.w, o.x, o.y, o.z])
    R = np.zeros(9)
    import mujoco
    mujoco.mju_quat2Mat(R, quat_wxyz)
    T[:3, :3] = R.reshape(3, 3)
    return T


def _T_to_pos_quat(T: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Extract position and MuJoCo quaternion [w,x,y,z] from a 4x4 transform."""
    pos = T[:3, 3].copy()
    import mujoco
    quat = np.zeros(4)
    mujoco.mju_mat2Quat(quat, T[:3, :3].flatten())
    return pos, quat


def _quat_to_rpy_deg(q_wxyz: np.ndarray) -> np.ndarray:
    """Convert MuJoCo quaternion [w,x,y,z] to roll/pitch/yaw in degrees."""
    import mujoco
    R = np.zeros(9)
    mujoco.mju_quat2Mat(R, q_wxyz)
    R = R.reshape(3, 3)
    pitch = np.arcsin(-np.clip(R[2, 0], -1, 1))
    if np.abs(R[2, 0]) < 0.9999:
        roll = np.arctan2(R[2, 1], R[2, 2])
        yaw = np.arctan2(R[1, 0], R[0, 0])
    else:
        roll = np.arctan2(-R[1, 2], R[1, 1])
        yaw = 0.0
    return np.degrees([roll, pitch, yaw])


def check_keyboard_input() -> str | None:
    """Check if a key has been pressed (non-blocking)."""
    if HAS_MSVCRT and msvcrt.kbhit():
        return msvcrt.getch().decode("utf-8", errors="ignore").lower()
    if HAS_TERMIOS and select.select([sys.stdin], [], [], 0)[0]:
        return sys.stdin.read(1).lower()
    return None


@dataclass
class Arm:
    """Represents a single robotic arm with its motors and configuration."""

    position: str  # "left" or "right"
    can_bus: can.BusABC  # The CAN bus for this arm
    channel: str  # Channel name (e.g., "can0", "can1")
    motors: list[Motor | None] = field(default_factory=list)
    states: list = field(default_factory=list)  # Current states for each motor
    is_master: bool = False  # Whether this arm is a master
    is_slave: bool = False  # Whether this arm is a slave
    mirror_mode: bool = False  # Whether mirror mode is enabled (for slaves)
    follows: str | None = None  # Channel name of master (for slaves)

    @property
    def active_motors(self) -> list[Motor]:
        """Get list of active (non-None) motors."""
        return [m for m in self.motors if m is not None]

    @property
    def active_count(self) -> int:
        """Count of active motors."""
        return len(self.active_motors)

    async def disable_all_motors(self) -> None:
        """Safely disable all active motors."""
        for motor in self.motors:
            if motor is not None:
                try:
                    await motor.disable()
                except Exception as e:  # noqa: BLE001
                    logger.debug("Failed to disable motor: %s", e)

    async def enable_all_motors(self, control_mode: ControlMode) -> None:
        """Enable all active motors with specified control mode."""
        for idx, motor in enumerate(self.motors):
            if motor is not None:
                try:
                    await motor.enable()
                    await motor.set_control_mode(control_mode)
                    logger.info("Motor %d: Enabled", idx + 1)
                    sys.stdout.write(f"    Motor {idx + 1}: Enabled\n")
                except Exception as e:
                    logger.exception("Motor %d: Error", idx + 1)
                    sys.stderr.write(f"{RED}    Motor {idx + 1}: Error - {e}{RESET}\n")

    async def refresh_states(self) -> None:
        """Refresh states for all motors."""
        new_states = []
        for motor in self.motors:
            if motor:
                try:
                    state = await motor.refresh_status()
                    new_states.append(state)
                except Exception as e:  # noqa: BLE001
                    logger.debug("Failed to refresh motor status: %s", e)
                    new_states.append(None)
            else:
                new_states.append(None)
        self.states = new_states


async def main(args: argparse.Namespace) -> None:
    """Run the monitor with the provided arguments."""
    # Create CAN buses
    try:
        if hasattr(can, "detect_available_configs"):
            configs = can.detect_available_configs("socketcan")
        else:
            import subprocess
            result = subprocess.run(
                ["/opt/iproute2-root/bin/ip", "link", "show"],
                capture_output=True, text=True,
            )
            configs = []
            lines = result.stdout.splitlines()
            for i, line in enumerate(lines):
                if i + 1 < len(lines) and "link/can" in lines[i + 1]:
                    iface = line.split(":")[1].strip().split("@")[0]
                    configs.append({"channel": iface, "interface": "socketcan"})
        print(f"detect_available_configs: {configs}")
        can_buses = [
            can.Bus(channel=config["channel"], interface=config["interface"])
            for config in configs
        ]
    except Exception as e:  # noqa: BLE001
        print(f"Exception: {e}")
        can_buses = []

    if not can_buses:
        return None

    sys.stdout.write(f"\nDetected {len(can_buses)} CAN bus(es)\n")

    try:
        return await _main(args, can_buses)
    finally:
        for bus in can_buses:
            bus.shutdown()


async def _main(args: argparse.Namespace, can_buses: list[can.BusABC]) -> None:  # noqa: C901, PLR0912
    # Detect motors on each bus
    all_bus_motors = []
    has_missing_motor = False

    print(f"can_buses: {can_buses}")

    for bus_idx, can_bus in enumerate(can_buses):
        sys.stdout.write(f"\nScanning for motors on bus {bus_idx + 1}...\n")
        slave_ids = [config.slave_id for config in MOTOR_CONFIGS]

        # Detect motors using raw CAN bus
        detected = list(detect_motors(can_bus, slave_ids, timeout=0.01))

        sys.stdout.write(f"\nBus {bus_idx + 1} Motor Status:\n")

        # Create lookup for detected motors by slave ID
        detected_lookup = {info.slave_id: info for info in detected}

        # Check all expected motors and their status
        bus_motors = []
        for config in MOTOR_CONFIGS:
            if config.slave_id not in detected_lookup:
                # Motor is not detected
                sys.stderr.write(
                    f"  {RED}✗{RESET} {config.name}: ID 0x{config.slave_id:02X} "
                    f"(Master: 0x{config.master_id:02X}) {RED}[NOT DETECTED]{RESET}\n"
                )
                bus_motors.append(None)
                has_missing_motor = True
            elif detected_lookup[config.slave_id].master_id != config.master_id:
                # Motor is detected but master ID doesn't match
                detected_info = detected_lookup[config.slave_id]
                sys.stderr.write(
                    f"  {RED}✗{RESET} {config.name}: ID 0x{config.slave_id:02X} "
                    f"{RED}[MASTER ID MISMATCH: Expected 0x{config.master_id:02X}, "
                    f"Got 0x{detected_info.master_id:02X}]{RESET}\n"
                )
                bus_motors.append(None)
                has_missing_motor = True
            else:
                # Motor is connected and configured correctly
                sys.stdout.write(
                    f"  {GREEN}✓{RESET} {config.name}: ID 0x{config.slave_id:02X} "
                    f"(Master: 0x{config.master_id:02X})\n"
                )
                # Create motor instance
                bus = Bus(can_bus)
                motor = Motor(
                    bus,
                    slave_id=config.slave_id,
                    master_id=config.master_id,
                    motor_type=config.type,
                )
                bus_motors.append(motor)

        all_bus_motors.append(bus_motors)

    # Exit if any motor is missing
    if has_missing_motor:
        sys.stderr.write(
            f"\n{RED}Error: Not all motors are detected or configured "
            f"correctly. Exiting.{RESET}\n"
        )
        return

    # Count total detected motors
    total_motors = sum(
        1 for bus_motors in all_bus_motors for m in bus_motors if m is not None
    )
    if total_motors == 0:
        sys.stderr.write(f"\n{RED}Error: No motors detected on any bus.{RESET}\n")
        return

    sys.stdout.write(
        f"\n{GREEN}Total {total_motors} motors detected across "
        f"{len(can_buses)} bus(es){RESET}\n"
    )

    # Disable all motors on all buses
    sys.stdout.write("\nDisabling all motors...\n")
    all_state_results = []
    for bus_idx, bus_motors in enumerate(all_bus_motors):
        bus_states = []
        for motor in bus_motors:
            if motor:
                try:
                    state = await motor.disable()
                    bus_states.append(state)
                except Exception as e:
                    logger.exception("Error disabling motor on bus %d", bus_idx + 1)
                    sys.stderr.write(
                        f"{RED}Error disabling motor on bus {bus_idx + 1}: {e}{RESET}\n"
                    )
                    bus_states.append(None)
            else:
                bus_states.append(None)
        all_state_results.append(bus_states)

    # Call teleop or monitor based on flag
    if args.teleop:
        await teleop(can_buses, all_bus_motors, all_state_results, args)
    else:
        await monitor_motors(can_buses, all_bus_motors, all_state_results)


async def monitor_motors(  # noqa: C901, PLR0912
    can_buses: list[can.BusABC],
    all_bus_motors: list[list[Motor | None]],
    all_state_results: list[list],
) -> None:
    """Monitor motor angles continuously and display them in a table format.

    Args:
        can_buses: List of CAN bus interfaces.
        all_bus_motors: List of motor lists for each bus.
        all_state_results: Initial state results for each motor.

    """
    # Start continuous monitoring with column display
    sys.stdout.write("\nContinuously monitoring motor angles (Ctrl+C to stop):\n\n")

    # Print header with bus labels
    header = "  Motor"
    for bus_idx in range(len(can_buses)):
        header += f"        Bus {bus_idx + 1}     "
    sys.stdout.write(header + "\n")
    sys.stdout.write("  " + "-" * (len(header) - 2) + "\n")

    # Print initial lines for each motor
    for config in MOTOR_CONFIGS:
        line = f"  {config.name:<12}"
        for _ in range(len(can_buses)):
            line += "  Initializing...  "
        sys.stdout.write(line + "\n")

    # Number of motors (lines to move up)
    num_motors = len(MOTOR_CONFIGS)

    # Use disable results for first display
    all_current_states = all_state_results

    try:
        while True:

            # Small delay before refresh
            await asyncio.sleep(0.1)

            # Refresh states for all buses
            new_all_states = []
            for bus_motors in all_bus_motors:
                bus_states = []
                for motor in bus_motors:
                    if motor:
                        try:
                            state = await motor.refresh_status()
                            bus_states.append(state)
                        except Exception as e:  # noqa: BLE001
                            logger.debug("Failed to refresh motor status: %s", e)
                            bus_states.append(None)
                    else:
                        bus_states.append(None)
                new_all_states.append(bus_states)
            all_current_states = new_all_states

    except KeyboardInterrupt:
        # Move cursor below all motor lines
        sys.stdout.write(f"\033[{num_motors}B\n")
        sys.stdout.write("\nMonitoring stopped.\n")


class _FrameVis:
    """Live OpenCV visualization of coordinate frames in a dedicated thread."""

    W, H = 900, 600
    SCALE = 600.0
    OX, OY = 150, 500

    def __init__(self):
        self._lock = threading.Lock()
        self._frames: dict[str, np.ndarray] = {}
        self._running = True
        self._thread = None
        if HAS_CV2:
            self._thread = threading.Thread(target=self._loop, daemon=True)
            self._thread.start()

    def _to_px(self, pos):
        return (int(self.OX + pos[0] * self.SCALE),
                int(self.OY - pos[2] * self.SCALE))

    def _draw_frame(self, img, T, label, axis_len=0.03):
        pos = T[:3, 3]
        R = T[:3, :3]
        center = self._to_px(pos)
        colors = [(0, 0, 255), (0, 255, 0), (255, 0, 0)]  # X=red, Y=green, Z=blue
        for i, color in enumerate(colors):
            tip_px = self._to_px(pos + R[:, i] * axis_len)
            cv2.arrowedLine(img, center, tip_px, color, 2, tipLength=0.3)
        cv2.putText(img, label, (center[0] + 5, center[1] - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1)

    def _render(self):
        img = np.zeros((self.H, self.W, 3), dtype=np.uint8)
        cv2.putText(img, "Side (XZ)", (10, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (140, 140, 140), 1)

        self._draw_frame(img, np.eye(4), "W", axis_len=0.04)

        with self._lock:
            frames = dict(self._frames)

        for label, T in frames.items():
            self._draw_frame(img, T, label)

        return img

    def _loop(self):
        cv2.namedWindow("Pose Viewer", cv2.WINDOW_AUTOSIZE)
        while self._running:
            img = self._render()
            cv2.imshow("Pose Viewer", img)
            if cv2.waitKey(33) & 0xFF == 27:  # ~30fps, ESC to close
                break
        cv2.destroyWindow("Pose Viewer")

    def update(self, **named_transforms):
        with self._lock:
            for name, T in named_transforms.items():
                if T is not None:
                    self._frames[name] = T.copy()

    def close(self):
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=1.0)


class _PoseListener:
    """Thread-safe container for the latest FoundationPose PoseStamped."""

    def __init__(self):
        self._lock = threading.Lock()
        self._pose: PoseStamped | None = None
        self._node = None
        self._thread = None

    def start(self):
        if not HAS_ROS2:
            logger.warning("rclpy not available — FoundationPose pose subscription disabled")
            return
        try:
            rclpy.init()
        except RuntimeError:
            pass  # already initialised

        self._node = rclpy.create_node("teleop_pose_listener")
        self._node.create_subscription(
            PoseStamped, "/foundationpose/pose", self._cb, 1,
        )
        self._thread = threading.Thread(target=self._spin, daemon=True)
        self._thread.start()
        logger.info("Subscribed to /foundationpose/pose")

    def _cb(self, msg: PoseStamped):
        with self._lock:
            self._pose = msg

    def _spin(self):
        try:
            rclpy.spin(self._node)
        except Exception:
            pass

    @property
    def latest(self) -> PoseStamped | None:
        with self._lock:
            return self._pose

    def shutdown(self):
        if self._node is not None:
            self._node.destroy_node()
        try:
            rclpy.shutdown()
        except Exception:
            pass


async def teleop(  # noqa: C901, PLR0912
    can_buses: list[can.BusABC],
    all_bus_motors: list[list[Motor | None]],
    all_state_results: list[list],
    args: argparse.Namespace,
) -> None:
    """Teleoperation mode - masters use MIT, slaves use PosVel."""
    # Start FoundationPose pose subscriber
    pose_listener = _PoseListener()
    pose_listener.start()

    # Live frame visualizer
    frame_vis = _FrameVis()

    # Initialize gravity compensator if enabled
    gravity_comp = None
    if args.gravity:
        sys.stdout.write("Initializing gravity compensation...\n")
        gravity_comp = GravityCompensator()

    # Create Arm objects for each bus
    arms: list[Arm] = []
    channel_to_arm = {}  # Maps channel name to Arm object

    for bus_idx, (can_bus, bus_motors, bus_states) in enumerate(
        zip(can_buses, all_bus_motors, all_state_results)
    ):
        # Get channel info from the bus
        channel_info = (
            str(can_bus.channel_info)
            if hasattr(can_bus, "channel_info")
            else str(can_bus.channel)
        )
        # Extract channel name (e.g., "can0" from various formats)
        if "channel" in channel_info:
            # For socketcan: extract from "SocketcanBus channel 'can0'"
            match = re.search(r"channel ['\"]?(\w+)", channel_info)
            channel_name = match.group(1) if match else f"bus{bus_idx}"
        else:
            # For USB devices, use the product name or bus index
            channel_name = channel_info.split()[-1] if channel_info else f"bus{bus_idx}"

        # Create Arm object
        arm = Arm(
            position="unknown",  # Will be set based on --follow arguments
            can_bus=can_bus,
            channel=channel_name,
            motors=bus_motors,
            states=bus_states,
        )
        arms.append(arm)
        channel_to_arm[channel_name] = arm
        sys.stdout.write(f"Bus {bus_idx + 1}: Channel '{channel_name}'\n")

    # Parse --follow arguments if provided
    if args.follow:
        for follow_spec in args.follow:
            try:
                parts = follow_spec.split(":")
                if len(parts) != FOLLOW_SPEC_PARTS:
                    msg = f"Invalid format: {follow_spec}"
                    raise ValueError(msg)  # noqa: TRY301

                # Parse MASTER:POSITION:SLAVE:POSITION
                master_ch, master_pos, slave_ch, slave_pos = parts

                # Validate positions
                if master_pos not in ["left", "right"]:
                    msg = f"Invalid master position: {master_pos}"
                    raise ValueError(msg)  # noqa: TRY301
                if slave_pos not in ["left", "right"]:
                    msg = f"Invalid slave position: {slave_pos}"
                    raise ValueError(msg)  # noqa: TRY301

                # Validate channels exist
                if master_ch not in channel_to_arm:
                    sys.stderr.write(
                        f"{RED}Error: Master channel '{master_ch}' not found{RESET}\n"
                    )
                    return
                if slave_ch not in channel_to_arm:
                    sys.stderr.write(
                        f"{RED}Error: Slave channel '{slave_ch}' not found{RESET}\n"
                    )
                    return

                # Get Arm objects
                master_arm = channel_to_arm[master_ch]
                slave_arm = channel_to_arm[slave_ch]

                # Check for conflicts
                if slave_arm.is_slave:
                    sys.stderr.write(
                        f"{RED}Error: Slave '{slave_ch}' already follows "
                        f"'{slave_arm.follows}'{RESET}\n"
                    )
                    return

                # Configure master arm
                master_arm.position = master_pos
                master_arm.is_master = True

                # Configure slave arm
                slave_arm.position = slave_pos
                slave_arm.is_slave = True
                slave_arm.follows = master_ch
                slave_arm.mirror_mode = (
                    master_pos != slave_pos
                )  # Auto-detect mirror mode

            except ValueError:
                sys.stderr.write(
                    f"{RED}Error: Invalid follow format '{follow_spec}'. Use "
                    f"MASTER:POSITION:SLAVE:POSITION where POSITION is "
                    f"'left' or 'right'{RESET}\n"
                )
                sys.stderr.write(
                    f"{RED}Example: --follow can0:left:can1:right (mirror) or "
                    f"--follow can0:left:can1:left (no mirror){RESET}\n"
                )
                return

        # Validate no channel is both master and slave
        for arm in arms:
            if arm.is_master and arm.is_slave:
                sys.stderr.write(
                    f"{RED}Error: Channel {arm.channel} cannot be both "
                    f"master and slave{RESET}\n"
                )
                return

    sys.stdout.write("\nMaster-Slave Configuration:\n")
    # Group slaves by master
    for master_arm in [arm for arm in arms if arm.is_master]:
        slaves = []
        for slave_arm in [
            arm for arm in arms if arm.is_slave and arm.follows == master_arm.channel
        ]:
            mirror_str = "(mirror)" if slave_arm.mirror_mode else ""
            slave_str = f"{slave_arm.channel}:{slave_arm.position}{mirror_str}"
            slaves.append(slave_str)
        if slaves:
            sys.stdout.write(
                f"  Master: {master_arm.channel}:{master_arm.position} -> "
                f"Slaves: {', '.join(slaves)}\n"
            )

    # Enable all motors (masters with MIT, slaves with PosVel)
    sys.stdout.write("\nEnabling motors for teleoperation...\n")
    for arm in arms:
        if arm.is_slave:
            sys.stdout.write(
                f"  {arm.channel}: Enabling motors with "
                f"Position-Velocity control (slave)\n"
            )
            #await arm.enable_all_motors(ControlMode.POS_VEL)
            await arm.enable_all_motors(ControlMode.MIT)
        else:  # master
            sys.stdout.write(
                f"  {arm.channel}: Enabling motors with MIT control (master)\n"
            )
            await arm.enable_all_motors(ControlMode.MIT)

    # Start teleoperation with monitoring display
    sys.stdout.write("\nTeleoperation mode starting...\n\n")

    # Number of motors (lines to move up)
    num_motors = len(MOTOR_CONFIGS)

    # Print stop instruction before entering raw mode
    stop_msg = "Press 'Q' to stop | X/Y/Z pos hold | Btn1 waypoint | Btn2 ori lock"
    sys.stdout.write(stop_msg + "\n")

    # Impedance control state per side.
    # setpoints: side -> {0: x, 1: y, 2: z} (float values; None = needs latching)
    # lock_ori:  side -> bool  (True = lock full orientation)
    # ori_quats: side -> np.ndarray [w,x,y,z]  (latched target quaternion)
    impedance_setpoints: dict[str, dict[int, float | None]] = {
        "left": {},
        "right": {},
    }
    impedance_lock_ori: dict[str, bool] = {"left": False, "right": False}
    impedance_ori_quats: dict[str, np.ndarray | None] = {
        "left": None,
        "right": None,
    }
    _KEY_TO_AXIS = {"x": 0, "y": 1, "z": 2}
    _AXIS_LABEL = {0: "X", 1: "Y", 2: "Z"}

    # Set terminal to raw mode for keyboard detection
    old_settings = None
    raw_mode = False
    if HAS_TERMIOS:
        try:
            old_settings = termios.tcgetattr(sys.stdin)
            tty.setraw(sys.stdin.fileno())
            raw_mode = True
        except (OSError, termios.error) as e:
            # Might fail in some environments
            logger.debug("Failed to set raw mode: %s", e)

    # Helper for raw mode printing
    def raw_print(msg: str = "") -> None:
        if raw_mode:
            sys.stdout.write(msg.replace("\n", "\r\n"))
            sys.stdout.flush()
        else:
            sys.stdout.write(msg + "\n")

    async def execute_waypoint(
        waypoints: "list[tuple[list[float], float]]",
        master_arm: "Arm",
        slave_arm: "Arm | None",
        hz: float = 200.0,
        hold_time: float = 3.0,
    ) -> None:
        """Move through a sequence of (joint_angles, duration) waypoints with smooth cubic blending.

        Each segment uses a cubic ease-in-out (smoothstep) from the previous waypoint
        to the next. A hold phase at the final waypoint is appended automatically.
        """
        # Current positions as the implicit first waypoint
        start_q = []
        for motor, state in zip(master_arm.motors, master_arm.states):
            if motor is not None and state is not None:
                start_q.append(state.position)
            else:
                start_q.append(0.0)

        # Build knot sequence: list of (q_target, cumulative_time)
        all_q = [start_q]
        seg_durations = []
        for q_target, dur in waypoints:
            n = min(len(q_target), len(start_q))
            all_q.append(list(q_target[:n]) + start_q[n:])
            seg_durations.append(dur)

        cum_times = [0.0]
        for d in seg_durations:
            cum_times.append(cum_times[-1] + d)
        total_move_time = cum_times[-1]
        total_time = total_move_time + hold_time

        n_joints = len(start_q)
        final_q = all_q[-1]
        dt = 1.0 / hz
        steps = int(total_time * hz)

        joint_gains = [
            (150.0, 100.0),
            (150.0, 100.0),
            (150.0, 100.0),
            (150.0, 100.0),
            (4.0, 2.0),
            (4.0, 2.0),
            (4.0, 2.0),
            (4.0, 2.0),
        ]

        for step in range(steps + 1):
            t = step * dt

            if t >= total_move_time:
                interp_q = list(final_q)
            else:
                # Find which segment we're in
                seg = 0
                for si in range(len(seg_durations)):
                    if t < cum_times[si + 1]:
                        seg = si
                        break
                else:
                    seg = len(seg_durations) - 1

                seg_start = cum_times[seg]
                seg_dur = seg_durations[seg]
                alpha = (t - seg_start) / seg_dur if seg_dur > 0 else 1.0
                alpha = min(max(alpha, 0.0), 1.0)
                alpha = 3 * alpha**2 - 2 * alpha**3

                q_from = all_q[seg]
                q_to = all_q[seg + 1]
                interp_q = [
                    s + alpha * (e - s) for s, e in zip(q_from, q_to)
                ]

            # Command master arm
            for idx, motor in enumerate(master_arm.motors):
                if motor is None or idx >= n_joints:
                    continue
                kp, kd = joint_gains[idx] if idx < len(joint_gains) else (2.0, 1.0)
                params = MitControlParams(
                    q=interp_q[idx], dq=0, kp=kp, kd=kd, tau=0,
                )
                try:
                    encode_control_mit(motor._bus, motor._slave_id, motor._motor_limits, params)
                    time.sleep(FRAME_GAP)
                    state = decode_motor_state_sync(motor._bus, motor._master_id, motor._motor_limits)
                    if state is not None:
                        master_arm.states[idx] = state
                except Exception:
                    pass

            # Command slave arm (mirror if needed)
            if slave_arm is not None:
                for idx, slave_motor in enumerate(slave_arm.motors):
                    if slave_motor is None or idx >= n_joints:
                        continue
                    pos = interp_q[idx]
                    if (
                        slave_arm.mirror_mode
                        and idx < len(MOTOR_CONFIGS)
                        and MOTOR_CONFIGS[idx].inverted
                    ):
                        pos = -pos
                    kp, kd = joint_gains[idx] if idx < len(joint_gains) else (2.0, 1.0)
                    alpha_gain = 2.0
                    params = MitControlParams(
                        q=pos, dq=0, kp=kp * alpha_gain, kd=kd * alpha_gain, tau=0,
                    )
                    try:
                        encode_control_mit(slave_motor._bus, slave_motor._slave_id, slave_motor._motor_limits, params)
                        time.sleep(FRAME_GAP)
                        state = decode_motor_state_sync(slave_motor._bus, slave_motor._master_id, slave_motor._motor_limits)
                        if state is not None:
                            slave_arm.states[idx] = state
                    except Exception:
                        pass

            # Print tracking error every 50 steps and on the last step
            if (step % 50 == 0 or step == steps) and slave_arm is not None and gravity_comp is not None:
                actual_q = []
                for motor, st in zip(slave_arm.motors, slave_arm.states):
                    actual_q.append(st.position if motor is not None and st is not None else 0.0)
                actual_pos, actual_quat = gravity_comp.forward_kinematics(actual_q[:7], position=slave_arm.position)
                desired_pos, desired_quat = gravity_comp.forward_kinematics(final_q[:7], position=slave_arm.position)
                pos_err = desired_pos - actual_pos
                ori_err = np.zeros(3)
                tgt_q = desired_quat.copy()
                if np.dot(tgt_q, actual_quat) < 0:
                    tgt_q = -tgt_q
                mujoco.mju_subQuat(ori_err, tgt_q, actual_quat)
                joint_err_deg = [np.degrees(f - a) for f, a in zip(final_q[:n_joints], actual_q[:n_joints])]
                joint_err_str = " ".join(f"j{i}={e:.2f}" for i, e in enumerate(joint_err_deg))
                sys.stdout.write(
                    f"\r  step {step}/{steps}  "
                    f"pos_err: dx={pos_err[0]*1000:.1f} dy={pos_err[1]*1000:.1f} dz={pos_err[2]*1000:.1f} mm  "
                    f"ori_err: r={np.degrees(ori_err[0]):.1f} p={np.degrees(ori_err[1]):.1f} y={np.degrees(ori_err[2]):.1f} deg  "
                    f"q_err(deg): {joint_err_str}"
                    f"\033[K\r\n"
                )
                sys.stdout.flush()

            await asyncio.sleep(dt)

    FRAME_GAP = 0.0003

    # Arduino button reader (non-blocking via background thread)
    button_state = [0, 0, 0, 0]
    _button_lock = threading.Lock()

    def _serial_reader():
        try:
            ser = serial.Serial("/dev/ttyACM0", 115200, timeout=0.05)
            while True:
                line = ser.readline().decode("utf-8", errors="ignore").strip()
                if not line:
                    continue
                parts = line.split()
                if len(parts) == 4:
                    try:
                        vals = [int(p) for p in parts]
                        with _button_lock:
                            button_state[:] = vals
                    except ValueError:
                        pass
        except Exception as e:
            logger.warning("Serial reader failed: %s", e)

    _serial_thread = threading.Thread(target=_serial_reader, daemon=True)
    _serial_thread.start()
    _prev_buttons = [0, 0, 0, 0]

    try:
        # Small initial delay to ensure display is ready
        await asyncio.sleep(0.1)

        loop_count = 0
        start_time = time.time()
        last_loop_time = start_time
        while True:

            # print time taken for this loop iteration in the last column
            loop_start = time.time()
            loop_time = loop_start - last_loop_time
            last_loop_time = loop_start

            
            loop_count += 1
            # Check for key presses
            if raw_mode:
                key = check_keyboard_input()
                if key == "q":
                    raw_print("\nStopping teleoperation...")
                    break
                elif key in _KEY_TO_AXIS:
                    axis = _KEY_TO_AXIS[key]
                    any_was_on = any(axis in sp for sp in impedance_setpoints.values())
                    for side_sp in impedance_setpoints.values():
                        if any_was_on:
                            side_sp.pop(axis, None)
                        else:
                            side_sp[axis] = None
                    state = "OFF" if any_was_on else "ON"
                    raw_print(f"\n  Impedance {_AXIS_LABEL[axis]} {state}\n")

            # Check for button presses (rising edge)
            # Remap: hardware [1,2,3,4] → logical [1,4,3,2]
            _BTN_REMAP = [0, 3, 2, 1]
            with _button_lock:
                btns_raw = button_state[:]
            btns = [btns_raw[_BTN_REMAP[i]] for i in range(4)]
            for i in range(4):
                if btns[i] and not _prev_buttons[i]:
                    if i == 0:
                        # Button 1: move gripper to FoundationPose target
                        fp_pose = pose_listener.latest
                        if fp_pose is None:
                            raw_print(f"\n  Button 1: no FoundationPose pose yet\n")
                        elif gravity_comp is None:
                            raw_print(f"\n  Button 1: gravity comp not enabled\n")
                        else:
                            left_master = None
                            left_slave = None
                            for arm in arms:
                                if arm.is_master and arm.position == "left":
                                    left_master = arm
                                if arm.is_slave and arm.position == "left":
                                    left_slave = arm

                            if left_slave is None:
                                raw_print(f"\n  Button 1: no left follower arm\n")
                            else:
                                cur_q = []
                                for motor, st in zip(left_slave.motors, left_slave.states):
                                    if motor is not None and st is not None:
                                        cur_q.append(st.position)
                                    else:
                                        cur_q.append(0.0)

                                # T_world_tcp and T_world_camera from current joint angles
                                tcp_pos, tcp_quat = gravity_comp.forward_kinematics(
                                    cur_q[:7], position="left",
                                )
                                cam_pos, cam_quat = gravity_comp.kdl.compute_body_pose(
                                    np.array(cur_q[:7]), "openarm_left_camera", side="left",
                                )
                                T_world_tcp = _pose_to_T(tcp_pos, tcp_quat)
                                T_world_cam = _pose_to_T(cam_pos, cam_quat)

                                # T_camera_object from FoundationPose
                                T_cam_obj = _ros_pose_to_T(fp_pose.pose)
                                

                                T_world_obj = T_world_cam @ T_cam_obj

                                # Define waypoints as transforms relative to the object frame
                                # (name, obj_T_wp, duration, gripper_angle)
                                obj_T_wps = []
                                wp1 = np.eye(4)
                                wp1[2, 3] = -0.05
                                wp1[1, 3] = 0.02
                                obj_T_wps.append(("approach", wp1, 1.5, None))

                                wp2 = np.eye(4)
                                wp2[2, 3] = -0.05
                                wp2[1, 3] = 0.02
                                obj_T_wps.append(("approach", wp2, 0.25, -0.1))

                                wp3 = np.eye(4)
                                wp3[2, 3] = -0.1
                                wp3[1, 3] = 0.02
                                obj_T_wps.append(("approach", wp3, 0.5, -0.1))

                                wp4 = np.eye(4)
                                wp4[2, 3] = -0.1
                                wp4[1, 3] = 0.02
                                obj_T_wps.append(("approach", wp4, 0.25, -0.7))

                                raw_print(f"\n  Button 1: move to FPose target ({len(obj_T_wps)} waypoints)")
                                raw_print(f"    Current TCP: [{tcp_pos[0]:.3f}, {tcp_pos[1]:.3f}, {tcp_pos[2]:.3f}]")
                                raw_print(f"    Object (cam): x={fp_pose.pose.position.x:.3f} y={fp_pose.pose.position.y:.3f} z={fp_pose.pose.position.z:.3f}")

                                seed_q = cur_q[:7]
                                current_gripper = cur_q[7] if len(cur_q) > 7 else 0.0
                                ik_waypoints = []
                                aborted = False
                                for wp_idx, (wp_name, obj_T_wp, wp_dur, grip_angle) in enumerate(obj_T_wps):
                                    T_world_wp = T_world_obj @ obj_T_wp
                                    target_pos, target_quat = _T_to_pos_quat(T_world_wp)

                                    raw_print(f"    WP {wp_idx}/{len(obj_T_wps)-1} '{wp_name}': [{target_pos[0]:.3f}, {target_pos[1]:.3f}, {target_pos[2]:.3f}]")

                                    target_q = gravity_comp.inverse_kinematics(
                                        target_pos=target_pos,
                                        target_quat=target_quat,
                                        seed_angles=seed_q,
                                        position="left",
                                    )

                                    verify_pos, verify_quat = gravity_comp.forward_kinematics(
                                        list(target_q), position="left",
                                    )
                                    pos_err = np.linalg.norm(verify_pos - target_pos)
                                    ori_err_vec = np.zeros(3)
                                    vq = verify_quat.copy()
                                    if np.dot(vq, target_quat) < 0:
                                        vq = -vq
                                    mujoco.mju_subQuat(ori_err_vec, target_quat, vq)
                                    rot_err = np.degrees(np.linalg.norm(ori_err_vec))
                                    raw_print(f"      IK err: {pos_err*1000:.1f} mm, {rot_err:.1f} deg")
                                    if pos_err > 0.02 or rot_err > 15.0:
                                        raw_print(f"      WARNING: IK error too large, aborting sequence\n")
                                        aborted = True
                                        break

                                    grip_val = grip_angle if grip_angle is not None else current_gripper
                                    target_q_full = list(target_q) + [grip_val]
                                    ik_waypoints.append((target_q_full, wp_dur))
                                    seed_q = list(target_q)
                                    current_gripper = grip_val

                                if not aborted:
                                    total_dur = sum(d for _, d in ik_waypoints)
                                    raw_print(f"    Executing smooth trajectory ({total_dur:.1f}s, {len(ik_waypoints)} segments)...")
                                    await execute_waypoint(
                                        ik_waypoints, left_master, left_slave,
                                        hz=200.0,
                                    )

                                    side_sp = impedance_setpoints.get("left", {})
                                    for axis in (0, 1, 2):
                                        if axis in side_sp:
                                            side_sp[axis] = float(target_pos[axis])
                                    if impedance_lock_ori.get("left", False):
                                        impedance_ori_quats["left"] = target_quat.copy()
                                    raw_print(f"    All waypoints reached — resuming teleop\n")
                                else:
                                    raw_print(f"    Sequence aborted\n")
                    elif i == 1:
                        # Button 2: toggle orientation lock
                        was_on = any(impedance_lock_ori.values())
                        for side in impedance_lock_ori:
                            impedance_lock_ori[side] = not was_on
                            if was_on:
                                impedance_ori_quats[side] = None
                        state = "OFF" if was_on else "ON"
                        raw_print(f"\n  Button 2: orientation lock {state}\n")
                    else:
                        raw_print(f"\n  Button {i+1} pressed\n")
            _prev_buttons[:] = btns

            # Move cursor up to the first motor line (add +1 for the status line)
            sys.stdout.write(f"\033[{num_motors + 1}A")


            # Small delay before refresh
            await asyncio.sleep(0.001)

            # Control master arms with MIT (gravity comp or zero torque)
            master_arms = {arm.channel: arm for arm in arms if arm.is_master}

            # --- Prepare master arm tasks ---
            # Pre-compute gravity for each master (cheap, CPU-only)
            master_gravity = {}  # channel -> (combined_torques, active_indices)
            
            for master_arm in master_arms.values():
                gravity_torques = None
                active_indices = []

                t_gravity_start = time.time()
                if gravity_comp and master_arm.position in ["left", "right"]:
                    active_positions = []
                    active_velocities = []
                    for idx, (motor, state) in enumerate(
                        zip(master_arm.motors, master_arm.states)
                    ):
                        if motor is not None and state:
                            active_positions.append(state.position)
                            active_velocities.append(state.velocity)
                            active_indices.append(idx)

                    if active_positions:
                        gravity_torques = gravity_comp.compute(
                            active_positions, position=master_arm.position,
                        )

                        # Impedance control for this arm's side
                        side = master_arm.position
                        side_sp = impedance_setpoints.get(side, {})
                        lock_ori = impedance_lock_ori.get(side, False)
                        if side_sp or lock_ori:
                            needs_latch = any(v is None for v in side_sp.values())
                            needs_quat = lock_ori and impedance_ori_quats[side] is None

                            if needs_latch or needs_quat:
                                tcp_pos, tcp_quat = gravity_comp.forward_kinematics(
                                    active_positions, position=side,
                                )
                                for axis in list(side_sp):
                                    if side_sp[axis] is None:
                                        side_sp[axis] = float(tcp_pos[axis])
                                if needs_quat:
                                    impedance_ori_quats[side] = tcp_quat.copy()

                            imp_torques = gravity_comp.impedance_torques(
                                active_positions,
                                setpoint=side_sp,
                                lock_orientation=lock_ori,
                                ori_quat=impedance_ori_quats[side],
                                position=side,
                                stiffness=200.0,
                                rot_stiffness=5.0,
                            )

                            gravity_torques = [
                                g + i
                                for g, i in zip(gravity_torques, imp_torques)
                            ]

                t_gravity_end = time.time()

                master_gravity[master_arm.channel] = (gravity_torques, active_indices)

            # --- Define per-arm BLOCKING workers (run in threads) ---
            def run_master_arm_sync(m_arm: Arm) -> list:
                """Run MIT control for all motors on one master arm (blocking, same bus).

                Uses pipelined send/recv: send ALL commands first, then read ALL responses.
                This overlaps motor processing time with CAN transmission.
                """
                grav_torques, act_indices = master_gravity[m_arm.channel]

                # Phase 1: Send ALL commands as fast as possible
                active_motors = []  # (index, motor) pairs for motors we sent to
                for motor_idx, motor in enumerate(m_arm.motors):
                    if motor is None:
                        continue
                    try:
                        torque = 0.0
                        if grav_torques and motor_idx in act_indices:
                            active_idx = act_indices.index(motor_idx)
                            if active_idx < len(grav_torques):
                                torque = grav_torques[active_idx]

                        params = MitControlParams(
                            q=0, dq=0, kp=0, kd=0, tau=torque,
                        )
                        encode_control_mit(motor._bus, motor._slave_id, motor._motor_limits, params)
                        time.sleep(FRAME_GAP)
                        active_motors.append((motor_idx, motor))
                    except Exception as e:  # noqa: BLE001
                        logger.debug("MIT send failed: %s", e)

                # Phase 2: Read ALL responses
                results = [None] * len(m_arm.motors)
                for motor_idx, motor in active_motors:
                    try:
                        state = decode_motor_state_sync(motor._bus, motor._master_id, motor._motor_limits)
                        results[motor_idx] = state
                    except Exception as e:  # noqa: BLE001
                        logger.debug("MIT recv failed: %s", e)
                return results

            def run_slave_arm_sync(s_arm: Arm, m_arm: Arm) -> list:
                """Run PosVel control for all motors on one slave arm (blocking, same bus).

                Uses pipelined send/recv: send ALL commands first, then read ALL responses.
                """
                # Phase 1: Send ALL commands as fast as possible
                active_motors = []  # (index, slave_motor) pairs
                results = [None] * len(s_arm.motors)

                for idx, (slave_motor, master_state) in enumerate(
                    zip(s_arm.motors, m_arm.states)
                ):
                    if slave_motor is None or master_state is None:
                        continue
                    try:
                        position = master_state.position
                        if (
                            s_arm.mirror_mode
                            and idx < len(MOTOR_CONFIGS)
                            and MOTOR_CONFIGS[idx].inverted
                        ):
                            position = -position

                        # params = PosVelControlParams(
                        #     position=position, velocity=args.velocity
                        # )
                        # encode_control_pos_vel(slave_motor._bus, slave_motor._slave_id, params)

                        # Per-joint MIT control gains
                        #Kp: [240.0, 240.0, 240.0, 240.0, 24.0, 31.0, 25.0, 16.0]
                        #Kd: [3.0, 3.0, 3.0, 3.0, 0.2, 0.2, 0.2, 0.2]
                        joint_gains = [
                            (150.0, 100.0),   # Joint 0: kp=240, kd=3
                            (150.0, 100.0),   # Joint 1: kp=240, kd=3
                            (150.0, 100.0),   # Joint 2: kp=240, kd=3
                            (150.0, 100.0),   # Joint 3: kp=240, kd=3
                            (4.0, 2.0),    # Joint 4: kp=24, kd=0.2
                            (4.0, 2.0),    # Joint 5: kp=31, kd=0.2
                            (4.0, 2.0),    # Joint 6: kp=25, kd=0.2
                            (4.0, 2.0),    # Joint 7: kp=16, kd=0.2
                        ]
                        kp, kd = joint_gains[idx]

                        # scale gains by alpha
                        alpha = 1.5  # You can adjust this to increase/decrease responsiveness
                        kp *= alpha
                        kd *= alpha
                        
                        params = MitControlParams(
                            q=position, dq=0, kp=kp, kd=kd, tau=0,
                        )
                        encode_control_mit(slave_motor._bus, slave_motor._slave_id, slave_motor._motor_limits, params)
                        time.sleep(FRAME_GAP)
                        active_motors.append((idx, slave_motor))
                    except Exception as e:  # noqa: BLE001
                        logger.debug("PosVel send failed: %s", e)

                # Phase 2: Read ALL responses
                for idx, slave_motor in active_motors:
                    try:
                        state = decode_motor_state_sync(slave_motor._bus, slave_motor._master_id, slave_motor._motor_limits)

                        if state.status != 1:
                            sys.stdout.write(
                                f"Motor {slave_motor.slave_id}: Position={state.position:.2f}, "
                                f"Velocity={state.velocity:.2f}, Torque={state.torque:.2f}, "
                                f"Temp={state.temp_mos}°C, Status={state.status}\r\n"
                            )
                            sys.stdout.flush()

                        results[idx] = state
                    except Exception as e:  # noqa: BLE001
                        logger.debug("PosVel recv failed: %s", e)
                return results

            # --- Run all arms concurrently across buses using threads ---
            t_control_start = time.time()

            # Phase 1: Run ALL master arms concurrently (different buses)
            master_tasks = []
            master_arm_refs = []
            for m_arm in master_arms.values():
                loop = asyncio.get_event_loop()
                master_tasks.append(loop.run_in_executor(None, run_master_arm_sync, m_arm))
                master_arm_refs.append(m_arm)

            if master_tasks:
                master_results = await asyncio.gather(*master_tasks)
                for m_arm, result in zip(master_arm_refs, master_results):
                    m_arm.states = result

            t_master_end = time.time()

            # Phase 2: Run ALL slave arms concurrently (they now have fresh master states)
            slave_tasks = []
            slave_arm_refs = []
            for s_arm in [arm for arm in arms if arm.is_slave]:
                if s_arm.follows and s_arm.follows in master_arms:
                    m_arm = master_arms[s_arm.follows]
                    slave_tasks.append(loop.run_in_executor(None, run_slave_arm_sync, s_arm, m_arm))
                    slave_arm_refs.append(s_arm)

            if slave_tasks:
                slave_results = await asyncio.gather(*slave_tasks)
                for s_arm, result in zip(slave_arm_refs, slave_results):
                    s_arm.states = result

            t_control_end = time.time()

            # Print pose and update visualizer
            fp_pose = pose_listener.latest
            left_slave = next(
                (a for a in arms if a.is_slave and a.position == "left"), None,
            )
            if fp_pose is not None and gravity_comp is not None and left_slave is not None:
                cur_q = []
                for motor, st in zip(left_slave.motors, left_slave.states):
                    cur_q.append(st.position if motor is not None and st is not None else 0.0)

                tcp_pos, tcp_quat = gravity_comp.forward_kinematics(cur_q[:7], position="left")
                cam_pos, cam_quat = gravity_comp.kdl.compute_body_pose(
                    np.array(cur_q[:7]), "openarm_left_camera", side="left",
                )
                T_world_tcp = _pose_to_T(tcp_pos, tcp_quat)
                T_world_cam = _pose_to_T(cam_pos, cam_quat)
                T_world_obj = T_world_cam @ _ros_pose_to_T(fp_pose.pose)

                obj_in_tcp = (np.linalg.inv(T_world_tcp) @ T_world_obj)[:3, 3]
                obj_in_world = T_world_obj[:3, 3]
                rpy_tcp = _quat_to_rpy_deg(tcp_quat)
                rpy_obj = _quat_to_rpy_deg(_T_to_pos_quat(T_world_obj)[1])

                pose_str = (
                    f"  tcp_T_obj: x={obj_in_tcp[0]:.3f} y={obj_in_tcp[1]:.3f} z={obj_in_tcp[2]:.3f}"
                    f"  |  world_obj: x={obj_in_world[0]:.3f} y={obj_in_world[1]:.3f} z={obj_in_world[2]:.3f}"
                    f"  |  TCP rpy: [{rpy_tcp[0]:.1f}, {rpy_tcp[1]:.1f}, {rpy_tcp[2]:.1f}]"
                    f"  |  Obj rpy: [{rpy_obj[0]:.1f}, {rpy_obj[1]:.1f}, {rpy_obj[2]:.1f}]"
                )
                frame_vis.update(TCP=T_world_tcp, CAM=T_world_cam, OBJ=T_world_obj)
            elif fp_pose is not None:
                p = fp_pose.pose.position
                pose_str = f"  FPose(cam): x={p.x:.3f} y={p.y:.3f} z={p.z:.3f}"
            else:
                pose_str = "  FPose: waiting..."
            sys.stdout.write(f"\r{pose_str}\033[K\r\n")
            sys.stdout.flush()

    except KeyboardInterrupt:
        # Move cursor below all motor lines
        sys.stdout.write(f"\033[{num_motors}B\n")
        if not raw_mode:
            sys.stdout.write("\nTeleoperation stopped.\n")
        else:
            raw_print("\nTeleoperation stopped.")

    finally:
        # Restore terminal settings first
        if old_settings is not None and HAS_TERMIOS:
            try:
                termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old_settings)
            except (OSError, termios.error) as e:
                logger.debug("Failed to restore terminal settings: %s", e)

        # SAFETY: Disable ALL motors (not just slaves) for safety
        sys.stdout.write("\nDisabling ALL motors for safety...\n")
        for arm in arms:
            await arm.disable_all_motors()
        sys.stdout.write("All motors disabled.\n")

        pose_listener.shutdown()
        frame_vis.close()


def parse_arguments() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Monitor Damiao motor angles")

    parser.add_argument(
        "--interface",
        "-i",
        default="can0",
        help="CAN interface name (default: can0, ignored on Windows/macOS)",
    )

    parser.add_argument(
        "--teleop",
        "-t",
        action="store_true",
        default=False,
        help="Enable teleoperation mode (enables motors with control mode)",
    )

    parser.add_argument(
        "--follow",
        action="append",
        help=(
            "Define follower mappings as MASTER:POSITION:SLAVE:POSITION "
            "where POSITION is 'left' or 'right'. "
            "Mirror mode is automatic when positions differ. "
            "(e.g., --follow can0:left:can1:right for mirror, "
            "--follow can0:left:can1:left for no mirror)"
        ),
    )

    parser.add_argument(
        "--gravity",
        "-g",
        action="store_true",
        help="Enable gravity compensation (MIT mode only)",
    )

    parser.add_argument(
        "--velocity",
        "-v",
        type=float,
        default=1.0,
        help="Velocity parameter for slave motors (default: 1.0)",
    )

    return parser.parse_args()


def run() -> None:
    """Entry point for the monitor script."""
    args = parse_arguments()

    try:
        asyncio.run(main(args))
    except KeyboardInterrupt:
        sys.stderr.write("\nInterrupted by user.\n")
        sys.exit(0)
    except Exception as e:
        logger.exception("Fatal error")
        sys.stderr.write(f"Error: {e}\n")
        sys.exit(1)


if __name__ == "__main__":
    run()
