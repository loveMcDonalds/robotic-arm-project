"""
scoop.py — Fixed scooping motion using relative joint movements

Flow:
    1. move_above_food positions arm above food
    2. This script executes the scooping motion relative to current position

Usage:
    python scoop.py
"""

import sys
import time
from lerobot.robots.so_follower import SO101Follower, SO101FollowerConfig

import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from move_above_food import move_above_food

ROBOT_PORT = "/dev/ttyACM0"
ROBOT_ID = "my_awesome_follower_arm"

JOINT_NAMES = [
    "shoulder_pan.pos", "shoulder_lift.pos", "elbow_flex.pos",
    "wrist_flex.pos", "wrist_roll.pos", "gripper.pos",
]

MOVE_FPS = 10
MAX_DELTA = 3.0

# Relative movements calculated from recorded positions:
# ABOVE FOOD:  pan=-9.35,  lift=19.56, elbow=79.89, wrist_f=-6.21,  wrist_r=-3.79, gripper=0.0
# INSERT:      pan=-12.30, lift=28.16, elbow=72.76, wrist_f=1.78,   wrist_r=-3.79, gripper=0.0
# SCOOP UP:    pan=-11.12, lift=40.95, elbow=76.28, wrist_f=-22.50, wrist_r=-3.79, gripper=0.0
# LIFT UP:     pan=-10.60, lift=26.40, elbow=69.63, wrist_f=-21.57, wrist_r=-3.79, gripper=0.0

SCOOP_DELTAS = [
    # Step 1: INSERT into food
    {
        "shoulder_pan.pos":   -2.95,
        "shoulder_lift.pos":  +8.60,
        "elbow_flex.pos":     -7.13,
        "wrist_flex.pos":     +7.99,
        "wrist_roll.pos":      0.00,
        "gripper.pos":         0.00,
    },
    # Step 2: SCOOP up
    {
        "shoulder_pan.pos":   +1.18,
        "shoulder_lift.pos": +12.79,
        "elbow_flex.pos":     +3.52,
        "wrist_flex.pos":    -24.28,
        "wrist_roll.pos":      0.00,
        "gripper.pos":         0.00,
    },
    # Step 3: LIFT up
    {
        "shoulder_pan.pos":   +0.52,
        "shoulder_lift.pos": -14.55,
        "elbow_flex.pos":     -6.65,
        "wrist_flex.pos":     +0.93,
        "wrist_roll.pos":      0.00,
        "gripper.pos":         0.00,
    },
]

STEP_PAUSE = 0.8


def move_to_target(robot, target_pose, current_pose):
    """Smoothly move arm to target pose, step by step."""
    pose = dict(current_pose)
    while True:
        done = True
        for joint in JOINT_NAMES:
            target_val = target_pose.get(joint, pose[joint])
            delta = target_val - pose[joint]
            if abs(delta) > MAX_DELTA:
                pose[joint] += MAX_DELTA * (1 if delta > 0 else -1)
                done = False
            else:
                pose[joint] = target_val
        robot.send_action(pose)
        if done:
            break
        time.sleep(1.0 / MOVE_FPS)
    return pose


def execute_scoop(robot):
    """Execute scooping motion from current arm position using relative deltas."""
    obs = robot.get_observation()
    current = {k: obs[k] for k in JOINT_NAMES}

    print("Starting scoop sequence...")
    step_names = ["INSERT into food", "SCOOP up", "LIFT up"]

    for i, delta in enumerate(SCOOP_DELTAS):
        print(f"  Step {i+1}: {step_names[i]}")
        target = {k: current[k] + delta.get(k, 0.0) for k in JOINT_NAMES}
        current = move_to_target(robot, target, current)
        time.sleep(STEP_PAUSE)

    print("Scoop complete!")
    return current


if __name__ == "__main__":
    robot_config = SO101FollowerConfig(
        port=ROBOT_PORT,
        id=ROBOT_ID,
        cameras={},
        use_degrees=False,
        disable_torque_on_disconnect=False,
    )
    robot = SO101Follower(robot_config)
    robot.connect()
    print("Connected.\n")

    # Move above food first (change pixel coords as needed)
    FOOD_X, FOOD_Y = 517, 233
    print(f"Moving above food at ({FOOD_X}, {FOOD_Y})...")
    move_above_food(robot, FOOD_X, FOOD_Y)
    print("Arrived above food.\n")

    input("Press Enter to start scooping...")
    execute_scoop(robot)

    robot.disconnect()
    print("Done.")
