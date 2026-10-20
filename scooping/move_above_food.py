"""
move_above_food.py — Move spoon above food position using bilinear interpolation

Usage:
    conda activate lerobot
    sudo chmod 666 /dev/ttyACM1
    python move_above_food.py
"""

import time
from lerobot.robots.so_follower import SO101Follower, SO101FollowerConfig

ROBOT_PORT = "/dev/ttyACM1"
ROBOT_ID = "my_awesome_follower_arm"

IMG_W = 640
IMG_H = 480

# Recorded corner joint angles (pan, lift, elbow)
CORNERS = {
    "top_left":     {"pan": -6.92, "lift": 47.79, "elbow":  0.48},
    "top_right":    {"pan": 20.69, "lift": 44.91, "elbow":  7.77},
    "bottom_left":  {"pan": -7.81, "lift": -6.62, "elbow": 81.73},
    "bottom_right": {"pan": 27.17, "lift": -1.73, "elbow": 74.44},
}

# Center position — fixed pose for wrist and gripper joints
CENTER_POSE = {
    "shoulder_pan.pos":   8.91,
    "shoulder_lift.pos": 24.64,
    "elbow_flex.pos":    38.70,
    "wrist_flex.pos":    44.22,
    "wrist_roll.pos":    -3.98,
    "gripper.pos":        0.00,
}

MOVE_FPS = 10
MAX_DELTA = 3.0

JOINT_NAMES = [
    "shoulder_pan.pos", "shoulder_lift.pos", "elbow_flex.pos",
    "wrist_flex.pos", "wrist_roll.pos", "gripper.pos",
]


def bilinear_interpolate(cx, cy):
    """Convert pixel position to joint angles using bilinear interpolation"""
    tx = cx / IMG_W
    ty = cy / IMG_H

    tl = CORNERS["top_left"]
    tr = CORNERS["top_right"]
    bl = CORNERS["bottom_left"]
    br = CORNERS["bottom_right"]

    pan_top   = tl["pan"]   + tx * (tr["pan"]   - tl["pan"])
    pan_bot   = bl["pan"]   + tx * (br["pan"]   - bl["pan"])
    lift_top  = tl["lift"]  + tx * (tr["lift"]  - tl["lift"])
    lift_bot  = bl["lift"]  + tx * (br["lift"]  - bl["lift"])
    elbow_top = tl["elbow"] + tx * (tr["elbow"] - tl["elbow"])
    elbow_bot = bl["elbow"] + tx * (br["elbow"] - bl["elbow"])

    pan   = pan_top   + ty * (pan_bot   - pan_top)
    lift  = lift_top  + ty * (lift_bot  - lift_top)
    elbow = elbow_top + ty * (elbow_bot - elbow_top)

    return pan, lift, elbow


def move_to_pose(robot, target_pose, current_pose):
    """Move arm smoothly to target pose"""
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


def move_above_food(robot, cx, cy):
    """Move spoon above food at pixel position (cx, cy)"""
    # Flip both axes to match camera orientation
    cx = IMG_W - cx
    cy = IMG_H - cy

    pan, lift, elbow = bilinear_interpolate(cx, cy)

    print(f"Food at pixel ({cx}, {cy})")
    print(f"Target: pan={pan:.2f}, lift={lift:.2f}, elbow={elbow:.2f}")

    obs = robot.get_observation()
    current = {k: obs[k] for k in JOINT_NAMES}

    # Use CENTER_POSE as base, only change pan/lift/elbow
    target = dict(CENTER_POSE)
    target["shoulder_pan.pos"]  = pan
    target["shoulder_lift.pos"] = lift
    target["elbow_flex.pos"]    = elbow

    move_to_pose(robot, target, current)
    print("Arrived above food.\n")


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

    print("=== Testing move_above_food ===")
    print("Moving to plate CENTER (320, 240)...")
    move_above_food(robot, 320, 240)

    input("Check position. Press Enter to test TOP-LEFT (0, 0)...")
    move_above_food(robot, 0, 0)

    input("Press Enter to test TOP-RIGHT (640, 0)...")
    move_above_food(robot, 640, 0)

    input("Press Enter to test BOTTOM-LEFT (0, 480)...")
    move_above_food(robot, 0, 480)

    input("Press Enter to test BOTTOM-RIGHT (640, 480)...")
    move_above_food(robot, 640, 480)

    input("Press Enter to return to CENTER...")
    move_above_food(robot, 320, 240)
    print("Done.")

    robot.disconnect()
