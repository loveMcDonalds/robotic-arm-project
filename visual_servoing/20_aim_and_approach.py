"""
20_aim_and_approach.py — Full pipeline: aiming + approaching

Workflow:
    1. Move to HOME_POSE
    2. AIMING: elbow_flex aims according to error_y, while shoulder_pan
       corrects left/right according to error_x
    3. Once aligned (|error_y| < 40 px): lock elbow_flex and switch to APPROACHING
    4. APPROACHING: shoulder_lift moves forward at a constant rate, while
       elbow_flex remains locked
    5. Stop when the target distance is reached
       (mouth_width_px >= STOP_DISTANCE_PX)

Usage:
    conda activate lerobot
    sudo chmod 666 /dev/ttyACM0
    python 20_aim_and_approach.py
    (Press Ctrl+C to stop)
"""

import time
import cv2
import mediapipe as mp
from lerobot.robots.so_follower import SO101Follower, SO101FollowerConfig

ROBOT_PORT = "/dev/ttyACM0"
ROBOT_ID = "my_awesome_follower_arm"
MODEL_PATH = "face_landmarker.task"
IMG_W, IMG_H = 640, 480

# ── Parameters ──────────────────────────────────────
KP_PAN = 0.01
MAX_DELTA_PAN = 1.5

ELBOW_STEP_PER_PX = 0.03
ELBOW_MIN_STEP = 1.5
ELBOW_MAX_STEP = 3.0
ELBOW_SEARCH_STEP = 1.5
CENTER_TOLERANCE_PX = 40

FORWARD_STEP = 0.8
FORWARD_DIRECTION = 1.0  # Increasing shoulder_lift = forward; set to -1.0 if reversed
STOP_DISTANCE_PX = 135.0
MOVE_FPS = 6
# ──────────────────────────────────────────────

HOME_POSE = {
    "shoulder_pan.pos": 65.61,
    "shoulder_lift.pos": 30.01,
    "elbow_flex.pos": -27.80,
    "wrist_flex.pos": 65.71,
    "wrist_roll.pos": -2.56,
    "gripper.pos": 0.00,
}

JOINT_NAMES = [
    "shoulder_pan.pos", "shoulder_lift.pos", "elbow_flex.pos",
    "wrist_flex.pos", "wrist_roll.pos", "gripper.pos",
]


def clamp(val, lo, hi):
    return max(lo, min(hi, val))


BaseOptions = mp.tasks.BaseOptions
FaceLandmarker = mp.tasks.vision.FaceLandmarker
FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode
options = FaceLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=MODEL_PATH),
    running_mode=VisionRunningMode.VIDEO,
    num_faces=1,
)
landmarker = FaceLandmarker.create_from_options(options)

robot_config = SO101FollowerConfig(
    port=ROBOT_PORT, id=ROBOT_ID, cameras={}, use_degrees=False,
    disable_torque_on_disconnect=False,
)
robot = SO101Follower(robot_config)
robot.connect()

# Move to HOME_POSE
print("[INIT] Moving to HOME_POSE...")
obs = robot.get_observation()
pose = {k: obs[k] for k in JOINT_NAMES}
for joint, target in HOME_POSE.items():
    pose[joint] = target
robot.send_action(pose)
time.sleep(2.0)

cap = cv2.VideoCapture(0)
timestamp_ms = 0

# State variables
is_centered = False
is_approaching = False
elbow_search_target = None

print("Running. Ctrl+C to stop.\n")

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        obs = robot.get_observation()
        current_elbow = obs["elbow_flex.pos"]

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
        timestamp_ms += 33
        result = landmarker.detect_for_video(mp_image, timestamp_ms)

        if not result.face_landmarks:
            # Face lost: hold all joints at their current angles and only move
            # elbow_flex upwards to search for the face.
            # shoulder_lift is not reset; it stays at its current position.
            is_centered = False
            if elbow_search_target is None:
                elbow_search_target = current_elbow
            elbow_search_target += (-1.0 * ELBOW_SEARCH_STEP)
            pose = {k: obs[k] for k in JOINT_NAMES}  # Use current actual angles, not HOME_POSE
            pose["elbow_flex.pos"] = elbow_search_target
            robot.send_action(pose)
            print(f"[SEARCH] shoulder_lift={obs['shoulder_lift.pos']:.2f}(held)  "
                  f"elbow_now={current_elbow:.2f}  target={elbow_search_target:.2f}")
            time.sleep(1 / MOVE_FPS)
            continue

        # Face found: reset the search
        elbow_search_target = None

        landmarks = result.face_landmarks[0]
        h, w = frame.shape[:2]
        upper_lip = landmarks[13]
        lower_lip = landmarks[14]
        mouth_x = (upper_lip.x + lower_lip.x) / 2 * w
        mouth_y = (upper_lip.y + lower_lip.y) / 2 * h
        left_corner = landmarks[61]
        right_corner = landmarks[291]
        mouth_width_px = abs(right_corner.x - left_corner.x) * w

        error_x = mouth_x - IMG_W / 2
        error_y = mouth_y - IMG_H / 2

        # Distance check: stop when close enough
        if mouth_width_px >= STOP_DISTANCE_PX:
            print(f"[STOP] Reached target distance. mouth_width={mouth_width_px:.1f}")
            break

        # shoulder_pan left/right correction (active throughout)
        delta_pan = clamp(KP_PAN * error_x, -MAX_DELTA_PAN, MAX_DELTA_PAN)

        # ── Aiming phase ──
        if not is_centered:
            if abs(error_y) < CENTER_TOLERANCE_PX:
                is_centered = True
                elbow_search_target = None  # Reset the search target
                print(f"[CENTERED] error_y={error_y:+.1f} — starting approach")
            else:
                elbow_sign = 1.0 if error_y > 0 else -1.0
                raw_step = abs(error_y) * ELBOW_STEP_PER_PX
                if abs(error_y) < 80:
                    elbow_step = 0.8
                else:
                    elbow_step = max(ELBOW_MIN_STEP, min(ELBOW_MAX_STEP, raw_step))

                # Accumulate from the previous target rather than the actual position (avoids getting stuck)
                if elbow_search_target is None:
                    elbow_search_target = current_elbow
                elbow_search_target += elbow_sign * elbow_step
                elbow_target = elbow_search_target

                pose = {k: obs[k] for k in JOINT_NAMES}  # Use current actual angles
                pose["shoulder_pan.pos"] = obs["shoulder_pan.pos"] + delta_pan
                pose["elbow_flex.pos"] = elbow_target
                robot.send_action(pose)
                print(f"[AIMING] error_y={error_y:+.1f}  elbow_now={current_elbow:.2f}  "
                      f"target={elbow_target:.2f}  mouth_width={mouth_width_px:.1f}")
                time.sleep(1 / MOVE_FPS)
                continue

        # ── Approaching phase (aligned) ──
        pose = {k: obs[k] for k in JOINT_NAMES}
        pose["shoulder_pan.pos"] += delta_pan
        pose["shoulder_lift.pos"] += FORWARD_STEP * FORWARD_DIRECTION
        # elbow_flex remains locked
        robot.send_action(pose)
        print(f"[APPROACH] error_y={error_y:+.1f}  shoulder_lift={obs['shoulder_lift.pos']:.2f}  "
              f"mouth_width={mouth_width_px:.1f}")

        time.sleep(1 / MOVE_FPS)

except KeyboardInterrupt:
    print("\nStopped by user.")
finally:
    robot.disconnect()
    cap.release()
    print("Done.")
