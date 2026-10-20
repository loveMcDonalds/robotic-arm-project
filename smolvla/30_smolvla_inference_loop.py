"""
30_smolvla_inference_loop.py — Closed-loop SmolVLA inference on the SO-101

Flow:
    1. Connect the arm and the front/side cameras
    2. Load the trained SmolVLA policy
    3. Move the spoon above the food with move_above_food
    4. Run the policy in a loop until Ctrl+C or 'q' is pressed

Usage:
    conda activate lerobot
    python 30_smolvla_inference_loop.py
"""
import sys
import time
import numpy as np
import torch
import cv2

import os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'scooping'))
from move_above_food import move_above_food

from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
from lerobot.policies.smolvla.processor_smolvla import make_smolvla_pre_post_processors
from lerobot.robots.so_follower import SO101Follower, SO101FollowerConfig
from lerobot.cameras.opencv import OpenCVCameraConfig

# ── Configuration ──────────────────────────────────────
# Local path to the downloaded checkpoint (e.g. from yannnnnn6/scooping-smolvla-v5)
MODEL_PATH = "models/scooping-smolvla-v5/checkpoints/010000/pretrained_model"
ROBOT_PORT = "/dev/ttyACM0"
ROBOT_ID = "my_awesome_follower_arm"
FRONT_CAMERA_INDEX = 2
SIDE_CAMERA_INDEX = 0
TASK = "Scoop the mashed potatoes"
INFERENCE_FPS = 10

# Fixed food position (pixel coordinates in the front camera)
FOOD_PIXEL_X = 322
FOOD_PIXEL_Y = 306
# ──────────────────────────────────────────────

JOINT_NAMES = [
    "shoulder_pan.pos", "shoulder_lift.pos", "elbow_flex.pos",
    "wrist_flex.pos", "wrist_roll.pos", "gripper.pos",
]

# Dataset statistics of scooping-all-v3-merged (used to normalise state / unnormalise actions)
STATE_MEAN = np.array([6.2580, 32.9720, 26.5138, 48.6262, -2.4590, 28.5211], dtype=np.float32)
STATE_STD  = np.array([6.0268,  9.3468, 17.7920, 13.6975,  4.6124, 19.6435], dtype=np.float32)

def img_to_bchw(img):
    arr = np.array(img, dtype=np.float32) / 255.0
    return arr.transpose(2, 0, 1)[np.newaxis, ...]

print("=== SmolVLA closed-loop inference ===")
print(f"Model: {MODEL_PATH}")
print(f"Task: {TASK}\n")

# 1. Connect the arm and cameras first (while the cameras are warmed up)
print("Connecting arm and cameras...")
robot_config = SO101FollowerConfig(
    port=ROBOT_PORT,
    id=ROBOT_ID,
    cameras={
        "front": OpenCVCameraConfig(index_or_path=FRONT_CAMERA_INDEX, width=640, height=480, fps=30, warmup_s=5),
        "side": OpenCVCameraConfig(index_or_path=SIDE_CAMERA_INDEX, width=640, height=480, fps=30, warmup_s=5),
    },
    use_degrees=False,
    disable_torque_on_disconnect=False,
)
robot = SO101Follower(robot_config)
robot.connect()
print("Arm and cameras connected.\n")

# 2. Load the policy
print("Loading model...")
policy = SmolVLAPolicy.from_pretrained(MODEL_PATH)
device = "cuda" if torch.cuda.is_available() else "cpu"
policy.to(device)
policy.eval()
print(f"Model loaded on device: {device}\n")

# 3. Build the preprocessor / postprocessor
dataset_stats = policy.dataset_stats if hasattr(policy, 'dataset_stats') else None
preprocessor, postprocessor = make_smolvla_pre_post_processors(
    config=policy.config,
    dataset_stats=dataset_stats,
)

# 4. Move the arm above the (fixed) food position
print(f"Moving arm above food at ({FOOD_PIXEL_X}, {FOOD_PIXEL_Y})...")
move_above_food(robot, FOOD_PIXEL_X, FOOD_PIXEL_Y)
print("Arm is above the food.\n")

input("Press Enter to start SmolVLA inference...")

# 5. Inference loop
print("Starting inference loop... Ctrl+C to stop\n")
step = 0

try:
    while True:
        t_start = time.time()

        obs = robot.get_observation()

        cv2.imshow("Front Camera", cv2.cvtColor(np.array(obs["front"]), cv2.COLOR_RGB2BGR))
        cv2.imshow("Side Camera", cv2.cvtColor(np.array(obs["side"]), cv2.COLOR_RGB2BGR))
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        state_raw = np.array([[obs[k] for k in JOINT_NAMES]], dtype=np.float32)
        state_normalized = (state_raw - STATE_MEAN) / STATE_STD

        raw_obs = {
            "observation.images.front": img_to_bchw(obs["front"]),
            "observation.images.side": img_to_bchw(obs["side"]),
            "observation.state": state_normalized,
            "task": TASK,
        }

        processed = preprocessor(raw_obs)
        processed_gpu = {
            k: v.to(device) if isinstance(v, torch.Tensor) else v
            for k, v in processed.items()
        }

        with torch.no_grad():
            action = policy.select_action(processed_gpu)

        ACTION_MEAN = torch.tensor([6.3080, 32.0442, 25.2983, 48.3196, -2.5404, 53.7091], device=device)
        ACTION_STD  = torch.tensor([6.0139,  9.2046, 17.9127, 13.8381,  4.6021,  9.8337], device=device)
        action_real = action * ACTION_STD + ACTION_MEAN
        action_np = action_real.cpu().numpy().flatten()

        action_dict = {joint: float(action_np[i]) for i, joint in enumerate(JOINT_NAMES)}
        robot.send_action(action_dict)

        step += 1
        elapsed = time.time() - t_start
        print(f"step {step:4d} | "
              f"pan={action_np[0]:.2f} lift={action_np[1]:.2f} "
              f"elbow={action_np[2]:.2f} wrist_f={action_np[3]:.2f} "
              f"wrist_r={action_np[4]:.2f} gripper={action_np[5]:.2f} | "
              f"time={elapsed*1000:.0f}ms")

        sleep_time = max(0, 1/INFERENCE_FPS - elapsed)
        time.sleep(sleep_time)

except KeyboardInterrupt:
    print(f"\nStopped after {step} steps.")
finally:
    robot.disconnect()
    cv2.destroyAllWindows()
    print("Arm disconnected.")
