"""
click_food_position.py — Click on food in camera view to get pixel position,
then move arm above that position.

Usage:
    conda activate lerobot
    sudo chmod 666 /dev/ttyACM1
    python click_food_position.py
"""

import cv2
import numpy as np
from lerobot.robots.so_follower import SO101Follower, SO101FollowerConfig
from move_above_food import move_above_food, JOINT_NAMES

ROBOT_PORT = "/dev/ttyACM0"
ROBOT_ID = "my_awesome_follower_arm"
CAMERA_INDEX = 0  # front camera


def get_food_position_by_click():
    """Show camera view, let user click on food, return pixel position"""
    cap = cv2.VideoCapture(CAMERA_INDEX)
    clicked = []

    def on_click(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            clicked.append((x, y))
            print(f"Clicked at pixel ({x}, {y})")

    print("Camera opened. Click on the food, then press Enter.")
    cv2.namedWindow("Click on food")
    cv2.setMouseCallback("Click on food", on_click)

    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        # Draw crosshair at clicked position
        if clicked:
            cx, cy = clicked[-1]
            cv2.circle(frame, (cx, cy), 10, (0, 255, 0), 2)
            cv2.line(frame, (cx-20, cy), (cx+20, cy), (0, 255, 0), 2)
            cv2.line(frame, (cx, cy-20), (cx, cy+20), (0, 255, 0), 2)
            cv2.putText(frame, f"({cx}, {cy})", (cx+15, cy-15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        cv2.putText(frame, "Click on food, press Enter to confirm, q to quit",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        cv2.imshow("Click on food", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == 13 and clicked:  # Enter
            break
        if key == ord('q'):
            clicked = []
            break

    cap.release()
    cv2.destroyAllWindows()
    return clicked[-1] if clicked else None


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

    while True:
        print("\n=== Click Food Position ===")
        pos = get_food_position_by_click()

        if pos is None:
            print("No position selected. Exiting.")
            break

        cx, cy = pos
        print(f"Moving arm above food at ({cx}, {cy})...")
        move_above_food(robot, cx, cy)

        cmd = input("Move to another position? (y/n): ")
        if cmd.lower() != 'y':
            break

    robot.disconnect()
    print("Done.")
