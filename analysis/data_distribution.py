"""
data_distribution.py

Plot how the scooping demonstrations are distributed across the plate.

For each episode, the scooping position is approximated as the frame at which
the shoulder_pan joint deviates furthest from its initial value. The
shoulder_pan (left-right) and shoulder_lift (near-far) angles at that frame are
plotted as histograms.

Used in Section 5.2.2 of the report.

Usage (e.g. in Google Colab, with LeRobot installed):
    python data_distribution.py
"""

from lerobot.datasets.lerobot_dataset import LeRobotDataset
import numpy as np
import matplotlib.pyplot as plt

DATASET = "yannnnnn6/scooping-all-v3-merged"   # 135 episodes used to train v5 / v6
PAN_IDX, LIFT_IDX = 0, 1                        # action dims: shoulder_pan, shoulder_lift


def scooping_positions(repo_id):
    ds = LeRobotDataset(repo_id)
    data = ds.hf_dataset.with_format("numpy")
    ep = np.array(data["episode_index"])
    act = np.stack(data["action"])

    pan_pt, lift_pt = [], []
    for e in np.unique(ep):
        a = act[ep == e]
        i = np.argmax(np.abs(a[:, PAN_IDX] - a[0, PAN_IDX]))  # scooping moment
        pan_pt.append(a[i, PAN_IDX])
        lift_pt.append(a[i, LIFT_IDX])
    return np.array(pan_pt), np.array(lift_pt)


def main():
    pan, lift = scooping_positions(DATASET)
    print("episodes:", len(pan))

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].hist(pan, bins=15, edgecolor="black")
    axes[0].set_xlabel("shoulder_pan at scooping position (°)")
    axes[0].set_ylabel("number of episodes")
    axes[0].set_title("(a) Left–right direction")

    axes[1].hist(lift, bins=15, edgecolor="black")
    axes[1].set_xlabel("shoulder_lift at scooping position (°)")
    axes[1].set_title("(b) Near–far direction")

    plt.tight_layout()
    plt.savefig("distribution_135.png", dpi=300, bbox_inches="tight")
    plt.show()


if __name__ == "__main__":
    main()
