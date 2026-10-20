# SmolVLA Training Commands

All policies were trained on Google Colab (CUDA GPU) with LeRobot, for 10,000 steps with batch size 32.
Before training, log in to HuggingFace (`huggingface_hub.login()`) with a token that has write access.
**Do not commit your token.**

## v5 — SmolVLA architecture, no pretrained weights

```bash
lerobot-train \
    --policy.type=smolvla \
    --dataset.repo_id=yannnnnn6/scooping-all-v3-merged \
    --batch_size=32 \
    --steps=10000 \
    --output_dir=/content/smolvla_v5 \
    --policy.device=cuda \
    --save_freq=1000 \
    --wandb.enable=false \
    --policy.repo_id=yannnnnn6/scooping-smolvla-v5 \
    --policy.push_to_hub=true \
    --num_workers=0
```

Note: `--policy.type=smolvla` builds the architecture only. The saved config shows
`pretrained_path: null` and `load_vlm_weights: false`, i.e. the backbone is randomly
initialised (and frozen), and only the action expert is learned.

## v6 — fine-tuned from smolvla_base

```bash
lerobot-train \
    --policy.path=lerobot/smolvla_base \
    --dataset.repo_id=yannnnnn6/scooping-all-v3-merged \
    --rename_map='{"observation.images.front": "observation.images.camera1", "observation.images.side": "observation.images.camera2"}' \
    --policy.empty_cameras=1 \
    --batch_size=32 \
    --steps=10000 \
    --output_dir=/content/drive/MyDrive/smolvla_v6 \
    --policy.device=cuda \
    --save_freq=1000 \
    --wandb.enable=false \
    --policy.repo_id=yannnnnn6/scooping-smolvla-v6 \
    --policy.push_to_hub=true \
    --num_workers=0
```

smolvla_base expects three cameras (`camera1`–`camera3`), so `front` → `camera1`,
`side` → `camera2`, and the third input is left empty. **The same mapping must be used at deployment.**

## v7 — fine-tuned on the cleaned dataset (114 episodes)

Same as v6, with:

```bash
    --dataset.repo_id=yannnnnn6/scooping-all-v3-merged-clean \
    --output_dir=/content/drive/MyDrive/smolvla_v7 \
    --policy.repo_id=yannnnnn6/scooping-smolvla-v7 \
```

## Resuming an interrupted run

```bash
lerobot-train \
    --config_path=/content/drive/MyDrive/smolvla_v6/checkpoints/last/pretrained_model/train_config.json \
    --resume=true
```
