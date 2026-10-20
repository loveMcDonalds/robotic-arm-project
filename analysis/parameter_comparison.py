"""
parameter_comparison.py

Compare the trained parameters of SmolVLA policies (v5, v6) with the
pretrained smolvla_base checkpoint, layer by layer, grouped by module.

Used in Section 5.2.1 of the report to verify how each policy was initialised:
- frozen modules (vision / text) ~0% difference  -> initialised from pretrained weights
- frozen modules >>0% difference                 -> NOT initialised from pretrained weights

Usage (e.g. in Google Colab):
    pip install huggingface_hub safetensors numpy torch
    python parameter_comparison.py
"""

from huggingface_hub import list_repo_files, hf_hub_download
from safetensors.torch import load_file
import numpy as np

BASE_REPO = "lerobot/smolvla_base"
POLICY_REPOS = {
    "v5": "yannnnnn6/scooping-smolvla-v5",   # trained without pretrained weights
    "v6": "yannnnnn6/scooping-smolvla-v6",   # fine-tuned from smolvla_base
}


def load_repo_weights(repo):
    """Load model.safetensors from a HuggingFace repo (prefers step 010000 / last)."""
    files = list_repo_files(repo)
    cands = [f for f in files if f.endswith("model.safetensors")]
    if not cands:
        print(f"[WARN] no model.safetensors found in {repo}")
        return None
    pick = (next((f for f in cands if "010000" in f), None)
            or next((f for f in cands if "last" in f), None)
            or cands[0])
    print(f"{repo} -> {pick}")
    return load_file(hf_hub_download(repo, pick))


def norm(key):
    """Strip common prefixes so layer names from different checkpoints match."""
    for p in ["model.", "vlm_with_expert.", "policy."]:
        while key.startswith(p):
            key = key[len(p):]
    return key


def group(key):
    if "vision_model" in key:
        return "vision_model"
    if "text_model" in key:
        return "text_model"
    if "lm_expert" in key:
        return "action_expert"
    return "other"


def compare(policy, base_n):
    stats, missing, shape_bad = {}, 0, 0
    for k, w in policy.items():
        kn = norm(k)
        if kn not in base_n:
            missing += 1
            continue
        wb = base_n[kn]
        if wb.shape != w.shape:
            shape_bad += 1
            continue
        wb, w = wb.float(), w.float()
        abs_d = (w - wb).abs().mean().item()
        rel_d = abs_d / (wb.abs().mean().item() + 1e-8)
        stats.setdefault(group(kn), []).append((abs_d, rel_d))
    return stats, missing, shape_bad


def main():
    base = load_file(hf_hub_download(BASE_REPO, "model.safetensors"))
    base_n = {norm(k): v for k, v in base.items()}

    for name, repo in POLICY_REPOS.items():
        policy = load_repo_weights(repo)
        if policy is None:
            continue
        stats, missing, shape_bad = compare(policy, base_n)
        print(f"\n===== {name} vs smolvla_base =====")
        print(f"matched={sum(len(v) for v in stats.values())}  "
              f"name not found={missing}  shape mismatch={shape_bad}")
        for g in ["vision_model", "text_model", "action_expert", "other"]:
            if g in stats:
                vals = stats[g]
                print(f"{g:15s} layers={len(vals):4d}  "
                      f"mean abs diff={np.mean([v[0] for v in vals]):.4f}  "
                      f"mean rel diff={np.mean([v[1] for v in vals]) * 100:.1f}%")


if __name__ == "__main__":
    main()
