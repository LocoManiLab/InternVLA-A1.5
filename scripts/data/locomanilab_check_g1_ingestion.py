#!/usr/bin/env python
"""Gate ① for the LocoManiLab G1 SONIC integration: prove the converted dataset really
reaches the policy in the shape the policy expects -- before any GPU, any base
checkpoint and any training step is paid for.

It exercises the *real* path (`lerobot.datasets.factory.make_dataset` with an
`InternVLAA15DatasetConfig`), not a hand-rolled loader, because most of the ways
this can go wrong are silent and live inside that path:

  * an unknown `robot_type` does not raise -- `SchemaRegistry.get()` logs a warning
    and returns an identity schema (`observation.image -> image0`), so the model
    would train on keys that do not exist;
  * `NormalizeTransformFn` warns-and-skips a stats key it cannot find, so a missing
    entry means training on unnormalized values with a plausible loss curve;
  * `tokenize_state` exists on both the policy and the dataset config with OPPOSITE
    defaults, and the disagreement silently removes proprioception from the prompt;
  * a broken task string shows up as the literal "0" rather than an error.

Usage:
    python scripts/data/locomanilab_check_g1_ingestion.py \
        --root /path/to/internvla_a15_data --repo-id v30/all31_merged \
        [--robot-type unitree_g1_sonic] [--samples 8] [--norm-samples 200]
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

ACTION_DIM = 78          # motion_token 64 + left_hand 7 + right_hand 7
STATE_DIM = 43
CHUNK = 50
STATS_KEYS = ("observation.state", "action.motion_token",
              "teleop.left_hand_joints", "teleop.right_hand_joints")
STATS_WIDTHS = {"observation.state": STATE_DIM, "action.motion_token": 64,
                "teleop.left_hand_joints": 7, "teleop.right_hand_joints": 7}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--root", required=True, type=Path,
                   help="the v3.0 data root; exported as HF_LEROBOT_HOME")
    p.add_argument("--repo-id", required=True, help="e.g. v30/all31_merged")
    p.add_argument("--robot-type", default="unitree_g1_sonic")
    p.add_argument("--samples", type=int, default=8, help="samples to pull through the pipeline")
    p.add_argument("--norm-samples", type=int, default=200,
                   help="samples used for the normalization sanity check (0 to skip)")
    p.add_argument("--vlm-model-path", default=None,
                   help="local Qwen3.5-2B; needed under HF_HUB_OFFLINE=1")
    return p.parse_args()


FAILURES: list[str] = []


def _load_tasks(ds_dir) -> list[str]:
    """The task strings, from v3.0's meta/tasks.parquet (v2.1's tasks.jsonl as fallback)."""
    import json

    pq = ds_dir / "meta" / "tasks.parquet"
    if pq.is_file():
        import pandas as pd

        df = pd.read_parquet(pq)
        if "task" in df.columns:
            return [str(t) for t in df["task"].tolist()]
        # the converter writes the task string as the INDEX and task_index as the column
        return [str(t) for t in df.index.tolist()]
    jl = ds_dir / "meta" / "tasks.jsonl"
    if jl.is_file():
        return [json.loads(l)["task"] for l in jl.read_text().splitlines() if l.strip()]
    return []


def check(ok: bool, label: str, detail: str = "") -> None:
    print(f"  [{'PASS' if ok else 'FAIL'}] {label}" + (f"  ({detail})" if detail else ""))
    if not ok:
        FAILURES.append(label)


def main() -> None:
    args = parse_args()
    root = args.root.expanduser().resolve()
    # Read at lerobot import time; also the only dataset-dir resolution that both
    # LeRobotDatasetMetadata and factory.find_info_json_path_for_repo agree on.
    os.environ["HF_LEROBOT_HOME"] = str(root)

    import json

    import numpy as np
    import torch

    from lerobot.configs.train import TrainPipelineConfig
    from lerobot.dataset_schemas import get_schema
    from lerobot.datasets.factory import make_dataset
    from lerobot.policies.internvla_a1_5 import InternVLAA15Config
    from lerobot.policies.internvla_a1_5.configuration_internvla_a1_5 import (
        InternVLAA15DatasetConfig,
    )

    ds_dir = root / args.repo_id
    print(f"\n=== 1. dataset on disk: {ds_dir}")
    info = json.loads((ds_dir / "meta" / "info.json").read_text())
    check(info.get("codebase_version") == "v3.0", "codebase_version is v3.0",
          str(info.get("codebase_version")))
    check(info.get("robot_type") == args.robot_type,
          f"robot_type is {args.robot_type}", str(info.get("robot_type")))
    for col in ("observation.images.ego_view", *STATS_KEYS):
        check(col in info.get("features", {}), f"column {col} present")
    print(f"  episodes={info.get('total_episodes')} frames={info.get('total_frames')} "
          f"fps={info.get('fps')}")

    print("\n=== 2. stats (NormalizeTransformFn warns-and-skips a missing key)")
    stats = json.loads((ds_dir / "meta" / "stats.json").read_text())
    for key in STATS_KEYS:
        entry = stats.get(key)
        if entry is None:
            check(False, f"stats[{key}]", "missing -> would train UNNORMALIZED")
            continue
        width = len(np.asarray(entry["mean"]).reshape(-1))
        check(width == STATS_WIDTHS[key], f"stats[{key}] width", f"{width}")

    print(f"\n=== 3. schema registry for robot_type={args.robot_type!r}")
    schema = get_schema(args.robot_type)
    check(schema.robot_type == args.robot_type,
          "registry returned the real schema, not the identity fallback",
          schema.robot_type)
    check(schema.get_action_keys() == list(STATS_KEYS[1:]),
          "action keys in order motion_token, left_hand, right_hand",
          str(schema.get_action_keys()))
    check(schema.image_mapping == {"observation.images.ego_view": "observation.images.image0"},
          "single ego camera mapped to image0", str(schema.image_mapping))

    print("\n=== 4. the real pipeline (factory.make_dataset)")
    ds_cfg_kwargs = dict(
        repo_id=args.repo_id, action_mode="abs", tokenize_state=True,
        max_state_dim=STATE_DIM, max_action_dim=ACTION_DIM, chunk_size=CHUNK,
        use_fast_action_tokens=False,
        # True is the default and KeyErrors here: factory assigns
        # meta.stats[<camera key>][...] for every camera and LocoManiLab datasets carry no
        # image entry in episodes_stats.
        use_imagenet_stats=False, use_external_stats=False, dist_loading=False,
    )
    if args.vlm_model_path:
        ds_cfg_kwargs["vlm_model_name_or_path"] = args.vlm_model_path
    policy_kwargs = dict(max_action_dim=ACTION_DIM, max_state_dim=STATE_DIM,
                         chunk_size=CHUNK, n_action_steps=CHUNK,
                         action_loss_only=True, enable_vqa_loss=False,
                         tokenize_state=True)
    if args.vlm_model_path:
        policy_kwargs["vlm_model_name_or_path"] = args.vlm_model_path
    cfg = TrainPipelineConfig(
        dataset=InternVLAA15DatasetConfig(**ds_cfg_kwargs),
        policy=InternVLAA15Config(**policy_kwargs),
        vqa_dataset=None, num_workers=0, seed=42,
    )
    dataset, data_stats = make_dataset(cfg)
    check(list(data_stats) == [args.robot_type],
          "make_dataset keyed its stats by the robot_type the server will ask for",
          str(list(data_stats)))
    print(f"  dataset length: {len(dataset)}")

    print("\n=== 5. one sample end to end")
    sample = dataset[0]
    act, state = sample["action"], sample["observation.state"]
    check(tuple(act.shape) == (CHUNK, ACTION_DIM), "action shape (chunk, 78)",
          str(tuple(act.shape)))
    check(state.shape[-1] == STATE_DIM, "state width 43", str(tuple(state.shape)))
    check(torch.isfinite(act).all().item(), "action has no NaN/Inf")
    # The raw image keys are consumed by the chat processor; UnifyInternVLAA15InputsTransformFn
    # returns only pixel_values / image_grid_thw. So check the letterbox through the GRID:
    # a 240x320 frame resized-with-pad to a 224x224 square gives a SQUARE patch grid.
    grid = sample["observation.image_grid_thw"].reshape(-1, 3)
    check(grid.numel() > 0, "vision tokens produced", str(tuple(grid.shape)))
    check(bool((grid[:, 1] == grid[:, 2]).all()),
          "patch grid is square -> the 240x320 frame really was letterboxed to 224x224",
          str(grid.tolist()))
    # A failed video decode is NOT an error in lerobot: it logs
    # `[video_decode_fallback] ... using_zeros` and hands back a black frame, so a broken
    # torchcodec looks exactly like a working one until the model learns nothing from vision.
    px = sample["observation.pixel_values"].float()
    check(float(px.std()) > 1e-3,
          "pixel_values are a real frame, not the zero-fallback",
          f"mean={float(px.mean()):.4f} std={float(px.std()):.4f}")

    # Decode the prompt: the two things that break silently are a lost task string
    # (DreamZero's converter once turned every prompt into the literal "0") and a
    # missing State: block (the tokenize_state two-flag trap).
    from transformers import AutoTokenizer

    tok_src = args.vlm_model_path or "Qwen/Qwen3.5-2B"
    text = AutoTokenizer.from_pretrained(tok_src).decode(
        sample["observation.input_ids"], skip_special_tokens=False)
    tasks = _load_tasks(ds_dir)
    hit = next((t for t in tasks if t and t in text), None)
    check(hit is not None, "prompt carries a real task string from meta/tasks",
          repr(hit)[:70] if hit else "NOT FOUND")
    check("State:" in text, "prompt carries the discretized State: block")
    check("Control Mode: <joint>" in text, "prompt carries the schema's action_mode")
    print(f"  prompt head: {text[:200]!r}")

    if args.norm_samples > 0:
        print(f"\n=== 6. normalization sanity over {args.norm_samples} samples")
        idx = np.linspace(0, len(dataset) - 1, args.norm_samples).astype(int)
        acts = torch.stack([dataset[int(i)]["action"][0] for i in idx]).float().numpy()
        raw = np.asarray(stats["action.motion_token"]["std"]).reshape(-1)
        live = raw > 1e-6                       # constant dims normalize to 0, correctly
        m = acts[:, :64][:, live].mean(axis=0)
        s = acts[:, :64][:, live].std(axis=0)
        check(bool(np.abs(m).max() < 0.25), "normalized motion_token mean ~ 0",
              f"max|mean|={np.abs(m).max():.3f}")
        check(bool(0.6 < np.median(s) < 1.5), "normalized motion_token std ~ 1",
              f"median std={np.median(s):.3f}")

    print("\n" + "=" * 60)
    if FAILURES:
        print(f"GATE ① FAILED: {len(FAILURES)} check(s)")
        for f in FAILURES:
            print(f"  - {f}")
        raise SystemExit(1)
    print("GATE ① PASSED -- data ingestion is correct; proceed to the training smoke.")


if __name__ == "__main__":
    main()
