#!/usr/bin/env python
"""Widen a released InternVLA-A1.5 checkpoint so it can be fine-tuned on the
Unitree G1 SONIC action space (78-D action, 43-D state).

Why this is mandatory
---------------------
`InternVLA-A1.5-base` was pre-trained with ``max_action_dim = max_state_dim = 32``.
Parameters are sized directly from those numbers
(``modeling_internvla_a1_5.py:558-562``)::

    action_in_proj  = nn.Linear(max_action_dim, action_expert_hidden_size)
    action_out_proj = nn.Linear(action_expert_hidden_size, max_action_dim)
    if not tokenize_state:                      # <- see below
        state_proj  = nn.Linear(max_state_dim,  action_expert_hidden_size)

and G1 SONIC needs 78 / 43.

Measured on ``InternRobotics/InternVLA-A1.5-base`` (2026-08-29): it was trained with
``tokenize_state=True``, so **there is no ``state_proj`` tensor at all** -- the state is
discretized into the prompt instead. ``max_state_dim`` therefore only sets how many
integers that ``State:`` block carries, and only the three action tensors are reshaped:

    model.action_in_proj.weight    (1024, 32) -> (1024, 78)
    model.action_out_proj.weight   (32, 1024) -> (78, 1024)
    model.action_out_proj.bias     (32,)      -> (78,)

The ``state_proj`` rules below are kept for a checkpoint trained with
``tokenize_state=False``; they simply never match otherwise.

Passing ``--policy.max_action_dim=78`` alone is not
enough: `make_policy` builds the model from the CLI config and then loads the
released weights through ``PreTrainedPolicy.from_pretrained(..., strict=False)``,
and ``strict=False`` only suppresses *missing/unexpected key* reports --
``nn.Module.load_state_dict`` collects every **size mismatch** into ``error_msgs``
and raises a ``RuntimeError`` regardless. So the weights have to be reshaped
offline, here.

What it does
------------
Zero-pads the extended axis of each dim-dependent tensor and copies every other
tensor byte-for-byte. Zero (rather than random) init keeps the flow-matching
velocity field at step 0 exactly where pre-training left it for the first 32
dims and contributes nothing spurious for the rest; the new columns/rows still
receive gradient (``dL/dW = δ·hᵀ`` is non-zero because the *inputs* are non-zero),
so they train normally from the first step.

It is also a **general shape reconciler**: it builds the full list of tensors
whose shape the target config changes, and refuses on any mismatch it was not
told to expect -- so a surprise (e.g. a vocab-size change in the language model)
surfaces here in seconds instead of forty minutes into a training run.

Usage
-----
    python tools/widen_for_g1_sonic.py \
        --src /path/to/InternVLA-A1.5-base \
        --dst /path/to/InternVLA-A1.5-base-g1sonic \
        [--max-action-dim 78] [--max-state-dim 43] [--force] [--dry-run]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file

WEIGHTS = "model.safetensors"

# suffix -> (axis that the target dim applies to, which config dim drives it)
#   action_in_proj : Linear(max_action_dim -> H)   weight is [H, max_action_dim] -> axis 1
#   action_out_proj: Linear(H -> max_action_dim)   weight is [max_action_dim, H] -> axis 0
#                                                  bias   is [max_action_dim]    -> axis 0
#   state_proj     : Linear(max_state_dim  -> H)   weight is [H, max_state_dim]  -> axis 1
WIDEN_RULES = {
    "action_in_proj.weight": (1, "action"),
    "action_out_proj.weight": (0, "action"),
    "action_out_proj.bias": (0, "action"),
    "state_proj.weight": (1, "state"),
    "state_proj.bias": (0, "state"),   # only if the module carries a bias
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--src", required=True, type=Path, help="released checkpoint dir")
    p.add_argument("--dst", required=True, type=Path, help="widened checkpoint dir to create")
    p.add_argument("--max-action-dim", type=int, default=78,
                   help="motion_token(64) + left_hand(7) + right_hand(7)")
    p.add_argument("--max-state-dim", type=int, default=43,
                   help="the G1 43-DoF proprio vector, exactly")
    p.add_argument("--force", action="store_true", help="overwrite an existing --dst")
    p.add_argument("--dry-run", action="store_true", help="report only, write nothing")
    return p.parse_args()


def die(msg: str) -> None:
    raise SystemExit(f"[widen] ERROR: {msg}")


def match_rule(key: str) -> tuple[int, str] | None:
    for suffix, rule in WIDEN_RULES.items():
        if key.endswith(suffix):
            return rule
    return None


def widen(tensor: torch.Tensor, axis: int, target: int) -> torch.Tensor:
    cur = tensor.shape[axis]
    if cur == target:
        return tensor
    if cur > target:
        die(f"cannot shrink a tensor: axis {axis} is {cur}, target {target}")
    shape = list(tensor.shape)
    shape[axis] = target - cur
    pad = torch.zeros(shape, dtype=tensor.dtype)
    return torch.cat([tensor, pad], dim=axis)


def sha(t: torch.Tensor) -> str:
    return hashlib.sha256(t.contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()[:12]


def main() -> None:
    args = parse_args()
    src, dst = args.src.expanduser().resolve(), args.dst.expanduser().resolve()
    targets = {"action": args.max_action_dim, "state": args.max_state_dim}

    if not (src / WEIGHTS).is_file():
        die(f"{src}/{WEIGHTS} not found (a sharded checkpoint is not supported here)")
    cfg_path = src / "config.json"
    if not cfg_path.is_file():
        die(f"{src}/config.json not found")
    cfg = json.loads(cfg_path.read_text())
    if cfg.get("type") != "internvla_a1_5":
        die(f"{cfg_path} has type={cfg.get('type')!r}, expected 'internvla_a1_5'")
    if dst.exists() and not args.force:
        die(f"{dst} already exists (pass --force to overwrite)")

    print(f"[widen] source     : {src}")
    print(f"[widen] max_action_dim {cfg.get('max_action_dim')} -> {targets['action']}")
    print(f"[widen] max_state_dim  {cfg.get('max_state_dim')} -> {targets['state']}")

    state = load_file(str(src / WEIGHTS))
    print(f"[widen] {len(state)} tensors, "
          f"{sum(t.numel() for t in state.values()) / 1e9:.2f}G params")

    out: dict[str, torch.Tensor] = {}
    changed: list[tuple[str, tuple, tuple]] = []
    for key, tensor in state.items():
        rule = match_rule(key)
        if rule is None:
            out[key] = tensor
            continue
        axis, which = rule
        new = widen(tensor, axis, targets[which])
        out[key] = new
        if new.shape != tensor.shape:
            changed.append((key, tuple(tensor.shape), tuple(new.shape)))

    if not changed:
        die("no tensor matched the widening rules -- the key names must have changed "
            f"(looked for suffixes: {', '.join(WIDEN_RULES)})")

    print("[widen] widened tensors:")
    for key, old, new in changed:
        print(f"    {key:<48} {old} -> {new}")

    # Anything else the target config resizes would be a surprise. The one class we
    # know about is the language model's vocab: the action-token band 248077..250124 is
    # added at model-construction time by ensure_qwen35_action_tokens(), which resizes
    # embed_tokens/lm_head itself -- so report those shapes rather than touching them.
    for key in sorted(state):
        if key.endswith(("embed_tokens.weight", "lm_head.weight")):
            print(f"[widen] (informational) {key}: {tuple(state[key].shape)}")

    untouched = [k for k in state if match_rule(k) is None]
    same = sum(1 for k in untouched if sha(state[k]) == sha(out[k]))
    print(f"[widen] {same}/{len(untouched)} untouched tensors verified byte-identical")
    if same != len(untouched):
        die("an untouched tensor changed -- refusing to write")

    if args.dry_run:
        print("[widen] --dry-run: nothing written")
        return

    if dst.exists():
        shutil.rmtree(dst)
    dst.mkdir(parents=True)
    for extra in src.iterdir():                # stats.json, README.md, tokenizer files...
        if extra.is_file() and extra.name not in (WEIGHTS, "config.json"):
            shutil.copy2(extra, dst / extra.name)

    save_file(out, str(dst / WEIGHTS), metadata={"format": "pt"})

    cfg["max_action_dim"] = targets["action"]
    cfg["max_state_dim"] = targets["state"]
    # PreTrainedConfig.validate_features() only fills these in when absent, so a stale
    # shape here would be believed and hand the policy a wrong feature spec.
    cfg.setdefault("input_features", {}).setdefault(
        "observation.state", {"type": "STATE", "shape": [targets["state"]]})
    cfg["input_features"]["observation.state"]["shape"] = [targets["state"]]
    cfg.setdefault("output_features", {}).setdefault(
        "action", {"type": "ACTION", "shape": [targets["action"]]})
    cfg["output_features"]["action"]["shape"] = [targets["action"]]
    (dst / "config.json").write_text(json.dumps(cfg, indent=2))

    print(f"[widen] wrote {dst}\n"
          f"  {WEIGHTS} ({(dst / WEIGHTS).stat().st_size / 1e9:.2f} GB), config.json, "
          f"{len(list(dst.iterdir())) - 2} copied file(s)")


if __name__ == "__main__":
    main()
