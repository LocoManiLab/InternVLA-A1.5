#!/usr/bin/env python3
"""InternVLA-A1.5 policy server speaking Isaac-GR00T's ZMQ protocol
(tcp://0.0.0.0:5550 / get_action).

This repository ships only a *websocket* server for LIBERO
(`evaluation/LIBERO/policy_server/server_policy.py`), but LocoManiLab's real-robot loop,
its offline `infer-check` client and the tele_humanoid_benchmark sim client all
speak the gr00t REQ/REP wire format. This adapter puts that wire format in front
of the repo's own `InternVLAA15Backend`, so an InternVLA-A1.5 checkpoint is driven
by exactly the same client as gr00t / fastwam / starVLA / PRTS -- which is what
makes their numbers comparable.

Mirrors `LocoManiLab/external/PRTS/deployment/run_prts_server.py`; keep the two in sync
if the wire format changes.

Deliberately NOT reimplemented here
-----------------------------------
Checkpoint loading, the eval-mode prompt construction ("Task: ...; Control Mode:
<joint>; State: ..."), action denormalisation and clipping all live in
`evaluation/LIBERO/policy_server/backends/`. Reimplementing any of it is how a
serving path silently drifts from training (PRTS's own unnormaliser no-ops on a
multi-key action layout -- that bug cost a whole eval batch). The LIBERO directory
name is historical: `BasePolicyBackend` is benchmark-agnostic and resolves
everything from the checkpoint's `stats.json` plus the dataset schema registered
for `--robot-type`.

Wire protocol (matches gr00t.policy.server_client.PolicyServer):
    Transport : ZMQ REQ/REP
    Encoding  : msgpack-numpy, request = {"endpoint": str, "data": {...}}
    Endpoints : ping, kill, reset, get_action, get_modality_config

Action layout (78-D, identical to the other G1 SONIC models):
    action.motion_token(64) + action.left_hand_joints(7) + action.right_hand_joints(7)

Usage:
    python deployment/run_internvla_server.py \
        --ckpt-path outputs/internvla_a1_5/<run>/checkpoints/last/pretrained_model \
        --robot-type unitree_g1_sonic --port 5550 \
        --initial-prompt "walk over to the toilet and stand next to it"
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Any, Optional

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
for _p in (REPO_ROOT / "src", REPO_ROOT):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

MOTION_TOKEN_DIM = 64
HAND_DIM = 7
ACTION_DIM = MOTION_TOKEN_DIM + 2 * HAND_DIM      # 78

# The 43-D G1 state is these joint groups in THIS order (sim2real/vla/observation.py
# G1_43DOF_STATE_SLICES). Do not sort the dict: alphabetical order scrambles the
# vector, and "projected_gravity" is an EXTRA field that must not be concatenated
# into the 43.
STATE_ORDER = ("left_leg", "right_leg", "waist", "left_arm",
               "left_hand", "right_arm", "right_hand")


class InternVLAA15Gr00tPolicy:
    """Adapts InternVLAA15Backend.infer() to the gr00t get_action() contract."""

    def __init__(self, backend, initial_prompt: Optional[str] = None,
                 upsample_factor: int = 1):
        self.backend = backend
        self._current_prompt = initial_prompt
        self.upsample_factor = max(1, int(upsample_factor))

    # ---- gr00t endpoints ---------------------------------------------------
    def ping(self) -> dict:
        return {"status": "ok", "message": "internvla_a1_5 gr00t-compat server is running"}

    def get_modality_config(self) -> dict:
        return {}

    def reset(self, options: dict | None = None) -> dict:
        prompt = (options or {}).get("prompt")
        if prompt is not None:
            self._current_prompt = prompt
        return {"status": "reset", "prompt": self._current_prompt}

    def get_action(self, observation: dict[str, Any],
                   options: dict[str, Any] | None = None
                   ) -> tuple[dict[str, Any], dict[str, Any]]:
        t0 = time.monotonic()

        prompt = self._extract_prompt(observation) or self._current_prompt
        if prompt is None:
            raise RuntimeError(
                "No prompt set. Pass --initial-prompt, call reset(prompt=...), or include "
                "obs['language']['annotation.human.task_description'][0][0].")
        self._current_prompt = prompt

        frame = self._extract_image(observation)             # [H,W,3] uint8
        proprio = self._extract_proprio(observation)         # [43] float32
        if proprio is None:
            raise RuntimeError(f"No proprio state in observation; keys={list(observation)}")

        # One image, because the schema for this robot_type maps exactly one camera
        # slot (ego_view -> image0); backends/input_semantics.py validates the count
        # and canonical_preprocess.build_base_sample fills the two unused slots with
        # white + mask=False, exactly as RemapImageKeyTransformFn does at train time.
        example = {"image": [frame], "state": proprio, "lang": prompt}

        t_inf = time.monotonic()
        out = self.backend.infer({"examples": [example]})
        infer_ms = (time.monotonic() - t_inf) * 1000.0

        actions = np.asarray(out["actions"], dtype=np.float32)   # denormalised + clipped
        if actions.ndim == 3:
            actions = actions[0]
        if actions.ndim == 1:
            actions = actions[None, :]
        if actions.shape[-1] < ACTION_DIM:
            raise RuntimeError(
                f"InternVLA-A1.5 returned action dim {actions.shape[-1]}, expected >= "
                f"{ACTION_DIM}. Was the checkpoint trained with --policy.max_action_dim=78 "
                f"from a widened base (tools/widen_action_dim.py)?")
        actions = actions[:, :ACTION_DIM]

        motion = self._upsample(actions[:, :MOTION_TOKEN_DIM])
        left = self._upsample(actions[:, MOTION_TOKEN_DIM:MOTION_TOKEN_DIM + HAND_DIM])
        right = self._upsample(actions[:, MOTION_TOKEN_DIM + HAND_DIM:ACTION_DIM])

        print(f"[internvla_a1_5] get_action: T={actions.shape[0]} -> H_out={motion.shape[0]} "
              f"(x{self.upsample_factor}); infer={infer_ms:.0f} ms; prompt={prompt!r}",
              flush=True)

        action_dict = {
            "action.motion_token": np.ascontiguousarray(motion, dtype=np.float32),
            "action.left_hand_joints": np.ascontiguousarray(left, dtype=np.float32),
            "action.right_hand_joints": np.ascontiguousarray(right, dtype=np.float32),
        }
        return action_dict, {"inference_time_s": time.monotonic() - t0,
                             "chunk_horizon": int(motion.shape[0])}

    # ---- internals ---------------------------------------------------------
    @staticmethod
    def _extract_prompt(obs: dict) -> Optional[str]:
        lang = obs.get("language") or {}
        v = lang.get("annotation.human.task_description")
        while isinstance(v, (list, tuple, np.ndarray)) and len(v):
            v = v[0]
        if isinstance(v, bytes):
            v = v.decode()
        return v if isinstance(v, str) and v else None

    @staticmethod
    def _extract_image(obs: dict) -> np.ndarray:
        """Newest ego-view frame as HWC uint8.

        The tele sim client sends `img[np.newaxis, np.newaxis]`, i.e. (1, 1, H, W, 3)
        -- FIVE dims, under the key "ego_view" (not "video.ego_view"). Real-robot
        callers may send (T, H, W, 3) or bare (H, W, 3). Peel every leading
        singleton/time axis instead of assuming a fixed rank.
        """
        video = obs.get("video") or {}
        for key in ("video.ego_view", "ego_view", "video.webcam", "webcam"):
            arr = video.get(key) if key in video else obs.get(key)
            if arr is None:
                continue
            arr = np.asarray(arr)
            while arr.ndim > 3:          # drop batch/time axes, keeping the newest frame
                arr = arr[-1]
            if arr.ndim != 3 or arr.shape[-1] not in (1, 3, 4):
                raise RuntimeError(f"ego-view image has unexpected shape {np.shape(arr)}; "
                                   f"expected (..., H, W, 3)")
            if arr.dtype != np.uint8:
                arr = np.clip(arr, 0, 255).astype(np.uint8)
            return arr
        raise RuntimeError(f"No ego-view image in observation; keys={list(obs)} "
                           f"video={list(video)}")

    @classmethod
    def _extract_proprio(cls, obs: dict) -> Optional[np.ndarray]:
        state = obs.get("state") or {}
        if not state:
            return None
        keys = ([k for k in STATE_ORDER if k in state]
                if any(k in state for k in STATE_ORDER) else sorted(state))
        parts = []
        for k in keys:
            v = np.asarray(state[k], dtype=np.float32)
            while v.ndim > 1:          # client sends (1, 1, D); keep the newest row
                v = v[-1]
            parts.append(v.ravel())
        return np.concatenate(parts).astype(np.float32) if parts else None

    def _upsample(self, chunk: np.ndarray) -> np.ndarray:
        """Linearly interpolate the action chunk when the control loop wants more steps."""
        if self.upsample_factor == 1 or chunk.shape[0] < 2:
            return chunk
        t_in = np.arange(chunk.shape[0], dtype=np.float32)
        t_out = np.linspace(0, chunk.shape[0] - 1,
                            chunk.shape[0] * self.upsample_factor, dtype=np.float32)
        return np.stack([np.interp(t_out, t_in, chunk[:, d])
                         for d in range(chunk.shape[1])], axis=1).astype(np.float32)


def serve(policy: InternVLAA15Gr00tPolicy, host: str, port: int) -> None:
    import msgpack_numpy as mnp
    import zmq

    ctx = zmq.Context.instance()
    sock = ctx.socket(zmq.REP)
    sock.bind(f"tcp://{host}:{port}")
    print(f"[internvla_a1_5] PolicyServer ready on tcp://{host}:{port}", flush=True)

    while True:
        try:
            req = mnp.unpackb(sock.recv(), raw=False)
            endpoint = req.get("endpoint", "get_action")
            data = req.get("data") or {}

            if endpoint == "ping":
                resp = policy.ping()
            elif endpoint == "kill":
                sock.send(mnp.packb({"status": "ok"}))
                break
            elif endpoint == "reset":
                resp = policy.reset(data)
            elif endpoint == "get_modality_config":
                resp = policy.get_modality_config()
            elif endpoint == "get_action":
                action, info = policy.get_action(data.get("observation", data),
                                                 data.get("options"))
                # The sim client does `action, _info = resp[0], resp[1]` -- it expects a
                # LIST, not a dict. Returning the action dict directly makes the client
                # die with a misleading `KeyError: 0` while the server logs look fine.
                resp = [action, info]
            else:
                resp = {"error": f"unknown endpoint {endpoint!r}"}
            sock.send(mnp.packb(resp))
        except zmq.error.ContextTerminated:
            break
        except Exception as e:                       # keep serving after a bad request
            import traceback
            traceback.print_exc()
            try:
                sock.send(mnp.packb({"error": str(e)}))
            except Exception:
                pass
    print("[internvla_a1_5] server stopped", flush=True)


def load_policy(args) -> InternVLAA15Gr00tPolicy:
    from evaluation.LIBERO.policy_server.backends.policy_backend_internvla_a1_5 import (
        InternVLAA15Backend,
    )

    backend = InternVLAA15Backend(
        ckpt_path=args.ckpt_path,
        device=args.device,
        stats_key=args.stats_key or None,
        robot_type=args.robot_type,
        resize_size=args.resize_size,
        vlm_model_path=args.vlm_model_path or None,
        max_prompt_length=args.max_prompt_length,
        # action_loss_only=True skips loading the WAN video branch. Action prediction
        # (`predict_action_chunk` -> `sample_actions`) never touches the WAN DiT/VAE,
        # so this is correct for serving even for a checkpoint trained WITH the video
        # loss -- and it is required for inference_backend="optimized".
        action_loss_only=args.action_loss_only,
        inference_backend=args.inference_backend,
        wan_model_path=args.wan_model_path or None,
        wan_vae_path=args.wan_vae_path or None,
    )
    # ---- upstream bug: the eval-side letterbox is a no-op --------------------
    # InternVLAA15Backend.__init__ builds `ResizeImagesWithPadFn(height=224, width=224)`
    # with NO mapping and never calls .hydrate(dataset) (it has no dataset at eval
    # time), and ResizeImagesWithPadFn.__call__ iterates `self.mapping.keys()` --
    # empty, so nothing is resized. Training letterboxes our 240x320 ego frame to a
    # padded 224x224 square (ResizeImagesWithPadFn hydrated from the schema's
    # image_mapping); serving it raw would hand Qwen3VLProcessor a different aspect
    # ratio and a different vision-token grid than the policy ever saw. Give the
    # transform the sample's own image keys so the letterbox actually runs.
    # Scoped to this adapter on purpose: "fixing" the shared backend would silently
    # change the LIBERO numbers upstream published with the no-op in place.
    from lerobot.transforms.core import ResizeImagesWithPadFn

    backend.resize = ResizeImagesWithPadFn(
        height=args.resize_size, width=args.resize_size,
        mapping={f"observation.images.image{i}": f"observation.images.image{i}"
                 for i in range(3)},
    )

    meta = backend.metadata()
    print(f"[internvla_a1_5] loaded {args.ckpt_path}\n"
          f"  robot_type={meta['robot_type']} stats_key={meta['stats_key']} "
          f"action_dim={meta['action_dim']} chunk_size={meta['chunk_size']}\n"
          f"  action_mode={meta['action_mode']} denorm={meta['action_denorm_mode']} "
          f"expected_state_dim={meta['expected_state_dim']} "
          f"images={meta['expected_num_input_images']} backend={args.inference_backend}",
          flush=True)
    if int(meta["action_dim"]) != ACTION_DIM:
        raise RuntimeError(
            f"checkpoint stats declare action_dim={meta['action_dim']}, expected {ACTION_DIM}. "
            f"Wrong --stats-key/--robot-type, or the run was not the G1 SONIC one.")
    return InternVLAA15Gr00tPolicy(backend, args.initial_prompt, args.upsample_factor)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description="InternVLA-A1.5 policy server (gr00t-compatible, :5550/get_action).")
    p.add_argument("--ckpt-path", required=True,
                   help="a <run>/checkpoints/<step>/pretrained_model dir "
                        "(config.json + model.safetensors + train_config.json + stats.json)")
    p.add_argument("--host", default="0.0.0.0")
    p.add_argument("--port", type=int, default=5550)
    p.add_argument("--device", default="cuda")
    p.add_argument("--initial-prompt", default=None)
    p.add_argument("--robot-type", default="unitree_g1_sonic",
                   help="picks the stats.json key AND the dataset schema whose "
                        "image_mapping/action_mode the eval prompt must match")
    p.add_argument("--stats-key", default="",
                   help="override the stats.json key (default: --robot-type)")
    p.add_argument("--resize-size", type=int, default=224)
    p.add_argument("--max-prompt-length", type=int, default=650,
                   help="must match the dataset config's max_prompt_length at train time")
    p.add_argument("--vlm-model-path", default="",
                   help="local Qwen3.5-2B dir; overrides the (hub-id) path baked into the "
                        "checkpoint config, which fails on an offline cluster")
    p.add_argument("--inference-backend", default="standard",
                   choices=("standard", "optimized"),
                   help="'optimized' is upstream's low-latency action-only path and is "
                        "~1.6x faster, but it threw a sporadic `CUDA error: an illegal "
                        "memory access was encountered` mid-episode on an H100 "
                        "(2026-08-29) and poisons the CUDA context when it does, so every "
                        "later call fails too. 'standard' ran the same episodes with zero "
                        "faults and is still well inside the sim client's 0.4 s budget.")
    p.add_argument("--no-action-loss-only", dest="action_loss_only", action="store_false",
                   help="load the WAN video branch too (needs --wan-model-path and "
                        "flash-attn installed); only for future-frame visualisation")
    p.add_argument("--wan-model-path", default="")
    p.add_argument("--wan-vae-path", default="")
    p.add_argument("--upsample-factor", type=int, default=1)
    args = p.parse_args(argv)

    serve(load_policy(args), args.host, args.port)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
