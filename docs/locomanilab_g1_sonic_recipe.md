# G1 SONIC training recipe (LocoManiLab)

The exact command LocoManiLab's `internvla_a15` plugin builds, recorded here so the run is
reproducible without LocoManiLab. Every flag that is not a default is load-bearing; the ones
marked 🔴 fail **silently** if dropped.

## Environment

    bash scripts/setup_env.sh          # see the pins and why each one exists

## 1. Dataset: LeRobot v2.1 -> v3.0

This fork trains only on v3.0. The converter also stamps the controller lane into the
copy's `robot_type` (LocoManiLab's datasets say `G1FloatingBody`, which four different
controllers share):

    python scripts/data/locomanilab_lerobot_v21_to_v30.py \
        --src  /path/to/all31_merged \
        --root /path/to/internvla_a15_data

Then prove the data actually reaches the policy (this is the gate, not "it finished"):

    python scripts/data/locomanilab_check_g1_ingestion.py \
        --root /path/to/internvla_a15_data --repo-id v30/all31_merged \
        --vlm-model-path /path/to/Qwen3.5-2B

## 2. Base checkpoint: widen 32 -> 78

The released base is `max_action_dim=32`; G1 needs 78. `from_pretrained` passes
`strict=False`, but PyTorch raises on a **size** mismatch regardless, so this is not
optional:

    python tools/widen_for_g1_sonic.py \
        --src /path/to/InternVLA-A1.5-base \
        --dst /path/to/InternVLA-A1.5-base-g1sonic

## 3. Train

    HF_LEROBOT_HOME=/path/to/internvla_a15_data \
    WANDB_MODE=offline HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
    TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
    CUDA_HOME=/usr/local/cuda-12.8 CUDA_VISIBLE_DEVICES=4,5,6,7 \
    accelerate launch --multi_gpu --num_processes=4 --num_machines=1 --machine_rank=0 \
        --main_process_port=29521 \
        src/lerobot/scripts/lerobot_train.py \
        --policy.type=internvla_a1_5 \
        --policy.pretrained_path=/path/to/InternVLA-A1.5-base-g1sonic \
        --policy.max_action_dim=78 --policy.max_state_dim=43 \
        --policy.chunk_size=50 --policy.n_action_steps=50 \
        --policy.action_loss_only=true \
        --policy.enable_vqa_loss=false \
        --policy.tokenize_state=true \
        --policy.optimizer_lr=5e-05 \
        --policy.scheduler_warmup_steps=2000 \
        --policy.scheduler_decay_steps=60000 --policy.scheduler_decay_lr=5e-06 \
        --policy.dtype=bfloat16 --policy.push_to_hub=false \
        --policy.vlm_model_name_or_path=/path/to/Qwen3.5-2B \
        --dataset.type=internvla_a1_5 \
        --dataset.repo_id=v30/all31_merged \
        --dataset.action_mode=abs \
        --dataset.tokenize_state=true \
        --dataset.max_state_dim=43 --dataset.max_action_dim=78 \
        --dataset.chunk_size=50 \
        --dataset.use_fast_action_tokens=false \
        --dataset.use_imagenet_stats=false \
        --dataset.use_external_stats=false --dataset.dist_loading=false \
        --dataset.vlm_model_name_or_path=/path/to/Qwen3.5-2B \
        --output_dir=/path/to/outputs/all31_20k --job_name=all31_20k \
        --seed=42 --batch_size=8 --steps=20000 --save_freq=5000 --log_freq=100 \
        --num_workers=6 --wandb.enable=false

### Why these flags

| flag | why |
|---|---|
| 🔴 `--dataset.use_imagenet_stats=false` | default is **true** and KeyErrors on this data: `factory.py` assigns `meta.stats[<camera>][...]` for every camera and LocoManiLab datasets carry no image entry in `episodes_stats.jsonl`. Pointless here anyway (VISUAL norm is IDENTITY; the Qwen processor normalizes). |
| 🔴 `tokenize_state` on **both** `--policy.` and `--dataset.` | the two configs declare it with **opposite defaults** (policy `True`, dataset `False`). Disagreeing builds the model fine but drops the `State:` block from the prompt: it trains with **no proprioception at all**, and the loss curve looks normal. |
| 🔴 `--policy.enable_vqa_loss=false` | with no text labels the chat processor emits `labels = full_like(input_ids, -100)` (not `None`), so `lm_head` still materializes ~650 x 250k fp32 logits per sample for a loss that is identically 0. |
| `--policy.max_action_dim=78` / `--policy.max_state_dim=43` | must match the widened checkpoint. 78 = motion_token(64) + left_hand(7) + right_hand(7). |
| `--dataset.action_mode=abs` | `motion_token` is an absolute latent pose code, never a delta from state. Also drops `DeltaActionTransformFn`. |
| `--dataset.use_fast_action_tokens=false` | FAST text tokens over a 64-D latent carry no meaning. |
| `--policy.action_loss_only=true` | skips WAN weight loading entirely and pins `video_loss` to 0. Flip it off for the foresight branch -- but install flash-attn first, `wan/modules/model.py` calls `flash_attention()` with no SDPA fallback. |
| no `--dataset.root` | the fork resolves a dataset dir two contradictory ways; `HF_LEROBOT_HOME` + `repo_id` with `root` unset is the one form both agree on. |
| no `--dataset.episodes` | the upstream converter stamps an episode's chunk/file index *before* deciding to flush, so the episode straddling a 100 MB boundary is recorded against the previous file. Harmless whole-dataset, wrong for a subset. |

`output_dir` must **not** exist -- `TrainPipelineConfig.validate()` raises `FileExistsError`.

## 4. Serve

    python deployment/run_internvla_server.py \
        --ckpt-path <run>/checkpoints/last/pretrained_model \
        --robot-type unitree_g1_sonic --port 5550 \
        --vlm-model-path /path/to/Qwen3.5-2B

gr00t-compatible ZMQ `:5550/get_action`. `--inference-backend` defaults to `standard`;
`optimized` is ~1.6x faster but threw a sporadic CUDA illegal-memory-access mid-episode.

## Measured (2026-08-29, all31_merged, 4xH100)

20000 steps in 2h32m (2.2-2.4 it/s), `loss_action` 1.19 -> 0.44, no NaN.

Offline open-loop, 5 episodes x 150 frames: `motion_token` MSE mean 0.0144,
pred/GT temporal std 1.05-1.23x (not collapsed), temporal corr median +0.27..+0.77.

Closed-loop sim, 5 tasks x 20 episodes, genuine **58/100**.
⚠️ open-loop correlation did **not** predict closed-loop success (r = -0.35 over those
5 tasks) -- the offline check catches collapse and denormalization bugs, but cannot
rank task performance.
