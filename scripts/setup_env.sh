#!/usr/bin/env bash
# Build the conda env this fork's G1 SONIC recipe trains and serves in.
#
# Idempotent: every step is skipped if already satisfied, so re-running is cheap and
# safe. Four of the pins below are NOT cosmetic -- each was a failure first, and two of
# them fail SILENTLY (see the comments at each step). Read those before relaxing a pin.
#
# Usage:
#     bash scripts/setup_env.sh
#
# Overridable by environment:
#     ENV_NAME     conda env to create            (default: internvla_a1_5)
#     CONDA_SETUP  a script to source before      (default: none; on an isolated conda
#                  touching conda                  installation point this at it, e.g.
#                                                  CONDA_SETUP=/path/to/conda-env.sh)
set -uo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_NAME="${ENV_NAME:-internvla_a1_5}"

# An isolated conda (own envs_dirs/pkgs_dirs/pip index) is sourced through a site
# script rather than assumed, so this works both on a shared cluster and on a plain
# `conda init` machine.
if [ -n "${CONDA_SETUP:-}" ]; then
    # shellcheck disable=SC1090
    source "$CONDA_SETUP"
elif ! command -v conda >/dev/null; then
    echo "conda not on PATH; set CONDA_SETUP=<script that provides it>" >&2
    exit 1
fi
: "${CONDA_ENVS_PATH:=$(conda info --base)/envs}"

step() { echo; echo "=============== $* ==============="; }

step "1. create env"
if [ -d "$CONDA_ENVS_PATH/$ENV_NAME" ]; then
    echo "env already exists"
else
    conda create -y -n "$ENV_NAME" python=3.11 || exit 1
fi
conda activate "$ENV_NAME" || exit 1
# conda's python-3.11 metapackage does not always pull pip in this isolated
# installation; without it every `pip install` below dies with "command not found".
command -v pip >/dev/null || conda install -y -n "$ENV_NAME" pip || exit 1
python -V
echo "prefix=$CONDA_PREFIX"

step "2. ffmpeg 7.x + svt-av1 (conda-forge)"
# Must be the ENV's ffmpeg, not /usr/bin/ffmpeg: torchcodec dlopen()s the libav* shared
# libraries next to the interpreter. PIN 7.x -- conda-forge now defaults to ffmpeg 9,
# and torchcodec only ships decoder shims for FFmpeg 4/5/6/7 (libavutil.so.56..59), so
# an unpinned install leaves every `libtorchcodec_core*.so` unloadable.
"$CONDA_PREFIX/bin/ffmpeg" -version 2>/dev/null | head -1 | grep -q "version 7\." \
  || conda install -y -c conda-forge "ffmpeg=7.*" svt-av1 || exit 1

step "3. torch 2.10.0 + torchvision 0.25.0 (cu128)"
python -c "import torch,sys; print(torch.__version__); sys.exit(0 if torch.__version__.startswith('2.10') else 1)" \
  || pip install torch==2.10.0 torchvision==0.25.0 --index-url https://download.pytorch.org/whl/cu128 || exit 1

step "4. transformers 5.2.0"
python -c "import transformers,sys; print(transformers.__version__); sys.exit(0 if transformers.__version__=='5.2.0' else 1)" \
  || pip install transformers==5.2.0 || exit 1

step "5. pip install -e . (the lerobot fork)"
cd "$REPO" || exit 1
pip install -e . || exit 1

step "6. flash-linear-attention + causal-conv1d (PERFORMANCE ONLY -- pure-torch fallbacks exist)"
pip install flash-linear-attention==0.5.0 --no-build-isolation || echo "!! fla install FAILED (falls back to torch_chunk_gated_delta_rule)"
pip install causal-conv1d==1.6.1 --no-build-isolation || echo "!! causal-conv1d install FAILED (falls back to torch_causal_conv1d_update)"

step "7. patch transformers with the repo's replacement model code"
TDIR="$CONDA_PREFIX/lib/python3.11/site-packages/transformers"
[ -d "$TDIR" ] || { echo "no $TDIR"; exit 1; }
for p in pi0 pi05 internvla_a1_5; do
    cp -r "src/lerobot/policies/$p/transformers_replace/models" "$TDIR/" || exit 1
    echo "patched from $p"
done

step "7b. torchcodec matched to torch 2.10 + cu128"
# `pip install -e .` pulls `torchcodec>=0.2.1` from PyPI, which today resolves to 0.16.0
# built against CUDA 13 and a newer libtorch stable ABI -- it fails to dlopen, and
# lerobot then SILENTLY substitutes a zero frame for every video read
# (`[video_decode_fallback] ... using_zeros`): black images, normal-looking loss.
# Version compatibility measured on this machine (torch 2.10.0+cu128, ffmpeg 7.1.1):
#   0.16.0 (PyPI)   -> needs libcudart.so.13 / libnvrtc.so.13     FAIL
#   0.12.0 (PyPI)   -> undefined symbol: torch_from_blob          FAIL (needs torch >= 2.11)
#   0.11.1 +cu128   -> undefined symbol: torch_dtype_float4_...   FAIL
#   0.10.0 +cu128   -> decodes                                    OK   <= pinned
#    0.9.1 +cu128   -> could not load libtorchcodec               FAIL
pip install --no-deps "torchcodec==0.10.0" --index-url https://download.pytorch.org/whl/cu128 || exit 1

step "7c. conda activate hook so the loader can find this env's FFmpeg"
# `conda activate` does not add $CONDA_PREFIX/lib to LD_LIBRARY_PATH; torchcodec needs
# it. An activate.d hook makes this true for every consumer -- LocoManiLab's runner, the policy
# server, an interactive shell -- instead of each one having to remember the export.
mkdir -p "$CONDA_PREFIX/etc/conda/activate.d"
cat > "$CONDA_PREFIX/etc/conda/activate.d/zz_torchcodec_ffmpeg.sh" <<'HOOK'
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
HOOK

step "7d. tilelang -- REQUIRED on Hopper, not optional"
# fla's Triton backward for the Gated DeltaNet refuses to run here:
#   RuntimeError: Triton >= 3.4.0 on Hopper GPUs produces incorrect results for
#   gated chunk_bwd_dqkwg (see #640). Please install tilelang
# It is a hard stop by design -- the kernel is silently WRONG, so fla would rather
# fail than train garbage. tilelang supplies the alternative backend. (Removing fla
# entirely also works, via the pure-torch fallback, but is much slower.)
python -c "import tilelang" 2>/dev/null || pip install tilelang || exit 1

step "8. wire deps for the gr00t-compat policy server"
pip install pyzmq msgpack-numpy || exit 1

step "9. verify"
python - <<'PY'
import torch, transformers
print("torch      ", torch.__version__, "cuda", torch.version.cuda, "avail", torch.cuda.is_available())
print("transformers", transformers.__version__)
for m in ("fla", "causal_conv1d"):
    try:
        __import__(m); print(f"{m:<14} OK")
    except Exception as e:
        print(f"{m:<14} MISSING ({type(e).__name__}) -> pure-torch fallback")
from transformers.models.qwen3_5 import Qwen3_5Tokenizer
print("qwen3_5     OK")
from lerobot.policies.internvla_a1_5 import InternVLAA15Config, InternVLAA15Policy
print("InternVLAA15 OK, default max_action_dim =", InternVLAA15Config().max_action_dim)
from lerobot.dataset_schemas import get_schema
s = get_schema("unitree_g1_sonic")
print("schema      ", s.robot_type, s.get_action_keys(), s.image_mapping, s.action_mode)
import torchcodec
print("torchcodec  ", torchcodec.__version__)
PY
echo; echo "=============== DONE ==============="
