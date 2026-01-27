#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="/home/boxbunny/Desktop/doomsday_integration/boxbunny_ws"
MODEL_DIR="$ROOT_DIR/models"
LLM_DIR="$MODEL_DIR/llm"
POSE_DIR="$MODEL_DIR/pose"

mkdir -p "$LLM_DIR" "$POSE_DIR"

# LLM: Qwen2.5-3B-Instruct GGUF (Q4_K_M)
LLM_URL="https://huggingface.co/Qwen/Qwen2.5-3B-Instruct-GGUF/resolve/main/qwen2.5-3b-instruct-q4_k_m.gguf"
LLM_OUT="$LLM_DIR/qwen2.5-3b-instruct-q4_k_m.gguf"

if [ "${FORCE:-0}" = "1" ]; then
  rm -f "$LLM_OUT"
fi

if [ ! -f "$LLM_OUT" ]; then
  echo "Downloading Qwen2.5-3B-Instruct GGUF..."
  curl -L "$LLM_URL" -o "$LLM_OUT"
else
  echo "LLM already exists: $LLM_OUT"
fi


# Pose model: YOLO26n-pose (Ultralytics). Uses ultralytics auto-download to cache, then copies.
POSE_OUT="$POSE_DIR/yolo26n-pose.pt"
if [ ! -f "$POSE_OUT" ]; then
  echo "Downloading YOLO26n-pose via ultralytics..."
  export POSE_OUT
  python3 - <<'PY'
from pathlib import Path
import os
import shutil

from ultralytics import YOLO  # type: ignore

model_name = "yolo26n-pose.pt"
out_path = Path(os.environ["POSE_OUT"])

YOLO(model_name)
cache_dir = Path.home() / ".cache" / "ultralytics"
matches = list(cache_dir.rglob(model_name))
if not matches:
    local = Path.cwd() / model_name
    if local.exists():
        shutil.copy(local, out_path)
        print(f"Copied {local} -> {out_path}")
        raise SystemExit(0)
    raise SystemExit(f"Could not find {model_name} in {cache_dir}")
shutil.copy(matches[0], out_path)
print(f"Copied {matches[0]} -> {out_path}")
PY
else
  echo "YOLO26n-pose already exists: $POSE_OUT"
fi

echo "Models are ready in $MODEL_DIR"
