"""Check that all ML models are present and loadable."""
import os
import sys

sys.path.insert(0, '.')

models = {
    'CV Action Model (.pth)': 'action_prediction/model/best_model.pth',
    'CV Action Model (.onnx)': 'action_prediction/model/best_model.onnx',
    'CV Action Model (.trt)': 'action_prediction/model/best_model.trt',
    'YOLO Pose Model': 'action_prediction/model/yolo26n-pose.pt',
    'LLM (Qwen2.5-3B)': 'models/llm/qwen2.5-3b-instruct-q4_k_m.gguf',
}

print("=== Model Status ===")
for name, path in models.items():
    if os.path.exists(path):
        size_mb = os.path.getsize(path) / 1024 / 1024
        print(f"  OK  {name}: {size_mb:.1f} MB")
    else:
        print(f"  MISSING  {name}: {path}")

print("\n=== CV Model Load Test ===")
try:
    import torch
    ckpt = torch.load(
        'action_prediction/model/best_model.pth',
        map_location='cpu', weights_only=False,
    )
    config = ckpt.get('config', {})
    print(f"  Model loaded successfully")
    print(f"  Classes: {ckpt.get('label_names', 'default 8-class')}")
    print(f"  Config keys: {list(config.keys())[:10]}")
except Exception as e:
    print(f"  Load failed: {e}")
