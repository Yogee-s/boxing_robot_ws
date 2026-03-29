"""Quick status check of all hardware and dependencies."""
import os

checks = [
    ('RealSense Camera',
     lambda: __import__('pyrealsense2').context().query_devices().size() > 0),
    ('PyTorch',
     lambda: __import__('torch').__version__),
    ('CUDA Available',
     lambda: __import__('torch').cuda.is_available()),
    ('PySide6',
     lambda: __import__('PySide6').__version__),
    ('FastAPI',
     lambda: __import__('fastapi').__version__),
    ('MediaPipe',
     lambda: __import__('mediapipe').__version__),
    ('llama-cpp-python',
     lambda: (getattr(__import__('llama_cpp'), '__version__', None)
              or 'installed')),
    ('bcrypt',
     lambda: __import__('bcrypt').__version__),
    ('qrcode',
     lambda: 'installed' if __import__('qrcode') else ''),
    ('CV Model',
     lambda: os.path.exists('action_prediction/model/best_model.pth')),
    ('YOLO Model',
     lambda: os.path.exists('action_prediction/model/yolo26n-pose.pt')),
    ('LLM Model',
     lambda: os.path.exists(
         'models/llm/qwen2.5-3b-instruct-q4_k_m.gguf')),
    ('Main DB',
     lambda: os.path.exists('data/boxbunny_main.db')),
]

print("=== Hardware & Dependency Check ===")
for name, check_fn in checks:
    try:
        result = check_fn()
        if isinstance(result, bool):
            status = 'OK' if result else 'MISSING'
        else:
            status = f'OK ({result})'
    except Exception as e:
        status = f'MISSING ({type(e).__name__})'
    symbol = '+' if 'OK' in status else '-'
    print(f"  [{symbol}] {name}: {status}")
