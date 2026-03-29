"""Test the local LLM model (Qwen2.5-3B)."""
import os
import subprocess
import sys
import time

MODEL_PATH = 'models/llm/qwen2.5-3b-instruct-q4_k_m.gguf'

if not os.path.exists(MODEL_PATH):
    print(f"Model not found at {MODEL_PATH}")
    print("Run: bash scripts/download_models.sh")
    raise SystemExit(1)

# Auto-install llama-cpp-python if missing
try:
    from llama_cpp import Llama
except ImportError:
    print("Installing llama-cpp-python (one-time, may take a few minutes)...")
    subprocess.check_call(
        [sys.executable, "-m", "pip", "install",
         "llama-cpp-python", "--quiet"],
    )
    from llama_cpp import Llama

print("Loading LLM (this takes 10-30 seconds)...")
start = time.time()
llm = Llama(
    model_path=MODEL_PATH, n_gpu_layers=-1,
    n_ctx=512, verbose=False,
)
load_time = time.time() - start
print(f"Model loaded in {load_time:.1f}s")

print("\nGenerating coaching tip...")
start = time.time()
result = llm.create_chat_completion(
    messages=[
        {"role": "system",
         "content": "You are BoxBunny AI Coach, an expert boxing "
                    "trainer. Keep responses under 2 sentences."},
        {"role": "user",
         "content": "I just did a training session: 87 punches, "
                    "mostly jabs and crosses, accuracy 72%. "
                    "Give me a quick tip."},
    ],
    max_tokens=80,
    temperature=0.7,
)
gen_time = time.time() - start
tip = result['choices'][0]['message']['content']
print(f"AI Coach says ({gen_time:.1f}s): {tip}")

del llm
print("\nLLM test passed.")
