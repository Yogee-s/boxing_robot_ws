"""Test the local LLM model (Qwen2.5-3B)."""
import time

MODEL_PATH = 'models/llm/qwen2.5-3b-instruct-q4_k_m.gguf'

try:
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
except ImportError:
    print("llama-cpp-python not installed. "
          "Run: pip install llama-cpp-python")
except Exception as e:
    print(f"LLM test failed: {e}")
