# BoxBunny — Notable Changes Log

Running log of the larger changes shipped to BoxBunny. Newest entries go at the top.

---

# 2026-04-17 — BLE base control, live person-tracking, sparring & reaction-test fixes

One integrated set of changes across the Jetson, the base Arduino, and the GUI.

## 1. BLE bridge — height + base rotation via Arduino

The base has two motors (yaw rotation + height lead-screw) wired to an Arduino Uno R4 WiFi running `ble_control.ino`. Before this change, height control flowed through the old V4 GUI / Teensy path (`/boxbunny/robot/height_remote`). Now:

- New ROS node **`ble_bridge_node`** ([src/boxbunny_core/boxbunny_core/ble_bridge_node.py](../src/boxbunny_core/boxbunny_core/ble_bridge_node.py)) owns the single BLE link to `BoxBunny Base`.
- Main GUI (Settings page press-and-hold) and phone dashboard (`POST /api/remote/height`) both publish `HeightCommand` on `/boxbunny/robot/height`; the bridge translates to `HUP:<pwm>`, `HDOWN:<pwm>`, `HSTOP` over BLE.
- Teensy simulator already subscribes to `/boxbunny/robot/height` — unchanged, still works.

**Height motor safety — two stacked watchdogs**:

- **Arduino firmware** (`ble_control.ino`): 250 ms refresh-window watchdog. Every `HUP`/`HDOWN` stamps `lastHeightCmdMs`. If silent for 250 ms, firmware zeroes the PWM. Motor can only run while commands are actively arriving. BLE drop, Jetson crash, unplugged cable — all safe.
- **Jetson `ble_bridge_node`**: 500 ms deadman. If the publisher goes silent mid-motion while the link is still up, the bridge re-issues `HSTOP`.

Full detail: [hardware/ble_bridge.md](hardware/ble_bridge.md).

## 2. Live person-tracking integrated into Free Training and Sparring

Replaces the standalone `notebooks/scripts/test_base_tracking.py` workflow.

- `ble_bridge_node` subscribes to `/boxbunny/cv/person_direction` + `/boxbunny/cv/user_tracking` and emits `L:<rpm>/R:<rpm>/S` when tracking is enabled.
- Toggled via two new services: `/boxbunny/ble/tracking_start` and `/boxbunny/ble/tracking_stop` (both `std_srvs/Trigger`).
- **Near-range gating** (`config/boxbunny.yaml → ble_tracking:`): default `near_range_m = 0.8` with `hysteresis_m = 0.15`. A user at 2 m produces no rotation commands even if they're the only one in frame. Stops background walkers from driving the motor.
- **Centre dead-zone tightened**: `run_with_ros.py:send_direction` shrunk from middle 30% to middle 10% of the frame, so the robot tracks lateral movement sooner. 20 px hysteresis kept.

UI:
- **Sparring Setup page** and **Sparring Session page** both have a Start/Stop Tracking button.
- **Free Training Session page** has the same toggle.
- All three share state via `/boxbunny/ble/status` (1 Hz JSON). Buttons are disabled until BLE is connected.
- **Safety**: every session-end path (`_on_stop`, `_on_timer_done`, `on_leave`, BLE disconnect) calls `tracking_stop` → motor halts. The motor cannot be left running after a session.

## 3. Sparring — counter counter was stuck at 0 (two bugs, both fixed)

### Bug A — source tag lost in strike_complete

The V4 Arm Control GUI publishes its own `strike_complete` payload on `/boxbunny/robot/strike_complete` *without* the `source` field, racing with `robot_node`'s correctly-tagged version. The sparring UI's `_on_strike_complete` was checking `data.get("source") == "counter"` and getting `None` half the time, so the COUNTERS counter barely ticked.

**Fix**: [gui_bridge.py](../src/boxbunny_gui/boxbunny_gui/gui_bridge.py) now also subscribes to `/boxbunny/robot/command` directly and emits a new `robot_command_issued(dict)` Qt signal. The sparring session page increments the counter **the instant `sparring_engine` fires** a RobotCommand with `source="counter"` — bypasses the broken pipeline entirely and ticks ~5 s sooner than before.

### Bug B — `_robot_busy` sticking True

`sparring_engine._robot_busy` only cleared on `/robot/strike_feedback`. If the V4 GUI wasn't running, or a message dropped, the flag stayed True after the first attack and blocked every subsequent counter.

**Fix**: added a 6 s busy-timeout watchdog (`_clear_stale_busy`) in [sparring_engine.py](../src/boxbunny_core/boxbunny_core/sparring_engine.py). Auto-releases the flag if feedback never arrives, with a warning log telling you to check whether V4/teensy_simulator is up.

## 4. Reaction Test — too sensitive with background people

Both trigger paths (local pose-motion delta, ROS-fused `punch_confirmed`) fired on *any* movement in the frame. In a busy gym, background walkers tripped the timer.

**Fix**: subscribes to `/boxbunny/cv/user_tracking` (cv_node already picks the largest + most centred person per frame) and gates triggers on four conditions:
- `user_detected == True`
- `0 < depth ≤ 2.0 m`
- bbox centre within the middle 40% of the 960 px frame
- tracking message within the last 500 ms (fail-safe on stale data)

Constants are at the top of [reaction_test_page.py](../src/boxbunny_gui/boxbunny_gui/pages/performance/reaction_test_page.py) — tune tighter if needed.

## 5. File / config changes (quick reference)

- **NEW** `src/boxbunny_core/boxbunny_core/ble_bridge_node.py`
- **NEW** `docs/hardware/boxing_arm_control.md`, `docs/hardware/ble_bridge.md`
- `config/boxbunny.yaml` — added `ble_tracking:` section.
- `config/ros_topics.yaml` — added `ble_status`, `tracking_start`, `tracking_stop`; marked `height_remote` deprecated; updated `person_direction` note.
- `src/boxbunny_core/setup.py` + `boxbunny_full.launch.py` — register/launch `ble_bridge_node`.
- `src/boxbunny_core/boxbunny_core/sparring_engine.py` — busy-timeout watchdog + `_set_busy`/`_clear_stale_busy`.
- `src/boxbunny_gui/boxbunny_gui/gui_bridge.py` — new signals: `ble_status_changed`, `robot_command_issued`, `user_tracking_changed`; new services: `call_tracking_start`, `call_tracking_stop`.
- `src/boxbunny_gui/boxbunny_gui/pages/sparring/sparring_config_page.py` — tracking card, accepts `bridge` kwarg.
- `src/boxbunny_gui/boxbunny_gui/pages/sparring/sparring_session_page.py` — tracking row, listens to `robot_command_issued` for COUNTERS.
- `src/boxbunny_gui/boxbunny_gui/pages/training/training_session_page.py` — tracking row in free-training mode, safety stop hooks.
- `src/boxbunny_gui/boxbunny_gui/pages/performance/reaction_test_page.py` — foreground-user gating.
- `src/boxbunny_dashboard/boxbunny_dashboard/api/remote.py` — phone height endpoint publishes `HeightCommand`.
- `Boxing_Arm_Control/ros2_ws/ble_control/ble_control.ino` — 5-line height watchdog addition.

## 6. How to operate (for the next person)

- **Launch the stack**: `ros2 launch boxbunny_core boxbunny_full.launch.py` (or `colcon build --symlink-install` first). `ble_bridge_node` auto-starts. Power-cycle the Arduino if it doesn't reconnect within ~10 s.
- **Height**: press & hold Up/Down in GUI Settings or phone dashboard. Arduino firmware refresh-window is 250 ms — under normal load (10 Hz publish = 100 ms interval), one dropped BLE packet won't stutter the motor.
- **Tracking**: go to Sparring Setup *or* any session page, press Start Tracking. Disabled until BLE is up (status label shows `BLE: connected`). Stops automatically on session end — you never need to remember.
- **Known-safe failure modes**: GUI crash mid-hold (firmware stops in 250 ms), BLE drop mid-tracking (bridge forces tracking off locally, firmware stops motor), Jetson reboot (Arduino is standalone, motor stops the moment commands stop arriving).

---

# 2026-04-13 — LLM Upgrade & Image Chat Feature

## 1. LLM Model Switch: Qwen 2.5-3B -> Gemma 4 E2B

### Why

The previous model (Qwen 2.5-3B-Instruct Q4_K_M) was chosen when the system was documented as having 8GB shared memory. The Jetson Orin NX actually has 16GB LPDDR5 (102.4 GB/s bandwidth), which allows a larger, more capable model. Google's Gemma 4 E2B (released April 2, 2026) is edge-optimised with Per-Layer Embeddings and supports multimodal (image + text) input.

### Model Comparison

| | Qwen 2.5-3B (Previous) | Gemma 4 E2B (New) |
|---|---|---|
| Parameters | 3B | 5.1B total, 2.3B active |
| Architecture | Dense transformer | Per-Layer Embeddings (edge-optimised) |
| Quantisation | Q4_K_M GGUF | Q4_K_M GGUF |
| File size | ~2.0 GB | ~2.9 GiB (~3.1 GB) |
| VRAM usage | ~2 GB | ~3.1 GB |
| Multimodal | Text only | Text + Image (with mmproj) |
| Source | Qwen/Qwen2.5-3B-Instruct-GGUF | unsloth/gemma-4-E2B-it-GGUF |
| License | Apache 2.0 | Apache 2.0 |

### GPU Memory Budget (16GB Jetson Orin NX)

| Component | Previous | Now (text only) | Now (with vision) |
|---|---|---|---|
| TensorRT action model | ~200MB | ~200MB | ~200MB |
| TensorRT YOLO pose | ~150MB | ~150MB | ~150MB |
| LLM model | ~2.0GB | ~3.1GB | ~3.1GB |
| Vision projector (mmproj) | N/A | 0 (not loaded) | ~1GB (lazy) |
| Frame buffers | ~100MB | ~100MB | ~100MB |
| **Total GPU** | **~2.5GB** | **~3.6GB** | **~4.6GB** |
| **Free remaining** | ~13.5GB | ~12.4GB | ~11.4GB |

### Critical: Jetson Performance Tuning

Gemma 4 E2B requires two specific parameters to run fast on Jetson:

| Parameter | Value | Why |
|---|---|---|
| `flash_attn=True` | Enabled | Gemma 4 has variable-size V embeddings across layers. Without flash attention, llama.cpp pads all V caches to 512 dims, wasting memory and compute. Flash attention avoids this padding. |
| `n_threads=6` | All CPU cores | Gemma 4 uses Per-Layer Embeddings which places ~1.7GB of token embedding weights on CPU. With the default 1-2 threads, CPU-side compute is a bottleneck. Using all 6 Cortex-A78AE cores makes it fast. |

**Without these**: 0.3-0.5 tok/s (unusable). **With these**: 16-20 tok/s (100 tokens in ~5 seconds).

These are set in `llm_node.py` in the `_lazy_load_model()` method and in `tools/llm_chat_gui.py`.

### How to Switch Back

Edit `config/boxbunny.yaml` -- comment the Gemma line, uncomment the Qwen line:

```yaml
llm:
  # model_path: "models/llm/gemma-4-E2B-it-Q4_K_M.gguf"
  model_path: "models/llm/qwen2.5-3b-instruct-q4_k_m.gguf"
```

Both model files remain on disk. The `llm_models.yaml` config also lists all available models for the chat GUI tool.

### Files Changed for Model Switch

| File | Change |
|---|---|
| `config/boxbunny.yaml` | `model_path` switched to Gemma 4 E2B, old models listed as commented alternatives |
| `config/llm_models.yaml` | Added Gemma 4 E2B as default, kept Qwen models as alternatives |
| `src/boxbunny_core/boxbunny_core/config_loader.py` | Default `model_path` updated to Gemma 4 E2B |
| `src/boxbunny_core/launch/boxbunny_full.launch.py` | Model filename updated |
| `src/boxbunny_core/launch/boxbunny_dev.launch.py` | Model filename updated |
| `src/boxbunny_core/launch/headless.launch.py` | Model filename updated |
| `scripts/download_models.sh` | HF repo, filename, min size updated for Gemma 4 E2B; Qwen info kept as comment |
| `notebooks/scripts/test_llm.py` | MODEL_PATH updated |
| `notebooks/scripts/hardware_check.py` | LLM model path check updated |
| `tools/llm_chat_gui.py` | Default model path updated (2 locations) |
| `docs/system/technical-deep-dive.md` | Model info table, GPU memory table, rationale text updated |
| `docs/system/architecture.md` | LLM node description updated |

### Dependency Change

llama-cpp-python upgraded from 0.3.19 to 0.3.20 to add Gemma 4 (`gemma4`) architecture support. Built from source with CUDA enabled:

```bash
export CUDACXX=/usr/local/cuda-12.6/bin/nvcc
CMAKE_ARGS="-DGGML_CUDA=on -DCMAKE_CUDA_COMPILER=/usr/local/cuda-12.6/bin/nvcc" \
  pip3 install --upgrade llama-cpp-python==0.3.20 --no-cache-dir
```

---

## 2. Documentation Fix: Jetson Specs

The technical deep dive incorrectly listed the hardware as Jetson Orin NX 8GB with JetPack 5.x.

| Field | Was (Incorrect) | Now (Correct) |
|---|---|---|
| Model | Jetson Orin NX 8GB | Jetson Orin NX 16GB |
| RAM | 8GB LPDDR5 | 16GB LPDDR5 |
| OS | JetPack 5.x (Ubuntu 20.04 / L4T) | JetPack 6.2.1 (Ubuntu 22.04 / L4T) |

Fixed in `docs/system/technical-deep-dive.md` (5 locations).

---

## 3. Image Chat Feature (Phone Dashboard)

### Overview

Users can now send images alongside text messages in the phone dashboard chat. The Gemma 4 E2B model is multimodal (text + image), so it can answer questions about photos -- for example, identifying food and estimating calories, recognising sports equipment, or analysing a boxing stance from a photo.

### How It Works

```
User takes/selects photo on phone
        |
        v
Frontend compresses to 512px JPEG (0.8 quality)
        |
        v
POST /api/chat/message  { message: "...", image: "<base64>" }
        |
        v
Dashboard API passes image_base64 in context_json to ROS service
        |
        v
llm_node._handle_generate() detects image_base64 in context
        |
        v
Routes to _generate_with_image() instead of _generate()
        |
        v
_lazy_load_vision() loads mmproj-F16.gguf (ONLY on first image, ~1GB VRAM)
        |
        v
Multimodal inference via create_chat_completion with image_url content block
        |
        v
Response sent back through the same pipeline as text-only chat
```

### Performance Protection

The image feature is designed to NOT impact the robot's real-time performance:

1. **Lazy loading**: The vision projector (`mmproj-F16.gguf`, ~1GB) is only loaded into VRAM when the first image is sent. During normal text-only operation, it uses zero VRAM.

2. **Separate code paths**: Real-time coaching tips (`_tip_tick()`) always use `_generate()` (text-only). Image requests use `_generate_with_image()`. The two paths never cross.

3. **Separate timeouts**: Text-only inference has a 20-second timeout. Image inference has a 40-second timeout. Image processing cannot starve the coaching tips.

4. **CV inference is unaffected**: Computer vision (TensorRT action model, YOLO pose) runs in a separate process and is completely independent from LLM inference.

5. **No ROS service changes**: Image data is passed through the existing `context_json` string field as base64, so the GenerateLlm.srv definition is unchanged and backward compatible.

### New Model File

| File | Size | Source |
|---|---|---|
| `models/llm/mmproj-F16.gguf` | 940 MB | `unsloth/gemma-4-E2B-it-GGUF` on HuggingFace |

This is the multimodal projector that bridges the vision encoder to the language model. It is required for image processing but NOT for text-only operation.

### Files Changed for Image Chat

**Backend (LLM Node):**

| File | Change |
|---|---|
| `src/boxbunny_core/boxbunny_core/llm_node.py` | Added `_lazy_load_vision()` method to load mmproj on first image request. Added `_generate_with_image()` method with 40s timeout and multimodal message format. Modified `_handle_generate()` to detect `image_base64` in context_json and route to vision or text path. Added `mmproj_path` parameter and `IMAGE_INFERENCE_TIMEOUT_S` constant. |
| `src/boxbunny_core/boxbunny_core/config_loader.py` | Added `mmproj_path` field to `LLMConfig` dataclass |

**Backend (Dashboard API):**

| File | Change |
|---|---|
| `src/boxbunny_dashboard/boxbunny_dashboard/api/chat.py` | Added optional `image` field (base64 string) to `ChatRequest` model. Updated `_call_llm_sync()` and `_call_llm()` to accept and pass image data. Image requests get 45s ROS timeout vs 25s for text. Added image size validation (max 5MB). |

**Frontend (Vue Dashboard):**

| File | Change |
|---|---|
| `src/boxbunny_dashboard/frontend/src/views/ChatView.vue` | Added image picker button (gallery icon) next to text input. Added image preview bar with cancel button above input. Added `<img>` display in user message bubbles for attached images. Added `compressImage()` function (client-side resize to max 512px, JPEG quality 0.8). Added `onImageSelected()` and `clearImage()` handlers. Updated `handleSend()` to pass image alongside text. Send button enabled when image is attached even without text. |
| `src/boxbunny_dashboard/frontend/src/stores/chat.js` | `sendMessage()` now accepts optional `image` parameter. User messages include `image` field when present. Default prompt "What is in this image?" when image sent without text. |
| `src/boxbunny_dashboard/frontend/src/api/client.js` | `sendChatMessage()` now accepts optional `image` parameter and includes it in request body. |

**Configuration:**

| File | Change |
|---|---|
| `config/boxbunny.yaml` | Added `mmproj_path: "models/llm/mmproj-F16.gguf"` to LLM section |

**Scripts:**

| File | Change |
|---|---|
| `scripts/download_models.sh` | Added section 1b to download `mmproj-F16.gguf` from HuggingFace. Added mmproj status to summary output. |

**Documentation:**

| File | Change |
|---|---|
| `docs/system/technical-deep-dive.md` | GPU memory table updated to show mmproj as conditionally loaded row. Added note about lazy loading. |
| `docs/system/architecture.md` | LLM node description updated to mention vision support and lazy loading. |

---

## 4. All New/Modified Files (Complete List)

### New files downloaded
- `models/llm/gemma-4-E2B-it-Q4_K_M.gguf` (2.9 GiB) -- main LLM model
- `models/llm/mmproj-F16.gguf` (940 MB) -- vision projector

### Existing files kept (not deleted)
- `models/llm/qwen2.5-3b-instruct-q4_k_m.gguf` (2.0 GB) -- previous default model
- `models/llm/qwen2.5-1.5b-instruct-q4_k_m.gguf` (1.1 GB) -- lightweight alternative

### Modified files
1. `config/boxbunny.yaml`
2. `config/llm_models.yaml`
3. `scripts/download_models.sh`
4. `src/boxbunny_core/boxbunny_core/config_loader.py`
5. `src/boxbunny_core/boxbunny_core/llm_node.py`
6. `src/boxbunny_core/launch/boxbunny_full.launch.py`
7. `src/boxbunny_core/launch/boxbunny_dev.launch.py`
8. `src/boxbunny_core/launch/headless.launch.py`
9. `src/boxbunny_dashboard/boxbunny_dashboard/api/chat.py`
10. `src/boxbunny_dashboard/frontend/src/api/client.js`
11. `src/boxbunny_dashboard/frontend/src/stores/chat.js`
12. `src/boxbunny_dashboard/frontend/src/views/ChatView.vue`
13. `notebooks/scripts/test_llm.py`
14. `notebooks/scripts/hardware_check.py`
15. `tools/llm_chat_gui.py`
16. `docs/system/technical-deep-dive.md`
17. `docs/system/architecture.md`

---

## 5. Testing Checklist

- [ ] llama-cpp-python 0.3.20 build completes successfully
- [ ] Gemma 4 E2B model loads and generates text responses
- [ ] Coaching tips still arrive every 18s during active sessions
- [ ] Phone dashboard chat works for text-only messages
- [ ] Phone dashboard image picker opens camera/gallery
- [ ] Image preview shows before sending with cancel option
- [ ] Image + text message sends and gets a response
- [ ] Images display in chat message bubbles
- [ ] Vision projector loads lazily (check VRAM before/after first image with `tegrastats`)
- [ ] CV inference (pose detection, action recognition) unaffected during image chat
- [ ] Switching back to Qwen via config works correctly
- [ ] Reply depth toggle (Short/Normal/Detail) changes response length
- [ ] Drills are NOT suggested unless user explicitly asks for one
- [ ] Responses finish naturally without mid-sentence cutoffs

---

## 6. Chat Conversation Flow & Reply Depth

### Problem

The previous chat setup had several issues:
1. `max_tokens` was set to 128 for all requests (tips AND chat), causing chat replies to cut off mid-sentence
2. The system prompt was too rigid -- it always told the LLM to keep replies to "2-3 sentences max"
3. Drill suggestions were sometimes generated unprompted, cluttering casual conversation

### Changes

**Reply depth toggle** -- Users can now select Short / Normal / Detail in the chat header:

| Mode | Behaviour | Max tokens |
|---|---|---|
| Short | 1-2 sentences, straight to the point | 100 |
| Normal | 2-4 sentences, balanced | 256 |
| Detailed | In-depth with examples and breakdowns | 512 |

The toggle is a small segmented control in the chat header next to the info button.

**Separate token limits** -- Real-time coaching tips still use the config value (128 tokens, kept short for the robot screen). Chat requests from the dashboard get higher limits based on the depth setting, so responses finish naturally.

**Improved system prompt** -- The chat system prompt (`_build_system_prompt` in chat.py) was rewritten to:
- Be conversational and natural, matching the user's tone
- Only suggest drills when the user explicitly asks (with clear examples of what "asking" looks like)
- Always finish thoughts completely
- Use the user's name occasionally
- Handle image queries naturally

**Drill suggestion rules** -- The chat prompt now includes a "TRAINING DRILL SYSTEM" section with:
- Explicit trigger phrases the model recognises (e.g., "suggest a training", "suggest a drill", "give me a workout", "what should I practice")
- A concrete example of a correct response WITH a `[DRILL:]` tag so the model knows the exact format
- A concrete example of a response WITHOUT a tag so the model knows when to just talk
- Available combo numbers (1=jab, 2=cross, 3=L hook, etc.) so it can build custom combos
- Result: asking "suggest me a training" reliably generates drill cards, while normal conversation stays clean

### Files Changed

| File | Change |
|---|---|
| `src/boxbunny_core/boxbunny_core/llm_node.py` | Chat requests (`system_prompt_key == "coach_chat"`) get 100/256/512 max tokens based on depth. Tips unchanged at 128. Added "Always finish your sentences completely" to main system prompt. Uses dashboard's custom system prompt when provided. |
| `src/boxbunny_dashboard/boxbunny_dashboard/api/chat.py` | `_build_system_prompt()` rewritten with conversation style guidance, explicit drill rules with examples, reply depth instructions, and image query handling. Accepts `reply_depth` from context. |
| `src/boxbunny_dashboard/frontend/src/views/ChatView.vue` | Added reply depth toggle (Short/Normal/Detail) as segmented control in header. Passes depth to store. |
| `src/boxbunny_dashboard/frontend/src/stores/chat.js` | `sendMessage()` accepts `depth` parameter, passes it as `reply_depth` in context. |

---

## 7. Bug Fixes During Testing

### 7a. Gemma 4 E2B extremely slow inference (0.3 tok/s -> 20 tok/s)

**Problem**: Initial testing showed 0.3-0.5 tokens/second, making the model unusable. Gemma 4 E2B is designed for edge devices, so something was wrong.

**Root cause**: Two missing parameters in the Llama constructor:
1. Without `flash_attn=True`, llama.cpp pads all V caches to 512 dimensions because Gemma 4 has variable-size V embeddings across layers. This wastes memory and compute.
2. Without `n_threads=6`, only 1-2 CPU cores were used. Gemma 4's Per-Layer Embeddings place ~1.7GB of token embedding weights on CPU (`CPU_Mapped` buffer), making CPU thread count critical.

**Fix**: Added `flash_attn=True` and `n_threads=6` to `_lazy_load_model()` in `llm_node.py` and `tools/llm_chat_gui.py`.

**Result**: 16-20 tokens/second. 100 tokens in ~5 seconds.

### 7b. Image chat returning "I can't see images right now"

**Problem**: Sending an image through the dashboard chat resulted in the LLM saying it couldn't see the image, even though standalone testing worked.

**Root cause**: The vision `chat_handler` was being passed as a parameter to `create_chat_completion()`, but llama-cpp-python requires the handler to be set directly on the model instance (`model.chat_handler = handler`). The parameter-based approach silently fell back to text-only mode.

**Fix**: In `_generate_with_image()`, the handler is now set on the model before the call and unset after:
```python
self._llm.chat_handler = self._chat_handler
try:
    result = self._llm.create_chat_completion(...)
finally:
    self._llm.chat_handler = None
```
This ensures vision is active for image requests and inactive for text-only tips.

### 7c. Drill suggestion cards not appearing

**Problem**: Asking "suggest me a training" in the dashboard chat resulted in conversational advice but no `[DRILL:]` tag, so no clickable drill card appeared.

**Root cause**: The system prompt gave the model the tag format but didn't show a concrete example of a complete response with the tag. The model understood what `[DRILL:]` tags are but didn't know it should actually generate one when asked.

**Fix**: Rewrote the drill section of the system prompt to include:
- A "TRAINING DRILL SYSTEM" header explaining that the robot loads drills from tags
- Explicit trigger phrases the model should recognise
- A concrete example of a full response WITH a tag (so the model sees the pattern)
- A concrete example WITHOUT a tag (so it knows when NOT to use one)
- Available combo numbers (1=jab, 2=cross, etc.) so it can build custom drills

**Result**: "Suggest me a training" now reliably generates drill cards. Normal conversation stays clean -- no unwanted drill suggestions.

### 7d. Responses using markdown formatting

**Problem**: The model was generating `**bold**`, numbered lists, and headers despite being told not to.

**Fix**: Strengthened the no-markdown instruction to explicitly list every format to avoid: "NEVER use markdown formatting like ** or * or # or numbered lists. Write in plain text only. No bold, no bullet points, no headers."

**Result**: All test responses come back as clean plain text.

### 7e. Image chat not working in live system (missing launch parameter)

**Problem**: Image chat worked in standalone testing but failed when running through the full ROS system. The LLM would respond saying it couldn't see the image.

**Root cause**: The `mmproj_path` ROS parameter was never being passed to the LLM node in the launch files. The node declared the parameter with a default of `""`, and since none of the three launch files set it, `_lazy_load_vision()` always saw an empty path and returned `False`, falling back to text-only mode.

**Fix**: Added `"mmproj_path": str(ws_root / "models" / "llm" / "mmproj-F16.gguf")` to the parameters dict in all three launch files.

**Result**: Vision projector path is now passed to the node at launch. First image request lazy-loads the mmproj and enables vision inference.

### Files Changed for Bug Fixes

| File | Change |
|---|---|
| `src/boxbunny_core/boxbunny_core/llm_node.py` | Added `n_threads=6, flash_attn=True` to `_lazy_load_model()`. Fixed `_generate_with_image()` to set `chat_handler` on model directly instead of passing as parameter. |
| `src/boxbunny_dashboard/boxbunny_dashboard/api/chat.py` | Rewrote drill system prompt section with concrete examples and trigger phrases. Strengthened no-markdown instruction. |
| `src/boxbunny_core/launch/boxbunny_full.launch.py` | Added `mmproj_path` parameter to LLM node. |
| `src/boxbunny_core/launch/boxbunny_dev.launch.py` | Added `mmproj_path` parameter to LLM node. |
| `src/boxbunny_core/launch/headless.launch.py` | Added `mmproj_path` parameter to LLM node. |
| `tools/llm_chat_gui.py` | Added `flash_attn=True` to Llama constructor. |
