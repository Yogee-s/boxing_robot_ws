import os
import sys
import site
import json
import time
from pathlib import Path

try:
    user_site = site.getusersitepackages()
    if user_site and user_site not in sys.path:
        sys.path.append(user_site)
except Exception:
    pass

try:
    from PySide6 import QtCore, QtGui, QtWidgets  # type: ignore
except Exception as exc:
    raise SystemExit(
        f"PySide6 not available: {exc}\nInstall with: python3 -m pip install --user PySide6"
    ) from exc

try:
    from llama_cpp import Llama  # type: ignore
except Exception:
    Llama = None

try:
    import yaml
except Exception:
    yaml = None


SENTENCE_ENDINGS = (".", "!", "?")

APP_STYLESHEET = """
QWidget { background-color: #111317; color: #E6E6E6; font-family: 'DejaVu Sans'; }
QGroupBox { border: 1px solid #2A2E36; border-radius: 8px; margin-top: 8px; padding: 10px; }
QGroupBox::title { subcontrol-origin: margin; left: 8px; padding: 0 4px; color: #C0C4CC; }
QPushButton { background-color: #2B3240; border: 1px solid #394151; padding: 6px 10px; border-radius: 6px; }
QPushButton:hover { background-color: #394151; }
QPushButton:pressed { background-color: #202633; }
QLineEdit, QTextEdit, QPlainTextEdit, QComboBox { background-color: #1A1E25; border: 1px solid #2A2E36; padding: 4px 6px; border-radius: 6px; }
QLabel { color: #E6E6E6; }
"""


def _apply_theme(app: QtWidgets.QApplication) -> None:
    app.setStyleSheet(APP_STYLESHEET)


class LlmWorker(QtCore.QThread):
    token = QtCore.Signal(str)
    done = QtCore.Signal(str)
    error = QtCore.Signal(str)

    def __init__(self, llm, prompt: str, max_tokens: int, temperature: float, complete_sentence: bool):
        super().__init__()
        self._llm = llm
        self._prompt = prompt
        self._max_tokens = max_tokens
        self._temperature = temperature
        self._buffer = []
        self._complete_sentence = complete_sentence

    def run(self) -> None:
        try:
            for chunk in self._llm(
                self._prompt,
                max_tokens=self._max_tokens,
                temperature=self._temperature,
                stop=["\n"],
                stream=True,
            ):
                text = chunk["choices"][0]["text"]
                if text:
                    self._buffer.append(text)
                    self.token.emit(text)

            text = "".join(self._buffer).strip()
            if self._complete_sentence and text and not text.endswith(SENTENCE_ENDINGS):
                # Try a short continuation to finish the sentence.
                follow_prompt = f"{self._prompt}{text}"
                follow = self._llm(
                    follow_prompt,
                    max_tokens=min(32, max(8, self._max_tokens // 4)),
                    temperature=self._temperature,
                    stop=["\n"],
                )
                extra = follow["choices"][0]["text"].strip()
                if extra:
                    self._buffer.append(" " + extra)
                    self.token.emit(" " + extra)
                    text = (text + " " + extra).strip()

                if text and not text.endswith(SENTENCE_ENDINGS):
                    self._buffer.append(".")
                    self.token.emit(".")

            self.done.emit("".join(self._buffer))
        except Exception as exc:  # pragma: no cover
            self.error.emit(str(exc))


class LlmChatGui(QtWidgets.QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("BoxBunny LLM Chat")
        self.resize(920, 680)

        self.model_path = os.environ.get(
            "BOXBUNNY_LLM_MODEL",
            "/home/boxbunny/Desktop/doomsday_integration/boxing_robot_ws/models/llm/qwen2.5-3b-instruct-q4_k_m.gguf",
        )
        self.system_path = os.environ.get(
            "BOXBUNNY_LLM_SYSTEM",
            "/home/boxbunny/Desktop/doomsday_integration/boxing_robot_ws/src/boxbunny_llm/config/system_prompt.txt",
        )
        self.models_path = os.environ.get(
            "BOXBUNNY_LLM_MODELS",
            "/home/boxbunny/Desktop/doomsday_integration/boxing_robot_ws/src/boxbunny_llm/config/models.yaml",
        )
        self.singlish_path = os.environ.get(
            "BOXBUNNY_LLM_SINGLISH",
            "/home/boxbunny/Desktop/doomsday_integration/boxing_robot_ws/src/boxbunny_llm/config/singlish_prompt.txt",
        )

        self._llm = None
        self._worker = None
        self._current_reply_cursor = None
        self._models = []
        self._singlish_prompt = ""
        self._history = []
        self._settings_path = Path(os.path.expanduser("~/.boxbunny/llm_gui_settings.json"))

        self._build_ui()
        self._load_models()
        self._load_singlish_prompt()
        self._load_system_prompt()
        self._load_settings()
        self._load_llm()

    def _build_ui(self) -> None:
        central = QtWidgets.QWidget()
        self.setCentralWidget(central)

        root = QtWidgets.QVBoxLayout(central)

        header = QtWidgets.QLabel("BoxBunny LLM Chat")
        header.setStyleSheet("font-size: 20px; font-weight: 600;")

        self.chat_view = QtWidgets.QTextEdit()
        self.chat_view.setReadOnly(True)
        self.chat_view.setStyleSheet("background: #151515; color: #e6e6e6; font-size: 14px;")
        self.chat_view.setLineWrapMode(QtWidgets.QTextEdit.LineWrapMode.WidgetWidth)

        # Top row: input + quick toggles
        input_row = QtWidgets.QHBoxLayout()
        self.user_input = QtWidgets.QLineEdit()
        self.user_input.setPlaceholderText("Type your message...")
        self.user_input.returnPressed.connect(self._send)
        self.send_btn = QtWidgets.QPushButton("Send")
        self.send_btn.clicked.connect(self._send)
        self.stop_btn = QtWidgets.QPushButton("Stop")
        self.stop_btn.clicked.connect(self._stop_generation)
        self.clear_btn = QtWidgets.QPushButton("Clear")
        self.clear_btn.clicked.connect(self._clear_history)

        self.singlish_toggle = QtWidgets.QCheckBox("Singlish")
        self.singlish_toggle.setToolTip("Adds Singlish tone (lah/leh/lor) to replies")
        self.singlish_toggle.toggled.connect(lambda v: self._log_setting(f"Singlish {'ON' if v else 'OFF'}"))
        self.advice_toggle = QtWidgets.QCheckBox("Advice")
        self.advice_toggle.setToolTip("Bias replies toward practical boxing advice")
        self.advice_toggle.toggled.connect(lambda v: self._log_setting(f"Advice {'ON' if v else 'OFF'}"))
        self.memory_toggle = QtWidgets.QCheckBox("Remember")
        self.memory_toggle.setToolTip("Include recent chat history in prompts")
        self.memory_toggle.toggled.connect(lambda v: self._log_setting(f"Memory {'ON' if v else 'OFF'}"))

        input_row.addWidget(self.user_input)
        input_row.addWidget(self.send_btn)
        input_row.addWidget(self.stop_btn)
        input_row.addWidget(self.clear_btn)
        input_row.addWidget(self.singlish_toggle)
        input_row.addWidget(self.advice_toggle)
        input_row.addWidget(self.memory_toggle)

        # Presets row
        presets_row = QtWidgets.QHBoxLayout()
        self.presets_combo = QtWidgets.QComboBox()
        self.presets_combo.addItems(
            [
                "Preset prompts...",
                "Give me jab-cross tips",
                "How to improve reaction time?",
                "What should I focus on this round?",
                "Analyze my punch stats in one sentence",
                "Give me a Singlish pep talk",
            ]
        )
        self.presets_combo.currentIndexChanged.connect(self._apply_preset)
        presets_row.addWidget(self.presets_combo)

        # Advanced settings (hidden by default)
        self.advanced_group = QtWidgets.QGroupBox("Advanced Settings")
        adv_layout = QtWidgets.QGridLayout(self.advanced_group)

        self.advanced_toggle_btn = QtWidgets.QPushButton("Show Advanced Settings")
        self.advanced_toggle_btn.clicked.connect(self._toggle_advanced)

        self.model_combo = QtWidgets.QComboBox()
        self.model_path_label = QtWidgets.QLabel(self.model_path)
        self.model_path_label.setWordWrap(True)
        self.model_combo.currentIndexChanged.connect(self._on_model_change)

        self.temp_slider = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
        self.temp_slider.setRange(0, 100)
        self.temp_slider.setValue(70)
        self.temp_label = QtWidgets.QLabel("0.70")
        self.temp_slider.valueChanged.connect(lambda v: self.temp_label.setText(f"{v/100:.2f}"))
        self.temp_slider.sliderReleased.connect(lambda: self._log_setting(f"Temperature {self.temp_label.text()}"))

        self.tokens_slider = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
        self.tokens_slider.setRange(16, 256)
        self.tokens_slider.setValue(64)
        self.tokens_label = QtWidgets.QLabel("64")
        self.tokens_slider.valueChanged.connect(lambda v: self.tokens_label.setText(str(v)))
        self.tokens_slider.sliderReleased.connect(lambda: self._log_setting(f"Max tokens {self.tokens_label.text()}"))

        self.context_slider = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
        self.context_slider.setRange(128, 2048)
        self.context_slider.setValue(512)
        self.context_label = QtWidgets.QLabel("512")
        self.context_slider.valueChanged.connect(lambda v: self.context_label.setText(str(v)))
        self.context_slider.sliderReleased.connect(lambda: self._log_setting(f"Context {self.context_label.text()}"))

        self.threads_slider = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
        self.threads_slider.setRange(1, max(2, os.cpu_count() or 4))
        self.threads_slider.setValue(min(6, max(2, os.cpu_count() or 4)))
        self.threads_label = QtWidgets.QLabel(str(self.threads_slider.value()))
        self.threads_slider.valueChanged.connect(lambda v: self.threads_label.setText(str(v)))
        self.threads_slider.sliderReleased.connect(lambda: self._log_setting(f"Threads {self.threads_label.text()}"))

        self.batch_slider = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
        self.batch_slider.setRange(16, 512)
        self.batch_slider.setValue(128)
        self.batch_label = QtWidgets.QLabel(str(self.batch_slider.value()))
        self.batch_slider.valueChanged.connect(lambda v: self.batch_label.setText(str(v)))
        self.batch_slider.sliderReleased.connect(lambda: self._log_setting(f"Batch {self.batch_label.text()}"))

        self.history_spin = QtWidgets.QSpinBox()
        self.history_spin.setRange(0, 12)
        self.history_spin.setValue(4)
        self.history_spin.valueChanged.connect(lambda v: self._log_setting(f"History turns {v}"))

        self.reload_model_btn = QtWidgets.QPushButton("Reload Model")
        self.reload_model_btn.clicked.connect(self._load_llm)

        adv_layout.addWidget(QtWidgets.QLabel("Model"), 0, 0)
        adv_layout.addWidget(self.model_combo, 0, 1)
        adv_layout.addWidget(self.reload_model_btn, 0, 2)
        adv_layout.addWidget(QtWidgets.QLabel("Path"), 1, 0)
        adv_layout.addWidget(self.model_path_label, 1, 1, 1, 2)
        adv_layout.addWidget(QtWidgets.QLabel("Temperature"), 2, 0)
        adv_layout.addWidget(self.temp_slider, 2, 1)
        adv_layout.addWidget(self.temp_label, 2, 2)
        adv_layout.addWidget(QtWidgets.QLabel("Max tokens"), 3, 0)
        adv_layout.addWidget(self.tokens_slider, 3, 1)
        adv_layout.addWidget(self.tokens_label, 3, 2)
        adv_layout.addWidget(QtWidgets.QLabel("Context"), 4, 0)
        adv_layout.addWidget(self.context_slider, 4, 1)
        adv_layout.addWidget(self.context_label, 4, 2)
        adv_layout.addWidget(QtWidgets.QLabel("Threads"), 5, 0)
        adv_layout.addWidget(self.threads_slider, 5, 1)
        adv_layout.addWidget(self.threads_label, 5, 2)
        adv_layout.addWidget(QtWidgets.QLabel("Batch"), 6, 0)
        adv_layout.addWidget(self.batch_slider, 6, 1)
        adv_layout.addWidget(self.batch_label, 6, 2)
        adv_layout.addWidget(QtWidgets.QLabel("History turns"), 7, 0)
        adv_layout.addWidget(self.history_spin, 7, 1)

        self.advanced_group.setVisible(False)

        # System prompt (hidden by default)
        system_group = QtWidgets.QGroupBox("System Prompt")
        system_layout = QtWidgets.QVBoxLayout(system_group)
        self.system_editor = QtWidgets.QPlainTextEdit()
        self.system_editor.setPlaceholderText("System prompt for the model...")
        system_layout.addWidget(self.system_editor)

        sys_buttons = QtWidgets.QHBoxLayout()
        self.toggle_system_btn = QtWidgets.QPushButton("Show System Prompt")
        self.toggle_system_btn.clicked.connect(self._toggle_system_prompt)
        self.reload_btn = QtWidgets.QPushButton("Reload")
        self.save_btn = QtWidgets.QPushButton("Save")
        self.reload_btn.clicked.connect(self._load_system_prompt)
        self.save_btn.clicked.connect(self._save_system_prompt)
        sys_buttons.addWidget(self.toggle_system_btn)
        sys_buttons.addWidget(self.reload_btn)
        sys_buttons.addWidget(self.save_btn)
        system_layout.addLayout(sys_buttons)
        self.system_editor.setVisible(False)

        footer = QtWidgets.QHBoxLayout()
        self.status_label = QtWidgets.QLabel("Ready")
        self.export_btn = QtWidgets.QPushButton("Export")
        self.export_btn.clicked.connect(self._export_history)
        footer.addWidget(self.status_label)
        footer.addStretch(1)
        footer.addWidget(self.export_btn)

        root.addWidget(header)
        root.addWidget(self.chat_view, 1)
        root.addLayout(input_row)
        root.addLayout(presets_row)
        root.addWidget(self.advanced_toggle_btn)
        root.addWidget(self.advanced_group)
        root.addWidget(system_group)
        root.addLayout(footer)

        self.setStyleSheet(
            "QMainWindow { background: #0f0f0f; color: #e6e6e6; }"
            "QGroupBox { border: 1px solid #303030; margin-top: 8px; padding: 8px; }"
            "QGroupBox::title { color: #dcdcdc; padding-left: 6px; }"
            "QLabel { color: #e6e6e6; }"
            "QCheckBox { color: #e6e6e6; }"
            "QLineEdit { background: #1f1f1f; color: #e6e6e6; padding: 6px; }"
            "QPlainTextEdit { background: #1a1a1a; color: #e6e6e6; }"
            "QPushButton { background: #2e7d32; color: white; padding: 6px 12px; }"
            "QPushButton:hover { background: #36a23d; }"
        )

    def _toggle_advanced(self) -> None:
        visible = not self.advanced_group.isVisible()
        self.advanced_group.setVisible(visible)
        self.advanced_toggle_btn.setText("Hide Advanced Settings" if visible else "Show Advanced Settings")
        self._log_setting("Advanced settings shown" if visible else "Advanced settings hidden")

    def _log_setting(self, message: str) -> None:
        self._append_chat("System", message)

    def _load_models(self) -> None:
        self._models = []
        if yaml and Path(self.models_path).exists():
            data = yaml.safe_load(Path(self.models_path).read_text()) or {}
            self._models = data.get("models", []) if isinstance(data, dict) else []
        if not self._models:
            self._models = [{"name": "Default", "path": self.model_path}]

        self.model_combo.blockSignals(True)
        self.model_combo.clear()
        for entry in self._models:
            self.model_combo.addItem(entry.get("name", "Model"))
        self.model_combo.blockSignals(False)

        for idx, entry in enumerate(self._models):
            if entry.get("path") == self.model_path:
                self.model_combo.setCurrentIndex(idx)
                break
        self.model_path_label.setText(self.model_path)

    def _load_singlish_prompt(self) -> None:
        path = Path(self.singlish_path)
        self._singlish_prompt = path.read_text() if path.exists() else "Use Singlish tone."

    def _on_model_change(self, index: int) -> None:
        if index < 0 or index >= len(self._models):
            return
        self.model_path = self._models[index].get("path", self.model_path)
        self.model_path_label.setText(self.model_path)
        self._log_setting(f"Model set to {self.model_combo.currentText()}")

    def _load_llm(self) -> None:
        if Llama is None:
            self.status_label.setText("llama_cpp not installed")
            return
        if not Path(self.model_path).exists():
            self.status_label.setText(f"Model not found: {self.model_path}")
            self._log_setting("Model not found - check path")
            return
        try:
            self._llm = Llama(
                model_path=self.model_path,
                n_ctx=int(self.context_slider.value()),
                n_threads=int(self.threads_slider.value()),
                n_batch=int(self.batch_slider.value()),
            )
            self.status_label.setText("Model loaded")
            self._log_setting("Model loaded")
        except Exception as exc:
            self._llm = None
            self.status_label.setText(f"Model load failed: {exc}")
            self._append_chat("System", "Model failed to load. Re-download the GGUF model.")

    def _load_system_prompt(self) -> None:
        path = Path(self.system_path)
        if path.exists():
            self.system_editor.setPlainText(path.read_text())
        else:
            self.system_editor.setPlainText("You are a helpful boxing coach. Be concise and safe.")

    def _save_system_prompt(self) -> None:
        path = Path(self.system_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self.system_editor.toPlainText())
        self.status_label.setText(f"Saved: {path}")
        self._log_setting("System prompt saved")

    def _build_prompt(self, user_text: str) -> str:
        system = self.system_editor.toPlainText().strip()
        if self.singlish_toggle.isChecked():
            system += f"\n{self._singlish_prompt}"
        if self.advice_toggle.isChecked():
            system += (
                "\nProvide practical boxing advice and training tips."
                " If unsure, say so and suggest safe practice."
                " Avoid medical or injury diagnosis."
            )

        if not self.memory_toggle.isChecked():
            return f"{system}\nUser: {user_text}\nCoach:"

        turns = int(self.history_spin.value())
        history = []
        # take last N pairs from history
        for item in self._history[-turns * 2 :]:
            role = "User" if item["role"] == "user" else "Coach"
            history.append(f"{role}: {item['text']}")

        history_text = "\n".join(history)
        return f"{system}\n{history_text}\nUser: {user_text}\nCoach:"

    def _send(self) -> None:
        text = self.user_input.text().strip()
        if not text:
            return
        if self._worker is not None:
            return
        self.user_input.clear()
        self._append_chat("You", text)
        self._history.append({"role": "user", "text": text, "ts": time.time()})
        if self._llm is None:
            self._append_chat("Coach", "Model not loaded.")
            return

        prompt = self._build_prompt(text)

        self._current_reply_cursor = self.chat_view.textCursor()
        self._current_reply_cursor.movePosition(QtGui.QTextCursor.End)
        self._current_reply_cursor.insertHtml("<br><b>Coach:</b> ")

        self._worker = LlmWorker(
            self._llm,
            prompt,
            max_tokens=int(self.tokens_slider.value()),
            temperature=float(self.temp_slider.value()) / 100.0,
            complete_sentence=True,
        )
        self._worker.token.connect(self._append_stream)
        self._worker.done.connect(self._finish_stream)
        self._worker.error.connect(self._stream_error)
        self.status_label.setText("Generating...")
        self._worker.start()

    def _append_stream(self, text: str) -> None:
        if self._current_reply_cursor is None:
            return
        self._current_reply_cursor.insertText(text)
        self.chat_view.ensureCursorVisible()

    def _finish_stream(self, text: str) -> None:
        if text:
            self._history.append({"role": "assistant", "text": text, "ts": time.time()})
        self._worker = None
        self._current_reply_cursor = None
        self.status_label.setText("Ready")

    def _stream_error(self, message: str) -> None:
        self._append_chat("System", f"LLM error: {message}")
        self._worker = None
        self._current_reply_cursor = None
        self.status_label.setText("Error")

    def _stop_generation(self) -> None:
        if self._worker is not None:
            self._worker.terminate()
            self._worker = None
            self._current_reply_cursor = None
            self.status_label.setText("Stopped")

    def _toggle_system_prompt(self) -> None:
        visible = not self.system_editor.isVisible()
        self.system_editor.setVisible(visible)
        self.toggle_system_btn.setText("Hide System Prompt" if visible else "Show System Prompt")
        self._log_setting("System prompt shown" if visible else "System prompt hidden")

    def _apply_preset(self, index: int) -> None:
        if index <= 0:
            return
        prompt = self.presets_combo.currentText()
        self.presets_combo.setCurrentIndex(0)
        self._log_setting(f"Preset selected: {prompt}")
        self.user_input.setText(prompt)
        self._send()

    def _export_history(self) -> None:
        if not self._history:
            self.status_label.setText("No chat to export")
            return
        out_dir = Path(os.path.expanduser("~/boxbunny_logs"))
        out_dir.mkdir(parents=True, exist_ok=True)
        filename = time.strftime("llm_chat_%Y%m%d_%H%M%S.json")
        out_path = out_dir / filename
        out_path.write_text(json.dumps(self._history, indent=2))
        self.status_label.setText(f"Exported: {out_path}")
        self._log_setting("Chat exported")

    def _clear_history(self) -> None:
        self._history = []
        self.chat_view.clear()
        self.status_label.setText("Cleared")
        self._log_setting("Chat cleared")

    def _load_settings(self) -> None:
        if not self._settings_path.exists():
            return
        try:
            data = json.loads(self._settings_path.read_text())
        except Exception:
            return
        self.temp_slider.setValue(int(data.get("temperature", 70)))
        self.tokens_slider.setValue(int(data.get("max_tokens", 64)))
        self.context_slider.setValue(int(data.get("context", 512)))
        self.threads_slider.setValue(int(data.get("threads", self.threads_slider.value())))
        self.batch_slider.setValue(int(data.get("batch", self.batch_slider.value())))
        self.singlish_toggle.setChecked(bool(data.get("singlish", False)))
        self.advice_toggle.setChecked(bool(data.get("advice", False)))
        self.memory_toggle.setChecked(bool(data.get("memory", False)))
        self.history_spin.setValue(int(data.get("history_turns", 4)))
        model_path = data.get("model_path")
        if model_path:
            self.model_path = model_path
            self.model_path_label.setText(self.model_path)
        self.model_combo.setCurrentIndex(data.get("model_index", 0))

        if not Path(self.model_path).exists():
            self.model_path = "/home/boxbunny/Desktop/doomsday_integration/boxing_robot_ws/models/llm/qwen2.5-3b-instruct-q4_k_m.gguf"
            self.model_path_label.setText(self.model_path)
            self._log_setting("Model path reset to new workspace")

    def _save_settings(self) -> None:
        payload = {
            "temperature": self.temp_slider.value(),
            "max_tokens": self.tokens_slider.value(),
            "context": self.context_slider.value(),
            "threads": self.threads_slider.value(),
            "batch": self.batch_slider.value(),
            "singlish": self.singlish_toggle.isChecked(),
            "advice": self.advice_toggle.isChecked(),
            "memory": self.memory_toggle.isChecked(),
            "history_turns": self.history_spin.value(),
            "model_index": self.model_combo.currentIndex(),
            "model_path": self.model_path,
        }
        self._settings_path.parent.mkdir(parents=True, exist_ok=True)
        self._settings_path.write_text(json.dumps(payload, indent=2))

    def closeEvent(self, event: QtGui.QCloseEvent) -> None:
        self._save_settings()
        super().closeEvent(event)

    def _append_chat(self, who: str, text: str) -> None:
        self.chat_view.append(f"<br><b>{who}:</b> {text}")


def main() -> None:
    app = QtWidgets.QApplication([])
    _apply_theme(app)
    gui = LlmChatGui()
    gui.show()
    app.exec()


if __name__ == "__main__":
    main()
