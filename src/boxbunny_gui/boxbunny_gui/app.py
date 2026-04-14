"""BoxBunny GUI -- main application entry point.

Creates the QApplication, wires up all subsystems via constructor injection,
and launches the 1024x600 frameless window for the Jetson Orin NX touchscreen.
"""
from __future__ import annotations

import logging
import os
import sys
from pathlib import Path
from typing import Optional

# Fix Qt platform plugin discovery when conda or virtualenv overrides paths
if not os.environ.get("QT_QPA_PLATFORM_PLUGIN_PATH"):
    try:
        import PySide6
        _pyside_plugins = Path(PySide6.__path__[0]) / "Qt" / "plugins" / "platforms"
        if _pyside_plugins.exists():
            os.environ["QT_QPA_PLATFORM_PLUGIN_PATH"] = str(_pyside_plugins)
    except Exception:
        pass

from PySide6.QtCore import Qt
from PySide6.QtGui import QFontDatabase, QPainter
from PySide6.QtWidgets import (
    QApplication, QGraphicsScene, QGraphicsView,
    QMainWindow, QStackedWidget, QWidget,
)

from boxbunny_gui.gui_bridge import GuiBridge
from boxbunny_gui.nav.imu_nav_handler import ImuNavHandler, KeyboardNavFilter
from boxbunny_gui.nav.router import PageRouter
from boxbunny_gui.sound import SoundManager
from boxbunny_gui.theme import Color, GLOBAL_STYLESHEET, Size

logger = logging.getLogger(__name__)

# Resolve asset directory relative to this file's package
_PKG_DIR = Path(__file__).resolve().parent
_SRC_DIR = _PKG_DIR.parent  # boxbunny_gui source root
_ASSETS_DIR = _SRC_DIR / "assets"


class BoxBunnyApp:
    """Top-level application object.

    Owns every subsystem and wires them together.
    No globals, no singletons -- everything is passed by reference.
    """

    def __init__(self, argv: Optional[list[str]] = None) -> None:
        self._setup_logging()

        # ── Qt application ──────────────────────────────────────────────
        QApplication.setAttribute(
            Qt.ApplicationAttribute.AA_CompressHighFrequencyEvents, True
        )
        self._qapp = QApplication(argv or sys.argv)
        self._qapp.setApplicationName("BoxBunny")

        # ── Load Inter font before any widgets are created ─────────────
        assets_dir = self._resolve_assets_dir()
        font_path = assets_dir / "fonts" / "InterVariable.ttf"
        if font_path.exists():
            font_id = QFontDatabase.addApplicationFont(str(font_path))
            if font_id < 0:
                logger.warning("Failed to load Inter font from %s", font_path)
            else:
                families = QFontDatabase.applicationFontFamilies(font_id)
                logger.info("Loaded font families: %s", families)
        font_italic_path = assets_dir / "fonts" / "InterVariable-Italic.ttf"
        if font_italic_path.exists():
            QFontDatabase.addApplicationFont(str(font_italic_path))

        self._qapp.setStyleSheet(GLOBAL_STYLESHEET)

        # ── Main window (fixed size, standard title bar as backup close) ─
        self._window = QMainWindow()
        self._window.setFixedSize(Size.SCREEN_W, Size.SCREEN_H)
        self._window.setWindowTitle("BoxBunny")
        self._window.setStyleSheet(f"background-color: {Color.BG};")
        self._window._boxbunny_app = self  # allow pages to access app for fullscreen

        # ── Central stacked widget for page routing ─────────────────────
        self._stack = QStackedWidget()
        self._stack.setFixedSize(Size.SCREEN_W, Size.SCREEN_H)

        # Wrap in QGraphicsView for runtime scaling (fullscreen)
        self._scene = QGraphicsScene()
        self._proxy = self._scene.addWidget(self._stack)
        self._view = QGraphicsView(self._scene)
        self._view.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self._view.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self._view.setFrameShape(QGraphicsView.Shape.NoFrame)
        self._view.setStyleSheet(f"background-color: {Color.BG}; border: none;")
        self._view.setRenderHints(
            QPainter.RenderHint.Antialiasing
            | QPainter.RenderHint.SmoothPixmapTransform
        )
        # Allow touch/mouse events to pass through to embedded widgets
        self._view.setInteractive(True)
        self._view.setDragMode(QGraphicsView.DragMode.NoDrag)
        self._view.viewport().setAttribute(
            Qt.WidgetAttribute.WA_AcceptTouchEvents, True
        )
        self._window.setCentralWidget(self._view)

        # ── Subsystems ──────────────────────────────────────────────────
        self._bridge = GuiBridge(parent=self._window)
        self._router = PageRouter(self._stack, parent=self._window)
        self._sound = SoundManager(self._resolve_assets_dir())
        self._imu_nav = ImuNavHandler(parent=self._window)

        # ── Keyboard nav filter (dev convenience) ───────────────────────
        self._kb_filter = KeyboardNavFilter(self._imu_nav, parent=self._window)
        self._window.installEventFilter(self._kb_filter)

        # ── F12 debug toggle filter ────────────────────────────────────
        from PySide6.QtCore import QObject as _QObject
        from PySide6.QtGui import QKeyEvent as _QKeyEvent

        class _HotkeyFilter(_QObject):
            def __init__(self, app_ref, parent=None):
                super().__init__(parent)
                self._app = app_ref
            def eventFilter(self, obj, event):
                if (isinstance(event, _QKeyEvent)
                        and event.type() == _QKeyEvent.Type.KeyPress):
                    if event.key() == Qt.Key.Key_F12:
                        self._app.toggle_debug_panel()
                        return True
                    if event.key() == Qt.Key.Key_F11:
                        self._app.toggle_fullscreen()
                        return True
                    if event.key() == Qt.Key.Key_Escape and self._app._is_fullscreen:
                        self._app.toggle_fullscreen()
                        return True
                return super().eventFilter(obj, event)

        self._hotkey_filter = _HotkeyFilter(self, parent=self._window)
        self._window.installEventFilter(self._hotkey_filter)

        # ── Fullscreen state ────────────────────────────────────────────
        self._is_fullscreen = False

        # ── Developer mode overlay ──────────────────────────────────────
        from boxbunny_gui.widgets.dev_overlay import DevOverlay
        self._dev_overlay = DevOverlay(parent=self._window)
        self._dev_overlay.move(Size.SCREEN_W - 330, Size.SCREEN_H - 230)
        self._dev_overlay.set_developer_mode(False)
        self._dev_overlay.raise_()

        # ── Debug detection panel (full-page, toggled via F12) ──────────
        from boxbunny_gui.widgets.debug_panel import DebugDetectionPanel
        self._debug_panel = DebugDetectionPanel()
        self._debug_panel_idx = self._stack.addWidget(self._debug_panel)
        self._debug_active = False

        # ── Preset quick-launch overlay ─────────────────────────────────
        from boxbunny_gui.widgets.preset_overlay import PresetOverlay
        self._preset_overlay = PresetOverlay(
            on_select=self._on_preset_selected,
            parent=self._window,
        )

        # ── Wire signals ────────────────────────────────────────────────
        self._connect_signals()

        # ── Register pages ──────────────────────────────────────────────
        self._register_pages()

        # ── Start ROS bridge (non-blocking) ─────────────────────────────
        self._bridge.start()

        # ── Remote command polling (phone dashboard control) ────────────
        from PySide6.QtCore import QTimer
        self._remote_timer = QTimer()
        self._remote_timer.setInterval(100)  # 10Hz for responsive height control
        self._remote_timer.timeout.connect(self._poll_remote_commands)
        self._remote_timer.start()

        logger.info("BoxBunnyApp initialised (online=%s)", self._bridge.online)

    # ── Public API ──────────────────────────────────────────────────────

    def run(self) -> int:
        """Show the window and enter the Qt event loop."""
        self._router.navigate("auth")
        self._window.show()
        exit_code = self._qapp.exec()
        self._shutdown()
        return exit_code

    # ── Internals ───────────────────────────────────────────────────────

    def set_developer_mode(self, enabled: bool) -> None:
        """Toggle developer mode overlay on/off."""
        self._dev_overlay.set_developer_mode(enabled)
        logger.info("Developer mode: %s", "ON" if enabled else "OFF")

    def toggle_fullscreen(self) -> None:
        """Toggle between fullscreen and windowed mode (F11).

        Scales the entire UI proportionally using QGraphicsView transform.
        """
        if self._is_fullscreen:
            self._window.showNormal()
            self._window.setFixedSize(Size.SCREEN_W, Size.SCREEN_H)
            self._view.resetTransform()
            self._is_fullscreen = False
        else:
            self._window.setMinimumSize(0, 0)
            self._window.setMaximumSize(16777215, 16777215)
            self._window.showFullScreen()
            self._is_fullscreen = True
            from PySide6.QtCore import QTimer as _QT
            _QT.singleShot(100, self._apply_fullscreen_scale)
        logger.info("Fullscreen: %s", "ON" if self._is_fullscreen else "OFF")

    def _apply_fullscreen_scale(self) -> None:
        """Calculate and apply scale factor for fullscreen mode."""
        if not self._is_fullscreen:
            return
        screen_w = self._window.width()
        screen_h = self._window.height()
        sx = screen_w / Size.SCREEN_W
        sy = screen_h / Size.SCREEN_H
        scale = min(sx, sy)
        self._view.resetTransform()
        self._view.scale(scale, scale)
        self._view.centerOn(self._proxy)
        logger.info("Fullscreen scale: %.2f (%dx%d)", scale, screen_w, screen_h)

    def toggle_debug_panel(self) -> None:
        """Toggle the full-page debug detection panel (F12)."""
        self._debug_active = not self._debug_active
        self._debug_panel.set_active(self._debug_active)
        if self._debug_active:
            self._saved_stack_idx = self._stack.currentIndex()
            self._stack.setCurrentIndex(self._debug_panel_idx)
        else:
            if hasattr(self, "_saved_stack_idx"):
                self._stack.setCurrentIndex(self._saved_stack_idx)
        logger.info("Debug panel: %s", "ON" if self._debug_active else "OFF")

    def _connect_signals(self) -> None:
        """Wire bridge signals to subsystem handlers."""
        # IMU nav commands — intercept for preset overlay
        self._bridge.nav_command.connect(self._on_nav_command)
        self._bridge.session_state_changed.connect(
            self._imu_nav.on_session_state_changed
        )
        self._imu_nav.go_back.connect(self._router.back)
        # Dev overlay: flash pads on punches, show CV predictions
        self._bridge.punch_confirmed.connect(self._on_punch_for_dev)
        # Debug panel: CV detection info + confirmed punches
        self._bridge.debug_info.connect(self._debug_panel.on_debug_info)
        self._bridge.punch_confirmed.connect(self._debug_panel.on_punch_confirmed)

    def _on_nav_command(self, command: str) -> None:
        """Route IMU nav commands — preset overlay intercepts when visible."""
        # If preset overlay is open, all commands go to it
        if self._preset_overlay.is_visible:
            if command == "back":
                self._preset_overlay.slide_out()
            elif command == "prev":
                self._preset_overlay.navigate_left()
            elif command == "next":
                self._preset_overlay.navigate_right()
            elif command == "enter":
                self._preset_overlay.confirm()
            return

        # Head pad — toggle presets on home pages only
        if command == "back":
            current = self._router.current_page
            if current in ("home", "home_guest"):
                kwargs = self._router._page_kwargs.get(current, {})
                username = kwargs.get("username", "")
                self._preset_overlay.set_username(username)
                self._preset_overlay.toggle()
            return

        # Centre pad — trigger start button on config/test pages
        if command == "enter":
            current = self._router.current_page
            page = self._router._pages.get(current)
            if page and hasattr(page, "imu_start"):
                page.imu_start()
            return

        # Other IMU commands ignored outside preset overlay

    def _poll_remote_commands(self) -> None:
        """Check for commands from the phone dashboard."""
        import json

        # ── Height control from phone dashboard (file-based) ──────────────
        height_file = Path("/tmp/boxbunny_height_cmd.json")
        try:
            if height_file.exists():
                hdata = json.loads(height_file.read_text())
                h_action = hdata.get("action", "stop")
                h_ts = hdata.get("timestamp", 0.0)
                import time as _t
                age = _t.time() - h_ts
                # If the phone stopped sending (>300ms stale), force stop
                if age > 0.3:
                    if getattr(self, '_last_phone_height', 'stop') != 'stop':
                        h_action = "stop"
                    else:
                        h_action = None  # already stopped, skip publish
                if h_action and self._bridge:
                    self._bridge.publish_height_command(h_action)
                    self._last_phone_height = h_action
        except (json.JSONDecodeError, OSError):
            pass

        # Check for phone login notification
        # Skip if a QR popup dialog is currently open — it handles its own polling
        from PySide6.QtWidgets import QApplication
        active_modal = QApplication.activeModalWidget()
        if active_modal is None:
            login_file = Path("/tmp/boxbunny_gui_login.json")
            try:
                if login_file.exists():
                    login_data = json.loads(login_file.read_text())
                    login_file.unlink(missing_ok=True)
                    username = login_data.get("username", "")
                    if username:
                        logger.info("Phone login detected: %s", username)
                        if self._preset_overlay.is_visible:
                            self._preset_overlay.slide_out()
                        self._router.navigate("home", username=username)
                        return
            except (json.JSONDecodeError, OSError):
                pass

        # Check for remote commands
        cmd_file = Path("/tmp/boxbunny_gui_command.json")
        try:
            if not cmd_file.exists():
                return
            data = json.loads(cmd_file.read_text())
            cmd_file.unlink(missing_ok=True)
            action = data.get("action", "")
            config = data.get("config", {})
            username = data.get("username", "")
            logger.info("Remote command: %s from %s", action, username)

            if action == "start_preset":
                self._preset_overlay.set_username(username)
                self._on_preset_selected(config)
            elif action == "open_presets":
                self._preset_overlay.set_username(username)
                self._preset_overlay.toggle()
            elif action == "start_training":
                self._router.navigate(
                    "training_session",
                    config=config,
                    username=username,
                )
            elif action == "setup_drill":
                # Navigate to training config — user starts when ready
                combo = config.get("combo", {})
                self._router.navigate(
                    "training_config",
                    combo=combo,
                    difficulty=config.get("difficulty", "Beginner"),
                    username=username,
                )
            elif action == "height_adjust":
                # Height now goes directly via ROS from dashboard server
                return
            elif action == "navigate":
                route = config.get("route", "")
                nav_user = username or config.get("username", "")
                if route:
                    if self._preset_overlay.is_visible:
                        self._preset_overlay.slide_out()
                    self._router.navigate(route, username=nav_user)
        except (json.JSONDecodeError, OSError):
            pass

    def _on_preset_selected(self, preset: dict) -> None:
        """Called when user confirms a preset — route to config page with Start button."""
        route = preset.get("route", "training_config")
        username = self._preset_overlay._username

        if route in ("power_test", "stamina_test", "reaction_test"):
            # Performance tests — navigate directly (they have their own start)
            self._router.navigate(route, username=username)
        else:
            # Training drills — go to config page, NOT directly to session.
            # User presses centre pad (imu_start) on config page to begin.
            config = dict(preset.get("config", {}))
            config["combo"] = preset.get("combo", {})
            self._router.navigate(
                "training_config",
                config=config,
                combo=preset.get("combo", {}),
                combo_id=preset.get("combo", {}).get("id"),
                difficulty=preset.get("difficulty", "Beginner"),
                username=username,
            )

    def _on_punch_for_dev(self, data: dict) -> None:
        """Forward punch data to the dev overlay."""
        if not self._dev_overlay.isVisible():
            return
        pad = data.get("pad", "")
        level = data.get("level", "medium")
        punch_type = data.get("type", "idle")
        confidence = data.get("cv_conf", 0.0)
        if pad:
            self._dev_overlay.flash_pad(pad, level)
        self._dev_overlay.set_prediction(punch_type, confidence)

    def _register_pages(self) -> None:
        """Create and register all page widgets with the router.

        Each page receives its dependencies through its constructor -- no
        globals required.  Placeholder QWidgets are used for pages that
        have not yet been implemented.
        """
        # Shared deps dict for page factories
        deps = {
            "bridge": self._bridge,
            "router": self._router,
            "sound": self._sound,
            "imu_nav": self._imu_nav,
            "dev_overlay": self._dev_overlay,
            "preset_overlay": self._preset_overlay,
        }

        # Register all pages — each gets its actual class from the page_map
        page_names = [
            "auth", "home", "home_guest", "home_coach",
            "guest_assessment", "account_picker", "pattern_lock", "signup",
            "training_select", "training_config", "training_session",
            "training_rest", "training_results", "self_select",
            "sparring_select", "sparring_session", "sparring_results",
            "performance", "power_test", "stamina_test", "reaction_test",
            "history", "presets", "coach", "settings",
        ]
        for name in page_names:
            page = self._try_load_page(name, deps)
            self._router.register(name, page)

    def _try_load_page(self, name: str, deps: dict) -> QWidget:
        """Attempt to import and instantiate a page, falling back to a stub."""
        # Map route name -> (module_path, class_name)
        # These match the actual filenames and class names in pages/
        page_map = {
            "auth": ("boxbunny_gui.pages.auth.startup_page", "StartupPage"),
            "home": ("boxbunny_gui.pages.home.home_individual", "HomeIndividualPage"),
            "home_guest": ("boxbunny_gui.pages.home.home_guest", "HomeGuestPage"),
            "home_coach": ("boxbunny_gui.pages.home.home_coach", "HomeCoachPage"),
            "guest_assessment": ("boxbunny_gui.pages.auth.guest_assessment_page", "GuestAssessmentPage"),
            "account_picker": ("boxbunny_gui.pages.auth.account_picker_page", "AccountPickerPage"),
            "pattern_lock": ("boxbunny_gui.pages.auth.pattern_lock_page", "PatternLockPage"),
            "signup": ("boxbunny_gui.pages.auth.signup_page", "SignupPage"),
            "training_select": ("boxbunny_gui.pages.training.combo_select_page", "ComboSelectPage"),
            "training_config": ("boxbunny_gui.pages.training.training_config_page", "TrainingConfigPage"),
            "training_session": ("boxbunny_gui.pages.training.training_session_page", "TrainingSessionPage"),
            "training_rest": ("boxbunny_gui.pages.training.training_rest_page", "TrainingRestPage"),
            "training_results": ("boxbunny_gui.pages.training.training_results_page", "TrainingResultsPage"),
            "self_select": ("boxbunny_gui.pages.training.self_select_page", "SelfSelectPage"),
            "sparring_select": ("boxbunny_gui.pages.sparring.sparring_config_page", "SparringConfigPage"),
            "sparring_session": ("boxbunny_gui.pages.sparring.sparring_session_page", "SparringSessionPage"),
            "sparring_results": ("boxbunny_gui.pages.sparring.sparring_results_page", "SparringResultsPage"),
            "performance": ("boxbunny_gui.pages.performance.performance_menu_page", "PerformanceMenuPage"),
            "power_test": ("boxbunny_gui.pages.performance.power_test_page", "PowerTestPage"),
            "stamina_test": ("boxbunny_gui.pages.performance.stamina_test_page", "StaminaTestPage"),
            "reaction_test": ("boxbunny_gui.pages.performance.reaction_test_page", "ReactionTestPage"),
            "history": ("boxbunny_gui.pages.history.history_page", "HistoryPage"),
            "presets": ("boxbunny_gui.pages.presets.presets_page", "PresetsPage"),
            "coach": ("boxbunny_gui.pages.coach.station_page", "StationPage"),
            "settings": ("boxbunny_gui.pages.settings.settings_page", "SettingsPage"),
        }

        if name not in page_map:
            return self._make_placeholder(name)

        module_path, class_name = page_map[name]
        try:
            import importlib
            import inspect
            mod = importlib.import_module(module_path)
            cls = getattr(mod, class_name)
            # Only pass kwargs the constructor actually accepts
            sig = inspect.signature(cls.__init__)
            accepted = set(sig.parameters.keys()) - {"self"}
            if "kwargs" in str(sig):
                # Constructor accepts **kwargs, pass everything
                filtered_deps = deps
            else:
                filtered_deps = {k: v for k, v in deps.items() if k in accepted}
            # Always try with parent=None too
            if "parent" in accepted and "parent" not in filtered_deps:
                filtered_deps["parent"] = None
            return cls(**filtered_deps)
        except Exception as exc:
            logger.warning("Page '%s' failed to load: %s", name, exc, exc_info=True)
            return self._make_placeholder(name)

    @staticmethod
    def _make_placeholder(name: str) -> QWidget:
        """Create a minimal stub widget for an unimplemented page."""
        from PySide6.QtWidgets import QLabel, QVBoxLayout
        from boxbunny_gui.theme import Color, font

        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.setAlignment(Qt.AlignmentFlag.AlignCenter)
        label = QLabel(f"{name.replace('_', ' ').title()}")
        label.setFont(font(Size.TEXT_HEADER, bold=True))
        label.setStyleSheet(f"color: {Color.TEXT_SECONDARY};")
        label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(label)

        sub = QLabel("Coming soon")
        sub.setFont(font(Size.TEXT_BODY))
        sub.setStyleSheet(f"color: {Color.TEXT_DISABLED};")
        sub.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(sub)
        return widget

    @staticmethod
    def _resolve_assets_dir() -> Path:
        """Find the assets directory, checking source tree then install share."""
        if _ASSETS_DIR.is_dir():
            return _ASSETS_DIR
        # Fallback: ament install share directory
        try:
            from ament_index_python.packages import get_package_share_directory
            share = Path(get_package_share_directory("boxbunny_gui"))
            assets = share / "assets"
            if assets.is_dir():
                return assets
        except Exception:
            pass
        logger.warning("Assets directory not found -- sounds will be silent")
        return _ASSETS_DIR

    @staticmethod
    def _setup_logging() -> None:
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
            datefmt="%H:%M:%S",
        )

    def _shutdown(self) -> None:
        """Clean up subsystems on exit."""
        self._sound.stop_all()
        self._bridge.shutdown()
        logger.info("BoxBunnyApp shut down")
        # Force kill the process so ROS threads don't hang
        import os
        os._exit(0)


def main() -> None:
    """ROS 2 / console_scripts entry point."""
    app = BoxBunnyApp()
    sys.exit(app.run())


if __name__ == "__main__":
    main()
