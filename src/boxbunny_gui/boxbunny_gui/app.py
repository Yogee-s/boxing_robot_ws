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
from PySide6.QtGui import QFontDatabase
from PySide6.QtWidgets import QApplication, QMainWindow, QStackedWidget, QWidget

from boxbunny_gui.gui_bridge import GuiBridge
from boxbunny_gui.nav.imu_nav_handler import ImuNavHandler, KeyboardNavFilter
from boxbunny_gui.nav.router import PageRouter
from boxbunny_gui.sound import SoundManager
from boxbunny_gui.theme import GLOBAL_STYLESHEET, Size

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
        self._window.setStyleSheet("background-color: #0A0A0A;")

        # ── Central stacked widget for page routing ─────────────────────
        self._stack = QStackedWidget()
        self._window.setCentralWidget(self._stack)

        # ── Subsystems ──────────────────────────────────────────────────
        self._bridge = GuiBridge(parent=self._window)
        self._router = PageRouter(self._stack, parent=self._window)
        self._sound = SoundManager(self._resolve_assets_dir())
        self._imu_nav = ImuNavHandler(parent=self._window)

        # ── Keyboard nav filter (dev convenience) ───────────────────────
        self._kb_filter = KeyboardNavFilter(self._imu_nav, parent=self._window)
        self._window.installEventFilter(self._kb_filter)

        # ── Developer mode overlay ──────────────────────────────────────
        from boxbunny_gui.widgets.dev_overlay import DevOverlay
        self._dev_overlay = DevOverlay(parent=self._window)
        self._dev_overlay.move(Size.SCREEN_W - 330, Size.SCREEN_H - 230)
        self._dev_overlay.set_developer_mode(False)
        self._dev_overlay.raise_()

        # ── Wire signals ────────────────────────────────────────────────
        self._connect_signals()

        # ── Register pages ──────────────────────────────────────────────
        self._register_pages()

        # ── Start ROS bridge (non-blocking) ─────────────────────────────
        self._bridge.start()

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

    def _connect_signals(self) -> None:
        """Wire bridge signals to subsystem handlers."""
        self._bridge.nav_command.connect(self._imu_nav.handle_command)
        self._bridge.session_state_changed.connect(
            self._imu_nav.on_session_state_changed
        )
        self._imu_nav.go_back.connect(self._router.back)
        # Dev overlay: flash pads on punches, show CV predictions
        self._bridge.punch_confirmed.connect(self._on_punch_for_dev)

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
        }

        # Register all pages — each gets its actual class from the page_map
        page_names = [
            "auth", "home", "home_guest", "home_coach",
            "guest_assessment", "account_picker", "pattern_lock", "signup",
            "training_select", "training_config", "training_session",
            "training_rest", "training_results",
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
            logger.warning("Page '%s' failed to load: %s", name, exc)
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


def main() -> None:
    """ROS 2 / console_scripts entry point."""
    app = BoxBunnyApp()
    sys.exit(app.run())


if __name__ == "__main__":
    main()
