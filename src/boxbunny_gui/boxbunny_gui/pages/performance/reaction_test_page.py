"""Reaction time test — 3 attempts with YOLO pose detection.

Camera positioning → countdown → red WAIT → green PUNCH → result → repeat.
Results page: replay on top, big average, tier, per-attempt cards.
"""
from __future__ import annotations

import logging
import random
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional

import cv2
import numpy as np

from PySide6.QtCore import QObject, Qt, QThread, QTimer, Signal
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import (
    QHBoxLayout, QLabel, QStackedLayout, QVBoxLayout, QWidget,
)
from PySide6.QtWidgets import QPushButton as _Btn

from boxbunny_gui.theme import Color, Icon, Size, font, badge_style, back_link_style, PRIMARY_BTN
from boxbunny_gui.widgets import BigButton

if TYPE_CHECKING:
    from boxbunny_gui.gui_bridge import GuiBridge
    from boxbunny_gui.nav.router import PageRouter

logger = logging.getLogger(__name__)

_TRIALS = 3
_WS = Path(__file__).resolve().parents[5]
_YOLO = _WS / "action_prediction" / "model" / "yolo26n-pose.pt"
_MOTION_PX = 20.0

# ── Foreground-user gate ─────────────────────────────────────────────────────
# The reaction test rejects motion/punch events unless the primary tracked
# user is inside this gate. Without it, background walkers in a busy gym
# trip the trigger while the user is standing still. cv_node picks the
# largest + most centred person per frame and publishes UserTracking.
_CAM_WIDTH_PX = 960.0
# Middle 40% of the frame horizontally (x in [288, 672] for 960px).
_CENTRE_HALF_WIDTH_PX = _CAM_WIDTH_PX * 0.2
# Max depth (metres) at which a user is "close enough" to the camera.
# 2.0 m comfortably covers an arm-length-plus setup; raise if the test
# is run from further back.
_MAX_USER_DEPTH_M = 2.0
# Tracking freshness — if no UserTracking message in this window, treat
# the gate as closed (better to miss an event than accept a false one).
_TRACKING_STALE_S = 0.5

_TIERS = [(150, "Lightning", "#56D364"), (200, "Fast", "#58A6FF"),
          (280, "Average", "#FF6B35"), (380, "Developing", "#FFAB40"),
          (9999, "Slow", "#FF5C5C")]


def _color(ms): return "#56D364" if ms < 200 else "#58A6FF" if ms < 300 else "#FF6B35" if ms < 400 else "#FF5C5C"
def _tier(avg):
    for t, n, c in _TIERS:
        if avg <= t: return n, c
    return "Slow", "#FF5C5C"
def _ord(n): return f"{n}{({1:'st',2:'nd',3:'rd'}.get(n,'th'))}"
def _short(n): return f"#{n}"


# ── Camera Worker ────────────────────────────────────────────────────────────

class _Cam(QObject):
    """Camera worker for reaction test.

    Uses pose_frame from the main CV model — no separate YOLO loaded,
    no GPU contention. Tracking stays smooth the whole time.
    """

    frame = Signal(object)
    motion = Signal(float)

    def __init__(self):
        super().__init__()
        self._on = False; self._prev_gray = None; self._prev_kps = None
        self._rec = False; self._buf: list = []; self._best: list = []; self._best_ms = 99999.0

    def reset(self): self._prev_gray = None; self._prev_kps = None
    def rec_start(self): self._rec = True; self._buf.clear()
    def rec_stop(self, ms):
        self._rec = False
        if ms < self._best_ms and self._buf:
            self._best_ms = ms; self._best = list(self._buf)
        self._buf.clear()

    def run(self):
        self._on = True; self._prev_gray = None
        self._run_ros()

    def _run_ros(self):
        """Try pose_frame from main CV model, fall back to own camera + YOLO."""
        import rclpy
        from rclpy.executors import SingleThreadedExecutor
        from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
        from sensor_msgs.msg import Image
        from cv_bridge import CvBridge

        ctx = rclpy.Context()
        ctx.init()
        node = rclpy.create_node("reaction_cam", context=ctx)
        executor = SingleThreadedExecutor(context=ctx)
        executor.add_node(node)
        bridge = CvBridge()
        pose_frame = [None]
        raw_frame = [None]

        def _pose_cb(msg):
            try:
                pose_frame[0] = bridge.imgmsg_to_cv2(msg, "mono8")
            except Exception:
                pass

        def _raw_cb(msg):
            try:
                raw_frame[0] = bridge.imgmsg_to_cv2(msg, "bgr8")
            except Exception:
                pass

        qos = QoSProfile(
            depth=1,
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
        )
        node.create_subscription(Image, "/boxbunny/cv/pose_frame", _pose_cb, qos)
        node.create_subscription(Image, "/camera/color/image_raw", _raw_cb, 1)

        try:
            # Wait up to 15s for pose_frame (CV model may still be loading)
            t0 = time.time()
            while pose_frame[0] is None and raw_frame[0] is None and time.time() - t0 < 15.0 and self._on:
                executor.spin_once(timeout_sec=0.1)

            if pose_frame[0] is not None:
                # Main CV model is running — use pose_frame (grayscale + dots)
                logger.info("Using pose_frame from CV pipeline")
                while self._on:
                    executor.spin_once(timeout_sec=0.01)
                    if pose_frame[0] is not None:
                        self._proc(pose_frame[0])
            elif raw_frame[0] is not None:
                # Raw camera frames available (system running but no pose_frame)
                logger.info("No pose_frame — using raw camera + own YOLO")
                self._load_yolo()
                while self._on:
                    executor.spin_once(timeout_sec=0.01)
                    if raw_frame[0] is not None:
                        self._proc_bgr(raw_frame[0])
            else:
                # Nothing after 15s — check if CV inference is expected
                cv_nodes = node.get_node_names()
                cv_running = any("cv_ros_bridge" in n for n in cv_nodes)
                if cv_running:
                    # CV node exists but no frames yet — keep waiting
                    logger.info("CV node active but no frames yet — waiting for pose_frame")
                    while self._on:
                        executor.spin_once(timeout_sec=0.1)
                        if pose_frame[0] is not None:
                            logger.info("pose_frame now available")
                            break
                        if raw_frame[0] is not None:
                            break
                    # Process whichever stream appeared
                    if pose_frame[0] is not None:
                        while self._on:
                            executor.spin_once(timeout_sec=0.01)
                            if pose_frame[0] is not None:
                                self._proc(pose_frame[0])
                    elif raw_frame[0] is not None:
                        self._load_yolo()
                        while self._on:
                            executor.spin_once(timeout_sec=0.01)
                            if raw_frame[0] is not None:
                                self._proc_bgr(raw_frame[0])
                else:
                    # Standalone mode — no CV node, try direct camera
                    if not self._on:
                        return
                    logger.info("No ROS frames and no CV node — trying direct camera")
                    self._run_direct_camera(node, executor)
        finally:
            node.destroy_node()
            ctx.try_shutdown()

    def _run_direct_camera(self, node, executor):
        """Open RealSense directly when no ROS camera is available."""
        if not self._on:
            return
        try:
            import sys
            _conda_sp = str(Path.home() / "miniconda3/envs/boxing_ai/lib/python3.10/site-packages")
            if _conda_sp not in sys.path:
                sys.path.insert(0, _conda_sp)
            import pyrealsense2 as rs
            ctx = rs.context()
            if len(ctx.query_devices()) == 0:
                logger.warning("No RealSense camera connected — skipping direct camera")
                return
            pipeline = rs.pipeline()
            cfg = rs.config()
            cfg.enable_stream(rs.stream.color, 960, 540, rs.format.bgr8, 30)
            pipeline.start(cfg)
            if not self._on:
                pipeline.stop()
                return
            import time as _time
            _time.sleep(0.5)
            logger.info("Direct camera opened for reaction test")
            try:
                while self._on:
                    frames = pipeline.wait_for_frames(timeout_ms=1000)
                    color = frames.get_color_frame()
                    if color:
                        import numpy as np
                        bgr = np.asanyarray(color.get_data())
                        self._proc_bgr(bgr)
            finally:
                pipeline.stop()
        except Exception as e:
            logger.warning("Direct camera failed: %s", e)

    def stop(self): self._on = False

    _shared_mdl = None

    def _load_yolo(self):
        if _Cam._shared_mdl is not None:
            return
        try:
            import sys
            _conda_sp = str(Path.home() / "miniconda3/envs/boxing_ai/lib/python3.10/site-packages")
            if _conda_sp not in sys.path:
                sys.path.insert(0, _conda_sp)
            from ultralytics import YOLO
            path = str(_YOLO) if _YOLO.exists() else "yolo11s-pose.pt"
            _Cam._shared_mdl = YOLO(path)
            logger.info("Standalone YOLO loaded: %s", path)
        except Exception as e:
            logger.warning("YOLO load failed: %s", e)

    def _proc_bgr(self, bgr):
        """Standalone mode: run own YOLO, draw dots, detect motion."""
        mdl = _Cam._shared_mdl
        mv = 0.0
        if mdl is not None:
            try:
                r = mdl(bgr, verbose=False)
                kps = self._kps(r)
                if kps is not None:
                    for i in range(len(kps)):
                        x, y = int(kps[i][0]), int(kps[i][1])
                        if len(kps[i]) >= 3 and kps[i][2] > 0.3:
                            cv2.circle(bgr, (x, y), 5, (0, 255, 0), -1)
                    if self._prev_kps is not None:
                        for j in range(min(len(self._prev_kps), len(kps))):
                            if len(kps[j]) >= 3 and kps[j][2] < 0.3:
                                continue
                            mv = max(mv, float(np.sqrt(
                                (kps[j][0] - self._prev_kps[j][0]) ** 2 +
                                (kps[j][1] - self._prev_kps[j][1]) ** 2)))
                    self._prev_kps = kps
            except Exception:
                pass
        if self._rec:
            self._buf.append(bgr.copy())
        g = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        h, w = g.shape
        self.frame.emit(
            QImage(g.data, w, h, w, QImage.Format.Format_Grayscale8).copy())
        if mv > 0:
            self.motion.emit(mv)

    @staticmethod
    def _kps(r):
        if not r or not r[0].keypoints or r[0].keypoints.data is None:
            return None
        kps_data = r[0].keypoints.data.cpu().numpy()
        if kps_data.shape[0] == 0:
            return None
        if kps_data.shape[0] == 1:
            return kps_data[0]
        boxes = r[0].boxes
        if boxes is None or boxes.xyxy is None:
            return kps_data[0]
        xyxy = boxes.xyxy.cpu().numpy()
        img_w = 640
        best_idx, best_score = 0, -1
        for i in range(len(xyxy)):
            x1, y1, x2, y2 = xyxy[i]
            area = (x2 - x1) * (y2 - y1)
            cx = (x1 + x2) / 2
            centre_dist = abs(cx - img_w / 2) / (img_w / 2)
            score = area * (1.0 - 0.5 * centre_dist)
            if score > best_score:
                best_score = score; best_idx = i
        return kps_data[best_idx] if best_idx < kps_data.shape[0] else kps_data[0]

    def _proc(self, gray):
        """Process grayscale+skeleton frame. Motion via frame differencing."""
        mv = 0.0
        if self._prev_gray is not None:
            diff = cv2.absdiff(gray, self._prev_gray)
            _, mask = cv2.threshold(diff, 15, 255, cv2.THRESH_BINARY)
            mv = float(cv2.countNonZero(mask)) / max(1, mask.size) * 1000.0
        self._prev_gray = gray.copy()
        if self._rec:
            self._buf.append(gray.copy())
        h, w = gray.shape
        self.frame.emit(
            QImage(gray.data, w, h, w, QImage.Format.Format_Grayscale8).copy())
        if mv > 2.0:
            self.motion.emit(mv)


# ── Page ─────────────────────────────────────────────────────────────────────

class ReactionTestPage(QWidget):
    def __init__(self, router: PageRouter, bridge: Optional[GuiBridge]=None, parent=None):
        super().__init__(parent)
        self._router = router; self._bridge = bridge
        self._trial = 0; self._times: List[float] = []
        self._stim_on = False; self._stim_t = 0.0; self._session_id: str = ""
        self._cam_t: Optional[QThread] = None; self._cam_w: Optional[_Cam] = None
        self._delay = QTimer(self); self._delay.setSingleShot(True); self._delay.timeout.connect(self._on_delay)
        self._countdown_n = 0; self._state = "idle"
        self._replay_timer = QTimer(self); self._replay_timer.setInterval(120); self._replay_timer.timeout.connect(self._rtick)
        self._rframes: list = []; self._ri = 0
        # Foreground-user gate state — updated by cv_node UserTracking.
        self._user_ok: bool = False
        self._user_last_ts: float = 0.0
        self._build()
        if self._bridge:
            self._bridge.punch_confirmed.connect(self._on_punch)
            self._bridge.user_tracking_changed.connect(self._on_user_tracking)

    def _build(self):
        root = QVBoxLayout(self); root.setContentsMargins(24,8,24,12); root.setSpacing(0)

        # Top
        top = QHBoxLayout()
        b = _Btn(f"{Icon.BACK}  Back"); b.setStyleSheet(back_link_style()); b.setCursor(Qt.CursorShape.PointingHandCursor); b.clicked.connect(self._abort)
        top.addWidget(b)
        t = QLabel("Reaction Time Test"); t.setStyleSheet(f"font-size:18px;font-weight:700;color:{Color.TEXT};"); top.addWidget(t)
        top.addStretch()
        self._badge = QLabel(f"0/{_TRIALS}")
        self._badge.setStyleSheet(f"font-size:16px;font-weight:700;color:{Color.WARNING};background:{Color.SURFACE};border-radius:10px;padding:6px 16px;")
        top.addWidget(self._badge)

        self._rbtn = _Btn("Slow-Mo"); self._rbtn.setFixedSize(100, 34)
        self._rbtn.setCursor(Qt.CursorShape.PointingHandCursor)
        self._rbtn.setStyleSheet(f"""
            QPushButton{{background:{Color.PRIMARY};color:#FFF;font-size:12px;font-weight:700;border:none;border-radius:8px;}}
            QPushButton:hover{{background:{Color.PRIMARY_DARK};}}
        """)
        self._rbtn.clicked.connect(self._rtoggle); self._rbtn.setVisible(False)
        top.addWidget(self._rbtn)
        root.addLayout(top); root.addSpacing(6)

        # Camera area
        self._cam_area = QWidget(); self._cam_area.setMinimumHeight(200)
        self._cam_area.setStyleSheet(f"background:{Color.BG};border-radius:14px;")
        stk = QStackedLayout(self._cam_area); stk.setStackingMode(QStackedLayout.StackingMode.StackAll)
        self._vid = QLabel(); self._vid.setAlignment(Qt.AlignCenter); self._vid.setStyleSheet(f"background:{Color.BG};border-radius:14px;")
        self._vid.setCursor(Qt.CursorShape.PointingHandCursor)
        self._vid.mousePressEvent = lambda e: self._rplay() if self._state == "done" else None
        stk.addWidget(self._vid)
        self._msg = QLabel("Tap Start"); self._msg.setAlignment(Qt.AlignCenter); self._msg.setWordWrap(True)
        self._msg.setStyleSheet(f"background:transparent;color:{Color.TEXT_SECONDARY};font-size:28px;font-weight:700;")
        self._msg.setCursor(Qt.CursorShape.PointingHandCursor)
        self._msg.mousePressEvent = lambda e: self._rplay() if self._state == "done" else None
        stk.addWidget(self._msg)
        root.addWidget(self._cam_area, stretch=1); root.addSpacing(6)

        # Attempt cards (in a widget so we can hide during results)
        self._cards_w = QWidget()
        row = QHBoxLayout(self._cards_w); row.setContentsMargins(0,0,0,0); row.setSpacing(6)
        self._cards: list[QLabel] = []
        for i in range(_TRIALS):
            c = QLabel(f"{_ord(i+1)}\n--"); c.setAlignment(Qt.AlignCenter); c.setFixedHeight(80)
            c.setStyleSheet(f"font-size:22px;font-weight:700;color:{Color.TEXT_DISABLED};background:{Color.SURFACE};border-radius:12px;")
            row.addWidget(c); self._cards.append(c)
        root.addWidget(self._cards_w); root.addSpacing(8)

        # Start button
        self._btn = BigButton("Start Test", stylesheet=PRIMARY_BTN); self._btn.setFixedHeight(80); self._btn.clicked.connect(self._begin)
        root.addWidget(self._btn)

        # ── Results ──────────────────────────────────────────────────────
        self._res = QWidget()
        rl = QVBoxLayout(self._res); rl.setContentsMargins(0,0,0,0); rl.setSpacing(0)

        rl.addStretch(2)

        # Average (big, centred)
        self._ravg = QLabel("--"); self._ravg.setAlignment(Qt.AlignCenter)
        self._ravg.setStyleSheet(f"font-size:80px;font-weight:800;color:{Color.PRIMARY};")
        rl.addWidget(self._ravg)

        # Tier pill
        self._rtier = QLabel("--"); self._rtier.setAlignment(Qt.AlignCenter); self._rtier.setFixedHeight(48)
        self._rtier.setStyleSheet(f"font-size:22px;font-weight:700;color:#FFF;background:{Color.SURFACE};border-radius:24px;")
        rl.addWidget(self._rtier)
        rl.addSpacing(12)

        # (Replay button is in the top bar, added below)

        rl.addStretch(3)

        # Attempt cards
        self._res_cards_row = QHBoxLayout(); self._res_cards_row.setSpacing(6)
        self._res_card_lbls: list[QLabel] = []
        for i in range(_TRIALS):
            c = QLabel(f"{_ord(i+1)}\n--"); c.setAlignment(Qt.AlignCenter); c.setFixedHeight(60)
            c.setStyleSheet(f"font-size:18px;font-weight:700;color:{Color.TEXT_DISABLED};background:{Color.SURFACE};border-radius:10px;")
            self._res_cards_row.addWidget(c); self._res_card_lbls.append(c)
        rl.addLayout(self._res_cards_row)
        rl.addSpacing(8)

        # AI Coach analysis
        ai_box = QWidget()
        ai_box.setStyleSheet(f"background:{Color.SURFACE};border-radius:12px;padding:12px;")
        ai_lay = QVBoxLayout(ai_box); ai_lay.setContentsMargins(14, 10, 14, 10); ai_lay.setSpacing(4)
        ai_title = QLabel("AI COACH")
        ai_title.setStyleSheet(
            f"font-size: 10px; font-weight: 700; color: {Color.INFO};"
            " letter-spacing: 1px;"
        )
        ai_lay.addWidget(ai_title)
        self._coach_lbl = QLabel("")
        self._coach_lbl.setStyleSheet(f"font-size: 14px; color: {Color.TEXT};")
        self._coach_lbl.setWordWrap(True)
        self._coach_lbl.setMinimumHeight(40)
        ai_lay.addWidget(self._coach_lbl)
        rl.addWidget(ai_box)
        rl.addSpacing(8)

        # Buttons
        br = QHBoxLayout(); br.setSpacing(10)
        retry = BigButton("Try Again", stylesheet=PRIMARY_BTN)
        retry.setFixedHeight(64); retry.clicked.connect(self._begin)
        br.addWidget(retry, stretch=1)
        done = BigButton("Done", stylesheet=f"""
            QPushButton{{background:{Color.SURFACE_LIGHT};color:{Color.TEXT};font-size:16px;font-weight:700;border:none;border-radius:12px;}}
            QPushButton:hover{{background:{Color.SURFACE_HOVER};}}
        """)
        done.setFixedHeight(64); done.setCursor(Qt.CursorShape.PointingHandCursor)
        done.clicked.connect(lambda: self._router.navigate("performance"))
        br.addWidget(done, stretch=1)
        rl.addLayout(br)

        self._res.setVisible(False); root.addWidget(self._res)

    # ── Phase display ────────────────────────────────────────────────────

    def _phase(self, text, fg="#FFF", bg="transparent", sz=40):
        if bg == "transparent":
            self._msg.setStyleSheet(f"background:rgba(11,15,20,0.75);color:{fg};font-size:{sz}px;font-weight:800;border-radius:20px;padding:16px 32px;margin:20px;")
        else:
            self._msg.setStyleSheet(f"background:{bg};color:{fg};font-size:{sz}px;font-weight:800;border-radius:16px;margin:10px;")
        self._msg.setText(text)

    # ── Flow ─────────────────────────────────────────────────────────────

    def imu_start(self):
        if self._btn.isVisible(): self._begin()

    def _begin(self):
        self._trial = 0; self._times.clear()
        for i,c in enumerate(self._cards): c.setText(f"{_ord(i+1)}\n--"); c.setStyleSheet(f"font-size:22px;font-weight:700;color:{Color.TEXT_DISABLED};background:{Color.SURFACE};border-radius:12px;")
        self._btn.setVisible(False); self._res.setVisible(False)
        self._coach_lbl.setText("")
        self._cam_area.setVisible(True); self._vid.setVisible(True); self._cards_w.setVisible(True)
        self._replay_timer.stop(); self._rbtn.setVisible(False)
        self._start_ros_session()
        self._start_cam(); self._go_countdown()

    def _go_countdown(self):
        self._state = "countdown"; self._countdown_n = 3
        self._vid.setVisible(False)
        self._badge.setText(f"{_ord(self._trial+1)} Attempt")
        self._phase(str(self._countdown_n), sz=120)
        self._delay.start(800)

    def _on_delay(self):
        if self._state == "countdown":
            self._countdown_n -= 1
            if self._countdown_n > 0: self._phase(str(self._countdown_n), sz=120); self._delay.start(800)
            else: self._go_red()
        elif self._state == "red": self._go_green()

    def _go_red(self):
        self._state = "red"
        self._phase("WAIT...", bg="#CC2A2A", sz=72)
        self._delay.start(random.randint(1500, 4000))

    def _go_green(self):
        self._state = "green"; self._stim_on = True; self._stim_t = time.monotonic()
        if self._cam_w: self._cam_w.reset(); self._cam_w.rec_start()
        self._phase("PUNCH!", bg="#1B8C3D", sz=80)

    def _on_motion(self, m):
        if self._stim_on and m > _MOTION_PX and self._user_gate_open():
            self._record()

    def _on_punch(self, d):
        if self._stim_on and self._user_gate_open():
            self._record()

    def _on_user_tracking(self, data: Dict[str, Any]) -> None:
        """Update the foreground-user gate from cv_node tracking.

        Accepts only a user who is present, close to the camera, and
        roughly centred in the frame. Background walkers fail one of
        these checks and are filtered out.
        """
        self._user_last_ts = time.monotonic()
        if not data.get("user_detected"):
            self._user_ok = False
            return
        depth = float(data.get("depth", 0.0))
        if depth <= 0.0 or depth > _MAX_USER_DEPTH_M:
            self._user_ok = False
            return
        cx = float(data.get("bbox_centre_x", 0.0))
        if abs(cx - _CAM_WIDTH_PX / 2.0) > _CENTRE_HALF_WIDTH_PX:
            self._user_ok = False
            return
        self._user_ok = True

    def _user_gate_open(self) -> bool:
        """True iff the primary user is present, close, centred, and fresh."""
        if not self._user_ok:
            return False
        if time.monotonic() - self._user_last_ts > _TRACKING_STALE_S:
            return False
        return True

    def _record(self):
        if not self._stim_on: return
        self._stim_on = False
        ms = (time.monotonic() - self._stim_t) * 1000
        self._times.append(ms); self._trial += 1
        if self._cam_w: self._cam_w.rec_stop(ms)

        c = _color(ms)
        self._badge.setText(f"{_ord(self._trial)} Attempt")

        # Update card
        if self._trial-1 < len(self._cards):
            cd = self._cards[self._trial-1]
            cd.setText(f"{_ord(self._trial)}\n{ms:.0f}ms")
            cd.setStyleSheet(f"font-size:22px;font-weight:700;color:{c};background:{Color.SURFACE};border-radius:12px;")

        self._phase(f"{ms:.0f} ms", fg=c, sz=72)

        if self._trial >= _TRIALS: QTimer.singleShot(1200, self._results)
        else: QTimer.singleShot(1500, self._go_countdown)

    def _start_ros_session(self) -> None:
        """Start a ROS session so the imu_node switches to TRAINING mode."""
        if self._bridge is None:
            return
        import json
        self._bridge.call_start_session(
            mode="reaction_test",
            difficulty="medium",
            config_json=json.dumps({"test": "reaction"}),
            username="",
            callback=lambda ok, sid, msg: (
                setattr(self, '_session_id', sid) if ok else None,
                logger.info("Reaction test session %s: %s", "started" if ok else "failed", sid or msg),
            ),
        )

    def _end_ros_session(self) -> None:
        """End the ROS session so imu_node returns to NAVIGATION mode."""
        if self._bridge is None or not self._session_id:
            return
        self._bridge.call_end_session(
            session_id=self._session_id,
            callback=lambda ok, summary, msg: logger.info(
                "Reaction test session ended: ok=%s", ok),
        )
        self._session_id = ""

    def _results(self):
        self._state = "done"
        self._end_ros_session()
        has_rp = False
        if self._cam_w and hasattr(self._cam_w, '_best'):
            self._rframes = list(self._cam_w._best)
            has_rp = len(self._rframes) > 2
        self._stop_cam()

        avg = sum(self._times) / len(self._times)
        tn, tc = _tier(avg)

        self._ravg.setText(f"{avg:.0f} ms")
        self._ravg.setStyleSheet(f"font-size:72px;font-weight:800;color:{tc};")
        self._rtier.setText(tn)
        self._rtier.setStyleSheet(f"font-size:20px;font-weight:700;color:#FFF;background:{tc};border-radius:22px;")

        self._rbtn.setVisible(has_rp)
        self._rbtn.setText("Slow-Mo")
        self._cam_area.setVisible(False)

        # Fill result attempt cards
        for i, lbl in enumerate(self._res_card_lbls):
            if i < len(self._times):
                ms = self._times[i]
                c = _color(ms)
                lbl.setText(f"{_ord(i+1)}\n{ms:.0f}ms")
                lbl.setStyleSheet(f"font-size:18px;font-weight:700;color:{c};background:{Color.SURFACE};border-radius:10px;")
            else:
                lbl.setText(f"{_ord(i+1)}\n--")
                lbl.setStyleSheet(f"font-size:18px;font-weight:700;color:{Color.TEXT_DISABLED};background:{Color.SURFACE};border-radius:10px;")

        try:
            from boxbunny_gui.session_tracker import get_tracker
            get_tracker().add_session(
                mode="Performance", duration="Reaction Test",
                punches=str(_TRIALS), score=f"{avg:.0f}ms ({tn})",
            )
        except Exception:
            pass

        self._cam_area.setVisible(False)
        self._cards_w.setVisible(False)
        self._msg.setText("")
        self._res.setVisible(True)

        self._request_llm_summary(avg, tn, self._times)

    # ── AI Coach ─────────────────────────────────────────────────────────

    def _request_llm_summary(
        self, avg: float, tier: str, times: List[float],
    ) -> None:
        import json

        best = min(times)
        worst = max(times)
        spread = worst - best

        session_desc = (
            f"The user completed a {_TRIALS}-trial reaction time test. "
            f"Individual times: {', '.join(f'{t:.0f}ms' for t in times)}. "
            f"Average: {avg:.0f}ms ({tier} tier). "
            f"Best: {best:.0f}ms, worst: {worst:.0f}ms, spread: {spread:.0f}ms."
        )

        if self._bridge is None:
            self._coach_lbl.setText(self._local_summary(avg, tier, times))
            return

        context = {
            "test_type": "reaction_time",
            "num_trials": _TRIALS,
            "times_ms": [round(t, 1) for t in times],
            "avg_ms": round(avg, 1),
            "best_ms": round(best, 1),
            "worst_ms": round(worst, 1),
            "spread_ms": round(spread, 1),
            "tier": tier,
        }
        self._coach_lbl.setText("AI analysis loading...")
        self._bridge.call_generate_llm(
            prompt=(
                f"Give a brief 2-sentence coaching analysis of this reaction test. "
                f"Be specific about the actual numbers and give one actionable tip. "
                f"{session_desc}"
            ),
            context_json=json.dumps(context),
            system_prompt_key="coach_summary",
            callback=self._on_llm_response,
        )

    @staticmethod
    def _local_summary(avg: float, tier: str, times: List[float]) -> str:
        """Fallback summary when LLM is unavailable."""
        best = min(times)
        spread = max(times) - best
        if spread > 80:
            consistency = "Your times varied quite a bit — focus on staying relaxed and consistent between attempts."
        elif spread > 40:
            consistency = "Decent consistency across attempts — keep working on making every rep feel the same."
        else:
            consistency = "Very consistent times — great focus and composure."
        return (
            f"Average: {avg:.0f}ms ({tier}), best attempt: {best:.0f}ms. "
            f"{consistency}"
        )

    def _on_llm_response(
        self, success: bool, response: str, gen_time: float,
    ) -> None:
        if success and response.strip():
            self._coach_lbl.setText(response.strip())
        else:
            self._coach_lbl.setText(
                self._local_summary(
                    sum(self._times) / max(len(self._times), 1),
                    _tier(sum(self._times) / max(len(self._times), 1))[0],
                    self._times,
                )
            )

    # ── Replay ───────────────────────────────────────────────────────────

    def _rtoggle(self):
        """Button click: show/hide the replay in the camera area."""
        if self._cam_area.isVisible():
            # Hide
            self._replay_timer.stop()
            self._cam_area.setVisible(False)
            self._rbtn.setText("Slow-Mo")
        else:
            # Show camera area and play replay into it
            self._cam_area.setVisible(True)
            self._vid.setVisible(True)
            self._msg.setText("")
            self._rbtn.setText("Hide")
            self._rplay()

    def _rplay(self):
        """Play/replay the slow-mo video into the camera area."""
        if not self._rframes: return
        self._ri = 0
        self._msg.setText("")
        self._msg.setStyleSheet("background:transparent;")
        self._replay_timer.start()

    def _rtick(self):
        if self._ri >= len(self._rframes):
            self._replay_timer.stop()
            self._msg.setText("Tap to replay")
            self._msg.setStyleSheet(f"background:rgba(11,15,20,0.5);color:{Color.TEXT_SECONDARY};font-size:16px;font-weight:600;border-radius:14px;")
            return
        frame = self._rframes[self._ri]; self._ri += 1
        if len(frame.shape) == 3:
            g = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        else:
            g = frame
        h, w = g.shape
        qi = QImage(g.data, w, h, w, QImage.Format.Format_Grayscale8).copy()
        pm = QPixmap.fromImage(qi.scaled(self._vid.size(),Qt.AspectRatioMode.KeepAspectRatio,Qt.TransformationMode.FastTransformation))
        self._vid.setPixmap(self._round_pixmap(pm))

    # ── Abort / Camera ───────────────────────────────────────────────────

    def _abort(self):
        self._delay.stop(); self._replay_timer.stop(); self._stop_cam(); self._end_ros_session(); self._router.back()

    def _start_cam(self):
        if self._cam_t: return
        self._cam_w = _Cam(); self._cam_t = QThread(); self._cam_w.moveToThread(self._cam_t)
        self._cam_w.frame.connect(self._on_frame); self._cam_w.motion.connect(self._on_motion)
        self._cam_t.started.connect(self._cam_w.run); self._cam_t.start()

    _orphaned_threads: list = []  # prevent GC of still-running QThreads

    def _stop_cam(self):
        if self._cam_w:
            self._cam_w.stop()
        if self._cam_t:
            self._cam_t.quit()
            if not self._cam_t.wait(2000):
                logger.info("Camera thread still running — parking to avoid QThread crash")
                # Keep a reference so Qt doesn't destroy the C++ thread object
                self._orphaned_threads.append(self._cam_t)
                self._cam_t.finished.connect(
                    lambda t=self._cam_t: self._orphaned_threads.remove(t)
                )
            self._cam_t = None
            self._cam_w = None

    @staticmethod
    def _round_pixmap(pm, radius=14):
        from PySide6.QtGui import QPainter, QPainterPath
        rounded = QPixmap(pm.size())
        rounded.fill(Qt.GlobalColor.transparent)
        p = QPainter(rounded)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        path = QPainterPath()
        path.addRoundedRect(0, 0, pm.width(), pm.height(), radius, radius)
        p.setClipPath(path)
        p.drawPixmap(0, 0, pm)
        p.end()
        return rounded

    def _on_frame(self, qi):
        pm = QPixmap.fromImage(qi.scaled(self._vid.size(),Qt.AspectRatioMode.KeepAspectRatio,Qt.TransformationMode.FastTransformation))
        self._vid.setPixmap(self._round_pixmap(pm))
        # Show "Tap Start" once first frame arrives (replaces "Loading...")
        if self._state == "idle" and not self._btn.isEnabled():
            self._vid.setText("")
            self._vid.setStyleSheet(f"background:{Color.BG};border-radius:14px;")
            self._btn.setEnabled(True)
            self._btn.setText("Start Test")
            self._phase("Tap Start", fg=Color.TEXT_SECONDARY, sz=36)

    # ── Lifecycle ────────────────────────────────────────────────────────

    def on_enter(self, **kw):
        self._state="idle"; self._times.clear(); self._trial=0
        self._btn.setVisible(True); self._res.setVisible(False)
        self._cam_area.setVisible(True); self._vid.setVisible(True)
        self._badge.setText(f"0/{_TRIALS}")
        self._badge.setStyleSheet(f"font-size:16px;font-weight:700;color:{Color.WARNING};background:{Color.SURFACE};border-radius:10px;padding:6px 16px;")
        for i,c in enumerate(self._cards): c.setText(f"{_ord(i+1)}\n--"); c.setStyleSheet(f"font-size:22px;font-weight:700;color:{Color.TEXT_DISABLED};background:{Color.SURFACE};border-radius:12px;")
        self._vid.setText("Loading camera...")
        self._vid.setAlignment(Qt.AlignCenter)
        self._vid.setStyleSheet(
            f"background:{Color.SURFACE};color:{Color.PRIMARY};"
            f"font-size:24px;font-weight:700;border-radius:14px;"
        )
        self._msg.setText("")
        self._btn.setEnabled(False)
        self._start_cam()

    def on_leave(self):
        self._delay.stop(); self._replay_timer.stop(); self._stop_cam(); self._end_ros_session()
