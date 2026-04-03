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
    frame = Signal(object)
    motion = Signal(float)

    def __init__(self):
        super().__init__()
        self._on = False; self._mdl = None; self._prev = None
        self._rec = False; self._buf: list = []; self._best: list = []; self._best_ms = 99999.0

    def reset(self): self._prev = None
    def rec_start(self): self._rec = True; self._buf.clear()
    def rec_stop(self, ms):
        self._rec = False
        if ms < self._best_ms and self._buf:
            self._best_ms = ms; self._best = list(self._buf)
        self._buf.clear()

    def _lazy_load_model(self):
        """Load YOLO on first use — doesn't block camera startup."""
        if self._mdl is not None or self._mdl_failed: return
        try:
            from ultralytics import YOLO
            self._mdl = YOLO(str(_YOLO) if _YOLO.exists() else "yolo11s-pose.pt")
        except:
            self._mdl_failed = True

    def run(self):
        self._on = True; self._prev = None; self._mdl_failed = False

        try:
            import pyrealsense2 as rs
            p = rs.pipeline(); c = rs.config()
            c.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
            p.start(c)
            while self._on:
                try: f = p.wait_for_frames(timeout_ms=1000)
                except: continue
                cf = f.get_color_frame()
                if cf: self._proc(np.asanyarray(cf.get_data()))
            p.stop(); return
        except: pass

        for idx in range(6):
            try:
                cap = cv2.VideoCapture(idx, cv2.CAP_V4L2)
                if not cap.isOpened(): continue
                ok, fr = cap.read()
                if ok and fr is not None and len(fr.shape)==3 and fr.shape[2]==3:
                    if not np.array_equal(fr[:,:,0], fr[:,:,1]):
                        while self._on:
                            ok, bgr = cap.read()
                            if ok: self._proc(cv2.flip(bgr, 1))
                            else: time.sleep(0.01)
                        cap.release(); return
                cap.release()
            except: pass

    def stop(self): self._on = False

    def _proc(self, bgr):
        self._lazy_load_model()
        mv = 0.0
        if self._mdl:
            try:
                r = self._mdl(bgr, verbose=False)
                kps = self._kps(r)
                if kps is not None:
                    for i in range(len(kps)):
                        x, y = int(kps[i][0]), int(kps[i][1])
                        if len(kps[i])>=3 and kps[i][2]>0.3: cv2.circle(bgr,(x,y),5,(0,255,0),-1)
                    if self._prev is not None:
                        for j in range(min(len(self._prev),len(kps))):
                            if len(kps[j])>=3 and kps[j][2]<0.3: continue
                            mv=max(mv,float(np.sqrt((kps[j][0]-self._prev[j][0])**2+(kps[j][1]-self._prev[j][1])**2)))
                    self._prev = kps
            except: pass
        if self._rec: self._buf.append(bgr.copy())
        g = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        h,w = g.shape
        self.frame.emit(QImage(g.data,w,h,w,QImage.Format.Format_Grayscale8).copy())
        if mv > 0: self.motion.emit(mv)

    @staticmethod
    def _kps(r):
        """Pick the person closest to camera centre (largest + most centred bbox)."""
        if not r or not r[0].keypoints or r[0].keypoints.data is None:
            return None
        kps_data = r[0].keypoints.data.cpu().numpy()
        if kps_data.shape[0] == 0:
            return None
        if kps_data.shape[0] == 1:
            return kps_data[0]

        # Multiple people — pick by largest bbox area × closeness to centre
        boxes = r[0].boxes
        if boxes is None or boxes.xyxy is None:
            return kps_data[0]
        xyxy = boxes.xyxy.cpu().numpy()
        img_w = 640  # approximate
        best_idx = 0
        best_score = -1
        for i in range(len(xyxy)):
            x1, y1, x2, y2 = xyxy[i]
            area = (x2 - x1) * (y2 - y1)
            cx = (x1 + x2) / 2
            # Penalise distance from horizontal centre
            centre_dist = abs(cx - img_w / 2) / (img_w / 2)
            score = area * (1.0 - 0.5 * centre_dist)
            if score > best_score:
                best_score = score
                best_idx = i
        return kps_data[best_idx] if best_idx < kps_data.shape[0] else kps_data[0]


# ── Page ─────────────────────────────────────────────────────────────────────

class ReactionTestPage(QWidget):
    def __init__(self, router: PageRouter, bridge: Optional[GuiBridge]=None, parent=None):
        super().__init__(parent)
        self._router = router; self._bridge = bridge
        self._trial = 0; self._times: List[float] = []
        self._stim_on = False; self._stim_t = 0.0
        self._cam_t: Optional[QThread] = None; self._cam_w: Optional[_Cam] = None
        self._delay = QTimer(self); self._delay.setSingleShot(True); self._delay.timeout.connect(self._on_delay)
        self._countdown_n = 0; self._state = "idle"
        self._replay_timer = QTimer(self); self._replay_timer.setInterval(120); self._replay_timer.timeout.connect(self._rtick)
        self._rframes: list = []; self._ri = 0
        self._build()
        if self._bridge: self._bridge.punch_confirmed.connect(self._on_punch)

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
        self._cam_area.setVisible(True); self._vid.setVisible(True); self._cards_w.setVisible(True)
        self._replay_timer.stop(); self._rbtn.setVisible(False)
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
        if self._stim_on and m > _MOTION_PX: self._record()

    def _on_punch(self, d):
        if self._stim_on: self._record()

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

    def _results(self):
        self._state = "done"
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
        bgr = self._rframes[self._ri]; self._ri += 1
        g = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY); h,w = g.shape
        qi = QImage(g.data,w,h,w,QImage.Format.Format_Grayscale8).copy()
        pm = QPixmap.fromImage(qi.scaled(self._vid.size(),Qt.AspectRatioMode.KeepAspectRatio,Qt.TransformationMode.FastTransformation))
        self._vid.setPixmap(self._round_pixmap(pm))

    # ── Abort / Camera ───────────────────────────────────────────────────

    def _abort(self):
        self._delay.stop(); self._replay_timer.stop(); self._stop_cam(); self._router.back()

    def _start_cam(self):
        if self._cam_t: return
        self._cam_w = _Cam(); self._cam_t = QThread(); self._cam_w.moveToThread(self._cam_t)
        self._cam_w.frame.connect(self._on_frame); self._cam_w.motion.connect(self._on_motion)
        self._cam_t.started.connect(self._cam_w.run); self._cam_t.start()

    def _stop_cam(self):
        if self._cam_w: self._cam_w.stop()
        if self._cam_t: self._cam_t.quit(); self._cam_t.wait(2000); self._cam_t=None; self._cam_w=None

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

    # ── Lifecycle ────────────────────────────────────────────────────────

    def on_enter(self, **kw):
        self._state="idle"; self._times.clear(); self._trial=0
        self._btn.setVisible(True); self._res.setVisible(False)
        self._cam_area.setVisible(True); self._vid.setVisible(True)
        self._badge.setText(f"0/{_TRIALS}")
        self._badge.setStyleSheet(f"font-size:16px;font-weight:700;color:{Color.WARNING};background:{Color.SURFACE};border-radius:10px;padding:6px 16px;")
        for i,c in enumerate(self._cards): c.setText(f"{_ord(i+1)}\n--"); c.setStyleSheet(f"font-size:22px;font-weight:700;color:{Color.TEXT_DISABLED};background:{Color.SURFACE};border-radius:12px;")
        self._phase("Tap Start", fg=Color.TEXT_SECONDARY, sz=36)
        self._start_cam()

    def on_leave(self):
        self._delay.stop(); self._replay_timer.stop(); self._stop_cam()
