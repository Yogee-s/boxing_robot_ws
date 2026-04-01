"""Self-Select sequence builder — build custom punch sequences.

Numpad with punch buttons (1-6) + defense moves. Up to 5 sequence slots.
Delete shifts sequences up. Backspace goes to previous slot when empty.
"""
from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, List

from PySide6.QtCore import Qt, QPoint
from PySide6.QtWidgets import (
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QWidget,
)
from PySide6.QtGui import QCursor

from boxbunny_gui.theme import Color, Icon, Size, back_link_style, PRIMARY_BTN
from boxbunny_gui.widgets import BigButton

if TYPE_CHECKING:
    from boxbunny_gui.nav.router import PageRouter

logger = logging.getLogger(__name__)

_PUNCHES = [
    ("1", "Jab", Color.JAB),
    ("2", "Cross", Color.CROSS),
    ("3", "L Hook", Color.L_HOOK),
    ("4", "R Hook", Color.R_HOOK),
    ("5", "L Upper", Color.L_UPPERCUT),
    ("6", "R Upper", Color.R_UPPERCUT),
]
_DEFENSE = [
    ("slip", "Slip-L"), ("slipr", "Slip-R"),
    ("block", "Block-L"), ("blockr", "Block-R"),
]
_MAX_SEQUENCES = 5
_MAX_COMBO_LEN = 10


class SelfSelectPage(QWidget):
    """Custom punch sequence builder."""

    def __init__(self, router: PageRouter, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._router = router
        self._active: int = 0
        # Store sequences as lists of token strings
        self._sequences: List[List[str]] = [[] for _ in range(_MAX_SEQUENCES)]
        self._slot_widgets: list[dict] = []
        self._build_ui()

    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(32, 10, 32, 22)
        root.setSpacing(0)

        # Top bar
        top = QHBoxLayout()
        btn_back = QPushButton(f"{Icon.BACK}  Back")
        btn_back.setStyleSheet(back_link_style())
        btn_back.setCursor(Qt.CursorShape.PointingHandCursor)
        btn_back.clicked.connect(lambda: self._router.back())
        top.addWidget(btn_back)
        title = QLabel("Build Custom Sequence")
        title.setStyleSheet(
            f"font-size: 20px; font-weight: 700; color: {Color.TEXT};"
        )
        top.addWidget(title)
        top.addStretch()
        root.addLayout(top)

        root.addStretch(1)

        # Two columns
        body = QHBoxLayout()
        body.setSpacing(24)

        # ── Left: slots ──────────────────────────────────────────────────
        left = QVBoxLayout()
        left.setSpacing(6)

        slots_lbl = QLabel("Sequences")
        slots_lbl.setStyleSheet(
            f"font-size: 13px; font-weight: 700; color: {Color.TEXT_SECONDARY};"
            " letter-spacing: 0.5px;"
        )
        left.addWidget(slots_lbl)

        self._drag_idx: int = -1
        self._drag_start_y: int = 0
        self._row_height = 46

        for i in range(_MAX_SEQUENCES):
            row_w = QWidget()
            row_w.setFixedHeight(self._row_height)
            row_lay = QHBoxLayout(row_w)
            row_lay.setContentsMargins(0, 0, 0, 0)
            row_lay.setSpacing(6)

            # Drag handle — 3 horizontal lines icon
            drag_lbl = QLabel("\u2261")
            drag_lbl.setFixedSize(28, 36)
            drag_lbl.setAlignment(Qt.AlignCenter)
            drag_lbl.setCursor(Qt.CursorShape.OpenHandCursor)
            drag_lbl.setStyleSheet(
                f"font-size: 20px; color: {Color.TEXT_DISABLED};"
                " background: transparent; border: none;"
            )
            drag_lbl.setObjectName(f"drag_{i}")
            row_lay.addWidget(drag_lbl)

            # Number button
            num_btn = QPushButton(str(i + 1))
            num_btn.setFixedSize(32, 32)
            num_btn.setCursor(Qt.CursorShape.PointingHandCursor)
            num_btn.clicked.connect(lambda _c=False, idx=i: self._select(idx))
            row_lay.addWidget(num_btn)

            # Sequence text
            seq_lbl = QLabel("(empty)")
            seq_lbl.setStyleSheet(
                f"font-size: 14px; color: {Color.TEXT_DISABLED};"
            )
            row_lay.addWidget(seq_lbl, stretch=1)

            # Delete button
            del_btn = QPushButton(Icon.CLOSE)
            del_btn.setFixedSize(28, 28)
            del_btn.setCursor(Qt.CursorShape.PointingHandCursor)
            del_btn.setStyleSheet(f"""
                QPushButton {{
                    font-size: 12px; background: transparent;
                    color: {Color.TEXT_DISABLED}; border: none;
                    border-radius: 14px;
                }}
                QPushButton:hover {{ color: {Color.DANGER}; background: {Color.SURFACE_HOVER}; }}
            """)
            del_btn.clicked.connect(lambda _c=False, idx=i: self._delete_slot(idx))
            row_lay.addWidget(del_btn)

            left.addWidget(row_w)
            self._slot_widgets.append({
                "widget": row_w, "num": num_btn, "label": seq_lbl,
                "delete": del_btn, "drag": drag_lbl,
            })

        # Controls
        ctrl = QHBoxLayout()
        ctrl.setSpacing(8)

        back_btn = QPushButton(f"{Icon.BACK}  Backspace")
        back_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        back_btn.setFixedHeight(44)
        back_btn.setStyleSheet(f"""
            QPushButton {{
                font-size: 13px; font-weight: 600;
                background-color: {Color.SURFACE}; color: {Color.TEXT_SECONDARY};
                border: 1px solid {Color.BORDER}; border-radius: {Size.RADIUS}px;
            }}
            QPushButton:hover {{ color: {Color.TEXT}; border-color: {Color.PRIMARY}; }}
        """)
        back_btn.clicked.connect(self._backspace)
        ctrl.addWidget(back_btn)

        next_btn = QPushButton(f"Next  {Icon.NEXT}")
        next_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        next_btn.setFixedHeight(44)
        next_btn.setStyleSheet(f"""
            QPushButton {{
                font-size: 13px; font-weight: 600;
                background-color: {Color.SURFACE}; color: {Color.TEXT_SECONDARY};
                border: 1px solid {Color.BORDER}; border-radius: {Size.RADIUS}px;
            }}
            QPushButton:hover {{ color: {Color.TEXT}; border-color: {Color.PRIMARY}; }}
        """)
        next_btn.clicked.connect(self._next_slot)
        ctrl.addWidget(next_btn)

        left.addLayout(ctrl)
        body.addLayout(left, stretch=1)

        # ── Right: numpad ────────────────────────────────────────────────
        right = QVBoxLayout()
        right.setSpacing(8)

        punches_lbl = QLabel("Punches")
        punches_lbl.setStyleSheet(
            f"font-size: 13px; font-weight: 700; color: {Color.TEXT_SECONDARY};"
            " letter-spacing: 0.5px;"
        )
        right.addWidget(punches_lbl)

        grid = QGridLayout()
        grid.setSpacing(8)
        for i, (code, name, color) in enumerate(_PUNCHES):
            btn = QPushButton(f"{code}\n({name})")
            btn.setCursor(Qt.CursorShape.PointingHandCursor)
            btn.setFixedHeight(80)
            btn.setStyleSheet(f"""
                QPushButton {{
                    background-color: {Color.SURFACE}; color: {Color.TEXT};
                    border: 1px solid {Color.BORDER};
                    border-bottom: 3px solid {color};
                    border-radius: {Size.RADIUS}px;
                    font-size: 17px; font-weight: 600; padding: 8px 6px 6px 6px;
                }}
                QPushButton:hover {{
                    background-color: {Color.SURFACE_HOVER};
                    border-color: {color}; border-bottom: 3px solid {color};
                }}
                QPushButton:pressed {{
                    background-color: {color}; color: #FFFFFF;
                }}
            """)
            btn.clicked.connect(lambda _c=False, c=code: self._add_token(c))
            grid.addWidget(btn, i // 3, i % 3)
        right.addLayout(grid)

        # Defense
        def_lbl = QLabel("Defense")
        def_lbl.setStyleSheet(
            f"font-size: 13px; font-weight: 700; color: {Color.TEXT_SECONDARY};"
            " letter-spacing: 0.5px;"
        )
        right.addWidget(def_lbl)

        def_row = QHBoxLayout()
        def_row.setSpacing(8)
        for code, name in _DEFENSE:
            btn = QPushButton(name)
            btn.setCursor(Qt.CursorShape.PointingHandCursor)
            btn.setFixedHeight(56)
            btn.setStyleSheet(f"""
                QPushButton {{
                    background-color: {Color.SURFACE}; color: {Color.TEXT_SECONDARY};
                    border: 1px solid {Color.BORDER}; border-radius: {Size.RADIUS}px;
                    font-size: 16px; font-weight: 600;
                }}
                QPushButton:hover {{
                    background-color: {Color.SURFACE_HOVER};
                    border-color: {Color.TEXT_SECONDARY}; color: {Color.TEXT};
                }}
            """)
            btn.clicked.connect(lambda _c=False, c=code: self._add_token(c))
            def_row.addWidget(btn)
        right.addLayout(def_row)

        body.addLayout(right, stretch=1)
        root.addLayout(body)

        root.addStretch(1)

        # Continue
        self._btn_continue = BigButton(
            f"{Icon.PLAY}  Continue to Config", stylesheet=PRIMARY_BTN
        )
        self._btn_continue.setFixedHeight(70)
        self._btn_continue.clicked.connect(self._on_continue)
        root.addWidget(self._btn_continue)

    # ── Slot management ──────────────────────────────────────────────────

    def _select(self, idx: int) -> None:
        self._active = idx
        self._refresh_all()

    def _next_slot(self) -> None:
        self._active = (self._active + 1) % _MAX_SEQUENCES
        self._refresh_all()

    def _add_token(self, token: str) -> None:
        seq = self._sequences[self._active]
        if len(seq) < _MAX_COMBO_LEN:
            seq.append(token)
            self._refresh_all()

    def _backspace(self) -> None:
        seq = self._sequences[self._active]
        if seq:
            seq.pop()
        elif self._active > 0:
            # Empty slot — go to previous
            self._active -= 1
            if self._sequences[self._active]:
                self._sequences[self._active].pop()
        self._refresh_all()

    # ── Drag reordering via mouse events on the whole page ──────────────

    def mousePressEvent(self, event) -> None:
        # Check if press is on a drag handle
        pos = event.position().toPoint()
        for i, sw in enumerate(self._slot_widgets):
            drag = sw["drag"]
            # Map position to the drag label's coordinate space
            local = drag.mapFrom(self, pos)
            if drag.rect().contains(local):
                self._drag_idx = i
                self._drag_start_y = pos.y()
                self._select(i)
                self.setCursor(Qt.CursorShape.ClosedHandCursor)
                return
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event) -> None:
        if self._drag_idx < 0:
            return
        pos = event.position().toPoint()
        dy = pos.y() - self._drag_start_y
        # Swap when dragged more than half a row height
        if dy > self._row_height * 0.6 and self._drag_idx < _MAX_SEQUENCES - 1:
            j = self._drag_idx
            self._sequences[j], self._sequences[j + 1] = (
                self._sequences[j + 1], self._sequences[j]
            )
            self._drag_idx = j + 1
            self._active = j + 1
            self._drag_start_y = pos.y()
            self._refresh_all()
        elif dy < -self._row_height * 0.6 and self._drag_idx > 0:
            j = self._drag_idx
            self._sequences[j], self._sequences[j - 1] = (
                self._sequences[j - 1], self._sequences[j]
            )
            self._drag_idx = j - 1
            self._active = j - 1
            self._drag_start_y = pos.y()
            self._refresh_all()

    def mouseReleaseEvent(self, event) -> None:
        if self._drag_idx >= 0:
            self._drag_idx = -1
            self.setCursor(Qt.CursorShape.ArrowCursor)
        super().mouseReleaseEvent(event)

    def _delete_slot(self, idx: int) -> None:
        """Delete slot and shift everything below up."""
        del self._sequences[idx]
        self._sequences.append([])  # Add empty at end
        if self._active >= idx and self._active > 0:
            self._active = max(0, self._active - 1)
        self._refresh_all()

    def _refresh_all(self) -> None:
        """Update all slot visuals from the data."""
        for i in range(_MAX_SEQUENCES):
            sw = self._slot_widgets[i]
            seq = self._sequences[i]
            is_active = (i == self._active)
            has_data = len(seq) > 0

            # Number button
            if is_active:
                sw["num"].setStyleSheet(f"""
                    QPushButton {{
                        font-size: 14px; font-weight: 700; color: #FFFFFF;
                        background-color: {Color.PRIMARY}; border: none;
                        border-radius: 16px;
                    }}
                """)
            else:
                sw["num"].setStyleSheet(f"""
                    QPushButton {{
                        font-size: 14px; font-weight: 700; color: {Color.TEXT_DISABLED};
                        background-color: transparent;
                        border: 1px solid {Color.BORDER}; border-radius: 16px;
                    }}
                    QPushButton:hover {{ color: {Color.PRIMARY}; border-color: {Color.PRIMARY}; }}
                """)

            # Sequence label
            if has_data:
                sw["label"].setText("-".join(seq))
                color = Color.TEXT if is_active else Color.TEXT_SECONDARY
                sw["label"].setStyleSheet(
                    f"font-size: 14px; font-weight: 600; color: {color};"
                )
            else:
                sw["label"].setText("(empty)")
                color = Color.PRIMARY_LIGHT if is_active else Color.TEXT_DISABLED
                sw["label"].setStyleSheet(
                    f"font-size: 14px; color: {color};"
                )

            # Row background — active gets warm tint, inactive plain
            if is_active:
                sw["widget"].setStyleSheet(f"""
                    QWidget {{
                        background-color: #1A1510;
                        border: none;
                        border-radius: {Size.RADIUS_SM}px;
                    }}
                """)
            else:
                sw["widget"].setStyleSheet(f"""
                    QWidget {{
                        background-color: transparent;
                        border: none;
                        border-radius: {Size.RADIUS_SM}px;
                    }}
                """)

    def _on_continue(self) -> None:
        sequences = ["-".join(s) for s in self._sequences if s]
        if not sequences:
            return
        combo_data = {
            "name": "Custom Sequence",
            "seq": sequences[0],
            "id": None,
        }
        self._router.navigate(
            "training_config",
            combo=combo_data,
            difficulty="Self-Select",
            curriculum=None,
        )

    def on_enter(self, **kwargs: Any) -> None:
        # Only reset if explicitly told to (fresh entry from combo select)
        # Coming back from config via router.back() preserves sequences
        if kwargs.get("reset", False) or not any(self._sequences):
            self._active = 0
            self._sequences = [[] for _ in range(_MAX_SEQUENCES)]
        self._refresh_all()
        logger.debug("SelfSelectPage entered")

    def on_leave(self) -> None:
        pass
