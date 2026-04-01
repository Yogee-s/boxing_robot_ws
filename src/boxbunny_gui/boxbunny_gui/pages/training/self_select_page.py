"""Self-Select sequence builder — build custom punch sequences.

Numpad with punch buttons (1-6) + defense moves. Up to 5 sequence slots.
Matches the old GUI's SelfSelectSequencePage functionality.
"""
from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, List

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from boxbunny_gui.theme import Color, Icon, Size, font, back_link_style, PRIMARY_BTN
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
    ("slip", "Slip"),
    ("block", "Block"),
]

_MAX_SEQUENCES = 5
_MAX_COMBO_LEN = 10


class _SequenceSlot(QWidget):
    """Editable sequence slot showing the current combo."""

    def __init__(self, index: int, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._index = index
        self._tokens: List[str] = []
        self.setFixedHeight(40)
        self.setStyleSheet(f"""
            QWidget {{
                background-color: {Color.SURFACE};
                border: 1px solid {Color.BORDER};
                border-radius: {Size.RADIUS_SM}px;
            }}
        """)

        lay = QHBoxLayout(self)
        lay.setContentsMargins(10, 4, 10, 4)
        lay.setSpacing(6)

        self._num_lbl = QLabel(f"{index + 1})")
        self._num_lbl.setFixedWidth(24)
        self._num_lbl.setStyleSheet(
            f"font-size: 13px; font-weight: 700; color: {Color.TEXT_DISABLED};"
            " background: transparent; border: none;"
        )
        lay.addWidget(self._num_lbl)

        self._seq_lbl = QLabel("(empty)")
        self._seq_lbl.setStyleSheet(
            f"font-size: 13px; color: {Color.TEXT_DISABLED};"
            " background: transparent; border: none;"
        )
        lay.addWidget(self._seq_lbl, stretch=1)

        clear_btn = QPushButton(Icon.CLOSE)
        clear_btn.setFixedSize(24, 24)
        clear_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        clear_btn.setStyleSheet(f"""
            QPushButton {{
                font-size: 12px; background: transparent;
                color: {Color.TEXT_DISABLED}; border: none;
            }}
            QPushButton:hover {{ color: {Color.DANGER}; }}
        """)
        clear_btn.clicked.connect(self.clear)
        lay.addWidget(clear_btn)

    def add_token(self, token: str) -> None:
        if len(self._tokens) < _MAX_COMBO_LEN:
            self._tokens.append(token)
            self._refresh()

    def backspace(self) -> None:
        if self._tokens:
            self._tokens.pop()
            self._refresh()

    def clear(self) -> None:
        self._tokens.clear()
        self._refresh()

    def _refresh(self) -> None:
        if self._tokens:
            display = "-".join(self._tokens)
            self._seq_lbl.setText(display)
            self._seq_lbl.setStyleSheet(
                f"font-size: 13px; font-weight: 600; color: {Color.TEXT};"
                " background: transparent; border: none;"
            )
        else:
            self._seq_lbl.setText("(empty)")
            self._seq_lbl.setStyleSheet(
                f"font-size: 13px; color: {Color.TEXT_DISABLED};"
                " background: transparent; border: none;"
            )

    @property
    def sequence(self) -> str:
        return "-".join(self._tokens) if self._tokens else ""

    @property
    def is_empty(self) -> bool:
        return len(self._tokens) == 0

    def set_active(self, active: bool) -> None:
        if active:
            bg = "#1A1510"
            border = Color.PRIMARY
            num_color = Color.PRIMARY
            seq_color = Color.TEXT if self._tokens else Color.PRIMARY_LIGHT
        else:
            bg = Color.SURFACE
            border = Color.BORDER
            num_color = Color.TEXT_DISABLED
            seq_color = Color.TEXT if self._tokens else Color.TEXT_DISABLED

        self._num_lbl.setStyleSheet(
            f"font-size: 13px; font-weight: 700; color: {num_color};"
            " background: transparent; border: none;"
        )
        self._seq_lbl.setStyleSheet(
            f"font-size: 13px; font-weight: 600; color: {seq_color};"
            " background: transparent; border: none;"
        )
        self.setStyleSheet(f"""
            QWidget {{
                background-color: {bg};
                border: 1px solid {border};
                border-left: 3px solid {border};
                border-radius: {Size.RADIUS_SM}px;
            }}
        """)


def _punch_btn(code: str, name: str, color: str) -> QPushButton:
    btn = QPushButton(f"{code}\n{name}")
    btn.setCursor(Qt.CursorShape.PointingHandCursor)
    btn.setFixedHeight(56)
    btn.setStyleSheet(f"""
        QPushButton {{
            background-color: {Color.SURFACE};
            color: {Color.TEXT}; border: 1px solid {Color.BORDER};
            border-bottom: 3px solid {color};
            border-radius: {Size.RADIUS}px;
            font-size: 13px; font-weight: 600; padding: 4px;
        }}
        QPushButton:hover {{
            background-color: {Color.SURFACE_HOVER};
            border-color: {color};
            border-bottom: 3px solid {color};
        }}
        QPushButton:pressed {{
            background-color: {color}; color: #FFFFFF;
        }}
    """)
    return btn


class SelfSelectPage(QWidget):
    """Custom punch sequence builder."""

    def __init__(self, router: PageRouter, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._router = router
        self._active_slot: int = 0
        self._slots: list[_SequenceSlot] = []
        self._build_ui()

    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(32, 10, 32, 10)
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

        # Two-column layout: slots left, numpad right
        body = QHBoxLayout()
        body.setSpacing(20)

        # ── Left: sequence slots ─────────────────────────────────────────
        left = QVBoxLayout()
        left.setSpacing(6)

        slots_lbl = QLabel("Sequences")
        slots_lbl.setStyleSheet(
            f"font-size: 13px; font-weight: 700; color: {Color.TEXT_SECONDARY};"
            " letter-spacing: 0.5px;"
        )
        left.addWidget(slots_lbl)

        for i in range(_MAX_SEQUENCES):
            slot = _SequenceSlot(i, self)
            # Select button per slot
            select_btn = QPushButton()
            select_btn.setFixedSize(24, 24)
            select_btn.setCursor(Qt.CursorShape.PointingHandCursor)
            select_btn.setStyleSheet(f"""
                QPushButton {{
                    background: transparent; border: none;
                    font-size: 11px; color: {Color.TEXT_DISABLED};
                }}
                QPushButton:hover {{ color: {Color.PRIMARY}; }}
            """)
            select_btn.setText(f"{i + 1}")
            select_btn.clicked.connect(
                lambda _c=False, idx=i: self._set_active_slot(idx)
            )
            # Insert select button into the slot's layout
            slot.layout().insertWidget(0, select_btn)
            left.addWidget(slot)
            self._slots.append(slot)

        # Control buttons row
        ctrl_row = QHBoxLayout()
        ctrl_row.setSpacing(8)

        backspace_btn = QPushButton(f"{Icon.BACK}  Backspace")
        backspace_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        backspace_btn.setFixedHeight(34)
        backspace_btn.setStyleSheet(f"""
            QPushButton {{
                font-size: 12px; font-weight: 600;
                background-color: {Color.SURFACE}; color: {Color.TEXT_SECONDARY};
                border: 1px solid {Color.BORDER}; border-radius: {Size.RADIUS}px;
            }}
            QPushButton:hover {{ color: {Color.TEXT}; border-color: {Color.PRIMARY}; }}
        """)
        backspace_btn.clicked.connect(self._backspace)
        ctrl_row.addWidget(backspace_btn)

        next_btn = QPushButton(f"Next Slot  {Icon.NEXT}")
        next_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        next_btn.setFixedHeight(34)
        next_btn.setStyleSheet(f"""
            QPushButton {{
                font-size: 12px; font-weight: 600;
                background-color: {Color.SURFACE}; color: {Color.TEXT_SECONDARY};
                border: 1px solid {Color.BORDER}; border-radius: {Size.RADIUS}px;
            }}
            QPushButton:hover {{ color: {Color.TEXT}; border-color: {Color.PRIMARY}; }}
        """)
        next_btn.clicked.connect(self._next_slot)
        ctrl_row.addWidget(next_btn)

        left.addLayout(ctrl_row)

        body.addLayout(left, stretch=1)

        # ── Right: punch numpad ──────────────────────────────────────────
        right = QVBoxLayout()
        right.setSpacing(6)

        punches_lbl = QLabel("Punches")
        punches_lbl.setStyleSheet(
            f"font-size: 13px; font-weight: 700; color: {Color.TEXT_SECONDARY};"
            " letter-spacing: 0.5px;"
        )
        right.addWidget(punches_lbl)

        # 2x3 grid for punches
        punch_grid = QGridLayout()
        punch_grid.setSpacing(8)
        for i, (code, name, color) in enumerate(_PUNCHES):
            btn = _punch_btn(code, name, color)
            btn.clicked.connect(
                lambda _c=False, c=code: self._add_token(c)
            )
            punch_grid.addWidget(btn, i // 3, i % 3)
        right.addLayout(punch_grid)

        right.addSpacing(6)

        # Defense buttons
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
            btn.setFixedHeight(40)
            btn.setStyleSheet(f"""
                QPushButton {{
                    background-color: {Color.SURFACE};
                    color: {Color.TEXT_SECONDARY};
                    border: 1px solid {Color.BORDER};
                    border-radius: {Size.RADIUS}px;
                    font-size: 13px; font-weight: 600;
                }}
                QPushButton:hover {{
                    background-color: {Color.SURFACE_HOVER};
                    border-color: {Color.TEXT_SECONDARY};
                    color: {Color.TEXT};
                }}
            """)
            btn.clicked.connect(
                lambda _c=False, c=code: self._add_token(c)
            )
            def_row.addWidget(btn)
        right.addLayout(def_row)

        right.addStretch()

        body.addLayout(right, stretch=1)

        root.addLayout(body)

        root.addStretch(1)

        # ── Continue button ──────────────────────────────────────────────
        self._btn_continue = BigButton(
            f"{Icon.PLAY}  Continue to Config", stylesheet=PRIMARY_BTN
        )
        self._btn_continue.setFixedHeight(60)
        self._btn_continue.clicked.connect(self._on_continue)
        root.addWidget(self._btn_continue)

    def _set_active_slot(self, idx: int) -> None:
        self._active_slot = idx
        for i, slot in enumerate(self._slots):
            slot.set_active(i == idx)

    def _add_token(self, token: str) -> None:
        self._slots[self._active_slot].add_token(token)

    def _backspace(self) -> None:
        self._slots[self._active_slot].backspace()

    def _next_slot(self) -> None:
        nxt = (self._active_slot + 1) % _MAX_SEQUENCES
        self._set_active_slot(nxt)

    def _on_continue(self) -> None:
        # Collect all non-empty sequences
        sequences = [s.sequence for s in self._slots if not s.is_empty]
        if not sequences:
            return

        # Use first sequence as the combo
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
        self._active_slot = 0
        for slot in self._slots:
            slot.clear()
        self._set_active_slot(0)
        logger.debug("SelfSelectPage entered")

    def on_leave(self) -> None:
        pass
