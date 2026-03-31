"""String-based page router with history stack.

Pages are registered by name and swapped inside a QStackedWidget.
Supports optional ``on_enter(**kwargs)`` and ``on_leave()`` protocols
on page widgets.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Protocol, runtime_checkable

from PySide6.QtCore import QObject, Signal
from PySide6.QtWidgets import QStackedWidget, QWidget

logger = logging.getLogger(__name__)


@runtime_checkable
class Routable(Protocol):
    """Protocol that page widgets may implement for lifecycle hooks."""

    def on_enter(self, **kwargs: Any) -> None: ...
    def on_leave(self) -> None: ...


class PageRouter(QObject):
    """Manages page navigation inside a QStackedWidget.

    Parameters
    ----------
    stack : QStackedWidget
        The widget that displays the currently active page.
    """

    page_changed = Signal(str)  # emits new page name

    def __init__(self, stack: QStackedWidget, parent: Optional[QObject] = None) -> None:
        super().__init__(parent)
        self._stack: QStackedWidget = stack
        self._pages: Dict[str, QWidget] = {}
        self._history: List[str] = []
        self._page_kwargs: Dict[str, Dict[str, Any]] = {}
        self._current: Optional[str] = None

    # ── Registration ────────────────────────────────────────────────────

    def register(self, name: str, widget: QWidget) -> None:
        """Register a page widget under *name*."""
        if name in self._pages:
            logger.warning("Page '%s' already registered -- replacing", name)
            old_idx = self._stack.indexOf(self._pages[name])
            if old_idx >= 0:
                self._stack.removeWidget(self._pages[name])
        self._stack.addWidget(widget)
        self._pages[name] = widget
        logger.debug("Registered page: %s", name)

    # ── Navigation ──────────────────────────────────────────────────────

    def navigate(self, name: str, **kwargs: Any) -> None:
        """Navigate to the page registered as *name*.

        The current page receives ``on_leave()`` and the new page
        receives ``on_enter(**kwargs)`` if those methods exist.
        """
        if name not in self._pages:
            logger.error("Unknown page: '%s'", name)
            return
        if name == self._current:
            return

        # Leave current page
        if self._current is not None:
            self._history.append(self._current)
            current_widget = self._pages[self._current]
            if isinstance(current_widget, Routable):
                try:
                    current_widget.on_leave()
                except Exception:
                    logger.exception("Error in on_leave for '%s'", self._current)

        # Enter new page
        self._current = name
        self._page_kwargs[name] = kwargs
        new_widget = self._pages[name]
        self._stack.setCurrentWidget(new_widget)
        if isinstance(new_widget, Routable):
            try:
                new_widget.on_enter(**kwargs)
            except Exception:
                logger.exception("Error in on_enter for '%s'", name)

        self.page_changed.emit(name)
        logger.debug("Navigated to: %s", name)

    def replace(self, name: str, **kwargs: Any) -> None:
        """Navigate to *name* without pushing the current page to history."""
        if name not in self._pages:
            logger.error("Unknown page: '%s'", name)
            return

        # Leave current page (but don't push to history)
        if self._current is not None:
            current_widget = self._pages[self._current]
            if isinstance(current_widget, Routable):
                try:
                    current_widget.on_leave()
                except Exception:
                    logger.exception("Error in on_leave for '%s'", self._current)

        self._current = name
        new_widget = self._pages[name]
        self._stack.setCurrentWidget(new_widget)
        if isinstance(new_widget, Routable):
            try:
                new_widget.on_enter(**kwargs)
            except Exception:
                logger.exception("Error in on_enter for '%s'", name)

        self.page_changed.emit(name)
        logger.debug("Replaced to: %s", name)

    def back(self) -> bool:
        """Go back one step.  Returns False if history is empty."""
        if not self._history:
            logger.debug("History empty -- cannot go back")
            return False
        prev = self._history.pop()
        # We don't want navigate() to re-push to history, so call directly
        if self._current is not None:
            current_widget = self._pages[self._current]
            if isinstance(current_widget, Routable):
                try:
                    current_widget.on_leave()
                except Exception:
                    logger.exception("Error in on_leave for '%s'", self._current)

        self._current = prev
        widget = self._pages[prev]
        self._stack.setCurrentWidget(widget)
        saved_kwargs = self._page_kwargs.get(prev, {})
        if isinstance(widget, Routable):
            try:
                widget.on_enter(**saved_kwargs)
            except Exception:
                logger.exception("Error in on_enter for '%s'", prev)

        self.page_changed.emit(prev)
        logger.debug("Back to: %s", prev)
        return True

    # ── Queries ─────────────────────────────────────────────────────────

    @property
    def current_page(self) -> Optional[str]:
        return self._current

    @property
    def history(self) -> List[str]:
        return list(self._history)

    def has_page(self, name: str) -> bool:
        return name in self._pages
