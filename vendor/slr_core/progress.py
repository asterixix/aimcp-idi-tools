"""Lightweight console progress bars without external dependencies."""
from __future__ import annotations

import time
import sys
from dataclasses import dataclass

_ACTIVE_PROGRESS = None


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def _bar(current: int, total: int, width: int = 20) -> str:
    if total <= 0:
        return "[" + ("-" * width) + "]"
    ratio = _clamp(current / total, 0.0, 1.0)
    filled = int(ratio * width)
    return "[" + ("#" * filled) + ("-" * (width - filled)) + "]"


def _format_eta(seconds: float | None) -> str:
    if seconds is None:
        return "--:--"
    seconds = max(0, int(seconds))
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    if h > 0:
        return f"{h:02d}:{m:02d}:{s:02d}"
    return f"{m:02d}:{s:02d}"


@dataclass
class ProgressState:
    total: int = 0
    current: int = 0
    label: str = ""
    started_at: float | None = None


class ProgressBars:
    """Render one or two progress bars in a single updating line."""

    def __init__(self, enabled: bool = True):
        self.enabled = enabled and sys.stdout.isatty()
        self.overall = ProgressState()
        self.inner = ProgressState()
        self._initialized = False
        self._line_count = 2
        self._sticky = True
        if self.enabled:
            self.activate()

    def activate(self):
        global _ACTIVE_PROGRESS
        _ACTIVE_PROGRESS = self
        if self.enabled and not self._initialized:
            # Reserve two lines at the bottom.
            print("")
            print("")
            self._initialized = True
            self._render()

    def deactivate(self):
        global _ACTIVE_PROGRESS
        if _ACTIVE_PROGRESS is self:
            _ACTIVE_PROGRESS = None

    def set_overall(self, *, current: int, total: int, label: str = ""):
        started = self.overall.started_at
        if started is None and total > 0 and current >= 0:
            started = time.time()
        self.overall = ProgressState(total=total, current=current, label=label, started_at=started)
        self._render()

    def set_inner(self, *, current: int, total: int, label: str = ""):
        started = self.inner.started_at
        if started is None and total > 0 and current >= 0:
            started = time.time()
        self.inner = ProgressState(total=total, current=current, label=label, started_at=started)
        self._render()

    def clear_inner(self):
        self.inner = ProgressState()
        self._render()

    def _state_to_line(self, prefix: str, state: ProgressState) -> str:
        eta = None
        if state.started_at and state.current > 0 and state.total > 0:
            elapsed = time.time() - state.started_at
            eta = elapsed * (state.total - state.current) / max(1, state.current)
        return (
            f"{prefix} {_bar(state.current, state.total)} "
            f"{state.current}/{state.total} ETA {_format_eta(eta)} {state.label}".rstrip()
        )

    def clear_for_log(self):
        if not self.enabled or not self._initialized or not self._sticky:
            return
        # Move cursor up to progress block and clear lines.
        sys.stdout.write(f"\x1b[{self._line_count}F")
        for i in range(self._line_count):
            sys.stdout.write("\x1b[2K")
            if i < self._line_count - 1:
                sys.stdout.write("\n")
        sys.stdout.flush()

    def restore_after_log(self):
        if not self.enabled or not self._initialized:
            return
        self._render()

    def _render(self):
        if not self.enabled or not self._initialized:
            return
        overall_text = self._state_to_line("ALL", self.overall)
        if self.inner.total > 0:
            inner_text = self._state_to_line("DOC", self.inner)
        else:
            inner_text = "DOC [--------------------] 0/0 ETA --:--"

        # Move cursor to the beginning of reserved block, render 2 lines, keep cursor below block.
        sys.stdout.write(f"\x1b[{self._line_count}F")
        sys.stdout.write("\x1b[2K" + overall_text + "\n")
        sys.stdout.write("\x1b[2K" + inner_text + "\n")
        sys.stdout.flush()

    def finish(self, message: str | None = None):
        if not self.enabled:
            return
        self.clear_for_log()
        if message:
            print(message, flush=True)
        self.deactivate()


def get_active_progress():
    return _ACTIVE_PROGRESS

