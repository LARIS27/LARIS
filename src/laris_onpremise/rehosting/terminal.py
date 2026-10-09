from __future__ import annotations

from contextlib import AbstractContextManager
import threading
import time
from typing import TextIO


_BANNER = r"""+--------------------------------------------------------------+
|  _        _    ____  ___ ____          .-----------------.   |
| | |      / \  |  _ \|_ _/ ___|         | runtime window  |   |
| | |     / _ \ | |_) || |\___ \         | [KV][SEQ][META] |   |
| | |___ / ___ \|  _ < | | ___) |        | 0x7f.. -> snap  |   |
| |_____/_/   \_\_| \_\___|____/         '------> rehost --'   |
|  Locking Onto Runtime Inference State                        |
|  Local LLMs Rehost Console for Restored-State Inquiry        |
+--------------------------------------------------------------+"""

_RESET = "\x1b[0m"
_COLORS = {
    "cyan": "\x1b[36m",
    "green": "\x1b[32m",
    "amber": "\x1b[33m",
    "red": "\x1b[31m",
}
_SPINNER_FRAMES = ("|", "/", "-", "\\")


def render_banner(*, color: bool) -> str:
    if not color:
        return _BANNER
    return "\x1b[36m" + _BANNER + "\x1b[0m"


class TerminalPresenter:
    """TTY-aware presentation layer with no authority over rehost decisions."""

    def __init__(
        self,
        stream: TextIO,
        *,
        color: bool,
        motion: bool,
        interval_seconds: float = 0.08,
    ) -> None:
        if interval_seconds <= 0:
            raise ValueError("terminal animation interval must be positive")
        self.stream = stream
        self.color = color
        self.motion = motion
        self.interval_seconds = interval_seconds
        self._write_lock = threading.Lock()

    def banner(self) -> None:
        self.stream.write(render_banner(color=self.color) + "\n")
        self.stream.flush()

    def startup_scan(self) -> None:
        if not self.motion:
            return
        label = "Scanning local rehost environment"
        for frame in _SPINNER_FRAMES:
            self._draw_activity(frame, label)
            time.sleep(min(self.interval_seconds, 0.15))
        self._clear_activity(label)

    def stage(
        self,
        name: str,
        status: str,
        detail: str | None = None,
        *,
        tone: str = "cyan",
    ) -> None:
        prefix = f"[{name}] {status}"
        if self.color:
            prefix = self._color(prefix, tone)
        suffix = f" - {detail}" if detail else ""
        self.stream.write(prefix + suffix + "\n")
        self.stream.flush()

    def activity(self, label: str) -> AbstractContextManager[None]:
        return _TerminalActivity(self, label)

    def _color(self, value: str, tone: str) -> str:
        try:
            prefix = _COLORS[tone]
        except KeyError as error:
            raise ValueError("terminal tone is invalid") from error
        return prefix + value + _RESET

    def _draw_activity(self, frame: str, label: str) -> None:
        marker = f"[{frame}]"
        if self.color:
            marker = self._color(marker, "cyan")
        with self._write_lock:
            self.stream.write("\r" + marker + " " + label)
            self.stream.flush()

    def _clear_activity(self, label: str) -> None:
        width = len(label) + 5
        with self._write_lock:
            self.stream.write("\r" + (" " * width) + "\r")
            self.stream.flush()


class _TerminalActivity(AbstractContextManager[None]):
    def __init__(self, presenter: TerminalPresenter, label: str) -> None:
        self.presenter = presenter
        self.label = label
        self._stop = threading.Event()
        self._worker: threading.Thread | None = None

    def __enter__(self) -> None:
        if not self.presenter.motion:
            self.presenter.stream.write(f"[....] {self.label}\n")
            self.presenter.stream.flush()
            return None
        self.presenter._draw_activity(_SPINNER_FRAMES[0], self.label)
        self._worker = threading.Thread(
            target=self._animate,
            name="laris-terminal-activity",
            daemon=True,
        )
        self._worker.start()
        return None

    def __exit__(self, exc_type: object, exc_value: object, traceback: object) -> None:
        if not self.presenter.motion:
            return None
        self._stop.set()
        if self._worker is not None:
            self._worker.join()
        self.presenter._clear_activity(self.label)
        return None

    def _animate(self) -> None:
        index = 1
        while not self._stop.wait(self.presenter.interval_seconds):
            self.presenter._draw_activity(
                _SPINNER_FRAMES[index % len(_SPINNER_FRAMES)],
                self.label,
            )
            index += 1


__all__ = ["TerminalPresenter", "render_banner"]
