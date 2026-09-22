"""Spoken and audible feedback for the detected colour.

A colour-blind user gets three channels out of the instrument: the note (which
encodes the colour index as pitch), an earcon (which encodes the texture motif
as a rhythm), and -- if a text-to-speech engine happens to be installed -- the
colour described in words.

Speech is strictly optional and probed at start-up.  Nothing here imports a
synthesis library at module level, so the program runs identically whether or
not one is present; it just quietly stops talking.  That is deliberate: an
accessibility feature that turns into an import error is worse than no feature.
"""

from __future__ import annotations

import threading
import time

from .config import PATTERN_NAMES


class SpeechEngine:
    """Best-effort text to speech, using whatever is actually installed."""

    def __init__(self) -> None:
        self.available = False
        self.backend = "none"
        self.reason = ""
        self._engine = None
        self._lock = threading.Lock()
        self._probe()

    def _probe(self) -> None:
        if self._try_pyttsx3():
            return
        self.reason = "no text-to-speech engine found (install pyttsx3 to enable)"

    def _try_pyttsx3(self) -> bool:
        try:
            import pyttsx3
        except Exception:
            return False
        try:
            self._engine = pyttsx3.init()
            self._engine.setProperty("rate", 185)
            self.available = True
            self.backend = "pyttsx3"
            return True
        except Exception as error:
            self.reason = f"pyttsx3 is installed but would not start ({error})"
            return False

    def say(self, message: str) -> None:
        """Speak without blocking the caller."""
        if not self.available or not message:
            return

        def _worker() -> None:
            with self._lock:
                try:
                    self._engine.say(message)
                    self._engine.runAndWait()
                except Exception:
                    self.available = False

        threading.Thread(target=_worker, daemon=True, name="colorpiano-tts").start()

    def close(self) -> None:
        if self._engine is not None:
            try:
                self._engine.stop()
            except Exception:
                pass


class Announcer:
    """Decides what, if anything, to say about the current note.

    Rate limited to roughly one announcement per second: a continuous stream of
    speech while an object is moving is noise, not information.
    """

    def __init__(self, audio, enabled: bool = False,
                 min_interval: float = 1.1) -> None:
        self.audio = audio
        self.speech = SpeechEngine()
        self.enabled = bool(enabled)
        self.min_interval = float(min_interval)
        self._last_time = 0.0
        self._last_index = -1

    @property
    def speaking_available(self) -> bool:
        return self.speech.available

    def describe(self, index: int, note_name: str, color_name: str,
                 pattern: int) -> str:
        motif = PATTERN_NAMES[pattern % len(PATTERN_NAMES)]
        return f"{color_name}, note {note_name}, pattern {motif}"

    def announce(self, index: int, note_name: str, color_name: str,
                 pattern: int, force: bool = False) -> str:
        """Emit whatever feedback is enabled.  Returns the text, if any."""
        now = time.time()
        if not force:
            if not self.enabled:
                return ""
            if index == self._last_index:
                return ""
            if now - self._last_time < self.min_interval:
                return ""
        self._last_time = now
        self._last_index = index

        message = self.describe(index, note_name, color_name, pattern)
        if self.speech.available:
            self.speech.say(message)
        else:
            # No speech engine: fall back to a rhythm that identifies the
            # texture motif.  The pitch already came from the note itself.
            self.audio.schedule_earcon(pattern)
        return message

    def toggle(self) -> bool:
        self.enabled = not self.enabled
        return self.enabled

    def close(self) -> None:
        self.speech.close()
