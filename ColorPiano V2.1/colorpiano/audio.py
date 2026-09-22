"""Audio playback: banks, polyphony, and a scheduler that never blocks the video.

Three problems with the original playback code are fixed here.

1. **Blocking the preview.**  ``play_note_with_glide`` called ``time.sleep`` in
   the middle of the render loop, so every time the colour jumped, the video
   feed froze for 60 ms.  Timed events now go to a scheduler thread.

2. **One channel for everything.**  ``Sound.play()`` uses a single channel, so
   a new note cut off the previous one.  Here notes are placed on a pool of
   channels, which is what makes overlapping notes and chords possible.

3. **Volume applied to the wrong object.**  The original called
   ``set_volume`` on every ``Sound``, which meant the loudness set by a gesture
   also became the velocity of every future note.  Volume lives on the channel;
   the ``Sound`` objects stay untouched.
"""

from __future__ import annotations

import heapq
import os
import threading
import time
from dataclasses import dataclass

import numpy as np

from . import synth
from .notes import Note

#: The three timbres offered.
BANKS = ("piano", "synth", "pure")

BANK_DESCRIPTIONS = {
    "piano": "the bundled recordings",
    "synth": "additive tone with harmonics",
    "pure": "a single sine -- the pitch is unambiguous",
}


@dataclass
class AudioStatus:
    available: bool
    driver: str
    reason: str = ""
    banks_loaded: tuple[str, ...] = ()


class _Scheduler(threading.Thread):
    """Runs callables at a requested time, off the render thread."""

    def __init__(self) -> None:
        super().__init__(name="colorpiano-scheduler", daemon=True)
        self._queue: list[tuple[float, int, object]] = []
        self._wake = threading.Event()
        self._stopping = threading.Event()
        self._counter = 0
        self._lock = threading.Lock()

    def schedule(self, delay: float, function) -> None:
        with self._lock:
            self._counter += 1
            heapq.heappush(self._queue, (time.monotonic() + delay, self._counter, function))
        self._wake.set()

    def clear(self) -> None:
        with self._lock:
            self._queue.clear()

    def stop(self) -> None:
        self._stopping.set()
        self._wake.set()

    def run(self) -> None:                       # pragma: no cover - timing dependent
        while not self._stopping.is_set():
            with self._lock:
                if not self._queue:
                    entry = None
                else:
                    when, _, function = self._queue[0]
                    entry = (when, function)
                    if when > time.monotonic():
                        entry = None
                    else:
                        heapq.heappop(self._queue)
            if entry is None:
                with self._lock:
                    wait = 0.05 if not self._queue else max(
                        0.0, min(0.05, self._queue[0][0] - time.monotonic())
                    )
                self._wake.wait(wait)
                self._wake.clear()
                continue
            try:
                entry[1]()
            except Exception:
                # A failed note must never take the instrument down.
                pass


class AudioEngine:
    """Loads the banks and plays notes.  Safe to call from the render thread."""

    def __init__(self, notes: list[Note], volume: float = 0.8,
                 bank: str = "piano", polyphony: int = 8,
                 preload: tuple[str, ...] = ("piano",)) -> None:
        import pygame

        self.pygame = pygame
        self.notes = notes
        self.volume = float(np.clip(volume, 0.0, 1.0))
        self.status = AudioStatus(available=False, driver="none")
        self._banks: dict[str, list] = {}
        self._channel_cycle = 0
        self._sounds_cache: dict[tuple[str, float], object] = {}
        self._scheduler = _Scheduler()
        self._muted = False

        self._init_mixer()
        if self.status.available:
            pygame.mixer.set_num_channels(max(8, polyphony))
            self.channels = [pygame.mixer.Channel(i) for i in range(max(8, polyphony))]
            for name in preload:
                self.load_bank(name)
            self.bank = bank if bank in self._banks else next(iter(self._banks), bank)
            self._scheduler.start()

    # ------------------------------------------------------------------ setup
    def _init_mixer(self) -> None:
        """Initialise the mixer, falling back rather than crashing.

        A machine with no sound card is a legitimate place to run this (a
        headless self-test, a CI box).  The original would have raised on
        import; here a silent driver is used so every other code path is still
        exercised.
        """
        pygame = self.pygame
        try:
            pygame.mixer.init(frequency=synth.MIXER_RATE, size=-16,
                              channels=synth.MIXER_CHANNELS, buffer=1024)
            self.status = AudioStatus(True, "device")
            return
        except Exception as error:
            first_error = error
        try:
            pygame.mixer.quit()
        except Exception:
            pass
        os.environ["SDL_AUDIODRIVER"] = "dummy"
        try:
            pygame.mixer.init(frequency=synth.MIXER_RATE, size=-16,
                              channels=synth.MIXER_CHANNELS, buffer=1024)
            self.status = AudioStatus(True, "dummy (silent)",
                                      f"no audio device: {first_error}")
        except Exception as error:
            self.status = AudioStatus(False, "none", f"{first_error} / {error}")

    def load_bank(self, name: str) -> bool:
        """Build a bank lazily and remember it."""
        if not self.status.available or name in self._banks:
            return name in self._banks
        if name == "piano":
            self._banks[name] = self._build_piano()
        elif name == "synth":
            self._banks[name] = self._build_generated(synth.synth_note)
        elif name == "pure":
            self._banks[name] = self._build_generated(synth.pure_tone)
        else:
            raise ValueError(f"unknown bank {name!r}; choose from {BANKS}")
        self.status = AudioStatus(
            self.status.available, self.status.driver, self.status.reason,
            tuple(sorted(self._banks)),
        )
        return True

    def _samples_for(self, note: Note) -> np.ndarray:
        """Read (and transpose) one note's audio, memoised per (file, shift)."""
        if note.synthesized:
            return synth.synth_note(note.frequency)
        key = (str(note.source), round(float(note.shift), 3))
        cached = self._sounds_cache.get(key)
        if cached is None:
            cached = synth.load_note_at(note.source, note.shift)
            self._sounds_cache[key] = cached
        return cached

    def _build_piano(self) -> list:
        sounds = []
        for note in self.notes:
            try:
                samples = synth.fade_out(self._samples_for(note))
                sounds.append(self.pygame.mixer.Sound(
                    buffer=synth.to_stereo_buffer(samples)))
            except Exception:
                # One unreadable file must not cost the other 51 notes.
                sounds.append(None)
        return sounds

    def _build_generated(self, generator) -> list:
        sounds = []
        for note in self.notes:
            try:
                samples = synth.fade_out(generator(note.frequency))
                sounds.append(self.pygame.mixer.Sound(
                    buffer=synth.to_stereo_buffer(samples)))
            except Exception:
                sounds.append(None)
        return sounds

    # ---------------------------------------------------------------- playing
    @property
    def bank(self) -> str:
        return self._bank

    @bank.setter
    def bank(self, name: str) -> None:
        if name not in BANKS:
            name = BANKS[0]
        self.load_bank(name)
        self._bank = name

    @property
    def sounds(self) -> list:
        return self._banks.get(self._bank, [])

    @property
    def muted(self) -> bool:
        return self._muted

    @property
    def available(self) -> bool:
        return self.status.available

    def note_name(self, index: int) -> str:
        if 0 <= index < len(self.notes):
            return self.notes[index].name
        return "?"

    def set_volume(self, volume: float) -> None:
        """Set the master level.  Applied when the *next* note starts.

        Notes already sounding keep the level they were given, which is what
        makes per-note velocity possible at all.
        """
        self.volume = float(np.clip(volume, 0.0, 1.0))

    def set_muted(self, muted: bool) -> None:
        self._muted = bool(muted)
        if self._muted:
            self.stop_all()

    def _next_channel(self):
        channels = getattr(self, "channels", [])
        if not channels:
            return None
        # Prefer a channel that is free; otherwise take the least recently used.
        for offset in range(len(channels)):
            index = (self._channel_cycle + offset) % len(channels)
            if not channels[index].get_busy():
                self._channel_cycle = (index + 1) % len(channels)
                return channels[index]
        channel = channels[self._channel_cycle]
        self._channel_cycle = (self._channel_cycle + 1) % len(channels)
        return channel

    def play(self, index: int, velocity: float = 1.0,
             volume_scale: float = 1.0) -> bool:
        """Play one note immediately.  Returns whether anything sounded."""
        if not self.status.available or self._muted:
            return False
        sounds = self.sounds
        if not sounds:
            return False
        sound = sounds[index % len(sounds)]
        if sound is None:
            return False
        channel = self._next_channel()
        if channel is None:
            return False
        level = float(np.clip(self.volume * velocity * volume_scale, 0.0, 1.0))
        channel.set_volume(level)
        channel.play(sound)
        return True

    def stop_all(self) -> None:
        if hasattr(self, "channels"):
            for channel in self.channels:
                try:
                    channel.stop()
                except Exception:
                    pass
        self.clear_scheduled()

    def arpeggio(self, from_index: int, to_index: int, steps: int = 3,
                 spacing: float = 0.045, velocity: float = 0.8) -> None:
        """Play a few notes on the way between two far-apart indices.

        Non-blocking: the event times are queued and the render loop carries on,
        which is the whole point -- the original slept between notes and the
        video stuttered every time the colour changed.
        """
        if steps < 2 or from_index == to_index:
            self.play(to_index, velocity)
            return
        step = (to_index - from_index) / (steps + 1)
        for i in range(1, steps + 1):
            index = int(round(from_index + step * i))
            self._scheduler.schedule(
                spacing * i, lambda index=index: self.play(index, velocity)
            )

    def schedule_earcon(self, pattern: int, delay: float = 0.0,
                        volume_scale: float = 0.6) -> None:
        """Queue a texture-pattern earcon (see :mod:`colorpiano.synth`)."""
        if not self.status.available or self._muted:
            return

        def _play() -> None:
            try:
                samples = synth.fade_out(synth.earcon(int(pattern)), 0.02)
                sound = self.pygame.mixer.Sound(buffer=synth.to_stereo_buffer(samples))
                channel = self._next_channel()
                channel.set_volume(float(np.clip(self.volume * volume_scale, 0, 1)))
                channel.play(sound)
            except Exception:
                pass

        self._scheduler.schedule(delay, _play)

    def clear_scheduled(self) -> None:
        self._scheduler.clear()

    def shutdown(self) -> None:
        self._scheduler.stop()
        try:
            self.stop_all()
            self.pygame.mixer.quit()
        except Exception:
            pass


class NullAudioEngine:
    """Drop-in engine for tests: records what *would* have been played."""

    def __init__(self, notes: list[Note], volume: float = 0.8,
                 bank: str = "piano", polyphony: int = 8) -> None:
        self.notes = notes
        self.volume = volume
        self.bank = bank
        self.muted = False
        self.played: list[int] = []
        self.scheduled: list[int] = []
        self.status = AudioStatus(False, "null", "audio disabled")

    @property
    def available(self) -> bool:
        return False

    @property
    def sounds(self) -> list:
        return []

    def load_bank(self, name: str) -> bool:
        return True

    def set_volume(self, volume: float) -> None:
        self.volume = volume

    def set_muted(self, muted: bool) -> None:
        self.muted = muted

    def play(self, index: int, velocity: float = 1.0, volume_scale: float = 1.0) -> bool:
        if self.muted:
            return False
        self.played.append(index)
        return True

    def stop_all(self) -> None:
        pass

    def arpeggio(self, from_index: int, to_index: int, steps: int = 3,
                 spacing: float = 0.045, velocity: float = 0.8) -> None:
        self.scheduled.append(to_index)

    def schedule_earcon(self, pattern: int, delay: float = 0.0,
                        volume_scale: float = 0.6) -> None:
        self.scheduled.append(-1 - int(pattern))

    def clear_scheduled(self) -> None:
        self.scheduled.clear()

    def shutdown(self) -> None:
        pass
