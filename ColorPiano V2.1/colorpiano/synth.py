"""Runtime audio generation: WAV reading, pitch shifting, tone synthesis.

Why this exists
---------------
All 52 bundled samples are *white keys* (A0..C8).  That is fine for the
original instrument, which only had 52 colours, but it leaves the black keys
silent.  Rather than shipping 36 more megabytes of recordings, the missing
semitones are produced by resampling the nearest sample: raising a recording by
one semitone means reading it 5.95% faster, which is a linear interpolation and
about ten lines of numpy.

The same machinery generates the two synthetic banks, so the instrument still
makes a sound if the ``assets/notes`` folder is missing entirely -- the original
called ``sys.exit(1)`` in that case.
"""

from __future__ import annotations

import wave
from pathlib import Path

import numpy as np

#: Mixer output format.  48 kHz matches the bundled samples exactly, so loading
#: the piano bank needs no resampling at all.
MIXER_RATE = 48000
MIXER_CHANNELS = 2


# --------------------------------------------------------------------------- #
# WAV reading
# --------------------------------------------------------------------------- #
def read_wav_mono(path: str | Path) -> tuple[np.ndarray, int]:
    """Read a WAV file as mono float32 in [-1, 1].

    Only the stdlib ``wave`` module is used, so this works regardless of which
    optional audio libraries happen to be installed.
    """
    with wave.open(str(path), "rb") as handle:
        channels = handle.getnchannels()
        width = handle.getsampwidth()
        rate = handle.getframerate()
        frames = handle.readframes(handle.getnframes())

    if width == 2:
        data = np.frombuffer(frames, dtype="<i2").astype(np.float32) / 32768.0
    elif width == 1:
        data = (np.frombuffer(frames, dtype=np.uint8).astype(np.float32) - 128.0) / 128.0
    elif width == 4:
        data = np.frombuffer(frames, dtype="<i4").astype(np.float32) / 2147483648.0
    else:
        raise ValueError(f"{path}: unsupported sample width {width * 8} bit")

    if channels > 1:
        data = data.reshape(-1, channels).mean(axis=1)
    return data, rate


def resample(mono: np.ndarray, source_rate: int, target_rate: int) -> np.ndarray:
    """Linear-interpolation resample.  Good enough for a note onset."""
    if source_rate == target_rate or mono.size == 0:
        return mono
    count = int(round(mono.size * target_rate / source_rate))
    if count <= 1:
        return mono[:1]
    positions = np.linspace(0.0, mono.size - 1, count)
    return np.interp(positions, np.arange(mono.size), mono).astype(np.float32)


def pitch_shift(mono: np.ndarray, semitones: float) -> np.ndarray:
    """Change pitch by *semitones* without changing the sample rate.

    Reading faster makes the note both higher and shorter; for a percussive
    piano sample that is exactly what a real instrument does, so the shortening
    is not corrected.
    """
    if semitones == 0 or mono.size == 0:
        return mono
    ratio = 2.0 ** (semitones / 12.0)
    count = max(2, int(mono.size / ratio))
    positions = np.arange(count) * ratio
    return np.interp(positions, np.arange(mono.size), mono).astype(np.float32)


def load_note_at(path: str | Path, semitones: float = 0.0) -> np.ndarray:
    """Read *path*, resample to the mixer rate and shift it onto the pitch."""
    mono, rate = read_wav_mono(path)
    mono = resample(mono, rate, MIXER_RATE)
    return pitch_shift(mono, semitones)


# --------------------------------------------------------------------------- #
# Synthesis
# --------------------------------------------------------------------------- #
def envelope(count: int, attack: float = 0.006, decay: float = 1.6) -> np.ndarray:
    """Fast attack, exponential decay, short release -- a plucked shape."""
    t = np.arange(count) / MIXER_RATE
    attack_curve = 1.0 - np.exp(-t / max(attack, 1e-6))
    return (attack_curve * np.exp(-t / decay)).astype(np.float32)


def synth_note(
    frequency: float,
    duration: float = 2.0,
    harmonics: tuple[float, ...] = (1.0, 0.28, 0.14, 0.06),
    decay: float = 1.4,
) -> np.ndarray:
    """Additive tone with a percussive envelope."""
    count = max(2, int(MIXER_RATE * duration))
    t = np.arange(count) / MIXER_RATE
    wave_out = np.zeros(count, dtype=np.float32)
    for index, amplitude in enumerate(harmonics, start=1):
        if frequency * index >= MIXER_RATE / 2:      # never alias
            break
        wave_out += amplitude * np.sin(2.0 * np.pi * frequency * index * t)
    peak = float(np.max(np.abs(wave_out))) or 1.0
    return (wave_out / peak * envelope(count, decay=decay)).astype(np.float32)


def pure_tone(frequency: float, duration: float = 2.0, decay: float = 1.4) -> np.ndarray:
    """A single sine.

    Deliberately the third bank: with no harmonics the pitch is unambiguous,
    which is what a user relying on pitch to identify the colour needs.
    """
    return synth_note(frequency, duration, harmonics=(1.0,), decay=decay)


def earcon(pattern: int, frequency: float = 660.0, rate: float = 0.075) -> np.ndarray:
    """A short rhythmic motif identifying a texture pattern (0..7).

    Eight distinguishable motifs, so the pattern -- the one cue that survives
    every kind of colour blindness -- can be heard as well as seen.
    """
    count = max(1, pattern + 1)
    total = int(MIXER_RATE * (rate * (count + 1) + 0.10))
    out = np.zeros(total, dtype=np.float32)
    for step in range(count):
        start = int(MIXER_RATE * rate * step)
        length = min(int(MIXER_RATE * 0.05), total - start)
        if length <= 0:
            continue
        t = np.arange(length) / MIXER_RATE
        # Ascending for even patterns, descending for odd: a second axis to
        # tell motifs apart by ear alone.
        pitch = frequency * (1.0 + (step if pattern % 2 == 0 else count - 1 - step) * 0.12)
        out[start:start + length] += (
            np.sin(2.0 * np.pi * pitch * t) * np.exp(-t / 0.02) * 0.5
        ).astype(np.float32)
    return out


# --------------------------------------------------------------------------- #
# Mixer buffers
# --------------------------------------------------------------------------- #
def to_stereo_buffer(mono: np.ndarray, gain: float = 1.0) -> np.ndarray:
    """float mono -> interleaved int16 stereo, ready for ``mixer.Sound(buffer=)``."""
    mono = np.asarray(mono, dtype=np.float32)
    if gain != 1.0:
        mono = mono * float(gain)
    mono = np.clip(mono, -1.0, 1.0)
    pcm = (mono * 32767.0).astype(np.int16)
    stereo = np.repeat(pcm[:, None], MIXER_CHANNELS, axis=1)
    return np.ascontiguousarray(stereo)


def fade_out(mono: np.ndarray, seconds: float = 0.01) -> np.ndarray:
    """Taper the end so retriggered notes do not click."""
    count = min(int(MIXER_RATE * seconds), mono.size // 2)
    if count > 1:
        mono = mono.copy()
        mono[-count:] *= np.linspace(1.0, 0.0, count, dtype=np.float32)
    return mono
