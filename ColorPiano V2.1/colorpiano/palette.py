"""The 52 colours, the colour smoother and the colour namer.

Two palettes are offered:

``hue``
    The original: 52 fully saturated hues evenly spaced around the HSV wheel.
    Pretty, and perfect for a sighted user -- but neighbouring entries are only
    7 degrees apart, and a dichromat sees most of them as the same yellow-brown.

``cvd_safe``
    Built on the Okabe-Ito universal-design hues, then spread over the CIELAB
    lightness axis.  Lightness is the one dimension every dichromat still
    perceives, so the 7 lightness bands per hue family are what actually carry
    the information; the 8 hue families are what the 8 texture patterns are
    keyed to.  Verified numerically by ``tools/selftest.py``.

This module also owns two pieces of signal processing that the original code got
badly wrong:

``ColorSmoother``
    Averages colour in CIELAB, with the hue angle averaged *circularly*.  The
    original averaged the palette **index**, so a sample that flickered between
    note 50 and note 2 produced note 26 -- a note that was never on screen.

``NoteStabilizer``
    Hysteresis.  A note only becomes current once it has won N frames in a row,
    which is what stops the instrument stuttering between two adjacent notes
    when the object sits exactly between them.
"""

from __future__ import annotations

import colorsys
from collections import deque
from dataclasses import dataclass, field

import numpy as np

from . import config
from .colorspace import (
    bgr_to_lab,
    circular_distance_deg,
    circular_mean_deg,
    lab_chroma,
    lab_hue_deg,
)

# --------------------------------------------------------------------------- #
# Anchors
# --------------------------------------------------------------------------- #
#: Okabe & Ito (2008) "Colour Universal Design" hues -- the de-facto standard
#: for colour-blind safe categorical palettes -- plus one violet anchor added by
#: this project so that 52 divides evenly into 8 hue families.
OKABE_ITO = [
    ("orange", (230, 159, 0)),
    ("sky blue", (86, 180, 233)),
    ("bluish green", (0, 158, 115)),
    ("yellow", (240, 228, 66)),
    ("blue", (0, 114, 178)),
    ("vermillion", (213, 94, 0)),
    ("reddish purple", (204, 121, 167)),
    ("violet", (117, 76, 170)),
]

#: The colour-blind-safe palette, as RGB.
#:
#: These 52 values are the *output* of ``python -m tools.design_palette``, which
#: searches for 52 colours whose worst-case CIEDE2000 separation -- the minimum
#: over normal vision, protanopia, deuteranopia and tritanopia -- is as large as
#: possible.  The values are pasted in rather than recomputed at import so that
#: start-up stays instant; regenerate with ``--emit`` if the search changes.
#:
#: Measured: worst-case separation 4.87, versus 0.00 for the hue wheel and 0.38
#: for the hue wheel even with normal vision only.  Lightness spans L* 26..92.
CVD_SAFE_COLORS_RGB = [
    (0x01, 0x29, 0xFE), (0x9C, 0x41, 0xFE), (0x01, 0x2C, 0xAA), (0x7E, 0x36, 0xCC),
    (0xA5, 0x71, 0xFE), (0xB7, 0xA8, 0xFE), (0x8C, 0x50, 0xAF), (0xC0, 0x81, 0xE2),
    (0x3B, 0x35, 0x73), (0xAC, 0x67, 0xBF), (0xC5, 0xB9, 0xF4), (0x84, 0x47, 0x8B),
    (0xA6, 0x5A, 0xA3), (0x01, 0xF2, 0xFD), (0xDB, 0xA1, 0xD7), (0x01, 0xAC, 0xB0),
    (0x01, 0x91, 0x91), (0xFE, 0xB9, 0xE4), (0xD5, 0x93, 0xBC), (0xFE, 0x5F, 0xAD),
    (0x7B, 0x01, 0x4A), (0x9F, 0xF9, 0xE3), (0x77, 0xD1, 0xBB), (0xC8, 0x45, 0x7E),
    (0x6A, 0xB6, 0x96), (0xBA, 0x01, 0x61), (0xC5, 0x63, 0x7E), (0xFE, 0xA9, 0xB9),
    (0x01, 0x87, 0x5A), (0xB8, 0xF7, 0xCA), (0x6A, 0x28, 0x2E), (0x01, 0xDB, 0x95),
    (0x95, 0x47, 0x44), (0x2A, 0x5E, 0x29), (0xE8, 0x01, 0x5E), (0x35, 0x42, 0x0E),
    (0x3D, 0x85, 0x3D), (0x6F, 0xA9, 0x5F), (0xFD, 0xE7, 0xAB), (0x97, 0x2F, 0x24),
    (0xB0, 0xBA, 0x6F), (0xE1, 0x4F, 0x51), (0x82, 0x01, 0x08), (0x50, 0x75, 0x01),
    (0xFE, 0x4D, 0x49), (0xC4, 0xF9, 0x7C), (0x63, 0x8F, 0x01), (0xCD, 0xD5, 0x59),
    (0xFE, 0x20, 0x0D), (0xE1, 0x98, 0x01), (0xDC, 0xD2, 0x01), (0xD3, 0xF8, 0x01),
]

#: Texture motif per colour, assigned by ``tools.design_palette`` with a greedy
#: graph colouring over the "these two are hard to tell apart" relation -- so
#: whenever the colour is ambiguous, the pattern is not.  Only 2 of the 52
#: colours share a pattern with something they are confusable with.
CVD_SAFE_PATTERNS = [
    2, 0, 0, 1, 1, 2, 2, 0, 3, 3, 6, 4, 7, 3, 7, 2, 0, 5, 1, 6, 1, 0, 4, 5, 1, 0,
    2, 3, 3, 1, 0, 5, 6, 1, 1, 4, 4, 6, 2, 7, 3, 7, 2, 3, 0, 4, 2, 1, 0, 4, 6, 0,
]


# --------------------------------------------------------------------------- #
# Palette
# --------------------------------------------------------------------------- #
@dataclass
class Palette:
    """A set of colours, one per note."""

    name: str
    colors: list[tuple[int, int, int]]
    patterns: list[int]
    description: str = ""
    _lab: np.ndarray | None = field(default=None, repr=False)

    def __len__(self) -> int:
        return len(self.colors)

    @property
    def lab(self) -> np.ndarray:
        if self._lab is None:
            self._lab = bgr_to_lab(np.asarray(self.colors, dtype=np.float64))
        return self._lab

    @property
    def names(self) -> list[str]:
        return [color_name(c) for c in self.colors]

    def color(self, index: int) -> tuple[int, int, int]:
        return self.colors[index % len(self.colors)]

    def pattern(self, index: int) -> int:
        return self.patterns[index % len(self.patterns)]


def build_hue_wheel(count: int = 52) -> Palette:
    """52 fully saturated hues around the wheel -- the original palette."""
    colors, patterns = [], []
    for i in range(count):
        r, g, b = colorsys.hsv_to_rgb(i / count, 1.0, 1.0)
        colors.append((int(b * 255), int(g * 255), int(r * 255)))
        patterns.append(i % len(config.PATTERN_NAMES))
    return Palette(
        name="hue",
        colors=colors,
        patterns=patterns,
        description="52 evenly spaced, fully saturated hues (the original look)",
    )


def build_cvd_safe(count: int = 52) -> Palette:
    """52 colours chosen for maximum worst-case separation across viewer types.

    Built by farthest-point sampling in ``tools/design_palette.py`` rather than
    by hand.  A hand-built 8-hues x 7-bands grid looks reasonable and measures
    badly: within one lightness band the eight hues sit on a circle of constant
    ``L*``, which a dichromat projects onto a line, so opposite ends of the
    circle land on top of each other.  Measured, that construction left 51 of 52
    notes within a just-noticeable difference of a neighbour.
    """
    colors = [(b, g, r) for r, g, b in CVD_SAFE_COLORS_RGB]         # RGB -> BGR
    patterns = list(CVD_SAFE_PATTERNS)
    if count != len(colors):
        colors = colors[:count]
        patterns = patterns[:count]
    return Palette(
        name="cvd_safe",
        colors=colors,
        patterns=patterns,
        description="52 colours maximising worst-case separation across CVD types",
    )


def build_palette(name: str, count: int = 52) -> Palette:
    builders = {
        "hue": build_hue_wheel,
        "wheel": build_hue_wheel,
        "cvd_safe": build_cvd_safe,
        "cvd": build_cvd_safe,
        "safe": build_cvd_safe,
    }
    try:
        return builders[name.lower()](count)
    except KeyError:
        raise ValueError(
            f"unknown palette {name!r}; choose from {sorted(set(builders))}"
        ) from None


# --------------------------------------------------------------------------- #
# Smoothing
# --------------------------------------------------------------------------- #
@dataclass
class SmoothResult:
    lab: np.ndarray
    chroma: float
    lightness: float
    hue: float
    samples: int
    outliers: int

    @property
    def is_grey(self) -> bool:
        return self.chroma < config.MIN_CHROMA or self.lightness < config.MIN_LIGHTNESS


class ColorSmoother:
    """Rolling average of camera colour samples, computed in CIELAB.

    Hue is averaged as an *angle*.  Averaging it as a number is the classic
    wrap-around bug that the first version of this project shipped with: the
    mean of the indices 50 and 2 is 26, which is a red note when the object on
    screen is magenta.
    """

    def __init__(self, size: int = 8):
        self.size = max(1, int(size))
        self._samples: deque[np.ndarray] = deque(maxlen=self.size)

    def reset(self) -> None:
        self._samples.clear()

    @property
    def count(self) -> int:
        return len(self._samples)

    def push(self, lab) -> SmoothResult:
        self._samples.append(np.asarray(lab, dtype=np.float64))
        return self.value()

    def value(self) -> SmoothResult:
        if not self._samples:
            return SmoothResult(np.zeros(3), 0.0, 0.0, 0.0, 0, 0)
        if len(self._samples) == 1:
            lab = self._samples[0]
            return SmoothResult(lab, float(lab_chroma(lab)), float(lab[0]),
                                float(lab_hue_deg(lab)), 1, 0)

        stack = np.vstack(self._samples)
        hues = lab_hue_deg(stack)
        chromas = lab_chroma(stack)
        lightness = float(np.mean(stack[:, 0]))

        # Outlier rejection: a single sample whose hue is far from the circular
        # mean is a hand or a shadow crossing the sampling box.  Keeping it
        # would drag the reported note somewhere neither object is.
        mean_hue = circular_mean_deg(hues, chromas + 1e-6)
        keep = circular_distance_deg(hues, mean_hue) <= 60.0
        outliers = int(np.sum(~keep))
        if keep.sum() >= max(1, len(stack) // 2):
            hues, chromas = hues[keep], chromas[keep]
            lightness = float(np.mean(stack[keep, 0]))

        hue = circular_mean_deg(hues, chromas + 1e-6)
        chroma = float(np.mean(chromas))
        lab = np.array([
            lightness,
            chroma * np.cos(np.radians(hue)),
            chroma * np.sin(np.radians(hue)),
        ])
        return SmoothResult(lab, chroma, lightness, hue, len(stack), outliers)


class NoteStabilizer:
    """Hysteresis on the chosen palette index.

    ``needed`` consecutive frames must agree before the note changes; if the
    index reverts before that, the counter restarts.  Without this the
    instrument flickers between two notes when an object sits on a boundary,
    which is both unpleasant and unplayable.
    """

    def __init__(self, needed: int = 4):
        self.needed = max(1, int(needed))
        self.current: int | None = None
        self._candidate: int | None = None
        self._count = 0
        self.rejected = 0

    def reset(self) -> None:
        self.current = None
        self._candidate = None
        self._count = 0

    def update(self, index: int) -> tuple[int, bool]:
        """Feed a raw index; return ``(stable_index, changed)``."""
        if self.current is None:
            self.current = index
            self._candidate, self._count = index, 0
            return index, True

        if index == self.current:
            self._candidate, self._count = index, 0
            return self.current, False

        if index == self._candidate:
            self._count += 1
        else:
            self.rejected += 1
            self._candidate, self._count = index, 1

        if self._count >= self.needed:
            self.current = index
            self._candidate, self._count = index, 0
            return index, True
        return self.current, False


# --------------------------------------------------------------------------- #
# Naming colours in words
# --------------------------------------------------------------------------- #
#: Upper bounds of the CIELAB hue-angle bins, with the published hue angle of
#: each reference colour noted so the boundaries can be checked: red 40,
#: orange 55, yellow 103, green 136, cyan 196, blue 306, magenta 328.
_HUE_BINS = [
    (15.0, "rose"),
    (50.0, "red"),          # 40
    (72.0, "orange"),       # 55
    (122.0, "yellow"),      # 103
    (165.0, "green"),       # 136
    (215.0, "cyan"),        # 196
    (265.0, "azure"),
    (296.0, "violet"),
    (318.0, "blue"),        # 306
    (348.0, "magenta"),     # 328
    (361.0, "rose"),
]

_LIGHTNESS_WORDS = [
    (20.0, "very dark"),
    (32.0, "dark"),
    (45.0, "deep"),
    (60.0, "medium"),
    (75.0, "light"),
    (88.0, "pale"),
    (101.0, "very light"),
]


def color_name(bgr) -> str:
    """A short *word* for a colour.

    For a colour-blind user the words carry information the pixels do not, so
    this is deliberately verbose: "dark vivid orange", not "#A0522D".
    """
    lab = bgr_to_lab(np.asarray(bgr, dtype=np.float64))
    L = float(lab[0])
    C = float(lab_chroma(lab))
    h = float(lab_hue_deg(lab))

    if L < 10.0:
        return "black"
    if L > 95.0 and C < 8.0:
        return "white"
    if C < 8.0:
        return "light grey" if L > 60.0 else "dark grey"

    hue_word = next(word for limit, word in _HUE_BINS if h < limit)
    light_word = next(word for limit, word in _LIGHTNESS_WORDS if L < limit)
    if C < 25.0:
        chroma_word = "dull "
    elif C < 55.0:
        chroma_word = ""
    else:
        chroma_word = "vivid "
    return f"{light_word} {chroma_word}{hue_word}".replace("  ", " ")
