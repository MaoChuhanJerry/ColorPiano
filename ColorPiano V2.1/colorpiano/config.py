"""Global configuration, paths and the runtime Settings object.

Everything that used to be a magic number scattered through the original
``color_piano.py`` lives here so it can be tuned (or overridden from the command
line) in one place.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

# --------------------------------------------------------------------------- #
# Paths
# --------------------------------------------------------------------------- #
PACKAGE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = PACKAGE_DIR.parent
ASSETS_DIR = PROJECT_ROOT / "assets"
NOTES_DIR = ASSETS_DIR / "notes"
DOCS_DIR = PROJECT_ROOT / "docs"

APP_NAME = "ColorPiano Pro"
APP_VERSION = "2.0.0"


def resolve_notes_dir() -> Path:
    """Return the directory that holds the note samples.

    Search order (first hit wins):

    1. ``$COLORPIANO_NOTES`` environment variable
    2. ``<project>/assets/notes``        <- the packaged layout
    3. ``<project>/assets``
    4. the current working directory     <- the *old* layout, kept so that an
       existing checkout with the wav files sitting next to the script keeps
       working.

    The original code blindly used ``glob.glob("*.wav")`` against the process
    CWD, which silently produced an empty (or wrong) note table whenever the
    program was started from another folder.
    """

    candidates = []
    env = os.environ.get("COLORPIANO_NOTES")
    if env:
        candidates.append(Path(env))
    candidates += [
        NOTES_DIR,
        ASSETS_DIR,
        Path.cwd(),
        PACKAGE_DIR,
        PROJECT_ROOT,
    ]
    for candidate in candidates:
        try:
            if candidate.is_dir() and any(candidate.glob("*.wav")):
                return candidate
        except OSError:
            continue
    # Nothing found: still hand back the canonical location so callers get a
    # useful path in their error message.
    return NOTES_DIR


# --------------------------------------------------------------------------- #
# Music
# --------------------------------------------------------------------------- #
#: Semitone offset of each natural note inside an octave.  The 52 samples are
#: the white keys of a standard 88-key piano (A0 .. C8).
NOTE_OFFSETS = {"C": 0, "D": 2, "E": 4, "F": 5, "G": 7, "A": 9, "B": 11}
NOTE_ORDER = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
SOLFEGE = {
    "C": "do", "D": "re", "E": "mi", "F": "fa",
    "G": "sol", "A": "la", "B": "si",
}
A4_MIDI = 69
A4_FREQ = 440.0

#: The 8 pattern motifs used for redundant (texture based) colour coding.
PATTERN_NAMES = [
    "solid", "horizontal", "vertical", "cross",
    "diagonal", "ring", "grid", "dots",
]


# --------------------------------------------------------------------------- #
# Colour handling
# --------------------------------------------------------------------------- #
#: Number of distinct colours == number of notes.
PALETTE_SIZE = 52

#: 8 hues x 7 bands -> 52 entries for the colour-blind safe palette.  See
#: ``colorblind.OKABE_ITO``.
CVD_SAFE_HUES = 8
CVD_SAFE_BANDS = 7

#: A sample with a CIELAB chroma below this is considered "grey" and will not
#: trigger a note.  The original code mapped black/white/grey to whichever hue
#: happened to be numerically closest, which made the instrument fire random
#: notes at a blank wall.
MIN_CHROMA = 12.0

#: Minimum lightness so that a nearly black object reads as "no colour".
MIN_LIGHTNESS = 12.0


# --------------------------------------------------------------------------- #
# Vision
# --------------------------------------------------------------------------- #
DEFAULT_WIDTH = 1280
DEFAULT_HEIGHT = 720
ROI_SIZE_MIN = 10
ROI_SIZE_MAX = 160


# --------------------------------------------------------------------------- #
# Settings
# --------------------------------------------------------------------------- #
@dataclass
class Settings:
    """Runtime tunables.  Every field is reachable from the command line."""

    # --- input ---------------------------------------------------------- #
    camera_index: int = 0
    source: str = "camera"          # camera | video | image | synthetic
    source_path: str = ""
    mirror: bool = True             # selfie view, much easier to aim
    width: int = DEFAULT_WIDTH
    height: int = DEFAULT_HEIGHT

    # --- colour detection ----------------------------------------------- #
    roi_size: int = 40              # half-width of the sampling square, px
    palette: str = "hue"            # hue | cvd_safe
    white_balance: bool = True      # grey-world correction, kills colour cast
    smoothing: int = 8              # samples kept in the colour smoother
    stabilizer: int = 4             # frames a new note must win before switching
    min_chroma: float = MIN_CHROMA

    # --- playing -------------------------------------------------------- #
    volume: float = 0.8
    bank: str = "piano"             # piano | synth | pure
    note_interval: float = 0.35     # seconds between note onsets (debounce)
    sustain: bool = False           # retrigger the same note repeatedly
    range_mode: str = "white"       # white | chromatic
    range_start: int = 60           # MIDI note the chromatic range starts on
    glide: bool = False             # short arpeggio when jumping far
    velocity_from_brightness: bool = True
    polyphony: int = 8

    # --- accessibility -------------------------------------------------- #
    cvd_mode: str = "none"          # none|protanopia|deuteranopia|tritanopia|achromatopsia
    simulate_cvd: bool = False      # apply the simulation to the preview
    daltonize: bool = False         # shift confusable colours apart
    daltonize_strength: float = 0.8
    #: Resolution the CVD filters run at, relative to the frame.  They are
    #: per-pixel colour transforms, so they cost the same whether or not the
    #: pixels matter; at 0.5 they are four times cheaper and the preview is a
    #: little softer.  1.0 for full detail, 0.25 if the machine is struggling.
    filter_scale: float = 0.5
    patterns: bool = True           # redundant texture coding
    show_names: bool = True         # text colour name (CVD-safe label)
    show_keyboard: bool = True      # where am I on the 88-key strip
    show_help: bool = True
    large_text: bool = False
    high_contrast: bool = False
    announce: bool = False          # spoken / earcon feedback on note change
    confusion_warning: bool = True

    # --- gestures ------------------------------------------------------- #
    gestures: bool = True
    gesture_confidence: float = 0.6

    # --- misc ----------------------------------------------------------- #
    show_fps: bool = True
    headless: bool = False
    record: str = ""                # write the annotated preview to this file
    duration: float = 0.0           # auto-quit after N seconds (0 = never)

    def to_lines(self) -> list[str]:
        """Human readable dump used by ``--print-config`` and the log header."""
        width = max(len(k) for k in self.__dataclass_fields__)
        return [f"{k.ljust(width)} = {getattr(self, k)!r}"
                for k in self.__dataclass_fields__]

    def apply_overrides(self, **values) -> "Settings":
        """Set only the fields that were actually supplied (``None`` means "unset").

        Used by the CLI, so that a flag left at its default never overwrites a
        value that came from somewhere else.
        """
        for key, value in values.items():
            if value is None:
                continue
            if key not in self.__dataclass_fields__:
                raise KeyError(f"unknown setting {key!r}")
            setattr(self, key, value)
        if self.bank not in ("piano", "synth", "pure"):
            raise ValueError(f"unknown bank {self.bank!r}")
        if self.range_mode not in ("white", "chromatic"):
            raise ValueError(f"unknown range mode {self.range_mode!r}")
        if self.source not in ("camera", "video", "image", "synthetic"):
            raise ValueError(f"unknown source {self.source!r}")
        return self
