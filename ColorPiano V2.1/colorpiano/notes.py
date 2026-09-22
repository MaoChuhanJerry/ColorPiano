"""Note naming, sample discovery and the mapping from colour index to note.

A quick word about the sample set: the 52 ``.wav`` files shipped in
``assets/notes`` are the white keys of a standard 88-key piano, ``A0`` up to
``C8``.  That is not a coincidence -- 88-key pianos have exactly 52 white keys,
which is why the instrument has 52 notes and therefore 52 colours.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

from .config import (
    A4_FREQ,
    A4_MIDI,
    NOTE_OFFSETS,
    NOTE_ORDER,
    NOTES_DIR,
    SOLFEGE,
    resolve_notes_dir,
)

_NOTE_RE = re.compile(r"^([A-Ga-g])([#b]?)(-?\d+)$")

#: MIDI note numbers of the 52 white keys between A0 and C8.
WHITE_KEY_MIDI = [
    midi
    for midi in range(21, 109)          # A0 .. C8
    if NOTE_ORDER[midi % 12] in ("C", "D", "E", "F", "G", "A", "B")
]
assert len(WHITE_KEY_MIDI) == 52, len(WHITE_KEY_MIDI)


def midi_to_name(midi: int) -> str:
    """69 -> ``A4``."""
    return f"{NOTE_ORDER[midi % 12]}{midi // 12 - 1}"


def midi_to_freq(midi: int) -> float:
    """Equal temperament, A4 = 440 Hz."""
    return A4_FREQ * (2.0 ** ((midi - A4_MIDI) / 12.0))


def midi_to_solfege(midi: int) -> str:
    name = NOTE_ORDER[midi % 12]
    return SOLFEGE.get(name[0], name)


def parse_note_name(stem: str) -> int | None:
    """``"C#4"`` / ``"Db4"`` / ``"A0"`` -> MIDI number, or ``None``.

    The original implementation did ``name[0]`` and ``int(name[1:])`` and blew
    up with a ``ValueError`` on anything unexpected (a stray ``test.wav`` in the
    folder was enough to kill the program).  This version simply reports the
    files it does not understand.
    """
    match = _NOTE_RE.match(stem.strip())
    if not match:
        return None
    letter, accidental, octave = match.groups()
    semitone = NOTE_OFFSETS[letter.upper()] + (1 if accidental == "#" else -1 if accidental == "b" else 0)
    return (int(octave) + 1) * 12 + semitone


@dataclass(frozen=True)
class Note:
    """One playable note and how to produce its sound.

    ``source`` is the recording to read and ``shift`` how far to transpose it,
    so a note with no recording of its own is still playable -- it is borrowed
    from the nearest one that exists.  ``source is None`` means "synthesise
    it", which is the last resort when no samples are present at all.
    """

    midi: int
    source: Path | None = None      # recording to read
    shift: float = 0.0              # semitones to transpose `source` by
    index: int = 0                  # position in the table == colour index

    @property
    def name(self) -> str:
        return midi_to_name(self.midi)

    @property
    def octave(self) -> int:
        return self.midi // 12 - 1

    @property
    def letter(self) -> str:
        return NOTE_ORDER[self.midi % 12][0]

    @property
    def solfege(self) -> str:
        return midi_to_solfege(self.midi)

    @property
    def frequency(self) -> float:
        return midi_to_freq(self.midi)

    @property
    def is_black_key(self) -> bool:
        return NOTE_ORDER[self.midi % 12].endswith("#")

    @property
    def label(self) -> str:
        return self.name

    @property
    def transposed(self) -> bool:
        return self.shift != 0.0

    @property
    def synthesized(self) -> bool:
        return self.source is None


def _indexed(notes: list[Note]) -> list[Note]:
    """Re-stamp ``index`` so it matches position -- index is the colour index."""
    return [
        Note(midi=n.midi, source=n.source, shift=n.shift, index=i)
        for i, n in enumerate(notes)
    ]


def discover_samples(directory: Path | None = None) -> list[Note]:
    """Scan *directory* for ``<note>.wav`` files, sorted by pitch.

    Sorting is by MIDI number rather than by filename: the old code's
    ``note_order[note]``-based key duplicated the pitch table already in the
    file and raised ``KeyError`` for any file not named ``A-G`` + digits.
    """
    directory = Path(directory) if directory is not None else resolve_notes_dir()
    found: dict[int, Path] = {}
    if directory.is_dir():
        for path in sorted(directory.glob("*.wav")):
            midi = parse_note_name(path.stem)
            if midi is None:
                continue
            # First file wins, so a stray duplicate copy cannot shift the table.
            found.setdefault(midi, path)
    return _indexed([Note(midi=midi, source=path) for midi, path in sorted(found.items())])


def white_keys_from(start_midi: int, count: int) -> list[int]:
    """*count* white-key MIDI numbers ascending from *start_midi*.

    Raises ``ValueError`` rather than wrapping around if the request runs off
    the end of the MIDI range -- a palette entry that silently plays a note two
    octaves below where it should is worse than an error message.
    """
    out: list[int] = []
    midi = int(start_midi)
    while len(out) < count and midi <= 127:
        if NOTE_ORDER[midi % 12] in ("C", "D", "E", "F", "G", "A", "B"):
            out.append(midi)
        midi += 1
    if len(out) < count:
        raise ValueError(
            f"cannot fit {count} white keys above MIDI {start_midi}; "
            f"at most {len(out)} are available"
        )
    return out


def fill_gaps(notes: list[Note], count: int) -> list[Note]:
    """Guarantee *count* playable white-key notes, borrowing from the nearest
    recording.

    The sequence walked is the *white-key* sequence, not semitones: with the 52
    bundled samples that reproduces A0..C8 exactly, which is the instrument the
    project started as.  Any white key with no recording of its own is pitch
    shifted from the closest one that has -- which is why a folder holding 30
    WAV files still yields a complete instrument, where the original called
    ``sys.exit(1)``.
    """
    recorded = {n.midi: n.source for n in notes if n.source is not None}
    if not recorded:
        # Nothing on disk: fall back to synthesis.  Still a working instrument,
        # it just sounds like a test tone.
        return _indexed([Note(midi=midi, source=None)
                         for midi in white_keys_from(WHITE_KEY_MIDI[0], count)])

    # Start from the lowest recorded *white* key so the range lines up with the
    # recordings the user actually has.
    recorded_white = [m for m in sorted(recorded) if m in set(WHITE_KEY_MIDI)]
    start = recorded_white[0] if recorded_white else sorted(recorded)[0]

    out: list[Note] = []
    for midi in white_keys_from(start, count):
        source = recorded.get(midi)
        if source is not None:
            out.append(Note(midi=midi, source=source))
        else:
            nearest = min(recorded, key=lambda m: abs(m - midi))
            out.append(Note(midi=midi, source=recorded[nearest],
                            shift=float(midi - nearest)))
    return _indexed(out)


def chromatic_range(recordings: list[Note], start_midi: int, count: int) -> list[Note]:
    """*count* consecutive semitones starting at *start_midi*.

    The index -> pitch-order mapping is unchanged; the difference is that the 52
    slots now cover 52 semitones instead of 52 white keys, so the instrument
    plays in every key.  The number of colours stays at 52, because 52 is what
    the palette can keep mutually distinguishable.
    """
    recorded = {n.midi: n.source for n in recordings if n.source is not None}
    # Only clamp to the MIDI range -- not to the 88-key piano range, because a
    # 52-semitone span starting at C4 legitimately runs past C8, and silently
    # moving the user's chosen octave would be worse than a transposed top note.
    start_midi = int(max(0, min(start_midi, 127 - count)))
    out: list[Note] = []
    for midi in range(start_midi, start_midi + count):
        if midi in recorded:
            out.append(Note(midi=midi, source=recorded[midi]))
        elif recorded:
            nearest = min(recorded, key=lambda m: abs(m - midi))
            out.append(Note(midi=midi, source=recorded[nearest],
                            shift=float(midi - nearest)))
        else:
            out.append(Note(midi=midi, source=None))
    return _indexed(out)


def build_note_table(
    directory: Path | None = None,
    count: int = 52,
    range_mode: str = "white",
    range_start: int = 60,
) -> list[Note]:
    """Assemble the note table: ``count`` playable notes, in pitch order.

    * ``white``     -- the recorded samples from the lowest one upwards, with
      gaps filled by transposition.  With the bundled 52 white-key samples this
      reproduces the original A0..C8 instrument exactly.
    * ``chromatic`` -- *count* consecutive semitones from ``range_start``; every
      black key is transposed from a neighbour.
    """
    recordings = discover_samples(directory)
    if range_mode == "chromatic":
        return chromatic_range(recordings, range_start, count)
    return fill_gaps(recordings, count)


def describe_table(notes: list[Note], directory: Path | None = None) -> str:
    """One-line summary of what was loaded, for the start-up banner."""
    directory = Path(directory) if directory is not None else NOTES_DIR
    if not notes:
        return (f"no notes available (no samples in {directory} and none synthesised)")
    transposed = sum(1 for n in notes if n.transposed)
    synthesized = sum(1 for n in notes if n.synthesized)
    parts = [f"{len(notes)} notes {notes[0].name}..{notes[-1].name}"]
    if transposed:
        parts.append(f"{transposed} transposed from neighbours")
    if synthesized:
        parts.append(f"{synthesized} synthesised (no samples found)")
    return ", ".join(parts) + f"  [samples: {directory}]"
