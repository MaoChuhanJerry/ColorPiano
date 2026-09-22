"""Everything drawn on top of the video frame.

The overlay is where most of the accessibility work actually lands, because it
is the only place the program can say something the camera cannot show.  The
rules it follows:

* **Never rely on colour alone.**  Every colour on screen is accompanied by a
  texture motif, a note name, and a word.  Three redundant channels, so a user
  can ignore any two of them.
* **Text must survive any background.**  Every label is drawn twice, dark under
  light, so it stays readable over both a white wall and a black object.
* **Information has a fixed place.**  The note is always top-left, the status
  always top-right, the keyboard always along the bottom.  A user who cannot
  read the colours needs to find the text by position, not by scanning.
* **Say when you are unsure.**  When two colours are confusable for the active
  viewer type, the panel says so and points at the pattern instead.
"""

from __future__ import annotations

import cv2
import numpy as np

from . import config
from .config import NOTE_ORDER
from .notes import Note
from .palette import Palette

FONT = cv2.FONT_HERSHEY_SIMPLEX
MONO = cv2.FONT_HERSHEY_DUPLEX

WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
AMBER = (0, 200, 255)
GREEN = (120, 230, 120)
RED = (90, 90, 255)
GREY = (170, 170, 170)


# --------------------------------------------------------------------------- #
# Primitives
# --------------------------------------------------------------------------- #
def text(frame, message: str, origin, scale: float = 0.6, color=WHITE,
         thickness: int = 1, anchor: str = "left", outline: bool = True,
         font: int = FONT) -> tuple[int, int]:
    """Draw text with a contrasting outline.  Returns its width and height.

    Coordinates are coerced to ``int`` here rather than at every call site:
    OpenCV rejects float points with a bare "Bad argument" that gives no clue
    which drawing call was at fault.
    """
    (width, height), _ = cv2.getTextSize(message, font, scale, max(1, thickness))
    x, y = int(origin[0]), int(origin[1])
    if anchor == "center":
        x -= width // 2
    elif anchor == "right":
        x -= width
    if outline:
        cv2.putText(frame, message, (x, y), font, scale, BLACK,
                    thickness + 2, cv2.LINE_AA)
    cv2.putText(frame, message, (x, y), font, scale, color, thickness, cv2.LINE_AA)
    return width, height


def panel(frame, rect, fill=BLACK, border=WHITE, alpha: float = 0.62,
          thickness: int = 1) -> None:
    """A translucent filled rectangle with a border."""
    x1, y1, x2, y2 = [int(v) for v in rect]
    x1, x2 = min(x1, x2), max(x1, x2)
    y1, y2 = min(y1, y2), max(y1, y2)
    x1, y1 = max(0, x1), max(0, y1)
    x2 = min(frame.shape[1], x2)
    y2 = min(frame.shape[0], y2)
    if x2 <= x1 or y2 <= y1:
        return
    if alpha < 1.0:
        overlay = frame[y1:y2, x1:x2].copy()
        overlay[:] = fill
        cv2.addWeighted(overlay, 1.0 - alpha, frame[y1:y2, x1:x2], alpha, 0,
                        frame[y1:y2, x1:x2])
    else:
        cv2.rectangle(frame, (x1, y1), (x2, y2), fill, -1)
    if thickness:
        cv2.rectangle(frame, (x1, y1), (x2, y2), border, thickness)


def contrast_text_color(bgr) -> tuple[int, int, int]:
    """Black or white, whichever will be legible on top of *bgr*."""
    b, g, r = (float(v) for v in bgr[:3])
    luminance = 0.114 * b + 0.587 * g + 0.299 * r
    return BLACK if luminance > 130 else WHITE


# --------------------------------------------------------------------------- #
# Texture patterns -- the redundant channel
# --------------------------------------------------------------------------- #
def draw_pattern(frame, center, radius: int, color, pattern: int,
                 weight: float = 0.42, line: int = 1) -> None:
    """Fill a disc with *color*, then overlay the motif for *pattern*.

    The motif is drawn into a copy and blended, so the underlying colour stays
    visible: the point is redundancy, not replacement.
    """
    x, y = int(center[0]), int(center[1])
    radius = max(3, int(radius))
    bgr = tuple(int(v) for v in np.asarray(color).ravel()[:3])

    cv2.circle(frame, (x, y), radius, bgr, -1)
    motif = pattern % len(config.PATTERN_NAMES)
    if motif == 0:
        return

    x1, y1 = max(0, x - radius), max(0, y - radius)
    x2, y2 = min(frame.shape[1], x + radius + 1), min(frame.shape[0], y + radius + 1)
    if x2 <= x1 or y2 <= y1:
        return
    patch = frame[y1:y2, x1:x2].copy()
    local = (x - x1, y - y1)
    ink = contrast_text_color(bgr)
    step = max(3, radius // 3)

    if motif == 1:                                     # horizontal
        for offset in range(-radius, radius + 1, step):
            cv2.line(patch, (local[0] - radius, local[1] + offset),
                     (local[0] + radius, local[1] + offset), ink, line)
    elif motif == 2:                                   # vertical
        for offset in range(-radius, radius + 1, step):
            cv2.line(patch, (local[0] + offset, local[1] - radius),
                     (local[0] + offset, local[1] + radius), ink, line)
    elif motif == 3:                                   # cross
        cv2.line(patch, (local[0] - radius, local[1]), (local[0] + radius, local[1]), ink, line + 1)
        cv2.line(patch, (local[0], local[1] - radius), (local[0], local[1] + radius), ink, line + 1)
    elif motif == 4:                                   # diagonal
        cv2.line(patch, (local[0] - radius, local[1] + radius),
                 (local[0] + radius, local[1] - radius), ink, line + 1)
        cv2.line(patch, (local[0] - radius, local[1] - radius),
                 (local[0] + radius, local[1] + radius), ink, line + 1)
    elif motif == 5:                                   # ring
        cv2.circle(patch, local, max(2, radius // 2), ink, line + 1)
    elif motif == 6:                                   # grid
        for offset in range(-radius, radius + 1, step):
            cv2.line(patch, (local[0] - radius, local[1] + offset),
                     (local[0] + radius, local[1] + offset), ink, 1)
            cv2.line(patch, (local[0] + offset, local[1] - radius),
                     (local[0] + offset, local[1] + radius), ink, 1)
    elif motif == 7:                                   # dots
        spacing = max(4, radius // 2)
        for dx in range(-radius + spacing // 2, radius, spacing):
            for dy in range(-radius + spacing // 2, radius, spacing):
                if dx * dx + dy * dy <= radius * radius:
                    cv2.circle(patch, (local[0] + dx, local[1] + dy), 2, ink, -1)

    # Blend the motif in only inside the disc, so strokes cannot spill past the
    # edge.  Where no stroke was drawn the patch equals the frame, so the mix is
    # a no-op and the underlying colour is left alone.
    mask = np.zeros(patch.shape[:2], dtype=np.uint8)
    cv2.circle(mask, local, radius - 1, 255, -1)
    region = frame[y1:y2, x1:x2]
    mixed = cv2.addWeighted(patch, weight, region, 1.0 - weight, 0)
    frame[y1:y2, x1:x2] = np.where(mask[..., None] > 0, mixed, region)


def pattern_swatch(frame, origin, size: int, color, pattern: int,
                   label: str = "", scale: float = 0.42) -> None:
    """A small square chip used in legends."""
    x, y = int(origin[0]), int(origin[1])
    half = size // 2
    draw_pattern(frame, (x, y), half, color, pattern, weight=0.5)
    cv2.rectangle(frame, (x - half, y - half), (x + half, y + half), WHITE, 1)
    if label:
        text(frame, label, (x + half + 5, y + half // 2), scale, WHITE, 1)


# --------------------------------------------------------------------------- #
# The sampling box
# --------------------------------------------------------------------------- #
def draw_sampling_box(frame, center, half_size: int, palette: Palette | None,
                      index: int | None, patterns: bool, show_pattern: bool,
                      warning: bool = False) -> None:
    """The box the camera colour is read from, with its colour and motif."""
    cx, cy = int(center[0]), int(center[1])
    half = max(4, int(half_size))

    if palette is not None and index is not None:
        color = palette.color(index)
        motif = palette.pattern(index) if (patterns and show_pattern) else 0
        draw_pattern(frame, (cx, cy), half, color, motif, weight=0.35,
                     line=1 if half < 30 else 2)

    corner = (255, 255, 255) if not warning else RED
    # Corner brackets rather than a full rectangle: the object inside stays
    # visible, and the box does not read as part of the scene.
    arm = max(6, half // 3)
    for dx, dy in ((-1, -1), (1, -1), (-1, 1), (1, 1)):
        px, py = cx + dx * half, cy + dy * half
        cv2.line(frame, (px, py), (px - dx * arm, py), corner, 2)
        cv2.line(frame, (px, py), (px, py - dy * arm), corner, 2)


def draw_aiming_cross(frame, center, enabled: bool) -> None:
    if not enabled:
        return
    cx, cy = int(center[0]), int(center[1])
    cv2.line(frame, (cx - 6, cy), (cx + 6, cy), WHITE, 1)
    cv2.line(frame, (cx, cy - 6), (cx, cy + 6), WHITE, 1)


# --------------------------------------------------------------------------- #
# Panels
# --------------------------------------------------------------------------- #
def note_panel(frame, origin, palette: Palette, index: int, note: Note | None,
               color_name: str, pattern_name: str, patterns: bool,
               scale: float = 1.0, warning: str = "") -> int:
    """Top-left: the note, the colour word, the motif.  Returns its bottom edge."""
    x, y = int(origin[0]), int(origin[1])
    big = 1.15 * scale
    small = 0.58 * scale
    note_label = note.name if note is not None else "?"

    lines = [note_label]
    sub = f"{note.solfege}  octave {note.octave}" if note is not None else "no note"
    lines.append(sub)
    lines.append(color_name)
    if patterns:
        lines.append(f"pattern: {pattern_name}")

    widest = 0
    for i, line in enumerate(lines):
        (w, _), _ = cv2.getTextSize(line, FONT, big if i == 0 else small, 1)
        widest = max(widest, w)
    if warning:
        (w, _), _ = cv2.getTextSize(warning, FONT, small, 1)
        widest = max(widest, w)

    height = int((34 + 20 * (len(lines) - 1)) * scale) + 24
    panel(frame, (x, y, x + widest + 60, y + height), alpha=0.66)

    # Colour chip + texture, so the note panel is self-contained.  Kept fully
    # inside the panel by starting one chip-radius plus a margin from the edge.
    chip = int(20 * scale)
    chip_center = (x + chip + 10, int(y + chip + 8 * scale))
    draw_pattern(frame, chip_center, chip, palette.color(index),
                 palette.pattern(index) if patterns else 0, weight=0.45)
    cv2.circle(frame, chip_center, chip + 2, WHITE, 1)

    tx = int(chip_center[0] + chip + 12)
    text(frame, lines[0], (tx, int(y + 34 * scale)), big, WHITE, 1)
    for i, line in enumerate(lines[1:], start=1):
        text(frame, line, (tx, int(y + (34 + 22 * i) * scale)), small, GREY, 1)
    bottom = int(y + height)
    if warning:
        text(frame, warning, (tx, bottom - 8), small, AMBER, 1)
        bottom += int(18 * scale)
    return bottom


def status_panel(frame, origin, rows: list[tuple[str, str]], scale: float = 1.0,
                 width: int | None = None) -> None:
    """Top-right: the mode readouts."""
    x, y = int(origin[0]), int(origin[1])
    small = 0.5 * scale
    rows = list(rows)
    widest = max((cv2.getTextSize(f"{k}: {v}", FONT, small, 1)[0][0] for k, v in rows),
                 default=120)
    box_width = width or widest + 28
    height = int(18 * len(rows) * scale) + 16
    panel(frame, (x - box_width, y, x, y + height), alpha=0.66)
    for i, (key, value) in enumerate(rows):
        baseline = int(y + 20 * (i + 1) * scale)
        text(frame, f"{key}", (x - box_width + 10, baseline), small, GREY, 1)
        text(frame, value, (x - 12, baseline), small, WHITE, 1, anchor="right")


#: The keyboard is always drawn as a real 88-key piano, A0..C8, whether or not
#: every key is playable.  Drawing only the keys that happen to be in the note
#: table produces a picture with no black keys at all, which is not a keyboard
#: and makes the octave numbers meaningless.
PIANO_LOW = 21          # A0
PIANO_HIGH = 108        # C8


def keyboard_strip(frame, rect, notes: list[Note], palette: Palette, index: int,
                   patterns: bool, scale: float = 1.0, label: str = "") -> None:
    """A real 88-key keyboard along the bottom, with the current key lit.

    This is the orientation aid the original had no equivalent of: a user who
    cannot see where a colour sits between "low" and "high" can read it off the
    keyboard instead of guessing from the note name.  Keys that are not playable
    in the current range are drawn dimmed rather than omitted.
    """
    x1, y1, x2, y2 = [int(v) for v in rect]
    if x2 - x1 < 60 or y2 - y1 < 12:
        return

    panel(frame, (x1, y1, x2, y2), fill=(18, 18, 18), border=WHITE, alpha=1.0)

    inner_x1, inner_x2 = x1 + 2, x2 - 2
    inner_y1 = y1 + 4
    inner_y2 = y2 - 2

    white_midis = [m for m in range(PIANO_LOW, PIANO_HIGH + 1)
                   if not NOTE_ORDER[m % 12].endswith("#")]
    if not white_midis:
        return
    white_width = (inner_x2 - inner_x1) / len(white_midis)
    white_x = {midi: inner_x1 + order * white_width
               for order, midi in enumerate(white_midis)}
    black_height = inner_y1 + int((inner_y2 - inner_y1) * 0.62)

    # midi -> note index, for the keys this instrument can actually play.
    playable = {note.midi: i for i, note in enumerate(notes)}

    def key_rect(midi: int) -> tuple[int, int, int, int]:
        if NOTE_ORDER[midi % 12].endswith("#"):
            previous = max((m for m in white_midis if m < midi), default=None)
            centre = (white_x[previous] + white_width) if previous is not None else inner_x1
            half = white_width * 0.30
            return int(centre - half), inner_y1, int(centre + half), black_height
        left = white_x[midi]
        return int(left), inner_y1, int(left + white_width), inner_y2

    for midi in range(PIANO_LOW, PIANO_HIGH + 1):
        left, top, right, bottom = key_rect(midi)
        if right <= left:
            continue
        is_black = NOTE_ORDER[midi % 12].endswith("#")
        active = midi in playable
        if is_black:
            fill = (25, 25, 25) if active else (38, 38, 38)
            cv2.rectangle(frame, (left, top), (right, bottom), fill, -1)
            cv2.rectangle(frame, (left, top), (right, bottom),
                          (90, 90, 90) if active else (58, 58, 58), 1)
        else:
            fill = (232, 232, 232) if active else (150, 150, 150)
            cv2.rectangle(frame, (left, top), (right, bottom), fill, -1)
            cv2.rectangle(frame, (left, top), (right, bottom), (60, 60, 60), 1)

    # Light the current key.
    if 0 <= index < len(notes):
        midi = notes[index].midi
        left, top, right, bottom = key_rect(midi)
        colour = palette.color(index)
        motif = palette.pattern(index) if patterns else 0
        pointer = (int((left + right) / 2), int((top + bottom) / 2))
        draw_pattern(frame, pointer, max(4, (right - left) // 2), colour, motif,
                     weight=0.45)
        cv2.rectangle(frame, (left, top), (right, bottom), AMBER, 2)

    # Octave ticks on the C keys, so the keyboard can be counted.
    for midi in white_midis:
        if NOTE_ORDER[midi % 12] != "C":
            continue
        left, top, right, bottom = key_rect(midi)
        if right - left < 14:
            continue
        text(frame, str(midi // 12 - 1), (int((left + right) / 2), bottom - 3),
             0.34 * scale, (70, 70, 70), 1, anchor="center", outline=False)

    # The current note is already named in the note panel and marked on the
    # keyboard; a caption here would only collide with whatever sits above.
    if label:
        text(frame, label, (x1 + 8, y1 + int(14 * scale)), 0.5 * scale, AMBER, 1)


def brightness_bar(frame, origin, height: int, brightness: float,
                   scale: float = 1.0) -> int:
    """A vertical light/dark scale with a marker at the current brightness.

    Lightness is the one dimension every viewer keeps, so it gets its own
    always-visible gauge rather than being implied by the colour.  Labels go
    *below* the bar, not beside it, so the gauge can sit in a narrow left-hand
    column without running into whatever is next to it.  Returns the bottom edge.
    """
    x, y = int(origin[0]), int(origin[1])
    width = max(12, int(18 * scale))
    text_height = int(16 * scale)
    panel(frame, (x - 3, y - 3, x + width + 3, y + height + text_height * 2 + 8),
          alpha=0.6)
    for i in range(16):
        level = int(255 * (1 - i / 15.0))
        top = y + int(height * i / 16)
        bottom = y + int(height * (i + 1) / 16)
        cv2.rectangle(frame, (x, top), (x + width, bottom), (level, level, level), -1)
    marker_y = y + int(height * (1.0 - max(0.0, min(1.0, brightness))))
    cv2.rectangle(frame, (x - 4, marker_y - 2), (x + width + 4, marker_y + 2), AMBER, -1)
    text(frame, f"{brightness * 100:.0f}%", (x + width // 2, y + height + text_height),
         0.42 * scale, WHITE, 1, anchor="center")
    return y + height + text_height * 2 + 8


def help_panel(frame, origin, rows: list[tuple[str, str]], scale: float = 1.0,
               anchor: str = "left", columns: int = 2) -> tuple[int, int, int, int]:
    """The key list.  Returns the rectangle it occupied.

    Laid out in columns rather than one long list: twenty rows stacked in a
    single column has to be set at a size that makes consecutive lines touch,
    which reads as a smear rather than a list.  Two columns keeps the same
    information at a legible size, and the height fits above the keyboard.

    Row baselines come from the same expression as the box height -- the two
    drifting apart is exactly what puts the last few rows outside the panel.
    """
    small = 0.46 * scale
    row_height = max(14.0, 18.0 * scale)
    columns = max(1, min(columns, len(rows)))

    per_column = (len(rows) + columns - 1) // columns
    column_rows = [rows[i * per_column:(i + 1) * per_column] for i in range(columns)]
    key_width = max((cv2.getTextSize(k, FONT, small, 1)[0][0] for k, _ in rows), default=40)
    value_width = max((cv2.getTextSize(v, FONT, small, 1)[0][0] for _, v in rows), default=80)
    column_width = key_width + value_width + 30
    box_width = column_width * len(column_rows) + 20
    height = int(row_height * per_column + 34 * scale)

    x = int(origin[0])
    if anchor == "right":
        x -= box_width
    y = int(origin[1])                       # the *bottom* edge
    top = y - height
    panel(frame, (x, top, x + box_width, y), alpha=0.8)

    text(frame, "keys", (x + 10, top + int(17 * scale)), small, AMBER, 1)
    for column, entries in enumerate(column_rows):
        column_x = x + 12 + column * column_width
        for row, (key, value) in enumerate(entries):
            baseline = int(top + 34 * scale + row_height * row)
            text(frame, key, (column_x, baseline), small, WHITE, 1)
            text(frame, value, (column_x + column_width - 20, baseline), small, GREY, 1,
                 anchor="right")
    return (x, top, x + box_width, y)


def warning_banner(frame, message: str, scale: float = 1.0) -> None:
    """A centred advisory, used for the colour-confusion warning."""
    if not message:
        return
    small = 0.58 * scale
    (width, height), _ = cv2.getTextSize(message, FONT, small, 1)
    x = (frame.shape[1] - width) // 2
    y = int(60 * scale)
    panel(frame, (x - 14, y - height - 10, x + width + 14, y + 12),
          fill=(20, 40, 90), border=AMBER, alpha=0.8)
    text(frame, message, (x, y), small, (120, 230, 255), 1)


def gesture_panel(frame, origin, gesture_label: str, action: str, active: bool,
                  available: bool, reason: str, scale: float = 1.0) -> None:
    x, y = int(origin[0]), int(origin[1])
    small = 0.5 * scale
    label = f"{gesture_label} -> {action}" if action else gesture_label
    if not available:
        label = f"gestures off: {reason[:48]}" if reason else "gestures off"
    (width, _), _ = cv2.getTextSize(label, FONT, small, 1)
    panel(frame, (x, y - 20, x + width + 24, y + 10),
          fill=(10, 60, 20) if active else (30, 30, 30),
          border=GREEN if active else GREY, alpha=0.75)
    text(frame, label, (x + 12, y), small, GREEN if active else GREY, 1)


def legend(frame, origin, palette: Palette, patterns: bool, scale: float = 1.0,
           columns: int = 2, count: int = 8) -> tuple[int, int, int, int] | None:
    """A motif legend, so the texture code can be learned from the screen.

    Returns the rectangle it occupied, so callers can stack something above it
    without guessing at the height.
    """
    if not patterns:
        return None
    x, y = int(origin[0]), int(origin[1])
    small = 0.44 * scale
    chip = int(9 * scale)
    row_height = int(21 * scale)
    column_width = int(104 * scale)
    rows = (count + columns - 1) // columns
    width = columns * column_width + 12
    height = rows * row_height + int(26 * scale)
    panel(frame, (x, y, x + width, y + height), alpha=0.66)
    text(frame, "patterns", (x + 8, y + int(16 * scale)), small, AMBER, 1)
    for motif in range(count):
        column, row = motif // rows, motif % rows
        cx = x + 22 + column * column_width
        cy = y + int(34 * scale) + row * row_height
        example = next((i for i in range(len(palette)) if palette.pattern(i) == motif),
                       motif % len(palette))
        pattern_swatch(frame, (cx, cy), chip, palette.color(example), motif,
                       config.PATTERN_NAMES[motif], small)
    return (x, y, x + width, y + height)


def fps_badge(frame, origin, fps: float) -> None:
    text(frame, f"{fps:4.1f} fps", origin, 0.46, GREEN if fps > 20 else RED, 1,
         anchor="right")
