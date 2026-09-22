"""Start-up self checks.

``python -m tools.selftest`` prints this report.  The point is that "it works"
should be a measurement rather than a claim: the note table, the palette, the
audio device, the camera and the gesture model are each probed, and anything
that is missing is reported with what to do about it.
"""

from __future__ import annotations

import platform
import sys
from dataclasses import dataclass, field

from . import config
from .colorblind import (
    CVD_MODES,
    build_confusion_tables,
    palette_score,
    worst_case_separation,
)
from .notes import build_note_table, describe_table, discover_samples
from .palette import build_palette


@dataclass
class Check:
    name: str
    ok: bool
    detail: str = ""
    advice: str = ""
    fatal: bool = False
    lines: list[str] = field(default_factory=list)


def check_python() -> Check:
    version = sys.version_info
    ok = version >= (3, 9)
    return Check(
        "python",
        ok,
        f"{platform.python_version()} on {platform.system()} {platform.release()}",
        "Python 3.9 or newer is required." if not ok else "",
        fatal=not ok,
    )


def check_dependencies() -> Check:
    lines, missing = [], []
    for module, minimum, why in (
        ("numpy", "1.20", "colour maths"),
        ("cv2", "4.5", "camera and overlay"),
        ("pygame", "2.0", "audio playback"),
    ):
        try:
            imported = __import__(module)
            version = getattr(imported, "__version__", "?")
            lines.append(f"{module:<8} {version:<10} {why}")
            try:
                if tuple(int(p) for p in version.split(".")[:2]) < tuple(
                    int(p) for p in minimum.split(".")
                ):
                    lines.append(f"         ^ older than the required {minimum}")
            except ValueError:
                pass
        except Exception as error:
            missing.append(module)
            lines.append(f"{module:<8} MISSING    ({error.__class__.__name__}) {why}")
    return Check(
        "dependencies",
        not missing,
        "all present" if not missing else f"missing: {', '.join(missing)}",
        "pip install -r requirements.txt" if missing else "",
        fatal=bool(missing),
        lines=lines,
    )


def check_notes(count: int = 52) -> Check:
    directory = config.resolve_notes_dir()
    recordings = discover_samples(directory)
    lines = [describe_table(recordings, directory)]
    if not recordings:
        lines.append("no .wav files were found -- the instrument will synthesise "
                     "every note instead")
    try:
        table = build_note_table(directory, count=count)
    except Exception as error:
        return Check("note table", False, str(error), fatal=True, lines=lines)
    lines.append(describe_table(table, directory))
    shift = max((abs(n.shift) for n in table), default=0.0)
    if shift > 2.0:
        lines.append(f"warning: notes are transposed by up to {shift:.0f} semitones; "
                     "they will sound thin at the extremes")
    return Check("note table", len(table) == count, f"{len(table)} notes ready",
                 lines=lines)


def check_palettes(count: int = 52) -> Check:
    lines = []
    ok = True
    reference = worst_case_separation(build_palette("hue", count).colors)
    for name in ("hue", "cvd_safe"):
        palette = build_palette(name, count)
        colours = palette.colors
        separation = worst_case_separation(colours)
        score = palette_score(colours)
        deuteranopia = score.get("deuteranopia", {})
        lines.append(
            f"{name:<9} n={len(colours):<3} worst-case separation {separation:5.2f}   "
            f"deuteranopia: {deuteranopia.get('ambiguous', '?')}/{len(colours)} "
            f"notes have a confusable partner"
        )
        if name == "cvd_safe" and separation <= reference:
            ok = False
            lines.append("         ^ no better than the plain hue wheel; "
                         "regenerate with tools/design_palette.py")
    lines.append(f"reference: the plain hue wheel scores {reference:.2f}, so the "
                 "colour-blind palette is what a CVD user should be on")
    return Check("palettes", ok, "measured against the full CVD reference set",
                 lines=lines)


def check_confusion(count: int = 52, palette: str = "cvd_safe") -> Check:
    table = build_palette(palette, count)
    tables = build_confusion_tables(table.colors)
    lines = [tables[mode].report() for mode in CVD_MODES if mode in tables]
    worst = max((t.ambiguous_count for t in tables.values()), default=0)
    return Check(
        "confusion warnings",
        True,
        f"{palette} palette: worst case {worst}/{count} notes flagged for one CVD type",
        lines=lines,
    )


def check_audio(quick: bool = False) -> Check:
    if quick:
        return Check("audio", True, "skipped (--skip-hardware)")
    try:
        from .audio import AudioEngine
        from .notes import build_note_table
    except Exception as error:
        return Check("audio", False, f"import failed: {error}", fatal=True)

    notes = build_note_table(count=52)
    try:
        engine = AudioEngine(notes, preload=("piano",))
    except Exception as error:
        return Check("audio", False, f"engine would not start: {error}",
                     "check the sound device; the app will still run silently")
    status = engine.status
    lines = [f"driver: {status.driver}", f"banks loaded: {status.banks_loaded or 'none'}"]
    if status.reason:
        lines.append(status.reason)
    notes_ok = sum(1 for s in engine.sounds if s is not None)
    lines.append(f"piano bank: {notes_ok}/{len(notes)} notes decoded")
    played = engine.play(30)
    lines.append(f"test note C4 (index 30) played: {played}")
    engine.shutdown()
    ok = status.available and notes_ok == len(notes)
    return Check(
        "audio", ok,
        f"mixer available via {status.driver}" if status.available else "no audio device",
        "" if ok else "the instrument will run with sound disabled",
        lines=lines,
    )


def check_camera(index: int = 0, quick: bool = False) -> Check:
    if quick:
        return Check("camera", True, "skipped (--skip-hardware)")
    try:
        import cv2
    except Exception as error:
        return Check("camera", False, f"opencv missing: {error}", fatal=True)
    from .vision import CameraSource

    try:
        source = CameraSource(index)
    except Exception as error:
        return Check("camera", False, str(error),
                     "connect a camera, close other apps using it, or run with "
                     "--source synthetic to try the program without one")
    lines = [f"opencv {cv2.__version__}", source.description]
    ok, frame = source.read()
    lines.append(f"read a frame: {ok}" + (f" {frame.shape}" if ok and frame is not None else ""))
    source.release()
    return Check("camera", ok, source.description, lines=lines)


def check_gestures(quick: bool = False) -> Check:
    if quick:
        return Check("gestures", True, "skipped (--skip-hardware)")
    from .gestures import (
        HandTracker,
        find_hand_model,
        load_mediapipe,
        locate_mediapipe,
    )

    lines = []
    located = locate_mediapipe()
    if located is None:
        return Check(
            "gestures", True, "disabled: mediapipe is not installed",
            "optional: pip install mediapipe to enable hand control. "
            "The keyboard and mouse do everything gestures do.",
        )
    lines.append(f"package at {located}")
    if not str(located).isascii():
        lines.append("path contains non-ASCII characters -- mediapipe cannot load "
                     "its model from here; the program stages a copy elsewhere")
    module, repaired, problem = load_mediapipe()
    if module is None:
        return Check("gestures", False, problem, fatal=False)
    lines.append(f"mediapipe {getattr(module, 'version', None) or getattr(module, '__version__', '?')}"
                 + (" (staged copy)" if repaired else ""))
    lines.append(f"API: {'solutions.hands' if hasattr(module, 'solutions') else 'tasks'}")
    model = find_hand_model()
    lines.append(f"hand model: {model if model else 'none found (needed by the tasks API)'}")

    tracker = HandTracker()
    if not tracker.available:
        return Check("gestures", True, f"disabled: {tracker.reason}",
                     "hand control is optional; the keyboard and mouse cover "
                     "everything it does",
                     lines=lines)
    try:
        import numpy as np

        blank = np.zeros((480, 640, 3), dtype="uint8")
        tracker.process(blank)
        lines.append(f"backend: {tracker.backend_name}")
        lines.append("a blank frame was processed without error")
    except Exception as error:
        tracker.close()
        return Check("gestures", False, f"failed on a blank frame: {error}",
                     lines=lines)
    tracker.close()
    return Check("gestures", True, f"hand tracking available ({tracker.backend_name})",
                 lines=lines)


def check_speech(quick: bool = False) -> Check:
    if quick:
        return Check("speech", True, "skipped (--skip-hardware)")
    from .announce import SpeechEngine

    engine = SpeechEngine()
    if engine.available:
        return Check("speech", True, f"text to speech via {engine.backend}")
    return Check("speech", True, engine.reason,
                 "optional: install pyttsx3 for spoken colour names; without it the "
                 "program uses texture-pattern earcons instead")


def run_all(settings=None, skip_hardware: bool = False) -> list[Check]:
    count = 52
    palette = "cvd_safe"
    index = 0
    if settings is not None:
        index = settings.camera_index
        palette = settings.palette
    return [
        check_python(),
        check_dependencies(),
        check_notes(count),
        check_palettes(count),
        check_confusion(count, palette),
        check_audio(skip_hardware),
        check_gestures(skip_hardware),
        check_speech(skip_hardware),
        check_camera(index, skip_hardware),
    ]


def format_report(checks: list[Check], width: int = 78) -> str:
    out = ["", "=" * width, f" {config.APP_NAME} {config.APP_VERSION} -- self check",
           "=" * width]
    for check in checks:
        mark = "ok  " if check.ok else ("FAIL" if check.fatal else "warn")
        out.append(f"[{mark}] {check.name:<16} {check.detail}")
        for line in check.lines:
            out.append(f"         {line}")
        if check.advice and not check.ok:
            out.append(f"         -> {check.advice}")
    fatal = [c for c in checks if c.fatal and not c.ok]
    out.append("-" * width)
    if fatal:
        out.append(f" {len(fatal)} blocking problem(s): "
                   + ", ".join(c.name for c in fatal))
    else:
        out.append(" no blocking problems found")
    out.append("=" * width)
    return "\n".join(out)


def exit_code(checks: list[Check]) -> int:
    return 1 if any(c.fatal and not c.ok for c in checks) else 0


__all__ = ["Check", "run_all", "format_report", "exit_code"]
