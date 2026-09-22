"""Run the installation checks, plus optionally a full headless pipeline run.

    python -m tools.selftest                    # checks only
    python -m tools.selftest --skip-hardware    # no camera, no sound card
    python -m tools.selftest --pipeline         # also drive 120 synthetic frames

``--pipeline`` is the interesting one: it feeds the instrument frames whose
colours are known, and asserts that the note that comes out is the one that
should.  That is a real end-to-end test of colour detection, smoothing,
stabilisation and note selection -- and it needs neither a camera nor a speaker.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from colorpiano import config                                          # noqa: E402
from colorpiano.diagnostics import exit_code, format_report, run_all   # noqa: E402
from colorpiano.config import Settings                                 # noqa: E402


def pipeline_check(frames: int = 240, period: int = 14, verbose: bool = False,
                   palette_name: str = "cvd_safe") -> int:
    """Drive the app over synthetic frames and verify the colour -> note path.

    The synthetic source paints one palette entry at a time, held for *period*
    frames.  Because the app averages the last ``smoothing`` samples, the first
    few frames after each change are still dominated by the previous colour --
    that lag is the deliberate price of not stuttering, so the check only
    asserts exact agreement once the average has had time to fill.
    """
    from colorpiano.app import ColorPianoApp
    from colorpiano.audio import NullAudioEngine
    from colorpiano.colorblind import indistinguishable_groups
    from colorpiano.vision import SyntheticSource

    settings = Settings(
        source="synthetic",
        headless=True,
        gestures=False,
        palette=palette_name,
        announce=False,
    )
    app = ColorPianoApp(settings)

    # A faster sweep than the default, so a few seconds of frames exercise many
    # colours rather than two.
    app.source.release()
    app.source = SyntheticSource(period=period, noise=2.0,
                                 colors=app.palette.colors)
    app.bind_source_palette()

    # Swap in the null engine so the test never touches a sound device, while
    # still exercising every call the real engine would receive.
    app.audio.shutdown()
    app.audio = NullAudioEngine(app.notes)
    app.announcer.audio = app.audio

    # Entries that no viewer can tell apart are legitimately allowed to be
    # confused with each other: reporting a neighbouring entry is then not a
    # detection error, it is the only honest answer.
    twins: dict[int, int] = {}
    for group in indistinguishable_groups(app.palette.colors):
        for member in group:
            twins[member] = group[0]
    n_twins = sum(1 for i in range(len(app.palette)) if twins.get(i, i) != i)

    settling = app.smoother.size + app.stabilizer.needed
    print()
    print("-" * 78)
    print(f" end-to-end pipeline: {frames} synthetic frames, one colour per "
          f"{period} frames")
    print(f"   palette {app.palette.name}: {n_twins} of {len(app.palette)} entries "
          "are indistinguishable from another entry to every viewer")
    print(f"   allowing {settling} frames to settle after each change")
    print("-" * 78)

    checked = 0
    exact = 0
    explained = 0
    mismatches: list[str] = []
    failures: list[str] = []
    steps = 0
    while steps < frames:
        if not app.step():
            failures.append("the frame source stopped early")
            break
        steps += 1
        target = getattr(app.source, "target_index", None)
        position = steps % period
        stable = settling <= position <= period - 2
        if stable and target is not None and app.result.index >= 0:
            checked += 1
            if target == app.result.index:
                exact += 1
            elif twins.get(target, target) == twins.get(app.result.index, app.result.index):
                explained += 1
            else:
                mismatches.append(
                    f"frame {steps}: painted index {target} but detected "
                    f"{app.result.index} ({app.result.note_name}, "
                    f"{app.result.color_name})"
                )
        if verbose and steps % 20 == 0:
            print(f"  frame {steps:>4}  painted {target if target is not None else '-':>3}"
                  f"  detected {app.result.index:>3} {app.result.note_name:<4} "
                  f"{app.result.color_name:<22} dE {app.result.distance:5.1f}")

    played = app.audio.played
    distinct = len(set(played))
    print(f"  frames processed        : {steps}")
    print(f"  settled frames compared : {checked}")
    print(f"    exact match           : {exact}")
    print(f"    twin entry (see above) : {explained}")
    print(f"    wrong                 : {len(mismatches)}")
    print(f"  distinct notes played   : {distinct} of {len(app.palette)}")
    print(f"  hysteresis rejections   : {app.stabilizer.rejected}")

    if not played:
        failures.append("no note was ever played")
    if distinct < 8:
        failures.append(f"only {distinct} distinct notes played; the sweep should "
                        "have produced many more")
    if checked == 0:
        failures.append("no settled frames were produced, so nothing was verified")
    failures.extend(mismatches[:6])

    app.shutdown()

    if failures:
        print()
        for problem in failures[:10]:
            print(f"  FAIL {problem}")
        return 1
    print(f"  PASS colour -> note -> audio: {exact} settled frames matched the painted "
          f"colour exactly, {explained} landed on an indistinguishable twin, "
          "0 wrong")
    return 0


def render_check() -> int:
    """Render one fully annotated frame and check the overlay really drew."""
    import numpy as np

    from colorpiano import hud
    from colorpiano.notes import build_note_table
    from colorpiano.palette import build_palette

    notes = build_note_table(count=52)
    palette = build_palette("cvd_safe", len(notes))
    frame = np.full((720, 1280, 3), 40, dtype=np.uint8)
    before = frame.copy()

    print()
    print("-" * 78)
    print(" overlay rendering")
    print("-" * 78)
    hud.draw_sampling_box(frame, (640, 360), 40, palette, 17, True, True)
    hud.note_panel(frame, (12, 12), palette, 17, notes[17], "dark vivid orange",
                   "diagonal", True)
    hud.keyboard_strip(frame, (0, 654, 1280, 720), notes, palette, 17, True)
    hud.brightness_bar(frame, (20, 200), 160, 0.62)
    hud.legend(frame, (12, 400), palette, True)
    hud.status_panel(frame, (1266, 12), [("timbre", "piano"), ("vision", "deuteranopia")])
    hud.warning_banner(frame, "under deuteranopia this also looks like D4")
    hud.help_panel(frame, (12, 640), [("q", "quit"), ("d", "daltonize")])

    changed = int(np.count_nonzero(frame != before))
    print(f"  overlay changed {changed} pixels of the frame")
    # Every panel is exercised above; anything under this means a panel silently
    # decided not to draw.
    if changed < 20_000:
        print("  FAIL the overlay drew far less than expected")
        return 1
    print("  PASS overlay rendering (all panels drew)")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--skip-hardware", action="store_true",
                        help="do not open the camera or the sound card")
    parser.add_argument("--pipeline", action="store_true",
                        help="also run the synthetic end-to-end test")
    parser.add_argument("--both-palettes", action="store_true",
                        help="with --pipeline, test the hue palette as well")
    parser.add_argument("--frames", type=int, default=240)
    parser.add_argument("--period", type=int, default=14,
                        help="frames each colour is held for during the sweep")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--config", action="store_true",
                        help="print the default settings first")
    args = parser.parse_args(argv)

    if args.config:
        print("\n".join(Settings().to_lines()))
        print()

    checks = run_all(None, skip_hardware=args.skip_hardware)
    print(format_report(checks))
    code = exit_code(checks)

    if args.pipeline and code == 0:
        code |= render_check()
        palettes = ["cvd_safe", "hue"] if args.both_palettes else ["cvd_safe"]
        for name in palettes:
            code |= pipeline_check(args.frames, args.period, args.verbose, name)
    return code


if __name__ == "__main__":
    raise SystemExit(main())
