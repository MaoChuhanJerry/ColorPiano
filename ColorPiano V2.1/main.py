"""ColorPiano Pro -- command line entry point.

    python main.py                          # start with the camera
    python main.py --source synthetic       # run without a camera
    python main.py --cvd deuteranopia --palette cvd_safe --daltonize
    python main.py --selftest               # check the installation

See ``python main.py --help`` for the full list, and README.md for what the
accessibility options actually do.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent))

from colorpiano import config                                 # noqa: E402
from colorpiano.colorblind import CVD_LABELS, CVD_MODES       # noqa: E402
from colorpiano.config import Settings                        # noqa: E402
from colorpiano.palette import build_palette                  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="colorpiano",
        description="Turn colours into piano notes, with colour-blind users first.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
key bindings while running:
  q / esc  quit            h  help panel         space  sustain
  b  timbre                p  palette            c  cycle colour-vision type
  d  daltonize             s  simulate CVD       t  texture patterns
  n  colour names          k  keyboard strip     a  announce
  g  hand gestures         L  large text         m  mute
  - / +  volume            arrows  move the box  [ ]  box size
  r  reset the box         1-5  pick a colour-vision type
""",
    )

    general = parser.add_argument_group("general")
    general.add_argument("--selftest", action="store_true",
                         help="run the checks and exit")
    general.add_argument("--skip-hardware", action="store_true",
                         help="with --selftest, do not touch the camera or sound card")
    general.add_argument("--print-config", action="store_true",
                         help="print every setting and exit")
    general.add_argument("--headless", action="store_true",
                         help="run without opening a window (for testing)")
    general.add_argument("--duration", type=float, default=None, metavar="SECONDS",
                         help="stop automatically after this long")
    general.add_argument("--record", default=None, metavar="FILE.mp4",
                         help="write the annotated view to a video file")
    general.add_argument("--version", action="version",
                         version=f"{config.APP_NAME} {config.APP_VERSION}")

    source = parser.add_argument_group("input")
    source.add_argument("--source", default=None,
                        choices=["camera", "video", "image", "synthetic"],
                        help="where frames come from (default: camera)")
    source.add_argument("--source-path", default=None,
                        help="file for --source video or --source image")
    source.add_argument("--camera", type=int, default=None, metavar="INDEX",
                        help="camera index (default: 0)")
    source.add_argument("--width", type=int, default=None)
    source.add_argument("--height", type=int, default=None)
    source.add_argument("--no-mirror", action="store_true",
                        help="do not flip the camera image")

    colour = parser.add_argument_group("colour detection")
    colour.add_argument("--palette", default=None, choices=["hue", "cvd_safe"],
                        help="52 hues, or the colour-blind safe palette (default: hue)")
    colour.add_argument("--roi-size", type=int, default=None, metavar="PX",
                        help="half-width of the sampling box, in pixels")
    colour.add_argument("--smoothing", type=int, default=None, metavar="N",
                        help="colour samples kept in the running average")
    colour.add_argument("--stabilizer", type=int, default=None, metavar="N",
                        help="frames a new note must win before it takes over")
    colour.add_argument("--min-chroma", type=float, default=None,
                        help="treat anything less colourful than this as grey")
    colour.add_argument("--no-white-balance", action="store_true",
                        help="disable the automatic colour-cast correction")
    colour.add_argument("--no-gestures", action="store_true")

    playing = parser.add_argument_group("playing")
    playing.add_argument("--bank", default=None, choices=["piano", "synth", "pure"],
                         help="timbre (default: piano)")
    playing.add_argument("--volume", type=float, default=None, metavar="0-1")
    playing.add_argument("--note-interval", type=float, default=None, metavar="SECONDS",
                         help="minimum gap between note onsets")
    playing.add_argument("--sustain", action="store_true",
                         help="retrigger a held note instead of waiting for a change")
    playing.add_argument("--glide", action="store_true",
                         help="arpeggiate when the colour jumps a long way")
    playing.add_argument("--range", dest="range_mode", default=None,
                         choices=["white", "chromatic"],
                         help="white keys only, or 52 semitones in every key")
    playing.add_argument("--range-start", type=int, default=None, metavar="MIDI",
                         help="lowest note of the chromatic range (60 = middle C)")
    playing.add_argument("--polyphony", type=int, default=None, metavar="N")
    playing.add_argument("--no-velocity", action="store_true",
                         help="do not scale loudness with brightness")

    access = parser.add_argument_group("accessibility")
    access.add_argument("--cvd", default=None, choices=CVD_MODES,
                        help="your colour-vision type; enables the confusion warnings")
    access.add_argument("--daltonize", action="store_true",
                        help="shift confusable colours apart in the live view")
    access.add_argument("--daltonize-strength", type=float, default=None, metavar="0-2")
    access.add_argument("--simulate-cvd", action="store_true",
                        help="show the preview as that viewer would see it")
    access.add_argument("--filter-scale", type=float, default=None, metavar="0.1-1",
                        help="resolution the colour-vision filters run at "
                             "(default 0.5; lower is faster and softer)")
    access.add_argument("--no-patterns", action="store_true",
                        help="hide the texture coding")
    access.add_argument("--no-names", action="store_true",
                        help="hide the colour names")
    access.add_argument("--no-keyboard", action="store_true",
                        help="hide the keyboard strip")
    access.add_argument("--no-help", action="store_true",
                        help="hide the key list on screen")
    access.add_argument("--large-text", action="store_true")
    access.add_argument("--high-contrast", action="store_true")
    access.add_argument("--announce", action="store_true",
                        help="speak the colour name, or play an earcon if there is "
                             "no speech engine")
    access.add_argument("--no-confusion-warning", action="store_true")

    return parser


def settings_from_args(args: argparse.Namespace) -> Settings:
    settings = Settings()
    settings.apply_overrides(
        source=args.source,
        source_path=args.source_path or "",
        camera_index=args.camera,
        width=args.width,
        height=args.height,
        mirror=False if args.no_mirror else None,
        palette=args.palette,
        roi_size=args.roi_size,
        smoothing=args.smoothing,
        stabilizer=args.stabilizer,
        min_chroma=args.min_chroma,
        white_balance=False if args.no_white_balance else None,
        gestures=False if args.no_gestures else None,
        bank=args.bank,
        volume=args.volume,
        note_interval=args.note_interval,
        sustain=True if args.sustain else None,
        glide=True if args.glide else None,
        range_mode=args.range_mode,
        range_start=args.range_start,
        polyphony=args.polyphony,
        velocity_from_brightness=False if args.no_velocity else None,
        cvd_mode=args.cvd,
        daltonize=True if args.daltonize else None,
        daltonize_strength=args.daltonize_strength,
        simulate_cvd=True if args.simulate_cvd else None,
        filter_scale=args.filter_scale,
        patterns=False if args.no_patterns else None,
        show_names=False if args.no_names else None,
        show_keyboard=False if args.no_keyboard else None,
        show_help=False if args.no_help else None,
        large_text=True if args.large_text else None,
        high_contrast=True if args.high_contrast else None,
        announce=True if args.announce else None,
        confusion_warning=False if args.no_confusion_warning else None,
        headless=True if args.headless else None,
        record=args.record or "",
        duration=args.duration,
    )
    # Raised as ValueError rather than SystemExit so that main() reports it the
    # same way as every other bad argument, with an "error:" prefix and exit 2.
    if not 0.0 <= settings.volume <= 1.0:
        raise ValueError(f"--volume must be between 0 and 1 (got {settings.volume})")
    if settings.roi_size is not None and not 0 < settings.roi_size <= 200:
        raise ValueError(f"--roi-size must be between 1 and 200 (got {settings.roi_size})")
    if settings.smoothing is not None and settings.smoothing < 1:
        raise ValueError(f"--smoothing must be at least 1 (got {settings.smoothing})")
    if settings.stabilizer is not None and settings.stabilizer < 1:
        raise ValueError(f"--stabilizer must be at least 1 (got {settings.stabilizer})")
    if not 0.1 <= settings.filter_scale <= 1.0:
        raise ValueError("--filter-scale must be between 0.1 and 1.0 "
                         f"(got {settings.filter_scale})")
    return settings


def describe_accessibility(settings: Settings) -> str:
    """A short explanation of what the accessibility options are doing."""
    lines = [f"colour-vision type : {CVD_LABELS.get(settings.cvd_mode, settings.cvd_mode)}"]
    if settings.cvd_mode != "none":
        if settings.daltonize:
            lines.append("daltonize          : on -- confusable colours are pushed "
                         "apart in the live view")
        else:
            lines.append("daltonize          : off -- press d to boost them")
        if settings.simulate_cvd:
            lines.append("simulate           : on -- the preview is what you would see; "
                         "press s to go back to the camera")
    lines.append(f"texture patterns   : {'on' if settings.patterns else 'off'}")
    lines.append(f"colour names       : {'on' if settings.show_names else 'off'}")
    lines.append(f"announcements      : {'on' if settings.announce else 'off'}")
    palette = build_palette(settings.palette, config.PALETTE_SIZE)
    lines.append(f"palette            : {palette.name} -- {palette.description}")
    if settings.palette == "hue" and settings.cvd_mode != "none":
        lines.append("                     note: with a colour-vision type set, the "
                     "'cvd_safe' palette")
        lines.append("                     is measurably easier to tell apart "
                     "(press p to switch)")
    return "\n".join(" " + line for line in lines)


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    if args.selftest:
        from colorpiano.diagnostics import exit_code, format_report, run_all

        settings = None
        try:
            settings = settings_from_args(args)
        except SystemExit:
            pass
        checks = run_all(settings, skip_hardware=args.skip_hardware)
        print(format_report(checks))
        return exit_code(checks)

    try:
        settings = settings_from_args(args)
    except (KeyError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2

    if args.print_config:
        print("\n".join(settings.to_lines()))
        return 0

    print(describe_accessibility(settings))

    try:
        from colorpiano.app import ColorPianoApp
    except ImportError as error:
        print(f"error: could not import the program ({error})", file=sys.stderr)
        print("       install the dependencies: pip install -r requirements.txt",
              file=sys.stderr)
        return 1

    try:
        app = ColorPianoApp(settings)
    except Exception as error:
        print(f"error: {error}", file=sys.stderr)
        print("       run 'python main.py --selftest' to see what is missing.",
              file=sys.stderr)
        return 1

    app.run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
