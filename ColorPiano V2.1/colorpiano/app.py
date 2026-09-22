"""The instrument: colour detection, note playing, overlay, and key handling.

Reading order, if you are looking for the interesting parts:

* :meth:`ColorPianoApp.step`  -- one frame, start to finish
* :meth:`ColorPianoApp.choose_index` -- sample -> note, with the grey guard
* :meth:`ColorPianoApp.play_if_changed` -- the debounce that makes it playable
* :meth:`ColorPianoApp.handle_key` -- every toggle, in one table
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import cv2
import numpy as np

from . import colorblind, config, hud
from .announce import Announcer
from .audio import BANK_DESCRIPTIONS, BANKS, AudioEngine
from .colorspace import bgr_to_lab, delta_e_ciede2000, lab_to_bgr
from .config import PATTERN_NAMES, Settings
from .gestures import Gesture, GestureStabilizer, HandTracker, gesture_help_rows
from .notes import Note, build_note_table, describe_table
from .palette import (
    ColorSmoother,
    NoteStabilizer,
    Palette,
    build_palette,
    color_name,
)
from .vision import FrameSource, gray_world_gains, open_source, sample_box

#: Millimetres of margin used when deciding a sample is "confidently" a colour.
AMBIGUITY_MARGIN = 1.5


@dataclass
class FrameResult:
    """What one frame produced -- returned by ``step`` so tests can assert on it."""

    index: int = -1
    note_name: str = ""
    color_name: str = ""
    played: bool = False
    sample_bgr: tuple[int, int, int] = (0, 0, 0)
    brightness: float = 0.0
    distance: float = 0.0
    runner_up: float = float("inf")
    warning: str = ""
    skipped: str = ""
    changed: bool = False


@dataclass
class GestureState:
    label: str = "none"
    action: str = ""
    active: bool = False
    tracking: bool = False
    paused: bool = False
    pinched_at: tuple[float, float] | None = None
    stats: dict = field(default_factory=dict)


class ColorPianoApp:
    """Colour -> note -> sound, with a colour-blind-aware overlay."""

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or Settings()

        # --- notes and colours ------------------------------------------ #
        self.notes: list[Note] = build_note_table(
            count=config.PALETTE_SIZE,
            range_mode=self.settings.range_mode,
            range_start=self.settings.range_start,
        )
        self.palette: Palette = build_palette(self.settings.palette, len(self.notes))
        # Precomputed per CVD type: which notes a viewer of that type cannot
        # tell apart.  This is what lets the overlay warn honestly instead of
        # silently guessing.
        self.confusion = colorblind.build_confusion_tables(self.palette.colors)

        # --- audio ------------------------------------------------------- #
        self.audio = AudioEngine(
            self.notes,
            volume=self.settings.volume,
            bank=self.settings.bank,
            polyphony=self.settings.polyphony,
        )
        self.announcer = Announcer(self.audio, enabled=self.settings.announce)

        # --- vision ------------------------------------------------------ #
        self.source: FrameSource = open_source(self.settings)
        self.bind_source_palette()

        # --- signal processing ------------------------------------------- #
        self.smoother = ColorSmoother(self.settings.smoothing)
        self.stabilizer = NoteStabilizer(self.settings.stabilizer)

        # --- gestures ---------------------------------------------------- #
        self.tracker: HandTracker | None = None
        self.gesture_stabilizer = GestureStabilizer()
        self.gesture = GestureState()
        if self.settings.gestures:
            self.tracker = HandTracker(confidence=self.settings.gesture_confidence)
            if not self.tracker.available:
                print(f"[gestures] disabled: {self.tracker.reason}")

        # --- runtime state ----------------------------------------------- #
        self.roi = [0.5, 0.5]              # normalised centre of the sampling box
        self.roi_size = int(self.settings.roi_size)
        self.volume = float(self.settings.volume)
        self.muted = False
        self.paused = False
        self.frame_index = 0
        self.fps = 0.0
        self.last_played_index = -1
        self.last_play_time = 0.0
        self.result = FrameResult()
        self.running = True
        self.writer = None
        self.window = "ColorPiano Pro"
        self._fps_window: list[float] = []

        self.help_rows = self._build_help_rows(
            gestures=self.tracker is not None and self.tracker.available
        )

    # ------------------------------------------------------------------ setup
    @staticmethod
    def _build_help_rows(gestures: bool = False) -> list[tuple[str, str]]:
        rows = [
            ("q / esc", "quit"),
            ("space", "sustain (retrigger the same note)"),
            ("b", "timbre: piano / synth / pure"),
            ("p", "palette: hue / colour-blind safe"),
            ("c", "cycle colour-vision type"),
            ("1-5", "set colour-vision type directly"),
            ("d", "daltonize (boost live colours)"),
            ("s", "simulate (see through their eyes)"),
            ("t", "texture patterns on/off"),
            ("n", "colour names on/off"),
            ("k", "keyboard strip on/off"),
            ("a", "announce (speech / earcon)"),
            ("g", "hand gestures on/off"),
            ("L", "large text"),
            ("m", "mute"),
            ("- / +", "volume"),
            ("arrows", "move the sampling box"),
            ("[ / ]", "sampling box size"),
            ("r", "reset the sampling box"),
            ("h", "this panel"),
        ]
        if gestures:
            # The gesture bindings belong on the same list: nothing else in the
            # program tells the user what the gestures do.
            rows.append(("-- hands --", ""))
            rows.extend(gesture_help_rows())
        return rows

    def bind_source_palette(self) -> None:
        """Let a synthetic source paint the real palette.

        Makes ``--source synthetic`` demonstrate the actual colours in use, and
        is what lets the self-test compare "colour painted" against "colour
        detected" without knowing anything about the sampler.
        """
        setter = getattr(self.source, "set_colors", None)
        if callable(setter):
            setter(self.palette.colors)

    # ------------------------------------------------------------------ colour
    def choose_index(self, lab: np.ndarray) -> tuple[int, float, int, float]:
        """Pick the palette entry closest to *lab*.

        Returns ``(index, distance, runner_up_index, runner_up_distance)``.
        Matching runs against the **raw** frame, never against the daltonized or
        simulated preview: those filters exist to help the user's eyes, and
        feeding them back into the decision would make the instrument's idea of
        the colour drift with a display setting.

        Distances are CIEDE2000 at full float precision -- see
        :mod:`colorpiano.colorspace` for why the original's 8-bit OpenCV Lab
        values made those distances meaningless.
        """
        distances = delta_e_ciede2000(lab, self.palette.lab)
        order = np.argsort(distances)
        index = int(order[0])
        if len(order) > 1:
            return index, float(distances[index]), int(order[1]), float(distances[order[1]])
        return index, float(distances[index]), index, float("inf")

    def generic_warning(self, index: int, distance: float,
                        runner_index: int, runner_distance: float) -> str:
        """A note when the sample sits almost exactly between two palette entries.

        Separate from :meth:`confusion_warning`: that one is about the viewer's
        colour vision, this one is about the *object* being an awkward colour
        for anybody.  Both are worth saying, and they need different fixes.
        """
        if runner_distance - distance >= AMBIGUITY_MARGIN:
            return ""
        name = self.note_label(index)
        other = self.note_label(runner_index)
        return (f"this colour sits between two notes: {distance:.1f} from {name}, "
                f"{runner_distance:.1f} from {other}")

    def note_label(self, index: int) -> str:
        return self.notes[index].name if 0 <= index < len(self.notes) else str(index)

    def confusion_warning(self, index: int) -> str:
        """A sentence when the active viewer type would mix this note up."""
        mode = self.settings.cvd_mode
        if mode in ("none", "") or not self.settings.confusion_warning:
            return ""
        table = self.confusion.get(mode)
        if table is None or not table.is_ambiguous(index):
            return ""
        partner = table.partner(index)
        partner_name = self.notes[partner].name if partner < len(self.notes) else str(partner)
        motif = PATTERN_NAMES[self.palette.pattern(index) % len(PATTERN_NAMES)]
        return (f"under {mode} this also looks like {partner_name} "
                f"(dE {table.distance[index]:.1f}) -- go by pattern '{motif}'")

    # ------------------------------------------------------------------ playing
    def play_if_changed(self, index: int, brightness: float) -> tuple[bool, bool]:
        """Play *index* if it is new and the debounce has elapsed.

        The original fired a note every 0.9 s whether or not anything had
        changed, so holding one object still produced a note every second.  This
        plays on *change*, and only retriggers a held note when sustain is on.
        """
        now = time.time()
        changed = index != self.last_played_index
        if not changed and not self.settings.sustain:
            return False, changed
        elapsed = now - self.last_play_time
        if elapsed < self.settings.note_interval:
            return False, changed

        velocity = 1.0
        if self.settings.velocity_from_brightness:
            # Quiet for dark objects, loud for bright ones: a second cue for
            # the lightness axis, which every viewer keeps.
            velocity = float(np.clip(0.35 + brightness * 0.85, 0.15, 1.0))

        far = changed and self.last_played_index >= 0 and abs(index - self.last_played_index) > 10
        if far and self.settings.glide:
            self.audio.arpeggio(self.last_played_index, index, steps=2, velocity=velocity * 0.8)
        else:
            self.audio.play(index, velocity)

        self.last_played_index = index
        self.last_play_time = now
        return True, changed

    # ------------------------------------------------------------------ gestures
    def apply_gestures(self) -> None:
        """Read the hand pose, aim the box, and fire discrete commands."""
        state = self.gesture
        if self.tracker is None or not self.tracker.available:
            state.tracking = False
            return

        pose = self.tracker.process(self.frame)
        self.tracker.draw(self.frame)
        state.tracking = self.tracker.hand_present
        state.label = pose.gesture.value
        state.action = ""

        now = time.time()
        gesture, fired = self.gesture_stabilizer.update(pose.gesture, now)
        state.active = gesture != Gesture.NONE

        # Continuous gestures -------------------------------------------------
        if gesture == Gesture.OPEN_PALM:
            self.paused = True
            state.action = "tracking paused"
        else:
            self.paused = False

        if gesture == Gesture.POINT and self.tracker.hand_present:
            target = pose.point
            self.steer_roi(target[0], target[1], rate=0.18)
            state.action = "aiming"

        if gesture == Gesture.PINCH and self.tracker.hand_present:
            self.steer_roi(pose.pinch_point[0], pose.pinch_point[1], rate=0.5)
            state.action = "jump"

        # Discrete gestures ---------------------------------------------------
        if not fired:
            return
        if gesture == Gesture.THUMBS_UP:
            self.set_volume(self.volume + 0.05)
            state.action = f"volume {self.volume * 100:.0f}%"
        elif gesture == Gesture.THUMBS_DOWN:
            self.set_volume(self.volume - 0.05)
            state.action = f"volume {self.volume * 100:.0f}%"
        elif gesture == Gesture.PEACE:
            self.cycle_bank()
            state.action = f"timbre {self.audio.bank}"
        elif gesture == Gesture.FIST:
            self.audio.stop_all()
            state.action = "stopped"

    def steer_roi(self, x: float, y: float, rate: float = 0.2) -> None:
        self.roi[0] += (float(x) - self.roi[0]) * rate
        self.roi[1] += (float(y) - self.roi[1]) * rate
        self.roi[0] = float(np.clip(self.roi[0], 0.02, 0.98))
        self.roi[1] = float(np.clip(self.roi[1], 0.02, 0.98))

    # ------------------------------------------------------------------ loop
    def step(self) -> bool:
        """Process one frame.  Returns ``False`` when the source is exhausted."""
        ok, frame = self.source.read()
        if not ok or frame is None:
            return False
        self.frame = frame
        self.frame_index += 1
        height, width = frame.shape[:2]
        result = FrameResult()

        # 1. gestures ------------------------------------------------------- #
        if self.tracker is not None and self.tracker.available:
            self.apply_gestures()

        center = (int(self.roi[0] * width), int(self.roi[1] * height))

        # 2. sample --------------------------------------------------------- #
        gains = None
        if self.settings.white_balance:
            gains = gray_world_gains(frame)
        sample = sample_box(frame, center[0], center[1], self.roi_size, gains=gains)
        result.sample_bgr = tuple(int(v) for v in sample.bgr)
        result.brightness = sample.brightness

        index = self.last_played_index if self.last_played_index >= 0 else 0
        if self.paused:
            result.skipped = "paused (open palm)"
        elif not sample.valid:
            result.skipped = sample.reason
        else:
            # Smooth *then* decide: the smoothing is what stops a noisy camera
            # from stuttering between neighbouring notes, so it has to come
            # before the nearest-colour search, not after it.  (The original
            # smoothed the resulting *index*, which on a hue wheel that wraps
            # produced notes that were never on screen.)
            smoothed = self.smoother.push(bgr_to_lab(sample.bgr.astype(np.float64)))
            if smoothed.chroma < self.settings.min_chroma or \
                    smoothed.lightness < config.MIN_LIGHTNESS:
                # Grey, black or white.  The original mapped these to whichever
                # hue was numerically nearest, so a blank wall played arbitrary
                # notes.  The history is dropped so the next real colour is not
                # dragged towards grey.
                result.skipped = "no colour in the box (grey / black / white)"
                self.smoother.reset()
            else:
                raw_index, distance, runner_index, runner_distance = \
                    self.choose_index(smoothed.lab)
                index, changed = self.stabilizer.update(raw_index)
                result.index = index
                result.distance = distance
                result.runner_up = runner_distance
                result.changed = changed
                result.note_name = self.note_label(index)
                result.color_name = color_name(lab_to_bgr(smoothed.lab))
                result.warning = (self.confusion_warning(index)
                                  or self.generic_warning(index, distance,
                                                          runner_index, runner_distance))
                result.played, _ = self.play_if_changed(index, sample.brightness)
                if result.played and self.announcer.enabled:
                    self.announcer.announce(index, result.note_name, result.color_name,
                                            self.palette.pattern(index))

        self.result = result

        # 3. overlay -------------------------------------------------------- #
        display = self.render(frame, result, center)
        if self.writer is not None:
            self.writer.write(display)
        if not self.settings.headless:
            cv2.imshow(self.window, display)

        self._tick_fps()
        return True

    def apply_vision_filters(self, frame: np.ndarray) -> np.ndarray:
        """Apply daltonization and/or CVD simulation to the preview.

        These are per-pixel colour transforms, so they cost the same whatever
        the pixels contain.  Running them at full 720p cost about 210 ms per
        frame with both enabled -- under 5 fps.  They are therefore evaluated at
        ``filter_scale`` resolution and scaled back up: the aid is about colour
        relationships, which survive the resampling, and the cost falls with the
        square of the scale.
        """
        settings = self.settings
        mode = settings.cvd_mode
        daltonize = settings.daltonize and mode not in ("none", "")
        simulate = settings.simulate_cvd and mode not in ("none", "")
        if not (daltonize or simulate):
            return frame

        scale = float(np.clip(settings.filter_scale, 0.1, 1.0))
        height, width = frame.shape[:2]
        if scale < 0.999:
            work = cv2.resize(frame, (max(1, int(width * scale)), max(1, int(height * scale))),
                              interpolation=cv2.INTER_AREA)
        else:
            work = frame

        if daltonize:
            work = colorblind.daltonize(work, mode, settings.daltonize_strength)
        if simulate:
            work = colorblind.simulate(work, mode)

        if scale < 0.999:
            return cv2.resize(work, (width, height), interpolation=cv2.INTER_LINEAR)
        return work

    def render(self, frame: np.ndarray, result: FrameResult,
               center: tuple[int, int]) -> np.ndarray:
        """Everything the user sees."""
        settings = self.settings
        scale = 1.35 if settings.large_text else 1.0
        height, width = frame.shape[:2]

        # Filters are for the *user's* eyes only; note selection uses the raw
        # frame, as documented in choose_index.
        display = self.apply_vision_filters(frame)

        index = result.index if result.index >= 0 else max(0, self.last_played_index)

        # Sampling box, then the overlay furniture.
        hud.draw_sampling_box(
            display, center, self.roi_size, self.palette,
            index if result.index >= 0 else None,
            patterns=settings.patterns,
            show_pattern=settings.patterns,
            warning=bool(result.skipped),
        )
        hud.draw_aiming_cross(display, center, result.skipped == "paused (open palm)")

        # ---------------------------------------------------------------- #
        # Layout.  Left column stacks downwards; reference panels sit on the
        # right; the keyboard owns the bottom strip.  Everything is placed from
        # the previous element's reported edge rather than from a hard-coded y,
        # because panels change height with the text scale and the note count.
        # ---------------------------------------------------------------- #
        left = 12
        top = 12

        if settings.show_names:
            note = self.notes[index] if index < len(self.notes) else None
            top = hud.note_panel(
                display, (left, top), self.palette, index, note,
                result.color_name or "--",
                PATTERN_NAMES[self.palette.pattern(index) % len(PATTERN_NAMES)],
                settings.patterns, scale=scale, warning=result.warning,
            )
        else:
            top += int(70 * scale)

        if settings.gestures:
            top += 8
            hud.gesture_panel(
                display, (left + 2, top + 14), self.gesture.label, self.gesture.action,
                self.gesture.active, self.tracker is not None and self.tracker.available,
                "" if self.tracker is None else self.tracker.reason, scale=scale,
            )
            top += 22

        # Brightness gauge under the gesture readout: lightness is the axis every
        # viewer keeps, so it gets a permanent gauge of its own.
        bar_height = int(max(90, min(180, height // 4)))
        hud.brightness_bar(display, (left + 8, top + 14), bar_height,
                           result.brightness, scale=scale)

        # Status, top right.
        rows = [
            ("timbre", self.audio.bank),
            ("palette", self.palette.name),
            ("vision", settings.cvd_mode),
            ("volume", f"{self.volume * 100:.0f}%" + (" (muted)" if self.muted else "")),
        ]
        if settings.daltonize:
            rows.append(("daltonize", f"on ({settings.daltonize_strength:.1f})"))
        if settings.simulate_cvd:
            rows.append(("simulate", "on (this is what they see)"))
        if settings.range_mode != "white":
            rows.append(("range", f"{settings.range_mode} from {self.notes[0].name}"))
        rows.append(("note", result.note_name or "-"))
        if result.distance:
            rows.append(("dE / next", f"{result.distance:.1f} / {result.runner_up:.1f}"))
        if result.skipped:
            rows.append(("skipped", result.skipped))
        hud.status_panel(display, (width - 14, 12), rows, scale=scale)

        # The bottom strip, then the panels that live above it.
        strip_top = height
        if settings.show_keyboard:
            strip_height = int(66 * scale)
            strip_top = height - strip_height
            hud.keyboard_strip(
                display, (0, strip_top, width, height),
                self.notes, self.palette, index, settings.patterns, scale=scale,
            )

        # Reference text lives at the bottom left and bottom right, one panel per
        # side, so the key list can never land on top of the pattern legend.
        #
        # These two grow more slowly than the rest: at full large-text scale the
        # key list is wide enough to cover the middle of the frame, which is
        # exactly where the sampling box has to be.  The readouts that matter --
        # note, colour name, keyboard -- still scale all the way.
        reference_scale = min(scale, 1.15)
        hud.legend(display, (left, strip_top - int(126 * reference_scale)),
                   self.palette, settings.patterns, scale=reference_scale)
        if settings.show_help:
            hud.help_panel(display, (width - 12, strip_top - 8), self.help_rows,
                           scale=reference_scale, anchor="right")

        if result.warning:
            hud.warning_banner(display, result.warning, scale=scale)

        if settings.show_fps:
            hud.fps_badge(display, (width - 14, height - 10), self.fps)

        if settings.high_contrast:
            cv2.rectangle(display, (0, 0), (width - 1, height - 1), (255, 255, 255), 3)
        return display

    def _tick_fps(self) -> None:
        now = time.perf_counter()
        self._fps_window.append(now)
        cutoff = now - 1.0
        while self._fps_window and self._fps_window[0] < cutoff:
            self._fps_window.pop(0)
        self.fps = float(len(self._fps_window))

    # ------------------------------------------------------------------ keys
    #: Extended key codes for the arrow keys.  These are checked *before* the
    #: value is masked to a byte, because masking turns them all into 0 -- and
    #: the Linux values (81..84) would then collide with the letters Q/R/S/T.
    ARROWS = {
        2424832: "left", 65361: "left",
        2555904: "right", 65363: "right",
        2490368: "up", 65362: "up",
        2621440: "down", 65364: "down",
    }

    def handle_key(self, raw_key: int) -> bool:
        """Apply a keypress.  Returns ``False`` to quit.

        ``raw_key`` must be the value straight from ``waitKey``, unmasked.
        """
        settings = self.settings
        arrow = self.ARROWS.get(raw_key)
        if arrow:
            delta = {"left": (-0.02, 0.0), "right": (0.02, 0.0),
                     "up": (0.0, -0.02), "down": (0.0, 0.02)}[arrow]
            self.roi[0] = float(np.clip(self.roi[0] + delta[0], 0.02, 0.98))
            self.roi[1] = float(np.clip(self.roi[1] + delta[1], 0.02, 0.98))
            return True

        key = raw_key & 0xFF
        if raw_key < 0:
            return True
        if key in (27, ord("q")):                       # esc / q
            return False
        if key == ord("h"):
            settings.show_help = not settings.show_help
        elif key == ord(" "):
            settings.sustain = not settings.sustain
            print(f"[sustain] {'on' if settings.sustain else 'off'}")
        elif key == ord("b"):
            self.cycle_bank()
        elif key == ord("p"):
            self.cycle_palette()
        elif key == ord("c"):
            self.cycle_cvd()
        elif key == ord("d"):
            settings.daltonize = not settings.daltonize
            print(f"[daltonize] {'on' if settings.daltonize else 'off'}")
        elif key == ord("s"):
            settings.simulate_cvd = not settings.simulate_cvd
            print(f"[simulate] {'on' if settings.simulate_cvd else 'off'}")
        elif key == ord("t"):
            settings.patterns = not settings.patterns
        elif key == ord("n"):
            settings.show_names = not settings.show_names
        elif key == ord("k"):
            settings.show_keyboard = not settings.show_keyboard
        elif key == ord("a"):
            enabled = self.announcer.toggle()
            kind = "speech" if self.announcer.speaking_available else "earcons"
            print(f"[announce] {'on' if enabled else 'off'} using {kind}")
        elif key == ord("g"):
            self.toggle_gestures()
        elif key == ord("L"):
            settings.large_text = not settings.large_text
        elif key == ord("m"):
            self.set_muted(not self.muted)
        elif key in (ord("+"), ord("=")):
            self.set_volume(self.volume + 0.05)
        elif key in (ord("-"), ord("_")):
            self.set_volume(self.volume - 0.05)
        elif key == ord("r"):
            self.roi = [0.5, 0.5]
            self.roi_size = int(Settings().roi_size)
            self.smoother.reset()
            self.stabilizer.reset()
        elif key == ord("["):
            self.roi_size = int(max(config.ROI_SIZE_MIN, self.roi_size - 5))
        elif key == ord("]"):
            self.roi_size = int(min(config.ROI_SIZE_MAX, self.roi_size + 5))
        elif ord("1") <= key <= ord("5"):
            settings.cvd_mode = colorblind.CVD_MODES[key - ord("1")]
            print(f"[vision] {settings.cvd_mode}")
        elif key == ord("0"):
            settings.cvd_mode = "none"
        return True

    def cycle_bank(self) -> str:
        self.audio.bank = BANKS[(BANKS.index(self.audio.bank) + 1) % len(BANKS)]
        print(f"[timbre] {self.audio.bank} -- {BANK_DESCRIPTIONS[self.audio.bank]}")
        return self.audio.bank

    def cycle_palette(self) -> str:
        names = ["hue", "cvd_safe"]
        current = self.palette.name if self.palette.name in names else names[0]
        return self.set_palette(names[(names.index(current) + 1) % len(names)])

    def set_palette(self, name: str) -> str:
        self.palette = build_palette(name, len(self.notes))
        self.confusion = colorblind.build_confusion_tables(self.palette.colors)
        self.stabilizer.reset()
        self.smoother.reset()
        self.bind_source_palette()
        print(f"[palette] {self.palette.name}")
        return self.palette.name

    def cycle_cvd(self) -> str:
        modes = colorblind.CVD_MODES
        current = modes.index(self.settings.cvd_mode) if self.settings.cvd_mode in modes else 0
        self.settings.cvd_mode = modes[(current + 1) % len(modes)]
        print(f"[vision] {self.settings.cvd_mode}")
        return self.settings.cvd_mode

    def set_volume(self, value: float) -> float:
        self.volume = float(np.clip(value, 0.0, 1.0))
        self.audio.set_volume(self.volume)
        return self.volume

    def set_muted(self, muted: bool) -> bool:
        self.muted = bool(muted)
        self.audio.set_muted(self.muted)
        return self.muted

    def toggle_gestures(self) -> bool:
        if self.tracker is None:
            self.tracker = HandTracker(confidence=self.settings.gesture_confidence)
        else:
            if self.tracker.available:
                self.tracker.close()
            else:
                self.tracker = HandTracker(confidence=self.settings.gesture_confidence)
        self.settings.gestures = self.tracker.available
        print(f"[gestures] {'on' if self.tracker.available else 'off'}"
              + ("" if self.tracker.available else f" ({self.tracker.reason})"))
        return self.tracker.available

    def on_mouse(self, event: int, x: int, y: int, flags: int, param) -> None:
        """Click to place the sampling box; wheel to resize it."""
        if event == cv2.EVENT_LBUTTONDOWN:
            self.steer_roi(x / max(1, self.frame.shape[1]),
                           y / max(1, self.frame.shape[0]), rate=1.0)
        elif event == cv2.EVENT_MOUSEWHEEL:
            step = 5 if flags > 0 else -5
            self.roi_size = int(np.clip(self.roi_size + step,
                                        config.ROI_SIZE_MIN, config.ROI_SIZE_MAX))

    # ------------------------------------------------------------------ run
    def run(self) -> FrameResult:
        print(self.banner())
        if not self.settings.headless:
            cv2.namedWindow(self.window, cv2.WINDOW_NORMAL)
            cv2.setMouseCallback(self.window, self.on_mouse)
        if self.settings.record:
            self._open_writer()

        started = time.time()
        try:
            while self.running and self.step():
                if not self.settings.headless:
                    key = cv2.waitKey(1)
                    if not self.handle_key(key):
                        break
                if self.settings.duration and time.time() - started >= self.settings.duration:
                    print(f"[run] stopping after {self.settings.duration:.0f}s")
                    break
        except KeyboardInterrupt:
            print("\n[run] interrupted")
        finally:
            self.print_summary(time.time() - started)
            self.shutdown()
        return self.result

    def print_summary(self, elapsed: float) -> None:
        """Say what actually happened.  Silence on exit is how bugs hide."""
        played = getattr(self.audio, "played", None)
        lines = [
            "-" * 72,
            f" processed {self.frame_index} frames in {elapsed:.1f}s "
            f"({self.frame_index / max(elapsed, 1e-6):.1f} fps)",
            f" last note  {self.result.note_name or '-'}"
            + (f"  ({self.result.color_name})" if self.result.color_name else ""),
        ]
        if played is not None:
            lines.append(f" notes played {len(played)} ({len(set(played))} distinct)")
        if self.result.skipped:
            lines.append(f" last frame skipped: {self.result.skipped}")
        lines.append("=" * 72)
        print("\n".join(lines))

    def _open_writer(self) -> None:
        frame = self.frame if hasattr(self, "frame") else None
        if frame is None:
            ok, frame = self.source.read()
            if not ok:
                return
        height, width = frame.shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        self.writer = cv2.VideoWriter(self.settings.record, fourcc, 25.0, (width, height))
        print(f"[record] writing {self.settings.record}")

    def banner(self) -> str:
        lines = [
            "=" * 72,
            f" {config.APP_NAME} {config.APP_VERSION}",
            "=" * 72,
            f" notes      {describe_table(self.notes)}",
            f" palette    {self.palette.name} -- {self.palette.description}",
            f" colour     {colorblind.CVD_LABELS[self.settings.cvd_mode]}"
            + (" (simulated on screen)" if self.settings.simulate_cvd else ""),
            f" vision aid daltonize {'on' if self.settings.daltonize else 'off'}, "
            f"patterns {'on' if self.settings.patterns else 'off'}, "
            f"names {'on' if self.settings.show_names else 'off'}",
            f" audio      {self.audio.status.driver}"
            + (f" -- {self.audio.status.reason}" if self.audio.status.reason else ""),
            f" gestures   {'available' if self.tracker and self.tracker.available else 'off'}",
            f" source     {self.source.description}",
            "-" * 72,
            " Aim the box at an object; the colour picks the note.",
            " Press h for the key list, q to quit.",
            "=" * 72,
        ]
        return "\n".join(lines)

    def shutdown(self) -> None:
        self.running = False
        try:
            self.source.release()
        except Exception:
            pass
        if self.writer is not None:
            self.writer.release()
            self.writer = None
        self.announcer.close()
        self.audio.shutdown()
        if self.tracker is not None:
            self.tracker.close()
        if not self.settings.headless:
            cv2.destroyAllWindows()
