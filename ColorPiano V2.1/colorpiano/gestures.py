"""Hand tracking and gesture recognition.

What the original did, and why it had to go
-------------------------------------------
Every version of this project detected volume gestures with::

    if landmarks[4].y < landmarks[3].y < landmarks[2].y:      # "volume up"
    elif landmarks[4].y > landmarks[3].y > landmarks[2].y:    # "volume down"

Those three points are the thumb tip, thumb IP and thumb MCP.  The test is
therefore "is the thumb pointing up", evaluated with no reference to the rest of
the hand -- so it fires while the hand is merely rotating, fires for a raised
index finger whenever the thumb happens to trail behind it, and cannot tell
"thumb up" from "whole hand up".  It is also unnormalised, so it behaves
differently depending on how far away the hand is, and it is applied on every
single frame with a three-second cooldown bolted on to hide the flicker.

Here every gesture is defined against the *pose of the whole hand*, in units of
the hand's own size, and must be held for several consecutive frames before it
is accepted.  That makes a gesture mean one thing, once, on purpose.
"""

from __future__ import annotations

import os
import shutil
import sys
import tempfile
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path

import numpy as np

#: MediaPipe hand landmark indices, named.
WRIST = 0
THUMB_CMC, THUMB_MCP, THUMB_IP, THUMB_TIP = 1, 2, 3, 4
INDEX_MCP, INDEX_PIP, INDEX_DIP, INDEX_TIP = 5, 6, 7, 8
MIDDLE_MCP, MIDDLE_PIP, MIDDLE_DIP, MIDDLE_TIP = 9, 10, 11, 12
RING_MCP, RING_PIP, RING_DIP, RING_TIP = 13, 14, 15, 16
PINKY_MCP, PINKY_PIP, PINKY_DIP, PINKY_TIP = 17, 18, 19, 20

FINGERS = {
    "index": (INDEX_MCP, INDEX_PIP, INDEX_DIP, INDEX_TIP),
    "middle": (MIDDLE_MCP, MIDDLE_PIP, MIDDLE_DIP, MIDDLE_TIP),
    "ring": (RING_MCP, RING_PIP, RING_DIP, RING_TIP),
    "pinky": (PINKY_MCP, PINKY_PIP, PINKY_DIP, PINKY_TIP),
}


class Gesture(str, Enum):
    NONE = "none"
    OPEN_PALM = "open_palm"
    FIST = "fist"
    PINCH = "pinch"
    POINT = "point"
    PEACE = "peace"
    THUMBS_UP = "thumbs_up"
    THUMBS_DOWN = "thumbs_down"


#: What each gesture does, and how it behaves.  Single source of truth: the app
#: builds its on-screen gesture list from this, so the help panel cannot drift
#: out of step with what the gestures actually do.
GESTURE_ACTIONS = {
    Gesture.THUMBS_UP: ("thumbs up", "volume up (hold to keep raising)"),
    Gesture.THUMBS_DOWN: ("thumbs down", "volume down (hold to keep lowering)"),
    Gesture.PEACE: ("peace sign", "switch timbre"),
    Gesture.FIST: ("fist", "stop all sound"),
    Gesture.OPEN_PALM: ("open palm", "pause colour tracking"),
    Gesture.POINT: ("point", "aim the sampling box"),
    Gesture.PINCH: ("pinch", "jump the sampling box here"),
}


def gesture_help_rows() -> list[tuple[str, str]]:
    """The gesture bindings, as ``(name, what it does)`` pairs."""
    return [entry for entry in GESTURE_ACTIONS.values()]


def _distance(a, b) -> float:
    return float(np.hypot(a.x - b.x, a.y - b.y))


def _extended(landmarks, name: str, wrist, scale: float) -> bool:
    """Is this finger straight?

    Measured as "the tip is further from the wrist than the middle joint", which
    is scale free and rotation free -- unlike comparing ``y`` coordinates, which
    only works while the hand is bolt upright.
    """
    mcp, _pip, _dip, tip = FINGERS[name]
    return _distance(landmarks[tip], wrist) > _distance(landmarks[mcp], wrist) + 0.35 * scale


def _thumb_extended(landmarks, scale: float) -> bool:
    """The thumb needs its own test: it does not lie along a knuckle line."""
    return _distance(landmarks[THUMB_TIP], landmarks[INDEX_MCP]) > 1.15 * scale


@dataclass
class HandPose:
    """A recognised pose plus the numbers used to reach it."""

    gesture: Gesture = Gesture.NONE
    extended: dict[str, bool] = field(default_factory=dict)
    thumb_extended: bool = False
    pinch_distance: float = 1.0
    scale: float = 1.0
    #: Tip of the index finger, normalised 0..1 in frame coordinates.
    point: tuple[float, float] = (0.5, 0.5)
    #: Midpoint of thumb and index tips -- where a pinch "is".
    pinch_point: tuple[float, float] = (0.5, 0.5)
    #: Palm centre, normalised, used for coarse aiming.
    palm: tuple[float, float] = (0.5, 0.5)


def classify(landmarks) -> HandPose:
    """Turn 21 landmarks into a :class:`HandPose`."""
    wrist = landmarks[WRIST]
    # Hand size reference: wrist to middle knuckle.  Everything below is
    # expressed as a multiple of this, so distance from the camera cancels out.
    scale = max(_distance(wrist, landmarks[MIDDLE_MCP]), 1e-6)

    extended = {name: _extended(landmarks, name, wrist, scale) for name in FINGERS}
    thumb = _thumb_extended(landmarks, scale)
    pinch = _distance(landmarks[THUMB_TIP], landmarks[INDEX_TIP]) / scale

    fingers_up = sum(extended.values())
    index_up = extended["index"]
    middle_up = extended["middle"]

    # A pinch needs the index finger *straight* and its tip meeting the thumb.
    # Without that requirement a closed fist -- where the thumb naturally rests
    # against the curled index -- reads as a pinch, every time.
    if pinch < 0.55 and extended["index"] and not extended["middle"] \
            and not extended["ring"]:
        gesture = Gesture.PINCH
    elif fingers_up == 4:
        gesture = Gesture.OPEN_PALM
    elif fingers_up == 0 and thumb:
        # Direction of the thumb relative to the hand, in image coordinates
        # (y grows downwards).  Normalised by hand size, so it survives being
        # close to or far from the camera.
        rise = (wrist.y - landmarks[THUMB_TIP].y) / scale
        gesture = Gesture.THUMBS_UP if rise > 0.6 else (
            Gesture.THUMBS_DOWN if rise < -0.6 else Gesture.FIST
        )
    elif fingers_up == 0:
        gesture = Gesture.FIST
    elif index_up and middle_up and not extended["ring"] and not extended["pinky"]:
        gesture = Gesture.PEACE
    elif index_up and not middle_up and not extended["ring"] and not extended["pinky"]:
        gesture = Gesture.POINT
    else:
        gesture = Gesture.NONE

    return HandPose(
        gesture=gesture,
        extended=extended,
        thumb_extended=thumb,
        pinch_distance=pinch,
        scale=scale,
        point=(landmarks[INDEX_TIP].x, landmarks[INDEX_TIP].y),
        pinch_point=((landmarks[THUMB_TIP].x + landmarks[INDEX_TIP].x) / 2.0,
                     (landmarks[THUMB_TIP].y + landmarks[INDEX_TIP].y) / 2.0),
        palm=((wrist.x + landmarks[MIDDLE_MCP].x) / 2.0,
              (wrist.y + landmarks[MIDDLE_MCP].y) / 2.0),
    )


#: Gestures whose whole point is to be repeated while held (you ramp the volume
#: by holding your thumb up).  Everything else is a *toggle*, and firing it
#: twice because the hand stayed put would be a bug: the gesture has to be
#: released and re-made before it acts again.
REPEATABLE = {Gesture.THUMBS_UP, Gesture.THUMBS_DOWN}

#: Gestures that mean something continuously rather than as a single action.
CONTINUOUS = {Gesture.NONE, Gesture.POINT, Gesture.OPEN_PALM}


class GestureStabilizer:
    """Requires a gesture to persist before it counts, and debounces repeats.

    ``hold_frames`` consecutive agreeing frames make a gesture *active*.  A
    repeatable gesture may then fire again once ``repeat_cooldown`` has passed;
    a toggle fires once and re-arms only after the gesture is dropped.  This is
    the piece that turns a jittery per-frame classifier into something a person
    can actually operate.
    """

    def __init__(self, hold_frames: int = 5, repeat_cooldown: float = 0.7) -> None:
        self.hold_frames = max(1, int(hold_frames))
        self.repeat_cooldown = float(repeat_cooldown)
        self.gesture = Gesture.NONE
        self._candidate = Gesture.NONE
        self._count = 0
        self._last_fired: dict[Gesture, float] = {}
        #: Gestures allowed to fire.  A toggle is removed when it fires and put
        #: back when the hand returns to a rest pose, so holding it does nothing
        #: more and making it again does something new.
        self._armed: set[Gesture] = set(Gesture)
        self.changed_at = 0.0

    def update(self, gesture: Gesture, now: float) -> tuple[Gesture, bool]:
        """Return ``(active_gesture, should_fire_now)``."""
        if gesture != self._candidate:
            self._candidate = gesture
            self._count = 1
            if gesture in CONTINUOUS:
                self._armed = set(Gesture)
            return self.gesture, False
        self._count += 1
        if self._count < self.hold_frames:
            return self.gesture, False

        if gesture != self.gesture:
            self.gesture = gesture
            self.changed_at = now

        if gesture in CONTINUOUS:
            # Read every frame rather than fired as an event.
            return self.gesture, False
        if gesture not in self._armed:
            return self.gesture, False

        last = self._last_fired.get(gesture, -1e9)
        if now - last < self.repeat_cooldown:
            return self.gesture, False
        self._last_fired[gesture] = now
        if gesture not in REPEATABLE:
            self._armed.discard(gesture)
        return self.gesture, True

    def reset(self) -> None:
        self.gesture = Gesture.NONE
        self._candidate = Gesture.NONE
        self._count = 0
        self._armed = set(REPEATABLE)


# --------------------------------------------------------------------------- #
# The non-ASCII install path problem
# --------------------------------------------------------------------------- #
def ascii_staging_dir() -> Path | None:
    """A writable directory whose path is guaranteed to be ASCII-only."""
    candidates = [
        Path(tempfile.gettempdir()) / "colorpiano_mediapipe",
        Path(os.environ.get("PROGRAMDATA", "C:/ProgramData")) / "ColorPiano" / "mediapipe",
        Path("C:/colorpiano_mediapipe"),
    ]
    for candidate in candidates:
        try:
            if not str(candidate).isascii():
                continue
            candidate.mkdir(parents=True, exist_ok=True)
            probe = candidate / ".writable"
            probe.write_text("ok", encoding="ascii")
            probe.unlink()
            return candidate
        except OSError:
            continue
    return None


def locate_mediapipe() -> Path | None:
    """Find the installed mediapipe package *without importing it*.

    The staging below has to happen before the first import: mediapipe's native
    extension modules register their resource directory when they load, and a
    second import of the same process reuses the already-loaded native module,
    so swapping ``sys.path`` afterwards has no effect.
    """
    for entry in sys.path:
        if not entry:
            continue
        candidate = Path(entry) / "mediapipe" / "__init__.py"
        try:
            if candidate.is_file():
                return candidate.parent
        except OSError:
            continue
    return None


def stage_mediapipe(source: Path, verbose: bool = True) -> Path | None:
    """Copy the mediapipe package somewhere ASCII; return the staging root.

    Only needed on Windows, and only when the interpreter lives under a path
    containing non-ASCII characters -- which is extremely common, because
    "make a copy of this folder" produces names like ``project - 副本``.

    MediaPipe's Python layer hands the path of its own ``.binarypb`` model down
    into C++, where it is opened through a narrow-character API.  The file is
    right there, but the open fails, and the error reads::

        The path does not exist: .../mediapipe/modules/hand_landmark/
        hand_landmark_tracking_cpu.binarypb

    That message -- about a file that demonstrably exists -- is why the original
    project, whose folder is literally named
    ``color_piano hand gestures - 副本``, printed "手势识别功能已禁用" and
    shipped a hand-gesture feature that could never run.
    """
    staging = ascii_staging_dir()
    if staging is None:
        return None
    target = staging / "mediapipe"
    marker = staging / ".mediapipe-version"
    version = "unknown"
    init = source / "__init__.py"
    try:
        for line in init.read_text(encoding="utf-8", errors="ignore").splitlines():
            if line.startswith("__version__"):
                version = line.split("=", 1)[1].strip().strip("'\"")
                break
    except OSError:
        pass
    try:
        if not target.is_dir() or not marker.is_file() or \
                marker.read_text(encoding="ascii").strip() != version:
            shutil.rmtree(target, ignore_errors=True)
            shutil.copytree(source, target)
            marker.write_text(version, encoding="ascii")
    except OSError:
        return None
    if verbose:
        print(f"[gestures] mediapipe lives under a non-ASCII path; staged a copy "
              f"at {target}")
    return staging


def load_mediapipe(repair_path: bool = True, verbose: bool = True):
    """Import mediapipe, working around a non-ASCII install path.

    Returns ``(module, repaired, reason)``.
    """
    located = locate_mediapipe()
    if located is None:
        return None, False, "mediapipe is not installed"
    repaired = False
    if repair_path and not str(located).isascii():
        staging = stage_mediapipe(located, verbose)
        if staging is not None:
            sys.path.insert(0, str(staging))
            repaired = True
    try:
        import importlib

        module = importlib.import_module("mediapipe")
    except Exception as error:
        return None, repaired, f"mediapipe would not import ({error})"
    return module, repaired, ""


# --------------------------------------------------------------------------- #
# Hand model for the modern (Tasks) API
# --------------------------------------------------------------------------- #
#: Where the Tasks API expects its model bundle.  MediaPipe 1.x ships no model
#: files at all, so it has to be fetched once; ``tools/fetch_model.py`` does it.
MODEL_FILENAME = "hand_landmarker.task"
MODEL_URL = ("https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
             "hand_landmarker/float16/1/hand_landmarker.task")


def find_hand_model() -> Path | None:
    """Look for the Tasks-API model bundle in the usual places."""
    from .config import ASSETS_DIR

    candidates = []
    env = os.environ.get("COLORPIANO_HAND_MODEL")
    if env:
        candidates.append(Path(env))
    candidates += [
        ASSETS_DIR / "models" / MODEL_FILENAME,
        ASSETS_DIR / MODEL_FILENAME,
        Path(__file__).resolve().parent.parent / MODEL_FILENAME,
    ]
    for candidate in candidates:
        try:
            if candidate.is_file() and candidate.stat().st_size > 100_000:
                return candidate
        except OSError:
            continue
    return None


class _LegacyBackend:
    """``mp.solutions.hands`` -- present up to mediapipe 0.10.x, gone in 1.0."""

    name = "solutions.hands"

    def __init__(self, module, max_hands: int, confidence: float,
                 model_complexity: int) -> None:
        solutions = module.solutions
        self._hands = solutions.hands.Hands(
            static_image_mode=False,
            max_num_hands=max_hands,
            model_complexity=model_complexity,
            min_detection_confidence=confidence,
            min_tracking_confidence=max(0.3, confidence - 0.2),
        )
        self._drawing = solutions.drawing_utils
        self._connections = solutions.hands.HAND_CONNECTIONS

    def detect(self, frame):
        import cv2

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rgb.flags.writeable = False
        result = self._hands.process(rgb)
        if not result.multi_hand_landmarks:
            return None
        return result.multi_hand_landmarks[0]

    def landmarks(self, hand):
        return hand.landmark

    def draw(self, frame, hand) -> None:
        self._drawing.draw_landmarks(
            frame, hand, self._connections,
            self._drawing.DrawingSpec(color=(0, 255, 255), thickness=2, circle_radius=3),
            self._drawing.DrawingSpec(color=(255, 255, 255), thickness=1),
        )

    def close(self) -> None:
        try:
            self._hands.close()
        except Exception:
            pass


class _TasksBackend:
    """``mediapipe.tasks.python.vision.HandLandmarker`` -- the 1.x API.

    Differences that matter: it takes an ``mp.Image`` rather than a numpy array,
    needs strictly increasing timestamps in video mode, and returns a plain list
    of landmarks rather than a wrapper object.  The landmark coordinates are the
    same normalised ``x``/``y``, so :func:`classify` is unchanged.
    """

    name = "tasks.HandLandmarker"

    def __init__(self, module, model_path: Path, max_hands: int,
                 confidence: float, _complexity: int = 0) -> None:
        # Imported here rather than at module scope so that a machine without
        # mediapipe never touches this path at all.
        from mediapipe.tasks.python import BaseOptions, vision

        options = vision.HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(model_path)),
            running_mode=vision.RunningMode.VIDEO,
            num_hands=max_hands,
            min_hand_detection_confidence=confidence,
            min_hand_presence_confidence=max(0.3, confidence - 0.2),
            min_tracking_confidence=max(0.3, confidence - 0.2),
        )
        self._module = module
        self._landmarker = vision.HandLandmarker.create_from_options(options)
        self._connections = getattr(
            getattr(vision, "HandLandmarksConnections", None), "HAND_CONNECTIONS", []
        )
        self._last_timestamp = -1

    def _timestamp(self) -> int:
        stamp = int(time.monotonic() * 1000)
        if stamp <= self._last_timestamp:
            stamp = self._last_timestamp + 1
        self._last_timestamp = stamp
        return stamp

    def detect(self, frame):
        import cv2

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = self._module.Image(image_format=self._module.ImageFormat.SRGB,
                                   data=rgb)
        result = self._landmarker.detect_for_video(image, self._timestamp())
        if not result.hand_landmarks:
            return None
        return result.hand_landmarks[0]

    def landmarks(self, hand):
        return hand

    def draw(self, frame, hand) -> None:
        import cv2

        height, width = frame.shape[:2]
        points = [(int(lm.x * width), int(lm.y * height)) for lm in hand]
        for connection in self._connections:
            start, end = connection.start, connection.end
            if start < len(points) and end < len(points):
                cv2.line(frame, points[start], points[end], (255, 255, 255), 1)
        for point in points:
            cv2.circle(frame, point, 3, (0, 255, 255), -1)

    def close(self) -> None:
        try:
            self._landmarker.close()
        except Exception:
            pass


def build_backend(module, model_path: Path | None, max_hands: int,
                  confidence: float, model_complexity: int):
    """Pick whichever backend the installed mediapipe supports.

    Returns ``(backend, reason)``.  The legacy API is preferred when present
    because it needs no model file -- one less thing to go wrong.
    """
    if hasattr(module, "solutions") and hasattr(getattr(module, "solutions"), "hands"):
        try:
            return _LegacyBackend(module, max_hands, confidence, model_complexity), ""
        except Exception as error:
            return None, f"the legacy hand API would not start ({error})"

    if hasattr(module, "tasks"):
        if model_path is None:
            return None, (
                "mediapipe 1.x is installed, which needs a separate hand model "
                "that was not found. Run 'python -m tools.fetch_model' to download "
                "it (about 8 MB, once), or install the older self-contained API "
                "with 'pip install \"mediapipe<1.0\"'."
            )
        try:
            return _TasksBackend(module, model_path, max_hands, confidence), ""
        except Exception as error:
            return None, f"the tasks hand API would not start ({error})"

    return None, "the installed mediapipe has no usable hand-tracking API"


class HandTracker:
    """Thin wrapper over MediaPipe hand tracking that degrades instead of crashing.

    The original probed a ``.binarypb`` file inside the mediapipe package and
    disabled gesture recognition when it was not found.  That check was a
    workaround for the non-ASCII path bug above: on a non-ASCII install it
    disabled a perfectly good mediapipe, and on an ASCII install the
    *construction* of the tracker is the only meaningful test anyway.  Here the
    construction is attempted (repairing the path first if it can), across both
    mediapipe APIs, and every failure comes with a reason worth reading.
    """

    def __init__(self, max_hands: int = 1, confidence: float = 0.6,
                 model_complexity: int = 0, repair_path: bool = True) -> None:
        self.available = False
        self.reason = ""
        self.backend_name = ""
        self.pose = HandPose()
        self.hand_present = False
        self.landmarks = None
        self.repaired_path = False
        self._backend = None

        mp, self.repaired_path, problem = load_mediapipe(repair_path)
        if mp is None:
            self.reason = problem
            return

        self.mp = mp
        backend, reason = build_backend(mp, find_hand_model(), max_hands,
                                        confidence, model_complexity)
        if backend is None:
            if "does not exist" in reason and "binarypb" in reason:
                reason = (
                    "mediapipe cannot reach its own model file. This usually means "
                    "the installation path contains non-ASCII characters (a folder "
                    "named like '副本'), which its native file loader cannot open. "
                    "Recreate the virtual environment under a plain-ASCII path."
                )
            self.reason = reason
            return
        self._backend = backend
        self.backend_name = backend.name
        self.available = True

    def process(self, frame) -> HandPose:
        """Detect the hand in *frame* and return the classified pose."""
        if not self.available or self._backend is None:
            return HandPose()
        try:
            hand = self._backend.detect(frame)
        except Exception as error:
            self.reason = f"hand tracking failed ({error})"
            self.available = False
            return HandPose()

        if hand is None:
            self.hand_present = False
            self.landmarks = None
            self.pose = HandPose()
            return self.pose

        self.hand_present = True
        self.landmarks = hand
        self.pose = classify(self._backend.landmarks(hand))
        return self.pose

    def draw(self, frame) -> None:
        """Skeleton overlay, in high-contrast colours.

        Yellow on white rather than MediaPipe's default green: the default is
        close to the hue that protanopes and deuteranopes confuse, and the
        overlay is supposed to be *helping*.
        """
        if not self.available or self.landmarks is None:
            return
        try:
            self._backend.draw(frame, self.landmarks)
        except Exception:
            pass

    def close(self) -> None:
        if self._backend is not None:
            self._backend.close()
            self._backend = None
        self.available = False
