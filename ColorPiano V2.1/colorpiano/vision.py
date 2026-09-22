"""Frame sources and colour sampling.

Everything the app needs from a camera is behind :class:`FrameSource`, which
means the instrument can also run from a video file, a still image, or a
synthetic generator.  That is not just a convenience: it is what makes the
program testable without a camera, so ``tools/selftest.py`` can exercise the
whole colour -> note -> overlay pipeline on a machine with no webcam.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from .colorspace import bgr_to_lab, lab_chroma
from .config import ROI_SIZE_MAX, ROI_SIZE_MIN, Settings


# --------------------------------------------------------------------------- #
# Sampling
# --------------------------------------------------------------------------- #
@dataclass
class Sample:
    """One colour reading from the sampling box."""

    bgr: np.ndarray
    brightness: float          # relative luminance, 0..1
    chroma: float              # CIELAB chroma, i.e. "how colourful"
    lightness: float           # CIELAB L*
    lit_fraction: float        # share of pixels that are not blown out or black
    valid: bool
    reason: str = ""

    @staticmethod
    def empty(reason: str) -> "Sample":
        return Sample(np.zeros(3, dtype=np.uint8), 0.0, 0.0, 0.0, 0.0, False, reason)


def gray_world_gains(frame: np.ndarray, strength: float = 1.0,
                     low: float = 0.86, high: float = 1.16) -> np.ndarray:
    """Per-channel gains that neutralise a colour cast.

    Measured over the *whole* frame, never over the sampling box -- a
    grey-world correction applied to the box alone would always pull the object
    towards grey and destroy the very colour being measured.

    The gains are clamped hard.  Unconstrained grey-world assumes the scene
    averages to grey, which is false for an instrument whose user is pointing
    the camera at one saturated object; clamping means it corrects a mild
    tungsten cast without recolouring the object.
    """
    small = cv2.resize(frame, (64, 36), interpolation=cv2.INTER_AREA)
    means = small.reshape(-1, 3).mean(axis=0).astype(np.float32)
    # Ignore channels that are essentially black -- dividing by them explodes.
    if float(np.min(means)) < 8.0:
        return np.ones(3, dtype=np.float32)
    grey = float(means.mean())
    gains = grey / np.maximum(means, 1e-6)
    gains = 1.0 + strength * (gains - 1.0)
    return np.clip(gains, low, high).astype(np.float32)


def sample_box(frame: np.ndarray, center_x: int, center_y: int, half_size: int,
               gains: np.ndarray | None = None,
               blur: int = 5) -> Sample:
    """Average colour of the square box centred on ``(center_x, center_y)``.

    ``half_size`` is half the *side*, so the box is ``2*half_size`` wide.  Note
    that the original passed the same style of argument but documented it as a
    size, which is a factor-of-two difference in how much of the scene is
    measured; here it is stated explicitly.
    """
    height, width = frame.shape[:2]
    half_size = int(np.clip(half_size, ROI_SIZE_MIN, ROI_SIZE_MAX))
    x1, x2 = max(0, center_x - half_size), min(width, center_x + half_size)
    y1, y2 = max(0, center_y - half_size), min(height, center_y + half_size)
    if x2 - x1 < 2 or y2 - y1 < 2:
        return Sample.empty("sampling box is outside the frame")

    roi = frame[y1:y2, x1:x2]
    if blur >= 3:
        roi = cv2.GaussianBlur(roi, (blur | 1, blur | 1), 0)

    pixels = roi.reshape(-1, 3).astype(np.float32)
    if gains is not None:
        pixels = pixels * gains
    mean = np.clip(pixels.mean(axis=0), 0, 255)
    bgr = np.round(mean).astype(np.uint8)

    luminance = (0.114 * mean[0] + 0.587 * mean[1] + 0.299 * mean[2]) / 255.0

    # "Lit" pixels: not blown out, not crushed.  A box that is 90% blown-out
    # white is not measuring anything, and the UI says so rather than playing a
    # note chosen by rounding noise.
    grey = pixels.mean(axis=1)
    lit = float(np.mean((grey > 20.0) & (grey < 245.0)))
    if lit < 0.25:
        reason = "too dark" if grey.mean() <= 20.0 else "blown out"
        return Sample(bgr, float(luminance), 0.0, 0.0, lit, False, reason)

    lab = bgr_to_lab(bgr.astype(np.float64))
    return Sample(bgr, float(luminance), float(lab_chroma(lab)), float(lab[0]), lit, True)


# --------------------------------------------------------------------------- #
# Sources
# --------------------------------------------------------------------------- #
class FrameSource:
    """A source of BGR frames."""

    description = "frames"
    static = False

    def read(self) -> tuple[bool, np.ndarray | None]:
        raise NotImplementedError

    def release(self) -> None:
        pass

    def __enter__(self) -> "FrameSource":
        return self

    def __exit__(self, *exc) -> None:
        self.release()


class CameraSource(FrameSource):
    """A webcam, opened with the best backend available for the platform."""

    def __init__(self, index: int = 0, width: int = 1280, height: int = 720,
                 mirror: bool = True) -> None:
        self.mirror = mirror
        self.capture = None
        self.opened_index = index
        self._description = f"camera {index} (not opened)"
        # DSHOW is the reliable backend on Windows; try it first, then whatever
        # OpenCV picks by default.  The original only ever tried index 0 and 1.
        for backend in (cv2.CAP_DSHOW, cv2.CAP_ANY):
            capture = cv2.VideoCapture(index, backend)
            if capture.isOpened():
                ok, _ = capture.read()
                if ok:
                    capture.set(cv2.CAP_PROP_FRAME_WIDTH, width)
                    capture.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
                    self.capture = capture
                    break
            capture.release()
        if self.capture is None:
            raise RuntimeError(
                f"no camera could be opened at index {index}. "
                "Check that it is connected and not in use by another program."
            )
        # Recorded at open time: the description is still wanted after the
        # capture has been released, and asking a released capture for its
        # properties raises.
        actual_width = int(self.capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_height = int(self.capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self._description = f"camera {index} ({actual_width}x{actual_height})"

    @property
    def description(self) -> str:
        return self._description

    @property
    def fps(self) -> float:
        if self.capture is None:
            return 0.0
        return float(self.capture.get(cv2.CAP_PROP_FPS) or 0.0)

    def read(self) -> tuple[bool, np.ndarray | None]:
        if self.capture is None:
            return False, None
        ok, frame = self.capture.read()
        if ok and self.mirror:
            frame = cv2.flip(frame, 1)
        return ok, frame

    def release(self) -> None:
        if self.capture is not None:
            self.capture.release()
            self.capture = None


class VideoFileSource(FrameSource):
    """A video file, looped.  Useful for demos and for testing."""

    def __init__(self, path: str, mirror: bool = False) -> None:
        self.capture = cv2.VideoCapture(str(path))
        if not self.capture.isOpened():
            raise RuntimeError(f"cannot open video {path!r}")
        self.path = path
        self.mirror = mirror
        self.frames = int(self.capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)

    @property
    def description(self) -> str:
        return f"video {Path(self.path).name} ({self.frames} frames)"

    def read(self) -> tuple[bool, np.ndarray | None]:
        ok, frame = self.capture.read()
        if not ok:                      # loop instead of stopping
            self.capture.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ok, frame = self.capture.read()
        if ok and self.mirror:
            frame = cv2.flip(frame, 1)
        return ok, frame

    def release(self) -> None:
        self.capture.release()


class ImageSource(FrameSource):
    """A single still image, returned every frame."""

    static = True

    def __init__(self, path: str) -> None:
        self.image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if self.image is None:
            raise RuntimeError(f"cannot read image {path!r}")
        self.path = path

    @property
    def description(self) -> str:
        height, width = self.image.shape[:2]
        return f"image {Path(self.path).name} ({width}x{height})"

    def read(self) -> tuple[bool, np.ndarray | None]:
        return True, self.image.copy()

    def release(self) -> None:
        pass


class SyntheticSource(FrameSource):
    """Generated frames: a moving patch of known colour under the sampling box.

    Exists so the whole pipeline can be verified without a camera.  The colours
    it paints come from a list that the app fills in with the active palette, so
    the self-test can assert that the colour put in is the colour that comes out
    the far end -- through sampling, smoothing, CIEDE2000 matching and
    stabilisation.
    """

    def __init__(self, width: int = 1280, height: int = 720,
                 period: float = 52.0, noise: float = 2.0,
                 colors=None, hold_index: int | None = None) -> None:
        self.width = width
        self.height = height
        self.period = period
        self.noise = noise
        #: BGR colours to sweep through.  The app replaces this with the active
        #: palette; the HSV-wheel fallback keeps the source usable standalone.
        self.colors = list(colors) if colors else None
        self.frame_index = 0
        self.rng = np.random.default_rng(12345)
        #: When set, the box keeps showing this entry instead of sweeping.
        self.hold_index = hold_index
        self.target_index = 0

    @property
    def description(self) -> str:
        count = len(self.colors) if self.colors else 52
        if self.hold_index is not None:
            return f"synthetic test pattern (held at index {self.hold_index})"
        return (f"synthetic test pattern ({count} colours, one step per sweep)")

    def set_colors(self, colors) -> None:
        """Point the generator at a palette (called by the app)."""
        self.colors = list(colors)
        self.frame_index = 0

    def _patch_color(self, index: int) -> np.ndarray:
        if self.colors:
            bgr = self.colors[index % len(self.colors)]
            return np.array([float(v) for v in bgr], dtype=np.float32)
        import colorsys

        r, g, b = colorsys.hsv_to_rgb((index % 52) / 52.0, 1.0, 1.0)
        return np.array([b * 255, g * 255, r * 255], dtype=np.float32)

    def read(self) -> tuple[bool, np.ndarray | None]:
        height, width = self.height, self.width
        # Background: three phase-shifted ramps.  It looks busy, but each
        # channel averages to the same value over a full period, so the frame is
        # neutral on average and the grey-world correction stays close to a
        # no-op -- left in deliberately, because it exercises that code path.
        ramp = np.linspace(0, 1, width, dtype=np.float32)
        background = np.zeros((height, width, 3), dtype=np.float32)
        for channel, offset in enumerate((0.0, 0.33, 0.66)):
            background[:, :, channel] = 60 + 120 * (
                0.5 + 0.5 * np.sin(2 * np.pi * (ramp + offset))
            )

        step = int(self.frame_index // self.period)
        self.target_index = int(self.hold_index) if self.hold_index is not None else step
        patch = self._patch_color(self.target_index)

        half = 120
        cy, cx = height // 2, width // 2
        background[cy - half:cy + half, cx - half:cx + half] = patch

        if self.noise > 0:
            background += self.rng.normal(0.0, self.noise, background.shape)
        frame = np.clip(background, 0, 255).astype(np.uint8)
        self.frame_index += 1
        return True, frame

    def release(self) -> None:
        pass


def open_source(settings: Settings) -> FrameSource:
    """Build the frame source named by the settings."""
    kind = (settings.source or "camera").lower()
    if kind == "camera":
        return CameraSource(settings.camera_index, settings.width, settings.height,
                            mirror=settings.mirror)
    if kind == "video":
        if not settings.source_path:
            raise ValueError("--source video needs --source-path")
        return VideoFileSource(settings.source_path, mirror=settings.mirror)
    if kind == "image":
        if not settings.source_path:
            raise ValueError("--source image needs --source-path")
        return ImageSource(settings.source_path)
    if kind == "synthetic":
        return SyntheticSource(settings.width, settings.height)
    raise ValueError(f"unknown source {settings.source!r}; "
                     "choose camera, video, image or synthetic")
