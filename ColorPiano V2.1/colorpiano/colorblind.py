"""Colour-vision-deficiency (CVD) maths: simulation, daltonization, analysis.

This module deliberately contains *no* palette and no UI text -- it is the
numerical core, and the numbers here are what make the rest of the project
honest:

* :func:`simulate` implements the Viénot/Brettel/Mollon dichromat projection so
  the preview can be shown the way a colour-blind user sees it.
* :func:`daltonize` implements the Daltonize error-redistribution filter, so
  colours that would collide can be pushed apart in real time.
* :func:`build_confusion_tables` precomputes, for each CVD type, which notes a
  user of that type cannot tell apart.  At runtime the app then *admits* the
  ambiguity and points at the redundant cue instead of pretending the colour
  identification was reliable.  See :mod:`colorpiano.palette` for how the
  precomputed table is used to choose between palettes.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .colorspace import (
    bgr_to_lab,
    decode_image,
    delta_e_2000_distance_matrix,
    encode_to_u8,
    linear_to_srgb,
    srgb_to_linear,
)

# --------------------------------------------------------------------------- #
# Cone response model
# --------------------------------------------------------------------------- #
#: Dichromat projection matrices, in **linear sRGB**, from Viénot, Brettel &
#: Mollon (1999) "Digital video colourmaps for checking the legibility of
#: displays by dichromats".
#:
#: These are the published results *already converted* to linear sRGB, which
#: matters more than it looks.  The raw Viénot coefficients are expressed in
#: Smith-Pokorny cone space; applying those coefficients inside a different LMS
#: space (Hunt-Pointer-Estevez, say) is a subtle and common mistake, and its
#: signature is that neutral greys come out tinted -- an obviously wrong thing
#: for a colour-blindness *simulator* to do.  Here every row sums to 1.0, so the
#: achromatic axis is mapped exactly onto itself, and a grey pixel stays grey.
#: ``tests/test_core.py`` asserts that property.
_RGB_MATRIX = {
    # long (red) cone missing: L is reconstructed from M and S
    "protanopia": np.array([
        [0.10889, 0.89111, 0.00000],
        [0.10889, 0.89111, 0.00000],
        [0.00447, -0.00447, 1.00000],
    ]),
    # medium (green) cone missing
    "deuteranopia": np.array([
        [0.29031, 0.70969, 0.00000],
        [0.29031, 0.70969, 0.00000],
        [-0.02197, 0.02197, 1.00000],
    ]),
    # short (blue) cone missing: S is reconstructed from L and M
    "tritanopia": np.array([
        [1.00000, 0.14461, -0.14461],
        [0.00000, 0.85924, 0.14076],
        [0.00000, 0.85924, 0.14076],
    ]),
}

#: Linear sRGB -> LMS (Hunt-Pointer-Estevez), used *only* for daltonization:
#: the Daltonize algorithm redistributes the error between cone responses, and
#: that bookkeeping is naturally expressed in cone space.  Simulation itself
#: does not go through here.
_M_RGB2LMS = np.array([
    [0.31399022, 0.63951294, 0.04649755],
    [0.15537241, 0.75789446, 0.08670142],
    [0.01775239, 0.10944209, 0.87256922],
])
_M_LMS2RGB = np.linalg.inv(_M_RGB2LMS)


def _lms_projection(missing: str) -> np.ndarray:
    """Reconstruct the missing cone from the two that remain, in LMS."""
    if missing == "l":
        return np.array([[0.0, 2.02344, -2.52581],
                         [0.0, 1.0, 0.0],
                         [0.0, 0.0, 1.0]])
    if missing == "m":
        return np.array([[1.0, 0.0, 0.0],
                         [0.494207, 0.0, 1.24827],
                         [0.0, 0.0, 1.0]])
    if missing == "s":
        return np.array([[1.0, 0.0, 0.0],
                         [0.0, 1.0, 0.0],
                         [-0.395913, 0.801109, 0.0]])
    raise ValueError(f"unknown cone: {missing!r}")


#: Daltonize error-shift matrices: where the lost information is re-injected.
_SHIFT = {
    "protanopia": np.array([[0.0, 0.0, 0.0],
                            [0.7, 1.0, 0.0],
                            [0.7, 0.0, 1.0]]),
    "deuteranopia": np.array([[1.0, 0.7, 0.0],
                              [0.0, 1.0, 0.0],
                              [0.0, 0.7, 1.0]]),
    "tritanopia": np.array([[1.0, 0.0, 0.7],
                            [0.0, 1.0, 0.7],
                            [0.0, 0.0, 1.0]]),
}

#: The missing-cone letter for each mode.
MISSING_CONE = {"protanopia": "l", "deuteranopia": "m", "tritanopia": "s"}

#: Daltonization, pre-composed, so the whole thing is two matrix multiplies
#: instead of four.  Matrix multiplication is associative, so this is the same
#: arithmetic -- ``tests/test_core.py`` checks the composed form against the
#: step-by-step one -- and it halves the cost of a full-frame filter.
#:
#:   error     = linear - simulated
#:             = linear - linear @ M.T
#:             = linear @ (I - M).T
#:   correction = strength * (error @ M_RGB2LMS.T) @ shift.T @ M_LMS2RGB.T
#:              = strength * error @ (M_LMS2RGB @ shift @ M_RGB2LMS).T
_ERROR_MATRIX = {
    mode: (np.eye(3) - matrix) for mode, matrix in _RGB_MATRIX.items()
}
_SHIFT_LINEAR = {
    mode: _M_LMS2RGB @ shift @ _M_RGB2LMS for mode, shift in _SHIFT.items()
}

CVD_MODES = ["none", "protanopia", "deuteranopia", "tritanopia", "achromatopsia"]

CVD_LABELS = {
    "none": "normal vision",
    "protanopia": "protanopia (no red cones)",
    "deuteranopia": "deuteranopia (no green cones)",
    "tritanopia": "tritanopia (no blue cones)",
    "achromatopsia": "achromatopsia (no colour at all)",
}

_COLOUR_MODES = ["protanopia", "deuteranopia", "tritanopia"]


# --------------------------------------------------------------------------- #
# Simulation and daltonization
# --------------------------------------------------------------------------- #
def _apply_matrix_linear(img_bgr, matrix: np.ndarray, quantize: bool):
    """Apply a matrix defined on *linear* RGB to a BGR image or colour list.

    ``quantize=True`` returns uint8 (what the preview needs); ``False`` keeps
    float precision, which the palette design and the confusion tables use --
    rounding to 8 bits before measuring a distance would fabricate collisions
    between colours that are merely close.

    The transfer functions go through lookup tables: they are the dominant cost
    of a per-pixel filter, and evaluating a fractional power over three million
    channels per frame is what dropped the preview from 30 fps to under 3.
    """
    image = img_bgr.astype(np.uint8) if quantize else np.asarray(img_bgr)
    if quantize:
        linear = decode_image(image[..., ::-1])
    else:
        linear = srgb_to_linear(image[..., ::-1].astype(np.float32) / 255.0).astype(np.float32)
    transformed = linear @ matrix.T.astype(np.float32)
    if quantize:
        return encode_to_u8(transformed)[..., ::-1]
    out = np.clip(linear_to_srgb(transformed), 0.0, 1.0)
    return out[..., ::-1] * 255.0


def _simulate_achromatopsia(img_bgr, quantize: bool):
    image = img_bgr.astype(np.uint8) if quantize else np.asarray(img_bgr)
    if quantize:
        linear = decode_image(image[..., ::-1])
    else:
        linear = srgb_to_linear(image[..., ::-1].astype(np.float32) / 255.0).astype(np.float32)
    lum = 0.2126 * linear[..., 0] + 0.7152 * linear[..., 1] + 0.0722 * linear[..., 2]
    if quantize:
        return np.repeat(encode_to_u8(lum)[..., None], 3, axis=2)
    grey = np.clip(linear_to_srgb(lum), 0.0, 1.0)[..., None] * 255.0
    return np.repeat(grey, 3, axis=2)


def simulate(img_bgr: np.ndarray, mode: str) -> np.ndarray:
    """Return *img_bgr* as it would look to somebody with *mode*, as uint8."""
    if mode in ("none", "", None):
        return img_bgr
    if mode == "achromatopsia":
        return _simulate_achromatopsia(img_bgr, quantize=True)
    matrix = _RGB_MATRIX.get(mode)
    if matrix is None:
        return img_bgr
    return _apply_matrix_linear(img_bgr, matrix, quantize=True).astype(np.uint8)


def _gamut_scale(linear: np.ndarray, correction: np.ndarray) -> np.ndarray:
    """Per-pixel factor in [0, 1] keeping ``linear + s * correction`` in gamut.

    Without this the correction is simply added and then clipped, and clipping
    is not a neutral operation: for two saturated reds the injected difference
    is cut off in both, so they come out *more* alike than they started -- one
    measured pair went from dE00 5.0 to 0.0, i.e. daltonization made them
    identical.  Scaling the correction to fit degrades to "do nothing" instead
    of "do harm".

    The headroom is tracked per direction: a channel with a positive correction
    is limited by how much room is left above it, one with a negative correction
    by how much is left below.  Using a single symmetric bound instead is
    tempting -- it is one reduction cheaper -- but it cripples the boost for
    dark saturated colours, and the effect is measurable: the recovered
    separation on the palette's hard protanopia pairs fell from 9 of 11 to 7 of
    11, which the test suite caught.
    """
    positive = np.maximum(correction, 1e-6)
    negative = np.maximum(-correction, 1e-6)
    room_up = ((1.0 - linear) / positive).min(axis=-1)
    room_down = (linear / negative).min(axis=-1)
    scale = np.minimum(room_up, room_down)
    return np.clip(scale, 0.0, 1.0)[..., None].astype(np.float32)


def daltonize(img_bgr: np.ndarray, mode: str, strength: float = 1.0) -> np.ndarray:
    """Push colours apart along the axis the viewer *can* still see.

    Unlike :func:`simulate` (which throws information away to demonstrate the
    problem) this is a boost: the error introduced by the deficiency is
    measured, then re-injected through the cones that still work.  How much of
    it can be injected depends on the pixel -- see :func:`_gamut_scale`.
    """
    if mode not in _SHIFT_LINEAR:
        return img_bgr
    strength = float(np.clip(strength, 0.0, 2.0))

    linear = decode_image(np.asarray(img_bgr, dtype=np.uint8)[..., ::-1])

    # The error is what the deficiency threw away, measured with the same
    # grey-preserving matrices the simulator uses, so a neutral pixel has an
    # error of exactly zero and daltonization cannot tint it.  Redistributing it
    # is naturally a cone-space operation; both steps are pre-composed into one
    # matrix each (see _ERROR_MATRIX / _SHIFT_LINEAR).
    error_linear = linear @ _ERROR_MATRIX[mode].T.astype(np.float32)
    correction = strength * (error_linear @ _SHIFT_LINEAR[mode].T.astype(np.float32))

    corrected = linear + _gamut_scale(linear, correction) * correction
    return encode_to_u8(corrected)[..., ::-1]


def simulate_colors(colors_bgr, mode: str, quantize: bool = False) -> np.ndarray:
    """Vectorised :func:`simulate` for an ``(N, 3)`` list of colours.

    Defaults to float output: these values feed distance measurements, not
    pixels, and 8-bit rounding would invent collisions.
    """
    arr = np.asarray(colors_bgr, dtype=np.uint8)
    if mode in ("none", "", None):
        return arr.astype(np.float64)
    if mode == "achromatopsia":
        return _simulate_achromatopsia(arr.reshape(1, -1, 3), quantize).reshape(-1, 3)
    matrix = _RGB_MATRIX.get(mode)
    if matrix is None:
        return arr.astype(np.float64)
    return _apply_matrix_linear(arr.reshape(1, -1, 3), matrix, quantize).reshape(-1, 3)


# --------------------------------------------------------------------------- #
# Confusion analysis
# --------------------------------------------------------------------------- #
#: Below this CIEDE2000 distance two colours count as "the same" for the
#: viewer.  ~2.3 is the classic just-noticeable difference; 6.0 is used here
#: because in a live camera view noise is far larger than in a lab.
CONFUSION_THRESHOLD = 6.0


@dataclass
class ConfusionTable:
    """For a single CVD mode: which palette entries cannot be told apart."""

    mode: str
    nearest: np.ndarray          # (N,) index of the closest *other* colour
    distance: np.ndarray         # (N,) CIEDE2000 distance to that neighbour
    matrix: np.ndarray           # (N, N) pairwise distances in simulated space

    @property
    def ambiguous_count(self) -> int:
        return int(np.sum(self.distance < CONFUSION_THRESHOLD))

    @property
    def median_distance(self) -> float:
        return float(np.median(self.distance))

    def is_ambiguous(self, index: int) -> bool:
        return bool(self.distance[index] < CONFUSION_THRESHOLD)

    def partner(self, index: int) -> int:
        return int(self.nearest[index])

    def report(self, names=None, limit: int = 10) -> str:
        n = len(self.nearest)
        names = names or [str(i) for i in range(n)]
        pairs, seen = [], set()
        for i in range(n):
            if not self.is_ambiguous(i):
                continue
            j = self.partner(i)
            key = (min(i, j), max(i, j))
            if key in seen:
                continue
            seen.add(key)
            pairs.append((float(self.distance[i]), names[i], names[j]))
        pairs.sort()
        lines = [
            f"[{self.mode}] {self.ambiguous_count}/{n} notes have a confusable "
            f"partner (dE00 < {CONFUSION_THRESHOLD}); {len(pairs)} distinct pairs, "
            f"median nearest-neighbour dE00 = {self.median_distance:.1f}"
        ]
        for dist, a, b in pairs[:limit]:
            lines.append(f"      {a:>4} ~ {b:<4} dE00 = {dist:4.1f}")
        if len(pairs) > limit:
            lines.append(f"      ... and {len(pairs) - limit} more")
        return "\n".join(lines)


def build_confusion_tables(colors_bgr, modes=None) -> dict[str, ConfusionTable]:
    """Precompute the nearest confusable partner of every colour, per CVD mode."""
    modes = modes or list(_COLOUR_MODES)
    tables: dict[str, ConfusionTable] = {}
    n = len(colors_bgr)
    for mode in modes:
        labs = bgr_to_lab(simulate_colors(colors_bgr, mode))
        matrix = delta_e_2000_distance_matrix(labs)
        np.fill_diagonal(matrix, np.inf)
        nearest = np.argmin(matrix, axis=1).astype(int)
        tables[mode] = ConfusionTable(
            mode=mode,
            nearest=nearest,
            distance=matrix[np.arange(n), nearest],
            matrix=matrix,
        )
    return tables


def indistinguishable_groups(colors_bgr, threshold: float = 2.5,
                             modes=None) -> list[list[int]]:
    """Groups of palette entries no viewer of these types could tell apart.

    Connected components of the "closer than a just-noticeable difference"
    graph, computed in simulated space so that a group means "the same to
    *everybody* in this list".  Two entries in one group are not a detection
    bug when the sampler reports the wrong one -- they are the same colour.
    """
    modes = modes or ["none"] + list(_COLOUR_MODES)
    matrix = worst_case_distances(colors_bgr, modes)
    n = len(colors_bgr)
    parent = list(range(n))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for i in range(n):
        for j in range(i + 1, n):
            if matrix[i, j] < threshold:
                root_i, root_j = find(i), find(j)
                if root_i != root_j:
                    parent[root_j] = root_i

    groups: dict[int, list[int]] = {}
    for i in range(n):
        groups.setdefault(find(i), []).append(i)
    return [members for members in groups.values() if len(members) > 1]


def worst_case_distances(colors_bgr, modes=None) -> np.ndarray:
    """``(N, N)`` matrix where entry ``[i, j]`` is the *smallest* CIEDE2000
    distance between colours ``i`` and ``j`` across normal vision and every
    supported CVD type.

    This is the metric worth optimising and worth quoting: a palette where the
    minimum over modes is large is one that no viewer, of any of these five
    kinds, can mix up -- whereas "minimum per mode" is always 0, because any
    palette with 52 entries contains *some* pair that collides for *somebody*.
    """
    modes = modes or ["none"] + list(_COLOUR_MODES)
    worst = None
    for mode in modes:
        labs = bgr_to_lab(simulate_colors(colors_bgr, mode))
        matrix = delta_e_2000_distance_matrix(labs)
        np.fill_diagonal(matrix, np.inf)
        worst = matrix if worst is None else np.minimum(worst, matrix)
    return worst


def worst_case_separation(colors_bgr, modes=None) -> float:
    """Smallest pairwise distance in the worst case over all viewer types."""
    return float(np.min(worst_case_distances(colors_bgr, modes)))


def palette_score(colors_bgr, modes=None) -> dict[str, dict]:
    """Summarise how well a palette survives each CVD type.

    ``ambiguous`` counts notes whose nearest neighbour is closer than
    :data:`CONFUSION_THRESHOLD`; ``median`` is the typical nearest-neighbour
    distance.  Both matter: a palette can have few collisions but still be
    cramped overall.
    """
    modes = modes or list(_COLOUR_MODES)
    out = {}
    for mode in modes:
        labs = bgr_to_lab(simulate_colors(colors_bgr, mode))
        matrix = delta_e_2000_distance_matrix(labs)
        np.fill_diagonal(matrix, np.inf)
        nearest = matrix[np.arange(len(labs)), np.argmin(matrix, axis=1)]
        out[mode] = {
            "ambiguous": int(np.sum(nearest < CONFUSION_THRESHOLD)),
            "median": float(np.median(nearest)),
            "worst": float(np.min(nearest)),
        }
    # Achromatopsia is the degenerate case: only lightness survives, so the
    # measure of interest is how well the palette spreads over L*.
    labs = bgr_to_lab(colors_bgr)
    out["achromatopsia"] = {
        "ambiguous": 0,
        "median": float(np.median(np.abs(np.diff(np.sort(labs[:, 0]))))),
        "worst": float(labs[:, 0].min()),
        "L_spread": float(labs[:, 0].max() - labs[:, 0].min()),
    }
    return out
