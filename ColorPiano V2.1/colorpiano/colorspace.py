"""Colour-space maths: sRGB <-> XYZ <-> CIELAB and the CIEDE2000 difference.

Why this module exists
----------------------
The original code obtained Lab values with::

    lab = cv2.cvtColor(np.uint8([[bgr]]), cv2.COLOR_BGR2LAB)[0][0]

which is wrong in two ways that matter as soon as you use the numbers for
anything other than a plain Euclidean sort:

1. OpenCV returns 8-bit Lab -- ``L`` squeezed into 0..255 instead of 0..100 and
   ``a``/``b`` into 0..255 instead of roughly -128..127, plus the rounding that
   comes with a ``uint8`` round trip.
2. More importantly, CIEDE2000's weighting functions (``S_L``, ``S_C``, ``S_H``,
   ``R_T``) are calibrated for ``L`` in 0..100 and ``a``/``b`` in their real
   units.  Feeding it 0..255-scaled values silently distorts every distance, so
   the whole point of "using CIEDE2000 instead of Euclidean RGB" is lost.

Here the conversion is done properly in float, and the CIEDE2000 implementation
is verified against all 34 reference pairs from Sharma, Wu & Dalal (2005), the
standard test set for the formula -- see ``tests/test_core.py``.
"""

from __future__ import annotations

import numpy as np

# D65, 2 degree observer.
_WHITE = np.array([0.95047, 1.00000, 1.08883], dtype=np.float64)

_M_RGB2XYZ = np.array(
    [
        [0.4124564, 0.3575761, 0.1804375],
        [0.2126729, 0.7151522, 0.0721750],
        [0.0193339, 0.1191920, 0.9503041],
    ],
    dtype=np.float64,
)

# Constant used by the CIE standard: (6/29)^3 and 3*(6/29)^2
_DELTA = 6.0 / 29.0
_DELTA3 = _DELTA ** 3
_THREE_DELTA2 = 3.0 * _DELTA ** 2
_TWENTY_FIVE_POW_7 = 25.0 ** 7


# --------------------------------------------------------------------------- #
# sRGB <-> CIELAB
# --------------------------------------------------------------------------- #
def srgb_to_linear(rgb: np.ndarray) -> np.ndarray:
    """Undo the sRGB transfer function.  *rgb* is in 0..1."""
    rgb = np.asarray(rgb, dtype=np.float64)
    return np.where(rgb <= 0.04045, rgb / 12.92, ((rgb + 0.055) / 1.055) ** 2.4)


def linear_to_srgb(rgb: np.ndarray) -> np.ndarray:
    rgb = np.asarray(rgb, dtype=np.float64)
    rgb = np.clip(rgb, 0.0, None)
    return np.where(rgb <= 0.0031308, rgb * 12.92, 1.055 * (rgb ** (1 / 2.4)) - 0.055)


def bgr_to_lab(bgr) -> np.ndarray:
    """Convert one colour or an ``(..., 3)`` BGR array to CIELAB (D65).

    Returns ``L`` in 0..100 and ``a``/``b`` in about -128..127, as float64.
    Works for a single ``(3,)`` pixel as well as an ``(H, W, 3)`` image; the
    result keeps the leading dimensions.
    """
    bgr = np.asarray(bgr, dtype=np.float64)
    rgb = bgr[..., ::-1] / 255.0
    linear = srgb_to_linear(rgb)
    xyz = linear @ _M_RGB2XYZ.T
    xyz = xyz / _WHITE
    f = np.where(xyz > _DELTA3, np.cbrt(xyz), xyz / _THREE_DELTA2 + 4.0 / 29.0)
    fx, fy, fz = f[..., 0], f[..., 1], f[..., 2]
    lab = np.empty(bgr.shape, dtype=np.float64)
    lab[..., 0] = 116.0 * fy - 16.0
    lab[..., 1] = 500.0 * (fx - fy)
    lab[..., 2] = 200.0 * (fy - fz)
    return lab


def lab_to_bgr(lab) -> np.ndarray:
    """Inverse of :func:`bgr_to_lab`; returns ``uint8`` BGR."""
    lab = np.asarray(lab, dtype=np.float64)
    L, a, b = lab[..., 0], lab[..., 1], lab[..., 2]
    fy = (L + 16.0) / 116.0
    fx = fy + a / 500.0
    fz = fy - b / 200.0
    f = np.stack([fx, fy, fz], axis=-1)
    xyz = np.where(f > _DELTA, f ** 3, _THREE_DELTA2 * (f - 4.0 / 29.0)) * _WHITE
    linear = xyz @ np.linalg.inv(_M_RGB2XYZ).T
    rgb = np.clip(linear_to_srgb(linear), 0.0, 1.0)
    out = np.empty_like(rgb)
    out[..., 0] = rgb[..., 2] * 255.0
    out[..., 1] = rgb[..., 1] * 255.0
    out[..., 2] = rgb[..., 0] * 255.0
    return np.round(out).astype(np.uint8)


def lab_chroma(lab) -> np.ndarray:
    lab = np.asarray(lab, dtype=np.float64)
    return np.hypot(lab[..., 1], lab[..., 2])


# --------------------------------------------------------------------------- #
# Fast transfer-function lookups
# --------------------------------------------------------------------------- #
#: The sRGB transfer functions are the expensive part of any per-pixel filter:
#: each one evaluates a fractional power over every channel of every frame.
#: Both directions are single-variable, so both can be tabulated.  Decoding is
#: exact (the input is 8-bit, so 256 entries cover it completely); encoding is
#: sampled and interpolated, with the table sized so the error stays far below
#: one 8-bit step.
_DECODE_TABLE = srgb_to_linear(np.arange(256, dtype=np.float64) / 255.0).astype(np.float32)

#: The encode table stores the finished 8-bit value rather than an intermediate
#: float, so encoding is one gather: no multiply, no rounding, no second cast.
#: 4096 rows puts the table in L1 and bounds the sampling error at a quarter of
#: one 8-bit step, which is well under the rounding nobody can see anyway.
_ENCODE_SIZE = 4096
_ENCODE_TABLE = np.round(
    linear_to_srgb(np.linspace(0.0, 1.0, _ENCODE_SIZE, dtype=np.float64)) * 255.0
).astype(np.uint8)


def decode_image(bgr_u8: np.ndarray) -> np.ndarray:
    """uint8 BGR -> linear-light float32 in 0..1.  Exact, via a 256-entry table."""
    return _DECODE_TABLE[bgr_u8]


def encode_to_u8(linear: np.ndarray) -> np.ndarray:
    """linear-light float32 in 0..1 -> uint8, via a 4096-entry table.

    Point-wise, so it is indifferent to channel order.  Out-of-range values are
    clipped, which is the caller's business: the daltonizer deliberately scales
    its correction so that this rarely bites.

    The table stores the finished 8-bit value, so this is one gather: no
    multiply, no rounding, no second cast.  4096 rows bounds the sampling error
    at a quarter of one 8-bit step.
    """
    index = (linear * (_ENCODE_SIZE - 1)).astype(np.int32)
    np.clip(index, 0, _ENCODE_SIZE - 1, out=index)
    return _ENCODE_TABLE[index]


def lab_hue_deg(lab) -> np.ndarray:
    """Hue angle in degrees, 0..360, measured in the a*/b* plane."""
    lab = np.asarray(lab, dtype=np.float64)
    return np.degrees(np.arctan2(lab[..., 2], lab[..., 1])) % 360.0


# --------------------------------------------------------------------------- #
# CIEDE2000
# --------------------------------------------------------------------------- #
def delta_e_ciede2000(lab1, lab2, kL: float = 1.0, kC: float = 1.0, kH: float = 1.0):
    """CIEDE2000 colour difference.

    ``lab1`` is a single ``(3,)`` colour, ``lab2`` may be anything that
    broadcasts -- most usefully an ``(N, 3)`` palette, which returns ``(N,)``
    distances.  Implementation follows Sharma, Wu & Dalal (2005) including the
    ``C1'*C2' == 0`` edge cases, which is where naive ports go wrong.
    """
    lab1 = np.asarray(lab1, dtype=np.float64)
    lab2 = np.asarray(lab2, dtype=np.float64)

    L1, a1, b1 = lab1[..., 0], lab1[..., 1], lab1[..., 2]
    L2, a2, b2 = lab2[..., 0], lab2[..., 1], lab2[..., 2]

    C1 = np.hypot(a1, b1)
    C2 = np.hypot(a2, b2)
    C_bar = 0.5 * (C1 + C2)
    C_bar7 = C_bar ** 7
    G = 0.5 * (1.0 - np.sqrt(C_bar7 / (C_bar7 + _TWENTY_FIVE_POW_7)))

    a1p = (1.0 + G) * a1
    a2p = (1.0 + G) * a2
    C1p = np.hypot(a1p, b1)
    C2p = np.hypot(a2p, b2)

    h1p = np.degrees(np.arctan2(b1, a1p)) % 360.0
    h2p = np.degrees(np.arctan2(b2, a2p)) % 360.0
    zero_chroma = (C1p * C2p) == 0.0
    # A chroma of zero has no defined hue; pin it to 0 so the arithmetic below
    # stays finite.  The terms it feeds are all multiplied by sqrt(C1'C2'), so
    # the convention cannot influence the result.
    h1p = np.where(C1p == 0.0, 0.0, h1p)
    h2p = np.where(C2p == 0.0, 0.0, h2p)

    dLp = L2 - L1
    dCp = C2p - C1p

    dhp = h2p - h1p
    dhp = np.where(zero_chroma, 0.0,
                   np.where(np.abs(dhp) <= 180.0, dhp,
                            np.where(dhp > 180.0, dhp - 360.0, dhp + 360.0)))
    dHp = 2.0 * np.sqrt(C1p * C2p) * np.sin(np.radians(dhp / 2.0))

    Lbp = 0.5 * (L1 + L2)
    Cbp = 0.5 * (C1p + C2p)

    h_sum = h1p + h2p
    hbp = np.where(
        zero_chroma,
        h_sum,
        np.where(np.abs(h1p - h2p) <= 180.0, 0.5 * h_sum,
                 np.where(h_sum < 360.0, 0.5 * (h_sum + 360.0), 0.5 * (h_sum - 360.0))),
    )

    T = (
        1.0
        - 0.17 * np.cos(np.radians(hbp - 30.0))
        + 0.24 * np.cos(np.radians(2.0 * hbp))
        + 0.32 * np.cos(np.radians(3.0 * hbp + 6.0))
        - 0.20 * np.cos(np.radians(4.0 * hbp - 63.0))
    )

    d_theta = 30.0 * np.exp(-(((hbp - 275.0) / 25.0) ** 2))
    Cbp7 = Cbp ** 7
    R_C = 2.0 * np.sqrt(Cbp7 / (Cbp7 + _TWENTY_FIVE_POW_7))
    S_L = 1.0 + (0.015 * (Lbp - 50.0) ** 2) / np.sqrt(20.0 + (Lbp - 50.0) ** 2)
    S_C = 1.0 + 0.045 * Cbp
    S_H = 1.0 + 0.015 * Cbp * T
    R_T = -np.sin(np.radians(2.0 * d_theta)) * R_C

    term_L = dLp / (kL * S_L)
    term_C = dCp / (kC * S_C)
    term_H = dHp / (kH * S_H)
    return np.sqrt(term_L ** 2 + term_C ** 2 + term_H ** 2 + R_T * term_C * term_H)


def delta_e_2000_distance_matrix(labs) -> np.ndarray:
    """Full pairwise CIEDE2000 matrix for an ``(N, 3)`` palette."""
    labs = np.asarray(labs, dtype=np.float64)
    n = labs.shape[0]
    out = np.zeros((n, n), dtype=np.float64)
    for i in range(n):
        out[i] = delta_e_ciede2000(labs[i], labs)
    return out


# --------------------------------------------------------------------------- #
# Circular helpers (the hue axis wraps, and that breaks naive averaging)
# --------------------------------------------------------------------------- #
def circular_mean_deg(angles_deg: np.ndarray, weights: np.ndarray | None = None) -> float:
    """Weighted mean of angles in degrees.

    Averaging hue *numbering* linearly is the classic wrap-around bug: the mean
    of 350 deg and 10 deg is 0 deg, never 180 deg.
    """
    angles = np.radians(np.asarray(angles_deg, dtype=np.float64))
    if weights is None:
        weights = np.ones_like(angles)
    weights = np.asarray(weights, dtype=np.float64)
    if weights.sum() <= 0:
        return 0.0
    x = float(np.sum(weights * np.cos(angles)))
    y = float(np.sum(weights * np.sin(angles)))
    if abs(x) < 1e-12 and abs(y) < 1e-12:
        return 0.0
    result = float(np.degrees(np.arctan2(y, x)) % 360.0)
    # arctan2 of a mean that lands exactly on 0 degrees returns -1e-15, which the
    # modulo sends to 359.999... -- indistinguishable from 0 geometrically, but
    # not numerically, and the difference leaks into any comparison.
    return 0.0 if result > 360.0 - 1e-9 else result


def circular_distance_deg(a, b) -> np.ndarray:
    """Shortest angular distance, 0..180."""
    d = np.abs(np.asarray(a, dtype=np.float64) - np.asarray(b, dtype=np.float64)) % 360.0
    return np.where(d > 180.0, 360.0 - d, d)
