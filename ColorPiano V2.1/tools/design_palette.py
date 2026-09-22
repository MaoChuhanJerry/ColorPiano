"""Design the colour-blind-safe 52-colour palette, numerically.

Run this to regenerate :data:`colorpiano.palette.CVD_SAFE_COLORS`::

    python -m tools.design_palette            # report
    python -m tools.design_palette --emit     # print the Python literal

Why a search and not a hand-picked list
---------------------------------------
Hand-picked "colour-blind safe" palettes are designed for eight *categorical*
colours.  This instrument needs fifty-two -- which is why the naive approach
(8 Okabe-Ito hues x 7 lightness bands) measures badly: within one lightness
band the eight hues sit on a circle of constant ``L*``, and a dichromat
projects that circle onto a line, so pairs at opposite ends of the circle land
on top of each other.  Measured, that palette put 51 of 52 notes within a
just-noticeable difference of a neighbour.

The fix is to choose the fifty-two colours *in the space the viewer actually
has*, which for a dichromat is two-dimensional: lightness plus the blue-yellow
axis.  The red-green axis, which they cannot see, is then free to be used for
making the palette look colourful to everyone else.

The search is farthest-point sampling: start from one colour, then repeatedly
add the candidate whose *worst-case* distance -- the minimum over normal
vision, protanopia, deuteranopia, tritanopia and achromatopsia -- to everything
already chosen is largest.  That is a 2-approximation to the max-min dispersion
problem, and it is more than good enough here.
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from colorpiano.colorspace import bgr_to_lab, delta_e_ciede2000, lab_hue_deg, lab_to_bgr  # noqa: E402
from colorpiano.colorblind import CVD_MODES, simulate_colors, worst_case_separation  # noqa: E402
from colorpiano.palette import OKABE_ITO, build_hue_wheel  # noqa: E402

COUNT = 52
MODES = ["none"] + [m for m in CVD_MODES if m not in ("none", "achromatopsia")]


# --------------------------------------------------------------------------- #
# Candidate colours
# --------------------------------------------------------------------------- #
def in_gamut_chroma(L: float, h_deg: float, steps: int = 24) -> float:
    """Largest CIELAB chroma that still fits inside sRGB at this ``L``/hue."""
    lo, hi = 0.0, 140.0
    h = np.radians(h_deg)
    for _ in range(steps):
        mid = 0.5 * (lo + hi)
        lab = np.array([L, mid * np.cos(h), mid * np.sin(h)])
        bgr = lab_to_bgr(lab).astype(np.float64)
        if np.all(bgr > 0.5) and np.all(bgr < 254.5):
            lo = mid
        else:
            hi = mid
    return lo


def build_candidates(
    l_range=(26.0, 92.0),
    l_steps=15,
    hue_steps=90,
    chroma_fractions=(1.0, 0.72, 0.5),
    min_chroma=32.0,
) -> np.ndarray:
    """A dense grid of usable colours: ``(N, 3)`` BGR.

    ``min_chroma`` keeps the palette vivid.  Without it the search happily
    picks muddy olive-browns, because near-grey colours are "far" from
    everything in CIEDE2000 terms while being useless as an instrument's
    palette -- the whole point is that the user can name the colour.
    """
    candidates = []
    for L in np.linspace(*l_range, l_steps):
        for h in np.linspace(0.0, 360.0, hue_steps, endpoint=False):
            c_max = in_gamut_chroma(float(L), float(h))
            if c_max < min_chroma:
                continue
            for fraction in chroma_fractions:
                C = max(min_chroma, c_max * fraction)
                if C > c_max:
                    continue
                lab = np.array([L, C * np.cos(np.radians(h)), C * np.sin(np.radians(h))])
                bgr = lab_to_bgr(lab).astype(np.float64)
                # Only keep colours that survive the round trip cleanly.
                if float(np.max(np.abs(bgr_to_lab(bgr) - lab))) > 4.0:
                    continue
                candidates.append(bgr)
    return np.unique(np.round(np.asarray(candidates)).astype(np.uint8), axis=0)


# --------------------------------------------------------------------------- #
# Distances / search
# --------------------------------------------------------------------------- #
def worst_case_matrix(colors: np.ndarray) -> np.ndarray:
    """``(N, N)`` minimum distance across all viewer types."""
    worst = None
    for mode in MODES:
        labs = bgr_to_lab(simulate_colors(colors, mode))
        matrix = delta_e_ciede2000(labs[:, None, :], labs[None, :, :])
        np.fill_diagonal(matrix, np.inf)
        worst = matrix if worst is None else np.minimum(worst, matrix)
    return worst


def farthest_point_select(candidates: np.ndarray, count: int, seed_index: int, vividness=0.12):
    """Greedy max-min dispersion over the candidate grid."""
    # Distance from every candidate to every candidate is too big; instead keep
    # a running "distance to the closest chosen colour" vector and update it one
    # chosen colour at a time.
    n = len(candidates)
    dmin = np.full(n, np.inf)

    # Pre-simulate the candidates once per mode.
    simulated = [bgr_to_lab(simulate_colors(candidates, mode)) for mode in MODES]
    chroma = np.linalg.norm(candidates.astype(np.float64) - 128.0, axis=1)  # rough vividness proxy

    chosen = [seed_index]
    _update(dmin, simulated, seed_index, candidates)

    for _ in range(count - 1):
        best = np.max(dmin)
        # Among near-optimal candidates prefer the more saturated one: the
        # geometry is what matters, but there is no reason to ship muddy colours
        # when a vivid one is equally well separated.
        viable = np.flatnonzero(dmin >= best - vividness * max(best, 1e-6))
        pick = int(viable[np.argmax(chroma[viable])])
        chosen.append(pick)
        _update(dmin, simulated, pick, candidates)

    return chosen


def _update(dmin: np.ndarray, simulated, index: int, candidates: np.ndarray) -> None:
    point = [sim[index] for sim in simulated]
    worst = None
    for sim, p in zip(simulated, point):
        d = delta_e_ciede2000(p, sim)
        worst = d if worst is None else np.minimum(worst, d)
    np.minimum(dmin, worst, out=dmin)


# --------------------------------------------------------------------------- #
# Ordering and pattern assignment
# --------------------------------------------------------------------------- #
def cvd_order(colors: np.ndarray) -> np.ndarray:
    """Order the palette the way a dichromat experiences it: blue-yellow, then light.

    ``b*`` under deuteranopia is the yellow-blue axis they still have, so
    walking the palette from yellow to blue is a visible, learnable sequence
    for every viewer -- which is what the on-screen keyboard strip relies on.
    """
    labs = bgr_to_lab(simulate_colors(colors, "deuteranopia"))
    return np.lexsort((labs[:, 0], labs[:, 2]))


def assign_patterns(
    colors: np.ndarray,
    max_patterns: int = 8,
    threshold: float = 14.0,
) -> tuple[list[int], int]:
    """Colour the "hard to tell apart" graph with the 8 texture motifs.

    Graph colouring, not a hue lookup.  Two colours closer than *threshold*
    under the worst-case viewer get different patterns wherever the graph is
    8-colourable, so the texture carries real information: whenever the colour
    is ambiguous, the pattern is not.  Returns ``(patterns, leftover_pairs)``
    where ``leftover_pairs`` counts the pairs that had to share anyway.
    """
    matrix = worst_case_matrix(colors)
    n = len(colors)
    # Most-constrained-first ordering makes the greedy colouring near-optimal.
    pressure = np.sort(matrix, axis=1)[:, :max_patterns].sum(axis=1)
    order = np.argsort(-pressure)

    patterns = [-1] * n
    for i in order:
        neighbours = np.flatnonzero(matrix[i] < threshold)
        used = {patterns[j] for j in neighbours if patterns[j] >= 0}
        free = [p for p in range(max_patterns) if p not in used]
        if free:
            patterns[i] = free[0]
        else:
            load = [
                sum(1 for j in neighbours if patterns[j] == p) for p in range(max_patterns)
            ]
            patterns[i] = int(np.argmin(load))

    leftover = 0
    for i in range(n):
        for j in np.flatnonzero(matrix[i] < threshold):
            if j > i and patterns[i] == patterns[j]:
                leftover += 1
    return patterns, leftover


# --------------------------------------------------------------------------- #
# Reporting
# --------------------------------------------------------------------------- #
def family_of(bgr) -> int:
    """Index of the closest Okabe-Ito anchor, by hue (reported for reference)."""
    hue = lab_hue_deg(bgr_to_lab(np.asarray(bgr, dtype=np.float64)))
    best, best_d = 0, 1e9
    for i, (_, rgb) in enumerate(OKABE_ITO):
        anchor_hue = lab_hue_deg(bgr_to_lab(np.array(rgb, dtype=np.float64)))
        d = min(abs(hue - anchor_hue), 360 - abs(hue - anchor_hue))
        if d < best_d:
            best, best_d = i, d
    return best


def main() -> int:

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--emit", action="store_true", help="print the Python literal")
    parser.add_argument("--count", type=int, default=COUNT)
    args = parser.parse_args()

    print("building candidate grid ...")
    candidates = build_candidates()
    print(f"  {len(candidates)} usable colours")

    # Seed on the Okabe-Ito orange, which is a strong, unambiguous starting point.
    target = np.array([0, 159, 230], dtype=np.float64)      # BGR of (230, 159, 0)
    seed = int(np.argmin(np.linalg.norm(candidates - target, axis=1)))

    print(f"selecting {args.count} colours by farthest-point sampling ...")
    chosen = farthest_point_select(candidates, args.count, seed)
    palette = candidates[chosen]
    palette = palette[cvd_order(palette)]
    patterns, leftover = assign_patterns(palette)

    print()
    print("worst-case separation (min over normal vision + 3 dichromacies):")
    for name, colors in (
        ("hue wheel (original)", build_hue_wheel(52).colors),
        ("designed palette", palette),
    ):
        sep = worst_case_separation(colors, MODES)
        hue_sep = worst_case_separation(colors, ["none"])
        print(f"  {name:<22} worst-case {sep:6.2f}   (normal vision only: {hue_sep:6.2f})")

    labs = bgr_to_lab(palette.astype(np.float64))
    print()
    print(f"  lightness range L*   = {labs[:,0].min():.0f} .. {labs[:,0].max():.0f}")
    print(f"  pattern load         = {sorted(Counter(patterns).values())}")
    print(f"  confusable pairs sharing a pattern: {leftover}")
    print(f"  nearest Okabe-Ito anchor per colour: "
          f"{sorted(Counter(family_of(c) for c in palette).items())}")

    if args.emit:
        print()
        print("CVD_SAFE_COLORS = [")
        for color, pattern in zip(palette, patterns):
            r, g, b = int(color[2]), int(color[1]), int(color[0])
            print(f"    (0x{r:02X}, 0x{g:02X}, 0x{b:02X}),  # pattern {pattern}")
        print("]")
        print()
        print(f"CVD_SAFE_PATTERNS = {patterns}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
