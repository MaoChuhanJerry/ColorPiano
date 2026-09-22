"""Unit tests for the parts that are easy to get wrong.

Run with ``python -m unittest discover -s tests -v`` (no pytest needed).

The first test in this file is the important one: it checks the CIEDE2000
implementation against all 34 reference pairs published in Sharma, Wu & Dalal
(2005).  Everything else in the colour pipeline is built on top of it, so if
that function is wrong, every distance in the program is wrong.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from colorpiano import colorblind, synth                                    # noqa: E402
from colorpiano.colorspace import (                                         # noqa: E402
    bgr_to_lab,
    circular_distance_deg,
    circular_mean_deg,
    delta_e_ciede2000,
    lab_chroma,
    lab_hue_deg,
    lab_to_bgr,
)
from colorpiano.config import Settings                                      # noqa: E402
from colorpiano.notes import (                                              # noqa: E402
    build_note_table,
    discover_samples,
    fill_gaps,
    midi_to_freq,
    midi_to_name,
    parse_note_name,
    white_keys_from,
)
from colorpiano.palette import (                                            # noqa: E402
    ColorSmoother,
    NoteStabilizer,
    build_cvd_safe,
    build_hue_wheel,
    build_palette,
    color_name,
)
from colorpiano.vision import gray_world_gains, sample_box                  # noqa: E402


# --------------------------------------------------------------------------- #
# CIEDE2000
# --------------------------------------------------------------------------- #
class TestCiede2000(unittest.TestCase):
    """Sharma, Wu & Dalal (2005), Table 1 -- the standard test set."""

    CASES = [
        ((50.0000, 2.6772, -79.7751), (50.0000, 0.0000, -82.7485), 2.0425),
        ((50.0000, 3.1571, -77.2803), (50.0000, 0.0000, -82.7485), 2.8615),
        ((50.0000, 2.8361, -74.0200), (50.0000, 0.0000, -82.7485), 3.4412),
        ((50.0000, -1.3802, -84.2814), (50.0000, 0.0000, -82.7485), 1.0000),
        ((50.0000, -1.1848, -84.8006), (50.0000, 0.0000, -82.7485), 1.0000),
        ((50.0000, -0.9009, -85.5211), (50.0000, 0.0000, -82.7485), 1.0000),
        ((50.0000, 0.0000, 0.0000), (50.0000, -1.0000, 2.0000), 2.3669),
        ((50.0000, -1.0000, 2.0000), (50.0000, 0.0000, 0.0000), 2.3669),
        ((50.0000, 2.4900, -0.0010), (50.0000, -2.4900, 0.0009), 7.1792),
        ((50.0000, 2.4900, -0.0010), (50.0000, -2.4900, 0.0010), 7.1792),
        ((50.0000, 2.4900, -0.0010), (50.0000, -2.4900, 0.0011), 7.2195),
        ((50.0000, 2.4900, -0.0010), (50.0000, -2.4900, 0.0012), 7.2195),
        ((50.0000, -0.0010, 2.4900), (50.0000, 0.0009, -2.4900), 4.8045),
        ((50.0000, -0.0010, 2.4900), (50.0000, 0.0010, -2.4900), 4.8045),
        ((50.0000, -0.0010, 2.4900), (50.0000, 0.0011, -2.4900), 4.7461),
        ((50.0000, 2.5000, 0.0000), (50.0000, 0.0000, -2.5000), 4.3065),
        ((50.0000, 2.5000, 0.0000), (73.0000, 25.0000, -18.0000), 27.1492),
        ((50.0000, 2.5000, 0.0000), (61.0000, -5.0000, 29.0000), 22.8977),
        ((50.0000, 2.5000, 0.0000), (56.0000, -27.0000, -3.0000), 31.9030),
        ((50.0000, 2.5000, 0.0000), (58.0000, 24.0000, 15.0000), 19.4535),
        ((50.0000, 2.5000, 0.0000), (50.0000, 3.1736, 0.5854), 1.0000),
        ((50.0000, 2.5000, 0.0000), (50.0000, 3.2972, 0.0000), 1.0000),
        ((50.0000, 2.5000, 0.0000), (50.0000, 1.8634, 0.5757), 1.0000),
        ((50.0000, 2.5000, 0.0000), (50.0000, 3.2592, 0.3350), 1.0000),
        ((60.2574, -34.0099, 36.2677), (60.4626, -34.1751, 39.4387), 1.2644),
        ((63.0109, -31.0961, -5.8663), (62.8187, -29.7946, -4.0864), 1.2630),
        ((61.2901, 3.7196, -5.3901), (61.4292, 2.2480, -4.9620), 1.8731),
        ((35.0831, -44.1164, 3.7933), (35.0232, -40.0716, 1.5901), 1.8645),
        ((22.7233, 20.0904, -46.6940), (23.0331, 14.9730, -42.5619), 2.0373),
        ((36.4612, 47.8580, 18.3852), (36.2715, 50.5065, 21.2231), 1.4146),
        ((90.8027, -2.0831, 1.4410), (91.1528, -1.6435, 0.0447), 1.4441),
        ((90.9257, -0.5406, -0.9208), (88.6381, -0.8985, -0.7239), 1.5381),
        ((6.7747, -0.2908, -2.4247), (5.8714, -0.0985, -2.2286), 0.6377),
        ((2.0776, 0.0795, -1.1350), (0.9033, -0.0636, -0.5514), 0.9082),
    ]

    def test_reference_pairs(self):
        worst = 0.0
        for lab1, lab2, expected in self.CASES:
            got = float(delta_e_ciede2000(np.array(lab1), np.array(lab2)))
            worst = max(worst, abs(got - expected))
            self.assertAlmostEqual(got, expected, places=4,
                                   msg=f"{lab1} vs {lab2}")
        self.assertLess(worst, 1e-4)

    def test_is_symmetric(self):
        a = np.array([50.0, 2.5, 0.0])
        for lab2, _, _ in self.CASES[:8]:
            b = np.array(lab2)
            self.assertAlmostEqual(float(delta_e_ciede2000(a, b)),
                                   float(delta_e_ciede2000(b, a)), places=9)

    def test_broadcasts_over_a_palette(self):
        palette = np.random.default_rng(0).uniform([20, -60, -60], [90, 60, 60],
                                                   size=(17, 3))
        got = delta_e_ciede2000(np.array([50.0, 0.0, 0.0]), palette)
        self.assertEqual(got.shape, (17,))
        for i in range(17):
            self.assertAlmostEqual(
                float(got[i]),
                float(delta_e_ciede2000(np.array([50.0, 0.0, 0.0]), palette[i])),
                places=10)


class TestLabConversion(unittest.TestCase):
    def test_round_trip(self):
        for bgr in [(0, 0, 0), (255, 255, 255), (255, 0, 0), (0, 255, 0),
                    (0, 0, 255), (19, 200, 250), (128, 128, 128)]:
            original = np.array(bgr, dtype=np.float64)
            back = lab_to_bgr(bgr_to_lab(original))
            np.testing.assert_array_equal(back.astype(int), np.array(bgr))

    def test_known_values(self):
        """sRGB blue/red must land on the published CIELAB coordinates."""
        # Pure blue, D65: L* 32.30, a* 79.19, b* -107.86
        lab = bgr_to_lab(np.array([255.0, 0.0, 0.0]))
        self.assertAlmostEqual(lab[0], 32.30, places=1)
        self.assertAlmostEqual(lab[1], 79.19, places=1)
        self.assertAlmostEqual(lab[2], -107.86, places=1)
        # White is L* 100 with no chroma.
        lab = bgr_to_lab(np.array([255.0, 255.0, 255.0]))
        self.assertAlmostEqual(lab[0], 100.0, places=3)
        self.assertAlmostEqual(lab_chroma(lab), 0.0, places=3)

    def test_ranges_are_real_units_not_opencv_bytes(self):
        """The original used OpenCV's 8-bit Lab, where L is 0..255.

        CIEDE2000's weights are calibrated for L* in 0..100; feeding it the
        8-bit scaling silently changes every distance, so pin the range down.
        """
        white = bgr_to_lab(np.array([255.0, 255.0, 255.0]))
        self.assertAlmostEqual(white[0], 100.0, delta=0.01)
        red = bgr_to_lab(np.array([0.0, 0.0, 255.0]))
        self.assertLess(red[1], 128.0)     # a* is roughly -128..127, not 0..255
        # OpenCV's own BGR2LAB would put L here in the 200s.
        self.assertLess(bgr_to_lab(np.array([128.0, 128.0, 128.0]))[0], 60.0)


class TestCircularMaths(unittest.TestCase):
    def test_mean_across_the_wrap_point(self):
        """The bug the original shipped: mean of 350 and 10 is 0, not 180."""
        for angles in ([350.0, 10.0], [10.0, 350.0], [350.0, 0.0, 10.0],
                       [350.0, 5.0, 355.0, 10.0]):
            got = circular_mean_deg(np.array(angles))
            self.assertLess(float(circular_distance_deg(got, 0.0)), 1e-6,
                            f"{angles} averaged to {got}")
        # And a value that wraps to 360 is reported as 0, not as 360.
        self.assertLess(circular_mean_deg(np.array([359.0, 1.0])), 1.0)

    def test_weighting(self):
        # Almost all the weight on 90 degrees.
        got = circular_mean_deg(np.array([90.0, 270.0]), np.array([100.0, 1.0]))
        self.assertAlmostEqual(got, 90.0, places=1)

    def test_distance(self):
        self.assertAlmostEqual(float(circular_distance_deg(10.0, 350.0)), 20.0)
        self.assertAlmostEqual(float(circular_distance_deg(0.0, 180.0)), 180.0)


# --------------------------------------------------------------------------- #
# Notes
# --------------------------------------------------------------------------- #
class TestNotes(unittest.TestCase):
    def test_parse(self):
        self.assertEqual(parse_note_name("A0"), 21)
        self.assertEqual(parse_note_name("C4"), 60)
        self.assertEqual(parse_note_name("C8"), 108)
        self.assertEqual(parse_note_name("C#4"), 61)
        self.assertEqual(parse_note_name("Db4"), 61)
        # The original raised ValueError on anything else, which killed the app.
        for junk in ("test", "hello", "C", "4C", "", "Cx4"):
            self.assertIsNone(parse_note_name(junk), junk)

    def test_names_and_frequencies(self):
        self.assertEqual(midi_to_name(69), "A4")
        self.assertAlmostEqual(midi_to_freq(69), 440.0)
        self.assertAlmostEqual(midi_to_freq(60), 261.6256, places=3)
        self.assertAlmostEqual(midi_to_freq(81), 880.0)

    def test_bundled_samples_are_the_52_white_keys(self):
        table = build_note_table(count=52)
        self.assertEqual(len(table), 52)
        self.assertEqual(table[0].name, "A0")
        self.assertEqual(table[-1].name, "C8")
        self.assertTrue(all(not n.is_black_key for n in table))
        self.assertTrue(all(not n.transposed for n in table),
                        "the bundled samples should cover the range exactly")

    def test_chromatic_range_borrows_neighbours(self):
        table = build_note_table(count=52, range_mode="chromatic", range_start=60)
        self.assertEqual(table[0].name, "C4")
        self.assertEqual(len(table), 52)
        self.assertEqual(sum(1 for n in table if n.is_black_key), 22)

    def test_gap_filling_from_a_handful_of_files(self):
        few = discover_samples()[:3]
        filled = fill_gaps(few, 52)
        self.assertEqual(len(filled), 52)
        self.assertEqual(filled[0].name, "A0")
        self.assertTrue(any(n.transposed for n in filled))

    def test_no_samples_at_all_still_plays(self):
        table = fill_gaps([], 52)
        self.assertEqual(len(table), 52)
        self.assertTrue(all(n.synthesized for n in table))

    def test_impossible_range_is_refused(self):
        with self.assertRaises(ValueError):
            white_keys_from(21, 88)

    def test_short_tables_are_allowed(self):
        for count in (1, 12, 30, 52):
            self.assertEqual(len(build_note_table(count=count)), count)


# --------------------------------------------------------------------------- #
# Palettes
# --------------------------------------------------------------------------- #
class TestPalettes(unittest.TestCase):
    def test_sizes_and_ranges(self):
        for name in ("hue", "cvd_safe"):
            palette = build_palette(name, 52)
            self.assertEqual(len(palette), 52)
            self.assertEqual(len(palette.patterns), 52)
            for colour in palette.colors:
                self.assertEqual(len(colour), 3)
                self.assertTrue(all(0 <= v <= 255 for v in colour))
            self.assertTrue(all(0 <= p < 8 for p in palette.patterns))

    def test_cvd_safe_beats_the_hue_wheel_under_every_cvd_type(self):
        """The claim the README makes, asserted rather than asserted-at."""
        hue = colorblind.worst_case_separation(build_hue_wheel(52).colors)
        safe = colorblind.worst_case_separation(build_cvd_safe(52).colors)
        self.assertGreater(safe, hue)
        self.assertGreater(safe, 3.0)

    def test_no_indistinguishable_twins_in_the_safe_palette(self):
        groups = colorblind.indistinguishable_groups(build_cvd_safe(52).colors)
        self.assertEqual(groups, [],
                         f"the CVD-safe palette should have no twins, got {groups}")

    def test_the_hue_wheel_does_have_twins(self):
        """Documented shortcoming of the original palette, pinned as a test."""
        groups = colorblind.indistinguishable_groups(build_hue_wheel(52).colors)
        self.assertTrue(groups, "the hue wheel is expected to contain twin entries")
        flagged = sum(len(g) for g in groups)
        self.assertGreater(flagged, 20)

    def test_each_pattern_is_used_something_like_evenly(self):
        counts = np.bincount(build_cvd_safe(52).patterns, minlength=8)
        self.assertTrue(all(counts > 0), counts)
        self.assertLess(counts.max() - counts.min(), 8, counts)

    def test_unknown_palette_name_is_rejected(self):
        with self.assertRaises(ValueError):
            build_palette("nonesuch")


class TestColorNames(unittest.TestCase):
    def test_names_are_sensible(self):
        """Names are read off CIELAB hue angles, so the boundaries matter."""
        cases = {
            (0, 0, 0): "black",
            (255, 255, 255): "white",
            (128, 128, 128): "grey",
            (0, 0, 255): "red",           # BGR
            (0, 255, 0): "green",
            (255, 0, 0): "blue",
            (0, 255, 255): "yellow",
            (255, 0, 255): "magenta",
            (255, 255, 0): "cyan",
        }
        for bgr, expected in cases.items():
            got = color_name(np.array(bgr, dtype=np.float64))
            self.assertIn(expected, got, f"{bgr} -> {got!r}")

    def test_names_describe_lightness_and_chroma(self):
        dark = color_name(np.array([0, 0, 90], dtype=np.float64))
        light = color_name(np.array([150, 150, 255], dtype=np.float64))
        self.assertNotEqual(dark, light)
        self.assertIn("vivid", color_name(np.array([3, 3, 255], dtype=np.float64)))


# --------------------------------------------------------------------------- #
# Smoothing and stabilisation
# --------------------------------------------------------------------------- #
class TestColorSmoother(unittest.TestCase):
    def test_wrap_around_regression(self):
        """The original averaged palette *indices*, so 50 and 2 gave 26.

        Here the smoothing happens in CIELAB, so samples either side of the hue
        wheel's seam must stay at the seam instead of jumping to the middle.
        """
        smoother = ColorSmoother(size=8)
        # Two colours 20 degrees apart, straddling 0 degrees in Lab hue.
        for hue in (350.0, 10.0, 355.0, 5.0, 0.0, 358.0):
            chroma = 60.0
            lab = np.array([55.0,
                            chroma * np.cos(np.radians(hue)),
                            chroma * np.sin(np.radians(hue))])
            result = smoother.push(lab)
        self.assertLess(float(circular_distance_deg(result.hue, 0.0)), 12.0,
                        f"smoothed hue drifted to {result.hue}")
        self.assertGreater(result.chroma, 40.0)

    def test_outliers_are_rejected(self):
        smoother = ColorSmoother(size=6)
        # BGR (255, 0, 0) is pure blue, whose CIELAB hue angle is 306 degrees.
        for _ in range(5):
            smoother.push(bgr_to_lab(np.array([255.0, 0.0, 0.0])))
        result = smoother.push(bgr_to_lab(np.array([0.0, 255.0, 0.0])))  # green blip
        self.assertEqual(result.outliers, 1)
        self.assertLess(float(circular_distance_deg(result.hue, 306.0)), 25.0,
                        f"the outlier moved the hue to {result.hue}")

    def test_single_sample_is_passed_through(self):
        smoother = ColorSmoother(size=4)
        result = smoother.push(bgr_to_lab(np.array([0.0, 0.0, 255.0])))
        self.assertEqual(result.samples, 1)
        self.assertFalse(result.is_grey)

    def test_grey_is_flagged(self):
        smoother = ColorSmoother(size=4)
        for _ in range(4):
            result = smoother.push(bgr_to_lab(np.array([128.0, 128.0, 128.0])))
        self.assertTrue(result.is_grey)

    def test_reset(self):
        smoother = ColorSmoother(size=4)
        smoother.push(bgr_to_lab(np.array([0.0, 0.0, 255.0])))
        smoother.reset()
        self.assertEqual(smoother.count, 0)


class TestNoteStabilizer(unittest.TestCase):
    def test_requires_consecutive_agreement(self):
        stabilizer = NoteStabilizer(needed=4)
        self.assertEqual(stabilizer.update(10), (10, True))     # first wins
        for _ in range(3):
            self.assertEqual(stabilizer.update(11), (10, False))
        index, changed = stabilizer.update(11)
        self.assertEqual((index, changed), (11, True))

    def test_does_not_flicker_between_two_notes(self):
        stabilizer = NoteStabilizer(needed=4)
        stabilizer.update(20)
        # Alternating input must never change the note.
        for i in range(20):
            index, changed = stabilizer.update(21 if i % 2 else 20)
            self.assertFalse(changed)
            self.assertIn(index, (20, 21))
        self.assertEqual(stabilizer.current, 20)

    def test_reset(self):
        stabilizer = NoteStabilizer(needed=3)
        stabilizer.update(5)
        stabilizer.reset()
        self.assertIsNone(stabilizer.current)
        self.assertEqual(stabilizer.update(9), (9, True))


# --------------------------------------------------------------------------- #
# Colour vision deficiency
# --------------------------------------------------------------------------- #
class TestColorblind(unittest.TestCase):
    def test_grey_stays_grey(self):
        grey = np.full((1, 4, 3), 128, dtype=np.uint8)
        for mode in colorblind.CVD_MODES:
            out = colorblind.simulate(grey, mode)
            spread = int(out.max()) - int(out.min())
            self.assertLessEqual(spread, 2, f"{mode} tinted a neutral gray")

    def test_none_is_the_identity(self):
        frame = np.random.default_rng(1).integers(0, 256, (8, 8, 3), dtype=np.uint8)
        np.testing.assert_array_equal(colorblind.simulate(frame, "none"), frame)

    def test_red_and_green_collapse_for_a_deuteranope(self):
        """The whole reason the project exists."""
        red = np.array([0, 0, 255], dtype=np.uint8)      # BGR
        green = np.array([0, 255, 0], dtype=np.uint8)
        original = float(delta_e_ciede2000(bgr_to_lab(red.astype(float)),
                                           bgr_to_lab(green.astype(float))))
        simulated = float(delta_e_ciede2000(
            bgr_to_lab(colorblind.simulate_colors([red], "deuteranopia")[0]),
            bgr_to_lab(colorblind.simulate_colors([green], "deuteranopia")[0])))
        self.assertGreater(original, 50.0)
        self.assertLess(simulated, original / 3.0,
                        "deuteranopia should collapse red and green")

    def _simulated_separation(self, pair, mode: str) -> float:
        image = np.asarray(pair, dtype=np.uint8).reshape(1, -1, 3)
        simulated = colorblind.simulate(image, mode).reshape(-1, 3).astype(float)
        return float(delta_e_ciede2000(bgr_to_lab(simulated[0]),
                                       bgr_to_lab(simulated[1])))

    def _daltonize_deltas(self, palette_name: str, mode: str,
                          threshold: float = 6.0) -> list[float]:
        palette = build_palette(palette_name, 52)
        deltas = []
        for i in range(52):
            for j in range(i + 1, 52):
                pair = np.stack([palette.colors[i], palette.colors[j]]).astype(np.uint8)
                before = self._simulated_separation(pair, mode)
                if before > threshold:
                    continue
                shifted = colorblind.daltonize(pair.reshape(1, 2, 3), mode, 1.0)[0]
                deltas.append(self._simulated_separation(shifted, mode) - before)
        return deltas

    def test_daltonize_helps_protanopia_on_a_good_palette(self):
        """Where daltonization measurably works, it works well.

        Protanopes lose the most information, so there is the most to recover --
        a median gain of over 3 CIEDE2000 units on the colour-blind palette's
        few remaining hard pairs.
        """
        deltas = self._daltonize_deltas("cvd_safe", "protanopia")
        self.assertGreaterEqual(len(deltas), 5)
        improved = sum(1 for d in deltas if d > 0.05)
        self.assertGreaterEqual(improved, len(deltas) * 3 // 4,
                                f"only {improved}/{len(deltas)} pairs improved")
        self.assertGreater(float(np.median(deltas)), 2.0)

    def test_daltonize_harm_is_bounded_and_rare(self):
        """The property that matters most: it must not do damage.

        The catastrophic failure mode is clipping.  A naive implementation adds
        the correction and lets it clip, and clipping is not neutral: two
        saturated reds both lose the injected difference, and one measured pair
        went from dE00 5.0 to 0.0 -- daltonization made them *identical*.
        Scaling the correction to fit the gamut removes that case entirely.

        What remains is bounded and palette-dependent, and the bound is stated
        rather than glossed over: on the recommended palette no pair loses more
        than a just-noticeable difference, while on the collapsed hue wheel a
        small minority of pairs can still lose up to about 3.5, which is one
        more reason the palette is the thing to fix first.
        """
        for mode in ("protanopia", "deuteranopia", "tritanopia"):
            deltas = self._daltonize_deltas("cvd_safe", mode)
            if deltas:
                self.assertGreater(min(deltas), -1.5,
                                   f"cvd_safe/{mode}: lost {abs(min(deltas)):.2f} dE00")

        hue = self._daltonize_deltas("hue", "protanopia")
        self.assertGreater(min(hue), -4.0,
                           f"hue/protanopia: lost {abs(min(hue)):.2f} dE00")
        harmed = sum(1 for d in hue if d < -0.05)
        self.assertLess(harmed / len(hue), 0.25,
                        f"{harmed}/{len(hue)} hue-wheel pairs got worse")

    def test_daltonization_is_not_a_substitute_for_a_good_palette(self):
        """A measured, slightly humbling result worth pinning down.

        Daltonization barely moves a badly-separated palette: on the original
        hue wheel the median change across its 139 confusable pairs is
        essentially zero, because the information the viewer needs was never in
        the image.  Choosing the colour-blind-safe palette is the actual fix.
        """
        hue = self._daltonize_deltas("hue", "protanopia")
        self.assertGreater(len(hue), 100)
        self.assertLess(abs(float(np.median(hue))), 1.0)
        safe = self._daltonize_deltas("cvd_safe", "protanopia")
        self.assertGreater(float(np.median(safe)), float(np.median(hue)) + 1.0,
                           "the palette should be where the benefit shows up")

    def test_daltonize_is_a_no_op_without_a_deficiency(self):
        frame = np.random.default_rng(2).integers(0, 256, (4, 4, 3), dtype=np.uint8)
        np.testing.assert_array_equal(colorblind.daltonize(frame, "none"), frame)

    def test_transfer_tables_are_accurate(self):
        """The filters run through lookup tables; the tables must be right.

        Decoding is exact (256 rows cover the whole 8-bit input domain).
        Encoding is sampled, so it is checked against the exact function and
        against the size of one 8-bit step -- anything coarser would be visible
        as banding in the preview.
        """
        from colorpiano.colorspace import (
            decode_image,
            encode_to_u8,
            linear_to_srgb,
            srgb_to_linear,
        )

        levels = np.arange(256, dtype=np.uint8)
        error = float(np.max(np.abs(srgb_to_linear(levels / 255.0) - decode_image(levels))))
        self.assertLess(error, 1e-6)

        linear = np.linspace(0.0, 1.0, 65536).astype(np.float32)
        exact = np.round(linear_to_srgb(linear) * 255.0).astype(np.int32)
        got = encode_to_u8(linear).astype(np.int32)
        # At most one 8-bit step, i.e. indistinguishable from the exact value.
        self.assertLessEqual(int(np.max(np.abs(exact - got))), 1)

    def test_daltonization_matrix_is_the_composed_form(self):
        """The pre-composed daltonize matrices must equal the step-by-step maths.

        The pipeline was collapsed from four matrix multiplies to two for speed;
        that is only valid because matrix multiplication is associative, and this
        checks the pre-computation did not silently transpose anything.
        """
        for mode in ("protanopia", "deuteranopia", "tritanopia"):
            error_step = np.eye(3) - colorblind._RGB_MATRIX[mode]
            np.testing.assert_allclose(colorblind._ERROR_MATRIX[mode], error_step,
                                       atol=1e-12)
            shift_step = (colorblind._M_LMS2RGB
                          @ colorblind._SHIFT[mode]
                          @ colorblind._M_RGB2LMS)
            np.testing.assert_allclose(colorblind._SHIFT_LINEAR[mode], shift_step,
                                       atol=1e-12)

    def test_confusion_table_is_self_consistent(self):
        palette = build_palette("cvd_safe", 52)
        tables = colorblind.build_confusion_tables(palette.colors)
        self.assertEqual(set(tables), {"protanopia", "deuteranopia", "tritanopia"})
        for mode, table in tables.items():
            self.assertEqual(table.nearest.shape, (52,))
            # The nearest *other* colour, never itself.
            self.assertTrue(all(table.nearest[i] != i for i in range(52)), mode)
            self.assertTrue(np.all(table.distance > 0), mode)
            # Symmetry of "is this pair confusable".
            for i in range(52):
                self.assertAlmostEqual(table.matrix[i, table.nearest[i]],
                                       table.distance[i], places=9)


# --------------------------------------------------------------------------- #
# Audio helpers
# --------------------------------------------------------------------------- #
class TestSynth(unittest.TestCase):
    def test_pitch_shift_raises_the_frequency_by_the_right_ratio(self):
        tone = synth.synth_note(220.0, duration=0.5, harmonics=(1.0,), decay=10.0)
        up = synth.pitch_shift(tone, 12)
        # An octave up is twice the frequency, so half the length.
        self.assertAlmostEqual(len(up) / len(tone), 0.5, places=2)

    def test_zero_shift_is_the_identity(self):
        tone = synth.synth_note(440.0, duration=0.1)
        np.testing.assert_array_equal(synth.pitch_shift(tone, 0), tone)

    def test_stereo_buffer_format(self):
        buffer = synth.to_stereo_buffer(np.zeros(1000, dtype=np.float32))
        self.assertEqual(buffer.shape, (1000, 2))
        self.assertEqual(buffer.dtype, np.int16)
        self.assertTrue(buffer.flags["C_CONTIGUOUS"])

    def test_clipping_is_prevented(self):
        loud = np.full(100, 5.0, dtype=np.float32)
        buffer = synth.to_stereo_buffer(loud)
        self.assertLessEqual(int(buffer.max()), 32767)
        self.assertGreaterEqual(int(buffer.min()), -32768)

    def test_earcons_differ_per_pattern(self):
        motifs = [synth.earcon(i) for i in range(8)]
        for i in range(8):
            self.assertGreater(len(motifs[i]), 0)
        self.assertNotAlmostEqual(len(motifs[0]), len(motifs[7]))

    def test_resample_preserves_duration(self):
        tone = synth.synth_note(300.0, duration=0.25)
        out = synth.resample(tone, 48000, 24000)
        self.assertAlmostEqual(len(out) / len(tone), 0.5, places=2)


# --------------------------------------------------------------------------- #
# Vision
# --------------------------------------------------------------------------- #
class TestVision(unittest.TestCase):
    def test_gray_world_is_bounded(self):
        """The correction must not be able to recolour a saturated object."""
        # A strongly blue-tinted scene.
        frame = np.zeros((40, 60, 3), dtype=np.uint8)
        frame[:, :, 0] = 60      # blue
        frame[:, :, 1] = 120
        frame[:, :, 2] = 180
        gains = gray_world_gains(frame)
        self.assertTrue(np.all(gains >= 0.85))
        self.assertTrue(np.all(gains <= 1.17))

    def test_gray_world_is_a_no_op_on_neutral_input(self):
        frame = np.full((40, 60, 3), 120, dtype=np.uint8)
        np.testing.assert_allclose(gray_world_gains(frame), np.ones(3), atol=0.02)

    def test_sample_box_reads_the_colour_it_is_aimed_at(self):
        frame = np.zeros((200, 200, 3), dtype=np.uint8)
        frame[80:120, 80:120] = (10, 20, 230)          # BGR
        sample = sample_box(frame, 100, 100, 20)
        self.assertTrue(sample.valid)
        np.testing.assert_allclose(sample.bgr, [10, 20, 230], atol=3)
        self.assertGreater(sample.chroma, 50.0)

    def test_sample_box_reports_darkness(self):
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        sample = sample_box(frame, 50, 50, 20)
        self.assertFalse(sample.valid)
        self.assertIn("dark", sample.reason)

    def test_sample_box_survives_a_box_off_the_edge(self):
        frame = np.full((60, 60, 3), 120, dtype=np.uint8)
        sample = sample_box(frame, 0, 0, 40)
        self.assertIsInstance(sample.valid, bool)      # must not raise

    def test_synthetic_source_is_deterministic_and_reports_its_colour(self):
        from colorpiano.vision import SyntheticSource

        source = SyntheticSource(320, 240, period=5, noise=0.0,
                                 colors=[(0, 0, 255), (0, 255, 0)])
        for _ in range(5):
            ok, frame = source.read()
            self.assertTrue(ok)
            self.assertEqual(source.target_index, 0)
        ok, frame = source.read()
        self.assertEqual(source.target_index, 1)
        np.testing.assert_allclose(frame[120, 160], [0, 255, 0], atol=1)


# --------------------------------------------------------------------------- #
# Gestures
# --------------------------------------------------------------------------- #
class _Landmark:
    __slots__ = ("x", "y")

    def __init__(self, x, y):
        self.x, self.y = x, y


def make_hand(up=("index", "middle", "ring", "pinky"), thumb_tip=(0.30, 0.70),
              thumb_out=True):
    """Build 21 landmarks for a hand with the named fingers straight.

    A compact hand (wrist to middle knuckle = 0.15 in normalised coordinates)
    so that the thumb can reach clearly above or below the wrist without leaving
    the frame.
    """
    wrist = _Landmark(0.50, 0.90)
    mcp = {
        "index": (0.44, 0.77), "middle": (0.50, 0.75),
        "ring": (0.56, 0.77), "pinky": (0.61, 0.80),
    }
    extended_tip = {
        "index": (0.44, 0.55), "middle": (0.50, 0.53),
        "ring": (0.56, 0.55), "pinky": (0.61, 0.57),
    }
    curled_tip = {
        "index": (0.44, 0.80), "middle": (0.50, 0.79),
        "ring": (0.56, 0.80), "pinky": (0.61, 0.82),
    }
    points = [wrist] + [_Landmark(0.0, 0.0) for _ in range(20)]
    points[1] = _Landmark(0.44, 0.86)          # thumb CMC
    points[2] = _Landmark(0.42, 0.85)          # thumb MCP
    points[3] = _Landmark(0.36, 0.84)          # thumb IP
    points[4] = _Landmark(*thumb_tip)          # thumb TIP

    for name, (mcp_index, pip, dip, tip) in {
        "index": (5, 6, 7, 8), "middle": (9, 10, 11, 12),
        "ring": (13, 14, 15, 16), "pinky": (17, 18, 19, 20),
    }.items():
        mx, my = mcp[name]
        points[mcp_index] = _Landmark(mx, my)
        tip_xy = extended_tip[name] if name in up else curled_tip[name]
        midway = ((mx + tip_xy[0]) / 2, (my + tip_xy[1]) / 2)
        points[pip] = _Landmark(*midway)
        points[dip] = _Landmark(*((midway[0] + tip_xy[0]) / 2,
                                  (midway[1] + tip_xy[1]) / 2))
        points[tip] = _Landmark(*tip_xy)
    return points


class TestGestures(unittest.TestCase):
    def setUp(self):
        from colorpiano.gestures import classify

        self.classify = classify

    def test_open_palm(self):
        pose = self.classify(make_hand())
        self.assertEqual(pose.gesture.value, "open_palm")
        self.assertTrue(all(pose.extended.values()))

    def test_fist(self):
        # Everything curled, and the thumb tucked in against the index knuckle.
        # The thumb resting on the curled index is exactly why the pinch branch
        # has to require a *straight* index finger.
        pose = self.classify(make_hand(up=(), thumb_tip=(0.44, 0.79)))
        self.assertEqual(pose.gesture.value, "fist")
        self.assertEqual(sum(pose.extended.values()), 0)

    def test_thumbs_up_and_down(self):
        up = self.classify(make_hand(up=(), thumb_tip=(0.36, 0.60)))
        self.assertEqual(up.gesture.value, "thumbs_up")
        down = self.classify(make_hand(up=(), thumb_tip=(0.36, 1.02)))
        self.assertEqual(down.gesture.value, "thumbs_down")

    def test_point_and_peace(self):
        self.assertEqual(self.classify(make_hand(up=("index",))).gesture.value, "point")
        self.assertEqual(self.classify(make_hand(up=("index", "middle"))).gesture.value,
                         "peace")

    def test_pinch(self):
        # Index straight, its tip meeting the thumb.
        pose = self.classify(make_hand(up=("index",), thumb_tip=(0.42, 0.58)))
        self.assertEqual(pose.gesture.value, "pinch")
        self.assertLess(pose.pinch_distance, 0.55)

    def test_pose_is_scale_invariant(self):
        """The original compared raw y coordinates, so it broke with distance.

        Every test here is in units of the hand's own size, so scaling the whole
        hand about the origin must not change the answer.
        """
        hand = make_hand(up=(), thumb_tip=(0.36, 0.60))
        scaled = [_Landmark(0.5 + (p.x - 0.5) * 2.5, 0.9 + (p.y - 0.9) * 2.5)
                  for p in hand]
        self.assertEqual(self.classify(scaled).gesture.value, "thumbs_up")

    def test_stabilizer_requires_a_hold(self):
        from colorpiano.gestures import Gesture, GestureStabilizer

        stabilizer = GestureStabilizer(hold_frames=4, repeat_cooldown=0.0)
        fired = [stabilizer.update(Gesture.PEACE, t * 0.1)[1] for t in range(8)]
        self.assertEqual(fired, [False, False, False, True, False, False, False, False])
        self.assertEqual(stabilizer.gesture, Gesture.PEACE)

    def test_a_toggle_fires_once_per_gesture(self):
        """Holding 'peace' must not keep switching the timbre back and forth."""
        from colorpiano.gestures import Gesture, GestureStabilizer

        stabilizer = GestureStabilizer(hold_frames=2, repeat_cooldown=0.0)
        fires = 0
        for step in range(30):
            gesture = Gesture.PEACE if step < 10 else Gesture.NONE
            fires += stabilizer.update(gesture, step * 0.1)[1]
        self.assertEqual(fires, 1)

    def test_a_toggle_re_arms_after_being_released(self):
        from colorpiano.gestures import Gesture, GestureStabilizer

        stabilizer = GestureStabilizer(hold_frames=2, repeat_cooldown=0.0)
        fires = []
        for step in range(40):
            # peace for 5 frames, then release for 5, four times over
            gesture = Gesture.PEACE if (step // 5) % 2 == 0 else Gesture.NONE
            if stabilizer.update(gesture, step * 0.1)[1]:
                fires.append(step)
        self.assertEqual(len(fires), 4, fires)

    def test_a_repeatable_gesture_repeats_but_respects_its_cooldown(self):
        """Thumbs-up is held to ramp the volume, so it must fire repeatedly."""
        from colorpiano.gestures import Gesture, GestureStabilizer

        stabilizer = GestureStabilizer(hold_frames=2, repeat_cooldown=0.5)
        fires = []
        for step in range(40):
            if stabilizer.update(Gesture.THUMBS_UP, step * 0.1)[1]:
                fires.append(step * 0.1)
        self.assertGreaterEqual(len(fires), 3, fires)
        for earlier, later in zip(fires, fires[1:]):
            self.assertGreaterEqual(later - earlier, 0.5 - 1e-9)

    def test_continuous_gestures_never_fire(self):
        from colorpiano.gestures import Gesture, GestureStabilizer

        stabilizer = GestureStabilizer(hold_frames=1, repeat_cooldown=0.0)
        for _ in range(10):
            _, fired = stabilizer.update(Gesture.POINT, 1.0)
            self.assertFalse(fired)

    def test_gesture_help_rows_are_pairs_of_strings(self):
        """The on-screen list unpacks each row as (key, value).

        Shipping a list of bare strings here instead does not fail until the
        overlay is drawn, which is late -- so it is checked here.
        """
        from colorpiano.gestures import gesture_help_rows

        rows = gesture_help_rows()
        self.assertTrue(rows)
        for row in rows:
            self.assertIsInstance(row, tuple)
            self.assertEqual(len(row), 2)
            for cell in row:
                self.assertIsInstance(cell, str)
                self.assertTrue(cell.strip())


# --------------------------------------------------------------------------- #
# Settings
# --------------------------------------------------------------------------- #
class TestSettings(unittest.TestCase):
    def test_overrides_ignore_none(self):
        settings = Settings()
        settings.apply_overrides(volume=0.3, bank=None, palette=None)
        self.assertEqual(settings.volume, 0.3)
        self.assertEqual(settings.bank, "piano")

    def test_overrides_validate(self):
        with self.assertRaises(ValueError):
            Settings().apply_overrides(bank="triangle")
        with self.assertRaises(ValueError):
            Settings().apply_overrides(source="telepathy")
        with self.assertRaises(KeyError):
            Settings().apply_overrides(not_a_setting=1)

    def test_defaults_are_sane(self):
        settings = Settings()
        self.assertEqual(settings.palette, "hue")
        self.assertTrue(settings.patterns)
        self.assertTrue(settings.white_balance)
        self.assertGreater(settings.smoothing, 0)
        self.assertGreater(settings.stabilizer, 0)

    def test_the_help_list_is_well_formed(self):
        """Every overlay row must unpack as (key, value)."""
        from colorpiano.app import ColorPianoApp

        for gestures in (False, True):
            rows = ColorPianoApp._build_help_rows(gestures=gestures)
            self.assertTrue(rows)
            for row in rows:
                self.assertIsInstance(row, tuple, row)
                self.assertEqual(len(row), 2, row)
                self.assertIsInstance(row[0], str, row)
                self.assertIsInstance(row[1], str, row)


if __name__ == "__main__":
    unittest.main(verbosity=2)
