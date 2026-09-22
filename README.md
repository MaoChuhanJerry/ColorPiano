# ColorPiano Pro

**See colour with your ears.**

> [中文版 README](README.zh-CN.md)

About 1 in 12 men and 1 in 200 women cannot reliably tell red from green. Their eyesight is not
degraded in any general sense. What they are missing is one specific channel of information: the
thing that separates a ripe tomato from an unripe one, a brake light from a tail light, one red
cable from a bundle of grey ones, a warning label from a label.

This program lets you listen to that channel instead.

Point a webcam at something and the colour in the middle of the frame becomes a piano note. A
piano has 52 white keys, and 52 is far more distinct values than a language supplies colour names
for. Each colour gets its own note, and a colour-blind user can learn to recognise any of them by
ear at a resolution their eyes do not offer.

The usual accessibility answer to this problem is to name the colour in words or draw a symbol
beside it. This program does that too, but names are coarse. "Green" and "blue" each have to cover
a large region of colour space, and that region is where a colour-blind person most needs finer
distinctions. A name gives back an answer less precise than the question. Pitch does not have that
problem: an untrained ear separates dozens of pitches without effort, so the user ends up with a
measuring instrument rather than a caption. Not "that's greenish" but "that is D4, and the one
beside it is F4", which is enough to sort, match and discriminate.

The rest of the code exists to make that dependable: colour maths that holds up, a set of colours
chosen so that no two of them sit close together, extra cues for when pitch alone is not enough,
and a way to say "I am not sure" when the program is not.

```
python main.py                                  # drive it from a webcam
python main.py --source synthetic               # no camera? still runs
python main.py --selftest --pipeline            # self test + end-to-end pipeline test
```

---

## Contents

- [How it works: four channels to the ear](#how-it-works-four-channels-to-the-ear)
- [What it does](#what-it-does)
- [The algorithms](#the-algorithms)
- [Quick start](#quick-start)
- [Usage](#usage)
- [Performance](#performance)
- [Verifying it works](#verifying-it-works)
- [Known limitations](#known-limitations)
- [Background](#background)
- [Troubleshooting](#troubleshooting)
- [Project layout](#project-layout)
- [License](#license)

---

## How it works: four channels to the ear

One colour becomes four independent signals. A user can rely on as many of them as they can hear.

| Channel | Encodes | Mechanism |
|---|---|---|
| Pitch | which colour | 52 notes, 52 colours. The colour index maps to a note index, so pitch is a stable and learnable identifier rather than an arbitrary label. |
| Loudness | lightness | Note velocity follows CIELAB `L*` (`0.35 + brightness × 0.85`, clamped to 0.15–1.0). Lightness is the one dimension every type of colour vision retains, so it anchors the other three. |
| Rhythm (earcon) | texture class | Each of the 8 texture patterns has a motif of 1 to 8 short pulses at 75 ms spacing. Even patterns ascend, odd patterns descend, and each pulse steps up 12% in pitch. Two axes, so the motifs can be told apart by ear alone. |
| Speech | the colour, in words | With a TTS engine installed: "dark vivid orange, note D4, pattern diagonal". The wording is intentionally verbose, since "dark vivid orange" carries information that `#A0522D` does not, and it is rate limited to about one announcement per second. A continuous stream of speech while an object moves is noise, not information. |

Pitch is the primary channel. The other three exist to resolve the cases where two colours land on
adjacent notes, or where the user wants confirmation. The texture motif matters most of the three,
because pattern is the only cue that survives every kind of colour blindness, which is why it is
represented both visually and audibly.

Speech degrades quietly in both directions. With no TTS engine the program plays the pattern
earcon instead and runs normally. With an engine installed but failing to start, it reports why
and stops talking rather than raising. Nothing in `announce.py` imports a speech library at module
level, since an accessibility feature that turns into an import error is worse than no feature.

---

## What it does

- 52 notes for 52 colours. The 52 `*.wav` files are the white keys of a standard 88-key piano
  (A0–C8; an 88-key piano has exactly 52 white keys). Colours are spread around the hue circle in
  52 steps and notes are sorted low to high.
- Colour matching by CIEDE2000, a difference formula calibrated to human perception rather than to
  RGB arithmetic. The implementation is checked against the published reference data.
- A palette computed to stay separated for colour-blind observers, rather than a hand-picked set of
  eight extended by lightness. Worst-case separation is 4.87 ΔE with no confusable pairs, against
  0.00 and 48 of 52 for the naive evenly-spaced hue wheel.
- A texture pattern on every colour, assigned by graph colouring so that colours which are hard to
  separate receive different patterns. Only 2 of 52 colours share a pattern with a neighbour they
  can be confused with.
- Runtime ambiguity reporting. If the current colour is hard to distinguish from another note under
  your colour vision, the display says so, names the other note, and points to the pattern.
- Live simulation and enhancement, on the preview only.
- Full keyboard and mouse operation. Gestures are an extra. The pipeline is testable without a
  camera, a microphone or a speech engine.

---

## The algorithms

### CIEDE2000, implemented properly

Colour matching is where projects of this kind usually come apart. The common shortcut is:

```python
lab = cv2.cvtColor(np.uint8([[bgr]]), cv2.COLOR_BGR2LAB)[0][0]
```

OpenCV returns 8-bit Lab: `L` squashed to 0–255 rather than 0–100, `a`/`b` also 0–255 rather than
roughly −128 to 127, plus uint8 rounding error. The consequence is worse than it looks, because
CIEDE2000's weighting functions are calibrated for `L∈[0,100]`. Feed them 0–255 and `S_L`, `S_C`,
`S_H` and `R_T` are all wrong. Using CIEDE2000 on 8-bit values costs you the whole formula and
returns none of its accuracy, which makes it worse than plain RGB distance.

`colorpiano/colorspace.py` carries the full floating-point chain: sRGB, linear, XYZ, CIELAB, then
CIEDE2000. It is checked against all 34 reference data sets from Sharma, Wu and Dalal (2005), the
test set the formula was published with:

```
CIEDE2000: 34 cases, 0 failures, worst error 4.95e-05
```

That number is a unit test, not a claim.

### Choosing colours that stay apart

The obvious approach is to pick 8 colour-blind-safe hues and vary lightness to fill out 52 colours.
It does not work, for a geometric reason. Lay 52 colours out as 8 hues across 7 lightness bands and
the 8 hues within a band sit on a circle in `L*`. A dichromat projects that circle onto a line, and
the two ends of the circle land on top of each other. Measured on that layout: 51 of the 52 notes
have a neighbour that some viewer cannot distinguish.

`tools/design_palette.py` works in the space the viewer actually has instead. For each dichromacy,
simulation maps colour space onto two dimensions, roughly lightness plus the blue-yellow axis.
Colours are placed by farthest-point sampling in that reduced space, minimising the worst case:
the smallest pairwise ΔE across normal vision and protanopia, deuteranopia and tritanopia. The
red-green axis a dichromat cannot see is then spent on differentiation for normal-sighted users,
where it costs nothing.

| Palette | Worst-case ΔE | Normal vision only | Entries with a twin |
|---|---|---|---|
| 52 evenly spaced hues | 0.00 | 0.38 | 48 / 52 |
| `cvd_safe` (computed, default) | 4.87 | 5.71 | 0 / 52 |

Worst-case ΔE is the minimum over normal vision, protanopia, deuteranopia and tritanopia, then the
minimum over all colour pairs. 4.87 is about two just-noticeable differences, which clears sensor
noise and webcam drift. Both palettes ship. `hue` is kept so users can compare against the naive
layout and hear the difference (press `p`).

The measurement also turned up something worth knowing about that naive layout: it is
indistinguishable even to normal vision. Its entries 17 and 18 are `(0,255,9)` and `(19,255,0)`, a
CIELAB hue difference of 0.6° and a ΔE of 0.38. Near the green corner of the sRGB gamut a 6.92°
step in HSV hue is worth only 0.45° of Lab hue, while elsewhere the same step is worth 25°. The 52
"evenly spaced" hues are not evenly spaced.

### Texture patterns by graph colouring

Each colour also carries one of 8 texture patterns: solid, horizontal, vertical, cross-hatch,
diagonal, rings, grid, dots. The pattern is drawn on the sampling box and the note panel and played
as an earcon.

Assignment is not a lookup by hue. It is a graph colouring: build a graph whose edges join colours
that are hard to separate under the worst-case viewer, then colour that graph with 8 patterns.
Colours that look alike end up with different patterns, which is the case where the pattern has to
carry the information by itself. Result: 2 of 52 colours share a pattern with a neighbour they can
be confused with.

### Runtime ambiguity analysis

At start-up the program computes a 52×52 simulated colour-difference matrix for every type of
colour vision. At runtime, if the sampled colour is hard to distinguish from another note under the
user's own colour vision, the display says so:

```
under deuteranopia this also looks like D4 (dE 4.2) -- go by pattern 'diagonal'
```

A tool that guesses wrong and stays silent is worse than one that says nothing, because the user
cannot tell the two cases apart. Stating the ambiguity sends attention to the cues that still work:
the earcon, the pattern, the note name.

There is a version of the same idea that does not depend on colour vision. If the sampled colour
falls midway between two notes, the display reports the distance to each: "this colour is between
two notes: 3.1 from A3, 4.6 from B3".

### Smoothing that does not wrap around the hue circle

Colour indices are circular, so 50 and 2 are two sides of the same hue. Averaging them
arithmetically gives 26, a colour that was never on screen. `palette.py` smooths in CIELAB and
takes a circular mean of hue in angle space, and rejects outlier samples so that a hand or a shadow
crossing the sampling box does not drag the estimate with it. A test pins this down.

Smoothing costs about 8 frames of lag, roughly 0.27 s by default. That is deliberate: an unsmoothed
view flickers at the slightest movement, and a flickering note is unplayable. `--smoothing` changes
it.

### Refusing to guess at grey

Grey, black and white have no hue. Run them through 52 hue comparisons anyway and you get a
confident answer that means nothing, and the user, unable to see that the object is colourless, has
no way to know the answer was arbitrary. Below a CIELAB chroma threshold the program declares no
colour, stays silent, and explains why on screen.

### Audio that does not block the video

Two things have to hold for the instrument to be playable. A note must not freeze the frame, and a
new note must not cut off the previous one.

- A scheduler thread runs timed events off the render loop, so sample playback, arpeggios and
  earcons are queued rather than slept on. The render loop never blocks.
- A channel pool lets notes overlap, the way a real instrument with a sustain pedal does.
- Volume lives on the channel, not on the `Sound` object. This is what makes per-note velocity
  possible: mutating shared `Sound` objects to change volume would contaminate the velocity of
  every following note.
- `--bank pure` substitutes a single sine with no harmonics, which gives the least ambiguous pitch.
  That is the right bank for a user identifying colours by ear rather than by timbre.

### Daltonization, and what it is actually worth

Two colour-vision filters are provided, and both affect the display only:

- `d` enhance (daltonize) moves the information a colour vision defect loses into the channels that
  remain.
- `s` simulate renders the preview as a selected colour vision type sees it, so a normally-sighted
  user can check whether an object is genuinely distinguishable.

The naive daltonize is "add the correction, then clip to gamut". Clipping is not a neutral
operation. On a pair of saturated reds, clipping removed the difference the correction had just
created and ΔE fell from 5.03 to 0.00, meaning enhancement had made two distinct colours identical.
The implementation here scales the correction by each pixel's remaining gamut headroom. It gives up
a little improvement and never does harm.

What it is worth, measured:

| Palette | Confusable pairs | Median ΔE change | Improved / worsened |
|---|---|---|---|
| `cvd_safe` + protanopia | 11 | +3.39 | 9 / 2 |
| 52 hues + protanopia | 139 | +0.01 | 60 / 30 |

Enhancement cannot rescue a bad palette, because the information the user needs is not in the image
to be re-routed. Choosing the right palette is what helps. Daltonization cleans up residual
confusions on a good palette and does little else.

Two invariants are enforced. Enhancement applies to the preview only, so note selection always uses
the original frame and "what the program thinks this colour is" cannot drift with a display toggle.
And filter cost is independent of image content, which is why `--filter-scale` exists.

### MediaPipe cannot load its own model from a non-ASCII path

This affects anyone installing the project, so it is worth stating. When MediaPipe's install path
contains non-ASCII characters, including Chinese, Cyrillic, accented letters, or a Windows "copy"
suffix, its C++ layer fails to open its own bundled model file and reports `The path does not
exist` about a file that is present. Native extensions register their resource directory at load
time, so patching `sys.path` afterwards has no effect.

`colorpiano/gestures.py` inspects the install path before the first `import mediapipe` and, if it
is non-ASCII, copies MediaPipe to a plain-ASCII temporary directory and imports from there, cached
by version and done once. Separately, recent MediaPipe wheels removed the bundled
`mp.solutions.hands` API and ship no model file, so an 8 MB download is now required. Both APIs are
supported and auto-detected, and the self test says which path you are on. A plain-ASCII install
path is still the better answer.

---

## Quick start

```bash
git clone <this-repo> ~/colorpiano        # or C:\dev\colorpiano on Windows
cd ~/colorpiano
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS / Linux:
source .venv/bin/activate

pip install -r requirements.txt
python main.py
```

On Windows, one script does the install and fetches the optional hand model:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\setup.ps1
```

Install into a plain-ASCII path. See the MediaPipe note above for why. The program works around it,
but a clean path is better.

Three dependencies are required: `opencv-contrib-python`, `numpy`, `pygame`. `mediapipe` for
gestures and `pyttsx3` for spoken colour names are optional, and the program reports at start-up
what it found.

### Check the installation

```bash
python main.py --selftest --skip-hardware     # does not touch camera or sound card
python main.py --selftest --pipeline          # adds the end-to-end pipeline test
```

---

## Usage

```bash
python main.py                                          # webcam
python main.py --palette cvd_safe --cvd deuteranopia    # recommended starting point for CVD users
python main.py --bank pure --announce                   # clearest pitch + speech/earcon feedback
python main.py --source synthetic                       # synthetic test pattern, no camera needed
python main.py --source video --source-path demo.mp4    # loop a video file
python main.py --range chromatic --range-start 60        # 52 semitones instead of 52 white keys
python main.py --record out.mp4                         # record the annotated view
python main.py --filter-scale 0.25                      # faster colour-vision filters
```

A reasonable first session for a colour-blind user is `--palette cvd_safe --bank pure --announce`:
maximum palette separation, unambiguous pitch, and an earcon or spoken confirmation on every change.

### Keys

| Key | Action | Key | Action |
|---|---|---|---|
| `q` / `Esc` | quit | `h` | toggle the key list |
| `Space` | sustained note (retrigger the same note) | `b` | bank: piano / synth / pure sine |
| `p` | palette: cvd_safe / hue | `c` | cycle colour-vision type |
| `1`–`5` | set colour-vision type directly | `0` | back to normal vision |
| `d` | live enhance (daltonize) | `s` | simulate (see what they see) |
| `t` | texture patterns on/off | `n` | colour names on/off |
| `k` | keyboard strip on/off | `a` | speech / pattern earcons |
| `g` | gestures on/off | `L` | large-text mode |
| `m` | mute | `-` / `+` | volume |
| Arrow keys | move the sampling box | `[` / `]` | sampling box size |
| `r` | reset the sampling box | Left click / wheel | sample here / resize box |

### Gestures

Gestures are for when your hands are busy and are never required. Every gesture is judged from the
whole hand's pose, normalised by the hand's own size, and must be held for several consecutive
frames, so it does not depend on distance from the camera and does not misfire when the hand
rotates.

| Gesture | Action | Trigger |
|---|---|---|
| Thumb up | volume +5% | hold to keep adjusting |
| Thumb down | volume −5% | hold to keep adjusting |
| Peace sign | next timbre | one-shot; release before repeating |
| Closed fist | stop all sound | one-shot; release before repeating |
| Open palm | pause recognition | continuous |
| Index point | sampling box follows the finger | continuous |
| Pinch | sampling box jumps to the pinch point | continuous |

---

## Performance

With the colour-vision filters off, the program measures about 30 fps at 1280×720 on an ordinary
laptop webcam, which is the camera's own ceiling. The bottleneck is not here.

The colour-vision filters are the only expensive part, because they are per-pixel colour transforms
whose cost does not depend on image content. The first version dropped the preview from 30 fps to
3.8 fps. Three rounds of optimisation followed:

| Optimisation | Effect |
|---|---|
| sRGB transfer function via lookup table, replacing a per-pixel 2.4 power | 350 ms → 210 ms per frame |
| Fused daltonize's four matrix multiplies into two | small |
| `_gamut_scale` reduced from 4 reductions to 2 | 47 ms → 41 ms |

One further "optimisation" was blocked by a test. Replacing the gain bound with a cheaper symmetric
form dropped separable colour pairs from 9/11 to 7/11, so the change was reverted.

What shipped instead is `--filter-scale` (default 0.5), which computes the filters at half
resolution and upscales. Assistive use cares about colour relationships, which downsampling does not
affect, and the cost falls quadratically. Measured with the synthetic source and both filters on:

| | Frame rate |
|---|---|
| No filters | 24 fps (the synthetic source's own ceiling; a real camera gives 30 fps) |
| Both filters, `--filter-scale 1.0` | 5.5 fps |
| Both filters, `--filter-scale 0.5` (default) | 15 fps |
| Both filters, `--filter-scale 0.25` | 21.7 fps |
| Real camera + daltonize + gestures | 27.8 fps |

Processing in horizontal strips for cache locality was tried and made no difference at all,
85.7 ms against 87.7 ms, so the limit is numpy throughput rather than cache. The numbers are in the
comments.

Two other trade-offs. Smoothing adds about 8 frames of lag, as described above. Audio banks are
built on demand, so only the piano bank loads by default, at 52 notes and roughly 23 MB, and the
synthetic banks are generated when you press `b`.

---

## Verifying it works

### Unit tests (67)

```bash
python -m unittest discover -s tests -v
```

Coverage includes the 34 CIEDE2000 reference sets, Lab round-trips, the circular-mean wrap-around
regression, outlier rejection in smoothing, hysteresis not oscillating, palette separation,
grey-invariance and red-green collapse in the colour-vision simulation, the boundedness of
enhancement, gesture pose classification and debouncing, pitch-shift ratios, and audio buffer
formats.

### End-to-end pipeline test

```bash
python -m tools.selftest --pipeline --both-palettes
```

The synthetic source paints known palette colours frame by frame, the program derives the note, and
the test asserts the two agree:

```
 end-to-end pipeline: 240 synthetic frames, one colour per 14 frames
   palette cvd_safe: 0 of 52 entries are indistinguishable from another entry to every viewer
    settled frames compared : 17
    exact match             : 17
    twin entry (see above)  : 0
    wrong                   : 0
  PASS colour -> note -> audio
```

Run against the 52-hue palette, the same test reports "twin entry" rather than "wrong". Those two
colours really are the same, so reporting either one is correct.

### Against a real camera

```bash
python main.py --headless --duration 5     # no window, just run the pipeline and print stats
```

```
 processed 120 frames in 4.0s (29.8 fps)
```

### Reproduce the palette design

```bash
python -m tools.design_palette            # redesign and print the comparison
python -m tools.design_palette --emit     # emit paste-ready constants
python main.py --selftest --skip-hardware # print the numbers quoted in this README
```

---

## Known limitations

1. Enhancement (daltonize) has limited effect. It cannot rescue a bad palette and only helps with
   residual confusable pairs on a good one. Choosing the palette is the mechanism that works.
2. The 52-hue palette has 48 of 52 entries with a twin. That is inherent to the layout, not a bug,
   and the palette ships only so users can hear the comparison. Use `cvd_safe`.
3. Smoothing has lag, about 8 frames to settle after an abrupt colour change.
4. 52 colours is a lot to learn. Pitch identifies a colour only once the mapping is learned. The
   keyboard strip, colour names and earcons exist to bootstrap that, but expect practice rather
   than instant fluency.
5. `--range chromatic` borrows its black keys by pitch shifting, derived from nearby white-key
   recordings. Within ±3 semitones it is acceptable and further out it audibly thins.
6. Gestures need an extra model download on modern MediaPipe, and a visible hand. Occlusion,
   backlighting and gloves all reduce the hit rate.
7. Spoken colour names need `pyttsx3`. Without it the program falls back to pattern earcons, so
   nothing is lost.
8. Pitch identification is affected by ambient noise and works best with headphones. The program
   does not warn you about this; you will notice it.

---

## Background

This is a rewrite of an earlier `color_piano` project. The rewrite started because the original's
gesture control had never executed, since MediaPipe cannot open its own model file from a non-ASCII
install path. The reason to keep going was different. The original's colour matching used 8-bit Lab
where CIEDE2000 requires 0–100 `L*`, its 52-hue palette had 48 entries indistinguishable from a
neighbour, and it had no way of telling the user when it was unsure.

So the colour maths was replaced with a verified implementation, the palette was computed rather
than inherited, and ambiguity reporting was added. The original's defects are not catalogued here.
What survives is what is still useful to a reader: the MediaPipe portability constraint above, and
the palette measurements, which are the evidence for why the palette is computed rather than chosen.

The original project is left untouched.

---

## Troubleshooting

**"Cannot access the camera" at launch**
Check whether another program is holding it, such as meeting software or a browser. Try
`--camera 1`. Or run `--source synthetic` first to confirm the program itself is fine.

**No sound**
The `audio` line in the self test will say. With no sound card the program switches to a silent
driver and keeps running, executing all the logic without producing sound, rather than crashing.

**Pitch is hard to distinguish**
`--bank pure` uses a single sine with no harmonics, which has the clearest pitch and is the right
choice if you identify colours by ear. Headphones help more than anything else.

**Gestures do not work; start-up prints "could not start hand tracking" or mentions a missing model**
Two possible causes, and the self test says which:

- `mediapipe 1.x is installed, which needs a separate hand model` — run
  `python -m tools.fetch_model` once.
- `mediapipe cannot reach its own model file` — the install path contains non-ASCII characters. The
  program copies MediaPipe to a temporary directory to work around it. If that still fails, create
  the venv under a plain-ASCII path such as `C:\dev\colorpiano\.venv`.

**The colour always registers as the neighbouring one**
Lower `--stabilizer` for a snappier switch, or raise `--smoothing` for more stability. If the
display says "under deuteranopia this also looks like ...", those two colours really are hard for
you to separate. Switch to `--palette cvd_safe`.

**Sound is odd, or notes are wrong**
`--velocity-from-brightness` makes loudness follow lightness. `--no-velocity` turns it off if you
would rather have a constant level.

**Text is too small, or contrast is too low**
Press `L` for large-text mode. `--high-contrast` adds a high-contrast border.

**Enabling the colour-vision filters makes it sluggish**
Both filters are per-pixel and cost a fixed amount. Use `--filter-scale 0.25` to trade image
sharpness for frame rate. Assistive use cares about colour relationships, which downsampling does
not change.

---

## Project layout

```
ColorPiano-Pro/
├── main.py                    CLI entry point (argument parsing + launch)
├── requirements.txt
├── assets/
│   └── notes/                 52 note wavs (A0–C8, every white key of an 88-key piano)
├── colorpiano/                the library
│   ├── config.py              paths, constants, Settings (every tunable in one place)
│   ├── notes.py               note-name parsing, sample discovery, fill-in, range modes
│   ├── colorspace.py          sRGB↔CIELAB, CIEDE2000 (verified on 34 reference sets)
│   ├── colorblind.py          CVD simulation, enhancement, confusion analysis
│   ├── palette.py             the two palettes, circular-mean smoothing, hysteresis, naming
│   ├── synth.py               WAV loading, pitch shifting, synthesised notes, pattern earcons
│   ├── audio.py               banks, channel pool, non-blocking scheduler thread
│   ├── vision.py              camera/video/image/synthetic sources, sampling, white balance
│   ├── gestures.py            MediaPipe wrapper, pose classification, debouncing, path repair
│   ├── hud.py                 all on-screen overlay drawing
│   ├── announce.py            speech / earcon feedback
│   ├── diagnostics.py         self-test checks
│   └── app.py                 the main loop, wiring everything together
├── tools/
│   ├── selftest.py            install self test + end-to-end pipeline test
│   └── design_palette.py      design/reproduce the CVD-safe palette
├── tests/
│   └── test_core.py           67 unit tests
└── scripts/                   one-shot install / run scripts
```

The separation is deliberate: numerics in one place, drawing in another, hardware behind an
interface, sound in its own module. `colorblind.py` has no UI text, `hud.py` has no colour maths,
`announce.py` imports no speech library at module level, and `vision.py` hides where frames come
from. That is why the whole pipeline can be tested on a machine with no camera, no microphone and
no speech engine.

---

## License

MIT. The 52 piano samples are used here for demonstration. Verify their licensing before commercial
use.
