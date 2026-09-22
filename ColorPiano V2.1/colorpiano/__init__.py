"""ColorPiano Pro -- turn colours into music, with colour-blind users first.

Package layout::

    config.py      paths, tunables, the Settings dataclass
    notes.py       note naming, sample discovery, colour-index <-> note mapping
    colorspace.py  sRGB <-> CIELAB, CIEDE2000 (verified against Sharma et al.)
    colorblind.py  CVD simulation, daltonization, CVD-safe palette, warnings
    palette.py     the 52-colour palettes and the colour smoother / stabiliser
    audio.py       pygame mixer wrapper: banks, polyphony, non-blocking scheduler
    vision.py      camera / video / image / synthetic frame sources
    gestures.py    MediaPipe hand tracking and the gesture -> command mapping
    hud.py         everything that gets drawn on top of the video frame
    synth.py       runtime synthesis and pitch shifting of missing notes
    announce.py    spoken / earcon feedback
    diagnostics.py start-up self checks
    app.py         the main loop that wires all of the above together
"""

from .config import APP_NAME, APP_VERSION

__all__ = ["APP_NAME", "APP_VERSION", "__version__"]
__version__ = APP_VERSION
