"""Download the hand-tracking model needed by MediaPipe 1.x.

    python -m tools.fetch_model           # download if missing
    python -m tools.fetch_model --force   # download again

Why this exists
---------------
MediaPipe changed its hand-tracking API in 1.0: the old ``mp.solutions.hands``
was removed, the new ``HandLandmarker`` is the only option, and -- unlike the
old API -- the new one ships **no model file in the wheel**.  So on a modern
``pip install mediapipe`` the program has no way to track a hand until this
model is fetched.

The model is about 8 MB and only needs downloading once.  The legacy API used by
mediapipe 0.10.x needs nothing, which is why ``requirements.txt`` pins
``mediapipe<1.0`` by default -- that install works with no extra step.  This
tool is for people who have 1.x, or who want the newer API.
"""

from __future__ import annotations

import argparse
import sys
import urllib.error
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from colorpiano.config import ASSETS_DIR                                  # noqa: E402
from colorpiano.gestures import MODEL_FILENAME, MODEL_URL, find_hand_model  # noqa: E402

TARGET_DIR = ASSETS_DIR / "models"


def download(url: str, destination: Path, timeout: int = 120) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_suffix(destination.suffix + ".part")
    print(f"  fetching {url}")
    request = urllib.request.Request(url, headers={"User-Agent": "colorpiano/2.0"})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        total = int(response.headers.get("Content-Length") or 0)
        done = 0
        with open(partial, "wb") as handle:
            while True:
                chunk = response.read(1 << 16)
                if not chunk:
                    break
                handle.write(chunk)
                done += len(chunk)
                if total:
                    percent = done * 100 // total
                    print(f"\r  {percent:3d}%  {done / 1e6:.1f} / {total / 1e6:.1f} MB",
                          end="", flush=True)
    print()
    partial.replace(destination)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--force", action="store_true",
                        help="download even if a model is already present")
    parser.add_argument("--url", default=MODEL_URL)
    args = parser.parse_args(argv)

    existing = find_hand_model()
    if existing and not args.force:
        size = existing.stat().st_size / 1e6
        print(f"hand model already present: {existing} ({size:.1f} MB)")
        print("nothing to do (use --force to download it again)")
        return 0

    target = TARGET_DIR / MODEL_FILENAME
    print(f"hand model -> {target}")
    try:
        download(args.url, target)
    except (urllib.error.URLError, OSError) as error:
        print(f"\nerror: download failed: {error}", file=sys.stderr)
        print("  Download it manually from:", file=sys.stderr)
        print(f"    {args.url}", file=sys.stderr)
        print(f"  and save it as:", file=sys.stderr)
        print(f"    {target}", file=sys.stderr)
        print("  Alternatively, install the self-contained older API instead:",
              file=sys.stderr)
        print('    pip install "mediapipe<1.0"', file=sys.stderr)
        return 1

    size = target.stat().st_size / 1e6
    print(f"done: {size:.1f} MB")
    if size < 0.1:
        print("warning: the downloaded file looks too small to be the model",
              file=sys.stderr)
        return 1
    print("hand gestures are now available; start the program normally.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
