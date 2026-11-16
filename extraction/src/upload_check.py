"""A quick check of a photo before it goes to Textract (cutoffs set on bad copies of our own photos).

Blurry, dark, and very small photos made Textract skip lines, and a skipped line is an entry no check can flag. These
three measures take a fraction of a second, so the app can ask for a retake before paying for Textract. The check
warns rather than blocks: if the assistant sends the photo anyway, its entries get the poor_photo reason and go to
review.

Only for photos (note photos and photographed sheets). Screenshots and PDFs are left out: a dark-mode screenshot
is dark on purpose, and a cropped screenshot can be small and still sharp.
"""
import numpy as np
from PIL import Image, ImageOps

from intake import add_reason

MIN_SHARPNESS = 60     # originals 100 to 1,186; blurry copies 14 to 38
MIN_BRIGHTNESS = 55    # originals 72 to 111; dark copies 24 to 38
MIN_MEGAPIXELS = 4.0   # originals 12 to 49; 1,600-pixel copies (1.9) and 800-pixel copies (0.5) both lost entries
MESSAGES = {
    "blurry": "The photo looks blurry. Hold the phone steady and tap the page to focus.",
    "dark": "The photo looks dark. Turn on a light or move closer to a window.",
    "small": "The photo is very small. Send the original photo, not a shrunk copy.",
}


def image_quality(image, longest=800):
    """Sharpness (variance of the Laplacian, higher is sharper), brightness (mean gray, 0 to 255), and megapixels.
    Sharpness and brightness are measured on a copy shrunk to `longest` pixels, so photos of different sizes compare."""
    gray = ImageOps.grayscale(image)
    gray.thumbnail((longest, longest))
    a = np.asarray(gray, dtype=float)
    laplacian = a[:-2, 1:-1] + a[2:, 1:-1] + a[1:-1, :-2] + a[1:-1, 2:] - 4 * a[1:-1, 1:-1]
    return {"sharpness": round(float(laplacian.var()), 1), "brightness": round(float(a.mean()), 1),
            "megapixels": round(image.size[0] * image.size[1] / 1e6, 2)}


def check_photo(source):
    """{"ok", "problems", "message", sharpness, brightness, megapixels} for a photo (file path or PIL image)."""
    image = source if isinstance(source, Image.Image) else ImageOps.exif_transpose(Image.open(source))
    measures = image_quality(image)
    problems = [name for name, bad in [("blurry", measures["sharpness"] < MIN_SHARPNESS),
                                       ("dark", measures["brightness"] < MIN_BRIGHTNESS),
                                       ("small", measures["megapixels"] < MIN_MEGAPIXELS)] if bad]
    return {"ok": not problems, "problems": problems, "message": " ".join(MESSAGES[p] for p in problems), **measures}


def add_photo_check(entries, check):
    """The entries from a photo that failed the check and was sent anyway, with the poor_photo reason added."""
    if check["ok"] or not len(entries):
        return entries
    return add_reason(entries, [True] * len(entries), "poor_photo")
