"""Image cleanup before Textract, and bad copies of photos (blurry, dark, small, tilted) for testing the photo check
(upload_check.image_quality). Pillow only.

A cleaned or bad copy is saved as its own file, so Textract reads it like any other photo or page.
"""
import io

from PIL import Image, ImageEnhance, ImageFilter, ImageOps

from upload_check import image_quality

BAD_UPLOADS = ["blurry", "dark", "small", "medium", "tilted"]


def load_image(source):
    """A photo (file path) or page (PNG bytes), upright. Phone photos are stored sideways with a rotation tag."""
    image = Image.open(io.BytesIO(source) if isinstance(source, bytes) else source)
    return ImageOps.exif_transpose(image)


def clean(image, name):
    """name "contrast": grayscale, with the lightest and darkest 2% stretched to white and black.
    name "sharp": the same, then sharpened so thin pen marks and dots stand out."""
    gray = ImageOps.autocontrast(ImageOps.grayscale(image), cutoff=2)
    if name == "sharp":
        gray = gray.filter(ImageFilter.UnsharpMask(radius=2, percent=150, threshold=3))
    return gray


def degrade(image, name):
    """A bad copy of a photo, like the uploads a phone often gives.

    blurry: out of focus or shaken (blur scaled to the photo's size). dark: taken in poor light. small: shrunk to
    800 pixels on the long side, as some messaging apps send photos. medium: 1,600 pixels, as email apps send a
    photo when asked to shrink it. tilted: turned 12 degrees, with the table showing around it."""
    image = image.convert("RGB")
    if name == "blurry":
        return image.filter(ImageFilter.GaussianBlur(max(image.size) / 800))
    if name == "dark":
        return ImageEnhance.Contrast(ImageEnhance.Brightness(image).enhance(0.35)).enhance(0.8)
    if name in ("small", "medium"):
        smaller = image.copy()
        smaller.thumbnail((800, 800) if name == "small" else (1600, 1600))
        return smaller
    if name == "tilted":
        return image.rotate(12, expand=True, fillcolor=(120, 120, 120))
    raise ValueError(f"{name} isn't one of {BAD_UPLOADS}")
