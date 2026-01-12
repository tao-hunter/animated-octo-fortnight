from PIL import Image, ImageEnhance

import io
import base64
from datetime import datetime
from typing import Optional
import os
import random
import numpy as np
import torch

from logger_config import logger
from schemas.trellis_schemas import TrellisResult

from config import settings

def composite_rgba_on_solid_background(image_rgba: Image.Image, bg_color: tuple[int, int, int]) -> Image.Image:
    """
    Composite an RGBA image onto a solid background color, returning RGB.
    """
    if image_rgba.mode != "RGBA":
        image_rgba = image_rgba.convert("RGBA")
    bg = Image.new("RGBA", image_rgba.size, color=(*bg_color, 255))
    out = Image.alpha_composite(bg, image_rgba)
    return out.convert("RGB")

def images_to_strip(images: list[Image.Image]) -> Image.Image:
    """
    Concatenate images horizontally for debugging/return payloads.
    """
    if not images:
        raise ValueError("images_to_strip: empty images")
    imgs = [img.convert("RGB") for img in images]
    h = max(i.height for i in imgs)
    w = sum(i.width for i in imgs)
    canvas = Image.new("RGB", (w, h), color=(0, 0, 0))
    x = 0
    for img in imgs:
        canvas.paste(img, (x, 0))
        x += img.width
    return canvas

def to_png_base64_any(image_or_images: Image.Image | list[Image.Image]) -> str:
    """
    Like `to_png_base64`, but supports lists by concatenating to a strip.
    """
    if isinstance(image_or_images, list):
        return to_png_base64(images_to_strip(image_or_images))
    return to_png_base64(image_or_images)

def make_tone_variant(
    image: Image.Image,
    *,
    contrast: float = 0.92,
    saturation: float = 0.96,
    brightness: float = 1.0,
) -> Image.Image:
    """
    Create a mild tone variant to reduce sensitivity to glare/lighting.
    Intended to be identity-preserving.
    """
    img = image.convert("RGB")
    if brightness != 1.0:
        img = ImageEnhance.Brightness(img).enhance(brightness)
    if contrast != 1.0:
        img = ImageEnhance.Contrast(img).enhance(contrast)
    if saturation != 1.0:
        img = ImageEnhance.Color(img).enhance(saturation)
    return img

def secure_randint(low: int, high: int) -> int:
    """ Return a random integer in [low, high] using os.urandom. """
    range_size = high - low + 1
    num_bytes = 4
    max_int = 2**(8 * num_bytes) - 1

    while True:
        rand_bytes = os.urandom(num_bytes)
        rand_int = int.from_bytes(rand_bytes, 'big')
        if rand_int <= max_int - (max_int % range_size):
            return low + (rand_int % range_size)

def set_random_seed(seed: int) -> None:
    """ Function for setting global seed. """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def decode_image(prompt: str) -> Image.Image:
    """
    Decode the image from the base64 string.

    Args:
        prompt: The base64 string of the image.

    Returns:
        The image.
    """
    # Decode the image from the base64 string
    image_bytes = base64.b64decode(prompt)
    return Image.open(io.BytesIO(image_bytes)).convert("RGB")

def to_png_base64(image: Image.Image) -> str:
    """
    Convert the image to PNG format and encode to base64.

    Args:
        image: The image to convert.

    Returns:
        Base64 encoded PNG image.
    """
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")

    # Convert to base64 from bytes to string
    return base64.b64encode(buffer.getvalue()).decode("utf-8")

def save_file_bytes(data: bytes, folder: str, prefix: str, suffix: str) -> None:
    """
    Save binary data to the output directory.

    Args:
        data: The data to save.
        folder: The folder to save the file to.
        prefix: The prefix of the file.
        suffix: The suffix of the file.
    """
    target_dir = settings.output_dir / folder
    target_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.utcnow().strftime("%Y-%m-%d_%H-%M-%S")
    path = target_dir / f"{prefix}_{timestamp}{suffix}"
    try:
        path.write_bytes(data)
        logger.debug(f"Saved file {path}")
    except Exception as exc:
        logger.error(f"Failed to save file {path}: {exc}")

def save_image(image: Image.Image, folder: str, prefix: str, timestamp: str) -> None:
    """
    Save PIL Image to the output directory.

    Args:
        image: The PIL Image to save.
        folder: The folder to save the file to.
        prefix: The prefix of the file.
        timestamp: The timestamp of the file.
    """
    target_dir = settings.output_dir / folder / timestamp
    target_dir.mkdir(parents=True, exist_ok=True)
    path = target_dir / f"{prefix}.png"
    try:
        image.save(path, format="PNG")
        logger.debug(f"Saved image {path}")
    except Exception as exc:
        logger.error(f"Failed to save image {path}: {exc}")

def save_files(
    trellis_result: Optional[TrellisResult], 
    image_edited: Image.Image, 
    image_without_background: Image.Image
) -> None:
    """
    Save the generated files to the output directory.

    Args:
        trellis_result: The Trellis result to save.
        image_edited: The edited image to save.
        image_without_background: The image without background to save.
    """
    # Save the Trellis result if available
    if trellis_result:
        if trellis_result.ply_file:
            save_file_bytes(trellis_result.ply_file, "ply", "mesh", suffix=".ply")

    # Save the images using PIL Image.save()
    timestamp = datetime.utcnow().strftime("%Y-%m-%d_%H-%M-%S")
    save_image(image_edited, "png", "image_edited", timestamp)
    save_image(image_without_background, "png", "image_without_background", timestamp)

