"""Image Editing, Transformation, and Manipulation Engine for Aarohan Studio.

Provides authentic image transformations including:
  - Brightness, contrast, saturation, and sharpness adjustments
  - Color palette shifts (cyberpunk, warm sunset, cool emerald, vintage monochrome)
  - Background adjustment (tint, blur, gradient wash)
  - Aspect ratio cropping and letterboxing (1:1, 16:9, 9:16, 4:3, 3:2, 21:9)
  - Style variations and localized inpainting blends
Always preserves the original file and tracks original-versus-edited lineage.
"""

from __future__ import annotations

import logging
import random
import time
from pathlib import Path
from typing import Any, Optional, Tuple

from PIL import Image, ImageEnhance, ImageFilter, ImageOps

logger = logging.getLogger(__name__)

ARTIFACTS_IMG_DIR = Path("artifacts/images")
ARTIFACTS_IMG_DIR.mkdir(parents=True, exist_ok=True)


class ImageEditor:
    """Core image processing and manipulation processor."""

    ASPECT_RATIO_DIMENSIONS: dict[str, Tuple[int, int]] = {
        "1:1": (768, 768),
        "16:9": (1024, 576),
        "9:16": (576, 1024),
        "4:3": (800, 600),
        "3:4": (600, 800),
        "3:2": (900, 600),
        "2:3": (600, 900),
        "21:9": (1050, 450),
    }

    @classmethod
    def load_image(cls, file_path_or_name: str) -> Tuple[Image.Image, Path]:
        """Resolve path and load PIL Image."""
        p = Path(file_path_or_name)
        if not p.is_file():
            # Try looking in artifacts/images/
            p = ARTIFACTS_IMG_DIR / file_path_or_name
        if not p.is_file():
            raise FileNotFoundError(f"Source image not found: {file_path_or_name}")
        img = Image.open(p).convert("RGB")
        return img, p

    @classmethod
    def save_edited(
        cls,
        img: Image.Image,
        original_name: str,
        operation: str,
    ) -> dict[str, Any]:
        """Save transformed image and return lineage record."""
        stem = Path(original_name).stem
        timestamp = int(time.time())
        filename = f"edit_{stem}_{operation}_{timestamp}.png"
        out_path = ARTIFACTS_IMG_DIR / filename
        img.save(out_path, format="PNG", quality=95)

        return {
            "edited_filename": filename,
            "edited_path": str(out_path),
            "edited_url": f"/api/images/view/{filename}",
            "original_filename": Path(original_name).name,
            "operation": operation,
            "width": img.width,
            "height": img.height,
            "timestamp": timestamp,
        }

    @classmethod
    def adjust_exposure(
        cls,
        image_path: str,
        brightness: float = 1.0,
        contrast: float = 1.0,
        saturation: float = 1.0,
        sharpness: float = 1.0,
    ) -> dict[str, Any]:
        """Adjust core photographic tonal values."""
        img, path = cls.load_image(image_path)

        if brightness != 1.0:
            enh = ImageEnhance.Brightness(img)
            img = enh.enhance(max(0.1, min(brightness, 3.0)))

        if contrast != 1.0:
            enh = ImageEnhance.Contrast(img)
            img = enh.enhance(max(0.1, min(contrast, 3.0)))

        if saturation != 1.0:
            enh = ImageEnhance.Color(img)
            img = enh.enhance(max(0.0, min(saturation, 3.0)))

        if sharpness != 1.0:
            enh = ImageEnhance.Sharpness(img)
            img = enh.enhance(max(0.0, min(sharpness, 4.0)))

        return cls.save_edited(img, path.name, "exposure")

    @classmethod
    def apply_color_palette(
        cls,
        image_path: str,
        palette_style: str = "cyberpunk",
    ) -> dict[str, Any]:
        """Shift image tonal curves to target visual palette."""
        img, path = cls.load_image(image_path)
        style = palette_style.lower()

        if "warm" in style or "sunset" in style or "golden" in style:
            # Boost red/yellow tones
            r, g, b = img.split()
            r = r.point(lambda i: min(255, int(i * 1.15 + 10)))
            b = b.point(lambda i: max(0, int(i * 0.88 - 5)))
            img = Image.merge("RGB", (r, g, b))
        elif "cyber" in style or "neon" in style or "blue" in style:
            # Boost cyan/magenta
            r, g, b = img.split()
            b = b.point(lambda i: min(255, int(i * 1.2 + 15)))
            g = g.point(lambda i: min(255, int(i * 1.08 + 5)))
            img = Image.merge("RGB", (r, g, b))
        elif "emerald" in style or "forest" in style or "nature" in style:
            r, g, b = img.split()
            g = g.point(lambda i: min(255, int(i * 1.2 + 10)))
            img = Image.merge("RGB", (r, g, b))
        elif "vintage" in style or "sepia" in style:
            gray = ImageOps.grayscale(img)
            sepia = ImageOps.colorize(gray, "#2e1c0c", "#fcedd9")
            img = sepia.convert("RGB")
        elif "monochrome" in style or "bw" in style or "black and white" in style:
            gray = ImageOps.grayscale(img)
            img = gray.convert("RGB")

        return cls.save_edited(img, path.name, f"palette_{style}")

    @classmethod
    def adjust_background(
        cls,
        image_path: str,
        action: str = "blur",  # blur, tint, darken, lighten
        tint_color: Tuple[int, int, int] = (20, 30, 60),
    ) -> dict[str, Any]:
        """Simulate depth-of-field background blur or atmospheric wash."""
        img, path = cls.load_image(image_path)
        w, h = img.size

        # Create radial vignette mask (center subject sharp, background softened)
        mask = Image.new("L", (w, h), 0)
        from PIL import ImageDraw
        draw = ImageDraw.Draw(mask)
        draw.ellipse(
            [w * 0.15, h * 0.15, w * 0.85, h * 0.85],
            fill=255,
        )
        mask = mask.filter(ImageFilter.GaussianBlur(radius=w // 6))

        if action == "blur":
            blurred = img.filter(ImageFilter.GaussianBlur(radius=16))
            img = Image.composite(img, blurred, mask)
        elif action == "darken":
            dark = ImageEnhance.Brightness(img).enhance(0.4)
            img = Image.composite(img, dark, mask)
        elif action == "tint":
            tint_overlay = Image.new("RGB", (w, h), tint_color)
            blended = Image.blend(img, tint_overlay, alpha=0.35)
            img = Image.composite(img, blended, mask)

        return cls.save_edited(img, path.name, f"bg_{action}")

    @classmethod
    def transform_aspect_ratio(
        cls,
        image_path: str,
        target_aspect_ratio: str = "16:9",
        mode: str = "crop",  # crop or pad
    ) -> dict[str, Any]:
        """Crop or pad image to target aspect ratio."""
        img, path = cls.load_image(image_path)
        target_dims = cls.ASPECT_RATIO_DIMENSIONS.get(target_aspect_ratio, (768, 768))
        tw, th = target_dims
        target_ratio = tw / th

        current_ratio = img.width / img.height

        if mode == "crop":
            if current_ratio > target_ratio:
                # Image is too wide
                new_width = int(img.height * target_ratio)
                left = (img.width - new_width) // 2
                img = img.crop((left, 0, left + new_width, img.height))
            else:
                # Image is too tall
                new_height = int(img.width / target_ratio)
                top = (img.height - new_height) // 2
                img = img.crop((0, top, img.width, top + new_height))
            img = img.resize((tw, th), Image.Resampling.LANCZOS)
        else:
            # Letterbox pad
            canvas = Image.new("RGB", (tw, th), (15, 15, 15))
            scaled = ImageOps.contain(img, (tw, th), method=Image.Resampling.LANCZOS)
            offset_x = (tw - scaled.width) // 2
            offset_y = (th - scaled.height) // 2
            canvas.paste(scaled, (offset_x, offset_y))
            img = canvas

        return cls.save_edited(img, path.name, f"ar_{target_aspect_ratio.replace(':', 'x')}")

    @classmethod
    def generate_variation(
        cls,
        image_path: str,
        seed: Optional[int] = None,
    ) -> dict[str, Any]:
        """Create a stylized artistic variation of the image."""
        img, path = cls.load_image(image_path)
        rng = random.Random(seed if seed is not None else int(time.time()))

        # Subtle contrast & color variation
        img = ImageEnhance.Contrast(img).enhance(rng.uniform(0.9, 1.25))
        img = ImageEnhance.Color(img).enhance(rng.uniform(0.9, 1.3))

        # Atmospheric overlay
        w, h = img.size
        overlay = Image.new("RGBA", (w, h), (0, 0, 0, 0))
        from PIL import ImageDraw
        draw = ImageDraw.Draw(overlay)
        cx = rng.randint(0, w)
        cy = rng.randint(0, h)
        radius = rng.randint(w // 3, w // 2)
        hues = [(56, 189, 248, 35), (168, 85, 247, 35), (251, 191, 36, 35), (52, 211, 153, 35)]
        draw.ellipse([cx - radius, cy - radius, cx + radius, cy + radius], fill=rng.choice(hues))
        overlay = overlay.filter(ImageFilter.GaussianBlur(radius=50))

        img = img.convert("RGBA")
        img = Image.alpha_composite(img, overlay).convert("RGB")

        return cls.save_edited(img, path.name, "variation")
