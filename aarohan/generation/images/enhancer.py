"""Prompt Intelligence and Artistic Enhancer for Aarohan Image Studio.

Translates natural language, informal instructions, and colloquial Hinglish
into professionally structured, high-fidelity diffusion prompts with tailored
lighting, composition, material textures, and negative safety bounds.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional


@dataclass
class EnhancedPromptResult:
    """Output of the prompt intelligence engine."""

    original_prompt: str
    enhanced_prompt: str
    negative_prompt: str
    detected_category: str
    detected_style: str
    suggested_aspect_ratio: str
    lighting: str
    composition: str
    details: list[str]


class PromptEnhancer:
    """Intelligent prompt expansion and visual design reasoning engine."""

    CATEGORIES = {
        "photography": [
            r"\b(photo|photograph|portrait|landscape|nature|wildlife|architecture|real life|dslr|85mm|macro)\b"
        ],
        "digital_art": [
            r"\b(digital art|illustration|concept art|anime|cartoon|fantasy|sci-fi|painting|drawing|artwork)\b"
        ],
        "business_branding": [
            r"\b(logo|banner|linkedin|brand|corporate|marketing|poster|advertisement|presentation|slide|business)\b"
        ],
        "education": [
            r"\b(diagram|anatomy|digestive|scientific|infographic|educational|microscopic|physics|chemical|cell|medical)\b"
        ],
        "ui_design": [
            r"\b(ui|ux|website|hero concept|landing page|mobile app|dashboard|mockup|interface)\b"
        ],
        "render_3d": [
            r"\b(3d|render|octane|isometric|cgi|unreal engine|blender|neural network)\b"
        ],
    }

    HINGLISH_MAP: list[tuple[re.Pattern, str]] = [
        (re.compile(r"\b(banao|banado|bana\s*do|create\s*karo|generate\s*karo)\b", re.I), "create"),
        (re.compile(r"\b(mast|badhiya|shandar|ekdum\s*mast|accha|sundar)\b", re.I), "stunning masterpiece"),
        (re.compile(r"\b(tasveer|tasvir|chitra|chhavi)\b", re.I), "photograph"),
        (re.compile(r"\b(pahad|parvat)\b", re.I), "mountain"),
        (re.compile(r"\b(shehar|shahar)\b", re.I), "cityscape"),
        (re.compile(r"\b(roshni|ujala)\b", re.I), "cinematic lighting"),
    ]

    LIGHTING_STYLES = {
        "photography": "golden hour natural sunlight with soft diffused shadows and volumetric atmospheric haze",
        "digital_art": "dramatic rim lighting with vibrant dual-tone luminous glows and ambient occlusion",
        "business_branding": "clean studio light with subtle softbox gradient illumination and sharp pristine highlights",
        "education": "neutral, high-clarity scientific laboratory lighting with zero shadow distortion",
        "ui_design": "modern translucent glassmorphism lighting with subtle frosted back-glows",
        "render_3d": "raytraced Octane studio lighting with global illumination, specular highlights, and subsurface scattering",
    }

    COMPOSITIONS = {
        "photography": "rule-of-thirds composition, shallow depth of field, sharp foreground focus, 85mm portrait lens",
        "digital_art": "dynamic visual perspective, wide-angle cinematic framing, layered atmospheric depth",
        "business_branding": "minimalist centered hero layout, clean negative space for typography, crisp vector borders",
        "education": "exploded anatomical orthographic view, cross-section detail, clean neutral backdrop",
        "ui_design": "isometric floating perspective, modular layout, high-contrast visual hierarchy",
        "render_3d": "dramatic low-angle hero framing with isometric isometric elevation and rich micro-details",
    }

    DEFAULT_NEGATIVE = (
        "low quality, blurry, pixelated, distorted, extra limbs, poorly drawn hands, "
        "oversaturated, mutated, gross proportions, grainy, watermark, signature, ugly, text artifacts"
    )

    @classmethod
    def normalize_informal(cls, text: str) -> str:
        """Translate informal words and colloquial Hinglish."""
        res = text.strip()
        for pat, repl in cls.HINGLISH_MAP:
            res = pat.sub(repl, res)
        return res

    @classmethod
    def detect_category(cls, prompt: str) -> str:
        """Infer target artistic and professional category."""
        lower = prompt.lower()
        for cat, patterns in cls.CATEGORIES.items():
            for pat in patterns:
                if re.search(pat, lower):
                    return cat
        if "logo" in lower or "banner" in lower:
            return "business_branding"
        if "mountain" in lower or "sunset" in lower or "city" in lower:
            return "photography"
        return "digital_art"

    @classmethod
    def enhance(
        cls,
        user_prompt: str,
        style_override: Optional[str] = None,
        aspect_ratio: Optional[str] = None,
    ) -> EnhancedPromptResult:
        """Construct an optimized, high-fidelity prompt while preserving user intent."""
        cleaned = cls.normalize_informal(user_prompt)
        category = cls.detect_category(cleaned)
        style = style_override or ("photorealistic" if category == "photography" else "cinematic")

        lighting = cls.LIGHTING_STYLES.get(category, cls.LIGHTING_STYLES["photography"])
        composition = cls.COMPOSITIONS.get(category, cls.COMPOSITIONS["photography"])

        # Fidelity tags
        detail_tokens = [
            "hyper-detailed textures",
            "award-winning visual quality",
            "8k resolution",
            "flawless color grading",
        ]

        if category == "business_branding":
            detail_tokens.extend(["minimalist elegance", "vector precision", "commercial design quality"])
            suggested_ar = "16:9" if "banner" in cleaned.lower() else "1:1"
        elif category == "education":
            detail_tokens.extend(["accurate scientific detailing", "educational clarity", "macro precision"])
            suggested_ar = "16:9"
        elif category == "photography":
            detail_tokens.extend(["Kodak Portra color palette", "photorealistic fine grain", "subtle bokeh"])
            suggested_ar = "16:9" if "landscape" in cleaned.lower() else "1:1"
        else:
            detail_tokens.extend(["unreal engine 5 aesthetic", "masterpiece composition", "volumetric depth"])
            suggested_ar = "1:1"

        if aspect_ratio:
            suggested_ar = aspect_ratio

        # Construct final enhanced prompt
        core_subject = cleaned.rstrip(".").strip()
        enhanced_prompt = (
            f"{core_subject}, {style} style, {lighting}, {composition}, "
            f"{', '.join(detail_tokens)}"
        )

        return EnhancedPromptResult(
            original_prompt=user_prompt,
            enhanced_prompt=enhanced_prompt,
            negative_prompt=cls.DEFAULT_NEGATIVE,
            detected_category=category,
            detected_style=style,
            suggested_aspect_ratio=suggested_ar,
            lighting=lighting,
            composition=composition,
            details=detail_tokens,
        )
