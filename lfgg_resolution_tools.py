"""Resolution helper node producing latent shells and previews."""

from __future__ import annotations

import os
from typing import Optional, Tuple

import torch
from PIL import Image, ImageDraw

import folder_paths
from comfy.comfy_types import IO


class LFGGResolutionTools:
    """Derive legal render dimensions from max bounds and an aspect ratio."""

    CATEGORY = "utils/LFGG"
    DESCRIPTION = (
        "Fit an aspect ratio into the provided max width/height, output the clamped "
        "dimensions, and emit an empty latent ready for downstream samplers."
    )
    FUNCTION = "resolve"

    _ASPECT_OPTIONS: Tuple[Tuple[str, Optional[Tuple[int, int]]], ...] = (
        ("SDXL - 1:1 square", (1, 1)),
        ("SDXL - 3:4 portrait", (3, 4)),
        ("SDXL - 5:8 portrait", (5, 8)),
        ("SDXL - 9:16 portrait", (9, 16)),
        ("SDXL - 9:21 portrait", (9, 21)),
        ("SDXL - 4:3 landscape", (4, 3)),
        ("SDXL - 3:2 landscape", (3, 2)),
        ("SDXL - 16:9 landscape", (16, 9)),
        ("SDXL - 21:9 landscape", (21, 9)),
        ("Custom", None),
    )

    @classmethod
    def INPUT_TYPES(cls):
        aspect_labels = [label for label, _ in cls._ASPECT_OPTIONS]
        return {
            "required": {
                "width": (
                    IO.INT,
                    {
                        "default": 1024,
                        "min": 64,
                        "max": 16384,
                        "step": 8,
                        "tooltip": "Maximum width to respect when fitting the aspect ratio.",
                    },
                ),
                "height": (
                    IO.INT,
                    {
                        "default": 1024,
                        "min": 64,
                        "max": 16384,
                        "step": 8,
                        "tooltip": "Maximum height to respect when fitting the aspect ratio.",
                    },
                ),
                "aspect_ratio": (
                    aspect_labels,
                    {
                        "default": "SDXL - 16:9 landscape",
                        "tooltip": "Common SDXL-friendly aspect ratios or Custom mode.",
                    },
                ),
                "aspect_ratio_y": (
                    IO.INT,
                    {
                        "default": 16,
                        "min": 1,
                        "max": 512,
                        "step": 1,
                        "tooltip": "First component of the custom aspect ratio (mapped to width unless swapped).",
                    },
                ),
                "aspect_ratio_x": (
                    IO.INT,
                    {
                        "default": 9,
                        "min": 1,
                        "max": 512,
                        "step": 1,
                        "tooltip": "Second component of the custom aspect ratio (mapped to height unless swapped).",
                    },
                ),
                "swap_dimensions": (
                    IO.BOOLEAN,
                    {
                        "default": False,
                        "tooltip": "Flip the aspect ratio components before fitting (e.g. landscape to portrait).",
                    },
                ),
            }
        }

    RETURN_TYPES = (IO.INT, IO.INT, IO.LATENT)
    RETURN_NAMES = ("height", "width", "empty_latent")
    OUTPUT_TOOLTIPS = (
        "Resolved height after clamping to bounds and multiples of 8.",
        "Resolved width after clamping to bounds and multiples of 8.",
        "Latent dictionary populated with zeros sized to the resolved dimensions.",
    )

    def resolve(
        self,
        width: int,
        height: int,
        aspect_ratio: str,
        aspect_ratio_y: int,
        aspect_ratio_x: int,
        swap_dimensions: bool,
    ):
        ratio_width, ratio_height = self._pick_ratio(aspect_ratio, aspect_ratio_y, aspect_ratio_x)
        if swap_dimensions:
            ratio_width, ratio_height = ratio_height, ratio_width

        final_width, final_height = self._fit_within_bounds(width, height, ratio_width, ratio_height)
        latent = self._make_empty_latent(final_width, final_height)
        ui_preview = self._write_preview(final_width, final_height)

        outputs = (final_height, final_width, latent)
        if ui_preview is not None:
            return outputs, {"ui": {"images": [ui_preview]}}
        return outputs

    @classmethod
    def _pick_ratio(cls, label: str, custom_y: int, custom_x: int) -> Tuple[int, int]:
        for option_label, ratio in cls._ASPECT_OPTIONS:
            if option_label == label and ratio is not None:
                return ratio
        return max(1, int(custom_y)), max(1, int(custom_x))

    @staticmethod
    def _fit_within_bounds(
        max_width: int,
        max_height: int,
        ratio_width: int,
        ratio_height: int,
    ) -> Tuple[int, int]:
        ratio_width = max(1, ratio_width)
        ratio_height = max(1, ratio_height)
        max_width = max(8, max_width)
        max_height = max(8, max_height)

        max_scale = min(max_width // ratio_width, max_height // ratio_height)
        scale = max_scale
        while scale > 0:
            width = ratio_width * scale
            height = ratio_height * scale
            if width % 8 == 0 and height % 8 == 0:
                return width, height
            scale -= 1

        fallback_width = max(8, (max_width // 8) * 8)
        fallback_height = max(8, (max_height // 8) * 8)
        return fallback_width or 8, fallback_height or 8

    @staticmethod
    def _make_empty_latent(width: int, height: int) -> dict:
        latent_width = max(1, width // 8)
        latent_height = max(1, height // 8)
        samples = torch.zeros((1, 4, latent_height, latent_width))
        return {"samples": samples}

    @staticmethod
    def _write_preview(width: int, height: int):
        if width <= 0 or height <= 0:
            return None

        temp_dir = folder_paths.get_temp_directory()
        os.makedirs(temp_dir, exist_ok=True)

        padding = 24
        max_display = 256
        scale = min(max_display / width, max_display / height)
        display_w = max(1, int(round(width * scale)))
        display_h = max(1, int(round(height * scale)))

        canvas_w = display_w + padding * 2
        canvas_h = display_h + padding * 2
        image = Image.new("RGB", (canvas_w, canvas_h), color=(24, 24, 24))
        drawer = ImageDraw.Draw(image)
        box = (
            padding,
            padding,
            padding + display_w - 1,
            padding + display_h - 1,
        )
        drawer.rectangle(box, outline=(255, 255, 255), width=2)
        drawer.rectangle(
            (box[0] + 1, box[1] + 1, box[2] - 1, box[3] - 1),
            outline=(128, 128, 128),
            width=1,
        )

        prefix = os.path.join("lfgg", "resolution_preview", f"{width}x{height}")
        full_output_folder, filename, counter, subfolder, _ = folder_paths.get_save_image_path(
            prefix,
            temp_dir,
            display_w,
            display_h,
        )
        os.makedirs(full_output_folder, exist_ok=True)
        file = f"{filename}_{counter:05}_.png"
        filepath = os.path.join(full_output_folder, file)
        try:
            image.save(filepath, format="PNG")
        except OSError:
            return None

        return {"filename": file, "subfolder": subfolder, "type": "temp", "width": display_w, "height": display_h}


NODE_CLASS_MAPPINGS = {
    "LFGG - Resolution Tools": LFGGResolutionTools,
}