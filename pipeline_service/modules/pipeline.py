from __future__ import annotations

import base64
import io
import time
from typing import Literal, Optional

from PIL import Image
import torch
import gc

from config import Settings, settings
from logger_config import logger
from schemas import (
    GenerateRequest,
    GenerateResponse,
    TrellisParams,
    TrellisRequest,
    TrellisResult,
)
from modules.image_edit.qwen_edit_module import QwenEditModule
from modules.background_removal.rmbg_manager import BackgroundRemovalService
from modules.gs_generator.trellis_manager import TrellisService
from modules.utils import (
    secure_randint,
    set_random_seed,
    decode_image,
    to_png_base64,
    to_png_base64_any,
    composite_rgba_on_solid_background,
    make_tone_variant,
    save_files,
)


class GenerationPipeline:
    def __init__(self, settings: Settings = settings):
        self.settings = settings
        self.qwen_edit = QwenEditModule(settings)
        self.rmbg = BackgroundRemovalService(settings)
        self.trellis = TrellisService(settings)

    async def startup(self) -> None:
        logger.info("Starting pipeline")
        self.settings.output_dir.mkdir(parents=True, exist_ok=True)
        await self.qwen_edit.startup()
        await self.rmbg.startup()
        await self.trellis.startup()
        # logger.info("Warming up generator...")
        # await self.warmup_generator()
        self._clean_gpu_memory()
        logger.success("Warmup is complete. Pipeline ready to work.")

    async def shutdown(self) -> None:
        logger.info("Closing pipeline")
        await self.qwen_edit.shutdown()
        await self.rmbg.shutdown()
        await self.trellis.shutdown()
        logger.info("Pipeline closed.")

    def _clean_gpu_memory(self) -> None:
        gc.collect()
        torch.cuda.empty_cache()

    # --- HÀM CỐT LÕI 1: CHUẨN BỊ ẢNH (CHỈ CHẠY 1 LẦN) ---
    async def prepare_input_images(
        self, image_bytes: bytes, seed: int = 42
    ) -> list[Image.Image]:
        """
        Prepare object-only images for 3D generation.

        Default path (fast, identity-preserving):
        - segment once (soft alpha)
        - composite on a neutral background
        - optionally return 3 multi-crop variants (tight/medium/loose)

        Optional path (slow, can drift identity): Qwen-edited left/right views.
        """
        image_base64 = base64.b64encode(image_bytes).decode("utf-8")
        image = decode_image(image_base64)
        if seed < 0:
            seed = secure_randint(0, 10000)
        set_random_seed(seed)

        # Fast path: object-only segmentation + neutral background + multi-crop variants
        if self.settings.use_multicrop_views and not self.settings.use_qwen_views:
            paddings = self.settings.multicrop_padding_factors
            rgba_crops = [
                self.rmbg.remove_background_rgba(image, padding_factor=paddings[0]),
                self.rmbg.remove_background_rgba(image, padding_factor=paddings[1]),
                self.rmbg.remove_background_rgba(image, padding_factor=paddings[2]),
            ]
            rgb_crops: list[Image.Image] = [
                composite_rgba_on_solid_background(rgba, self.settings.object_bg_color)
                for rgba in rgba_crops
            ]

            # Optional: add one mild tone variant (usually helps glossy objects + cluttered lighting)
            if self.settings.use_tone_variant and rgb_crops:
                mid = rgb_crops[min(1, len(rgb_crops) - 1)]
                rgb_crops.append(
                    make_tone_variant(
                        mid,
                        contrast=self.settings.tone_contrast,
                        saturation=self.settings.tone_saturation,
                        brightness=self.settings.tone_brightness,
                    )
                )

            # Optional: second neutral background (encourage invariance)
            if self.settings.object_bg_color_alt is not None:
                rgb_crops.append(
                    composite_rgba_on_solid_background(
                        rgba_crops[min(1, len(rgba_crops) - 1)],
                        self.settings.object_bg_color_alt,
                    )
                )

            # Keep count small for latency; Trellis doesn’t need lots of images here.
            return rgb_crops[:5]

        # Optional slow path: Qwen-generated views (may drift; use only if you must)
        if self.settings.use_qwen_views:
            left_image_edited = self.qwen_edit.edit_image(
                prompt_image=image,
                seed=seed,
                prompt=(
                    "Show this object in left three-quarters view and make sure it is fully visible. "
                    "Keep the SAME exact object. Do not change shape, details, logo, or text. "
                    "Turn background neutral solid color contrasting with the object. Delete background details. "
                    "Keep object colors. Sharpen image details."
                ),
            )
            right_image_edited = self.qwen_edit.edit_image(
                prompt_image=image,
                seed=seed,
                prompt=(
                    "Show this object in right three-quarters view and make sure it is fully visible. "
                    "Keep the SAME exact object. Do not change shape, details, logo, or text. "
                    "Turn background neutral solid color contrasting with the object. Delete background details. "
                    "Keep object colors. Sharpen image details."
                ),
            )
            left_rgb = composite_rgba_on_solid_background(
                self.rmbg.remove_background_rgba(left_image_edited),
                self.settings.object_bg_color,
            )
            right_rgb = composite_rgba_on_solid_background(
                self.rmbg.remove_background_rgba(right_image_edited),
                self.settings.object_bg_color,
            )
            orig_rgb = composite_rgba_on_solid_background(
                self.rmbg.remove_background_rgba(image),
                self.settings.object_bg_color,
            )
            return [left_rgb, right_rgb, orig_rgb]

        # Fallback: single image object-only
        orig_rgb = composite_rgba_on_solid_background(
            self.rmbg.remove_background_rgba(image),
            self.settings.object_bg_color,
        )
        return [orig_rgb]

    # --- HÀM CỐT LÕI 2: CHẠY TRELLIS (CHẠY NHIỀU LẦN VỚI SEED KHÁC NHAU) ---
    async def generate_trellis_only(
        self,
        processed_images: list[Image.Image],
        seed: int,
        mode: Literal[
            "single", "multi_multi", "multi_sto", "multi_with_voxel_count"
        ] = "multi_with_voxel_count",
    ) -> bytes:
        """Chỉ chạy tạo 3D từ ảnh đã xử lý."""
        trellis_params = TrellisParams.from_settings(self.settings)
        set_random_seed(seed)

        trellis_result = self.trellis.generate(
            TrellisRequest(
                images=processed_images,
                seed=seed,
                params=trellis_params,
            ),
            mode=mode,
        )

        if not trellis_result or not trellis_result.ply_file:
            raise ValueError("Trellis generation failed")

        return trellis_result.ply_file

    # --- API Wrapper Cũ (Refactored) ---
    async def generate_gs(self, request: GenerateRequest) -> GenerateResponse:
        t1 = time.time()
        logger.info(f"New generation request")

        if request.seed < 0:
            request.seed = secure_randint(0, 10000)

        # Decode từ request để lấy bytes cho hàm prepare
        img_bytes = base64.b64decode(request.prompt_image)

        # 1. Prepare Images
        processed_images = await self.prepare_input_images(img_bytes, request.seed)

        # 2. Generate Trellis
        ply_bytes = await self.generate_trellis_only(processed_images, request.seed)

        # 3. Tạo kết quả trả về (Mock lại TrellisResult để save file nếu cần)
        # Lưu ý: Logic save file cũ đang nằm rải rác, mình giả lập lại response
        if self.settings.save_generated_files:
            # Reconstruct dummy result object if needed for saving logic
            pass

        t2 = time.time()
        generation_time = t2 - t1
        logger.info(f"Total generation time: {generation_time} seconds")
        self._clean_gpu_memory()

        return GenerateResponse(
            generation_time=generation_time,
            ply_file_base64=ply_bytes,  # Trả về bytes trực tiếp, controller sẽ encode base64
            # Các trường image_edited tạm thời để None hoặc cần logic riêng để lấy ra từ processed_imgs nếu muốn trả về
            image_edited_file_base64=to_png_base64_any(processed_images)
            if self.settings.send_generated_files
            else None,
            image_without_background_file_base64=to_png_base64_any(processed_images)
            if self.settings.send_generated_files
            else None,
        )

    # API cũ wrapper
    async def generate_from_upload(self, image_bytes: bytes, seed: int) -> bytes:
        # Tái sử dụng logic mới
        processed_images = await self.prepare_input_images(image_bytes, seed)
        return await self.generate_trellis_only(processed_images, seed)
