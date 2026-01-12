from __future__ import annotations

import time
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from transformers import AutoModelForImageSegmentation
from torchvision import transforms
from torchvision.transforms.functional import to_pil_image, resized_crop

from config import Settings
from logger_config import logger


class BackgroundRemovalService:
    def __init__(self, settings: Settings):
        """
        Initialize the BackgroundRemovalService.
        """
        self.settings = settings

        # Set padding percentage, output size
        self.padding_percentage = self.settings.padding_percentage
        self.output_size = self.settings.output_image_size
        self.limit_padding = self.settings.limit_padding

        # Set device
        self.device = f"cuda:{settings.qwen_gpu}" if torch.cuda.is_available() else "cpu"

        # Set model
        self.model: AutoModelForImageSegmentation | None = None

        # Set transform
        self.transforms = transforms.Compose(
            [
                transforms.Resize(self.settings.input_image_size), 
                transforms.ToTensor(),
            ]
        )

        # Set normalize
        self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
       
    async def startup(self) -> None:
        """
        Startup the BackgroundRemovalService.
        """
        logger.info(f"Loading {self.settings.background_removal_model_id} model...")

        # Load model
        try:
            self.model = AutoModelForImageSegmentation.from_pretrained(
                self.settings.background_removal_model_id,
                torch_dtype=torch.float32,
                trust_remote_code=True,
            ).to(self.device)
            logger.success(f"{self.settings.background_removal_model_id} model loaded.")
        except Exception as e:
            logger.error(f"Error loading {self.settings.background_removal_model_id} model: {e}")
            raise RuntimeError(f"Error loading {self.settings.background_removal_model_id} model: {e}")

    async def shutdown(self) -> None:
        """
        Shutdown the BackgroundRemovalService.
        """
        self.model = None
        logger.info("BackgroundRemovalService closed.")

    def ensure_ready(self) -> None:
        """
        Ensure the BackgroundRemovalService is ready.
        """
        if self.model is None:
            raise RuntimeError(f"{self.settings.background_removal_model_id} model not initialized.")

    def remove_background(self, image: Image.Image) -> Image.Image:
        """
        Remove the background from the image.
        Returns an RGB image (foreground premultiplied over black).

        For object-only pipelines, prefer `remove_background_rgba()` and then
        composite onto a neutral background.
        """
        try:
            t1 = time.time()
            # If already has meaningful alpha, just premultiply it.
            if image.mode == "RGBA":
                rgba = np.array(image).astype(np.float32) / 255.0
                alpha = rgba[:, :, 3:4]
                if np.any(alpha < 0.999):
                    rgb = rgba[:, :, :3] * alpha
                    image_without_background = Image.fromarray(
                        (np.clip(rgb, 0, 1) * 255).astype(np.uint8),
                        mode="RGB",
                    )
                    removal_time = time.time() - t1
                    logger.success(
                        f"Background remove - Time: {removal_time:.2f}s - OutputSize: {image_without_background.size} - InputSize: {image.size}"
                    )
                    return image_without_background

            # PIL.Image (H, W, C) C=3
            rgb_image = image.convert("RGB")

            # Tensor (C, H', W') in [0,1]
            rgb_tensor = self.transforms(rgb_image).to(self.device)
            rgba_tensor = self._remove_background_rgba(rgb_tensor)
            image_without_background = to_pil_image(rgba_tensor[:3].clamp(0, 1))

            removal_time = time.time() - t1
            logger.success(f"Background remove - Time: {removal_time:.2f}s - OutputSize: {image_without_background.size} - InputSize: {image.size}")

            return image_without_background
            
        except Exception as e:
            logger.error(f"Error removing background: {e}")
            return image 

    def remove_background_rgba(self, image: Image.Image, padding_factor: float | None = None) -> Image.Image:
        """
        Remove the background and return an RGBA cutout with a soft alpha matte.
        """
        self.ensure_ready()
        rgb_image = image.convert("RGB")
        rgb_tensor = self.transforms(rgb_image).to(self.device)  # (C,H,W) in [0,1]
        rgba_tensor = self._remove_background_rgba(rgb_tensor, padding_factor=padding_factor)
        return to_pil_image(rgba_tensor.clamp(0, 1))

    def _remove_background(self, image_tensor: torch.Tensor) -> torch.Tensor:
        """
        Backwards-compatible method: returns a (4,H_out,W_out) tensor RGBA.
        """
        return self._remove_background_rgba(image_tensor)

    def _remove_background_rgba(self, image_tensor: torch.Tensor, padding_factor: float | None = None) -> torch.Tensor:
        """
        Core matting/cropping routine.
        Input: image_tensor (C,H,W) in [0,1]
        Output: (4,H_out,W_out) RGBA in [0,1]
        """
        self.ensure_ready()

        input_tensor = self.normalize(image_tensor).unsqueeze(0)
        with torch.no_grad():
            preds = self.model(input_tensor)[-1].sigmoid()  # (1,1,H,W)
            mask = preds[0, 0].float().clamp(0, 1)          # (H,W)

        # Suppress low-confidence shadowy regions (keeps strong foreground)
        try:
            gamma = float(getattr(self.settings, "mask_gamma", 1.0))
        except Exception:
            gamma = 1.0
        if gamma != 1.0:
            mask = mask.pow(gamma)

        # If the scene contains multiple objects, keep only the largest connected component.
        # This helps prevent "double bowls/heads" artifacts.
        try:
            keep_largest = bool(getattr(self.settings, "keep_largest_component", True))
        except Exception:
            keep_largest = True
        if keep_largest:
            thresh_cc = 0.25
            bin_mask = (mask > thresh_cc).detach().to("cpu").numpy().astype(np.uint8)
            if bin_mask.any():
                h0, w0 = bin_mask.shape
                visited = np.zeros((h0, w0), dtype=np.uint8)
                best_count = 0
                best_pixels: list[tuple[int, int]] = []

                # 4-neighborhood flood fill (fast enough for 1024x1024 in practice)
                for y in range(h0):
                    row = bin_mask[y]
                    vrow = visited[y]
                    for x in range(w0):
                        if row[x] == 0 or vrow[x] == 1:
                            continue
                        stack = [(y, x)]
                        visited[y, x] = 1
                        pixels: list[tuple[int, int]] = []
                        while stack:
                            cy, cx = stack.pop()
                            pixels.append((cy, cx))
                            # neighbors
                            ny = cy - 1
                            if ny >= 0 and bin_mask[ny, cx] and not visited[ny, cx]:
                                visited[ny, cx] = 1
                                stack.append((ny, cx))
                            ny = cy + 1
                            if ny < h0 and bin_mask[ny, cx] and not visited[ny, cx]:
                                visited[ny, cx] = 1
                                stack.append((ny, cx))
                            nx = cx - 1
                            if nx >= 0 and bin_mask[cy, nx] and not visited[cy, nx]:
                                visited[cy, nx] = 1
                                stack.append((cy, nx))
                            nx = cx + 1
                            if nx < w0 and bin_mask[cy, nx] and not visited[cy, nx]:
                                visited[cy, nx] = 1
                                stack.append((cy, nx))

                        if len(pixels) > best_count:
                            best_count = len(pixels)
                            best_pixels = pixels

                if best_pixels:
                    keep = np.zeros((h0, w0), dtype=np.float32)
                    ys, xs = zip(*best_pixels)
                    keep[np.array(ys), np.array(xs)] = 1.0
                    keep_t = torch.from_numpy(keep).to(mask.device)
                    mask = mask * keep_t

        # BBox from a modest threshold (keeps thin regions better than 0.8)
        thresh = 0.2
        ys_xs = torch.argwhere(mask > thresh)  # (N,2) -> (y,x)
        h, w = mask.shape

        if ys_xs.numel() == 0:
            top, left, bottom, right = 0, 0, h, w
        else:
            y_min = int(torch.min(ys_xs[:, 0]).item())
            y_max = int(torch.max(ys_xs[:, 0]).item())
            x_min = int(torch.min(ys_xs[:, 1]).item())
            x_max = int(torch.max(ys_xs[:, 1]).item())

            cy = (y_min + y_max) / 2.0
            cx = (x_min + x_max) / 2.0
            box_h = max(1, y_max - y_min + 1)
            box_w = max(1, x_max - x_min + 1)
            size = float(max(box_h, box_w))

            pad = 1.0 + float(self.padding_percentage)
            if padding_factor is not None:
                pad = float(padding_factor)
            size *= pad

            half = size / 2.0
            top = int(round(cy - half))
            bottom = int(round(cy + half))
            left = int(round(cx - half))
            right = int(round(cx + half))

            if self.limit_padding:
                top = max(0, top)
                left = max(0, left)
                bottom = min(h, bottom)
                right = min(w, right)

        crop_args = dict(
            top=top,
            left=left,
            height=max(1, bottom - top),
            width=max(1, right - left),
        )

        # Light feather to reduce jagged edges (cheap 5x5 avg blur)
        mask_4d = mask.unsqueeze(0).unsqueeze(0)  # (1,1,H,W)
        mask_blur = F.avg_pool2d(mask_4d, kernel_size=5, stride=1, padding=2).squeeze(0)  # (1,H,W)

        # Premultiply RGB by alpha and pack RGBA
        tensor_rgba = torch.cat([image_tensor * mask_blur, mask_blur], dim=0)  # (4,H,W)
        output = resized_crop(tensor_rgba, **crop_args, size=self.output_size, antialias=False)
        return output

