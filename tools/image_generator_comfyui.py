"""ComfyUI image generator for ViMax pipeline.

Connects to i9 ComfyUI API (Z-Image Turbo + LoRA) to generate images locally ($0).
Implements the ImageGenerator protocol from tools/protocols.py.
"""

import asyncio
import json
import logging
import random
import urllib.request
import urllib.parse
from io import BytesIO
from typing import List, Optional

from PIL import Image

from interfaces.image_output import ImageOutput
from utils.rate_limiter import RateLimiter

PORTRAIT_WORKFLOW = {
    "1": {"class_type": "UNETLoader", "inputs": {"unet_name": "z_image_turbo_bf16.safetensors", "weight_dtype": "default"}},
    "2": {"class_type": "CLIPLoader", "inputs": {"clip_name": "qwen_3_4b.safetensors", "type": "pixart"}},
    "3": {"class_type": "VAELoader", "inputs": {"vae_name": "ae.safetensors"}},
    "4": {"class_type": "LoraLoader", "inputs": {"model": ["1", 0], "clip": ["2", 0], "lora_name": "z-image-turbo-flow-dpo.safetensors", "strength_model": 0.5, "strength_clip": 0.5}},
    "5": {"class_type": "LoraLoader", "inputs": {"model": ["4", 0], "clip": ["4", 1], "lora_name": "z-image-turbo-sda.safetensors", "strength_model": 0.3, "strength_clip": 0.3}},
    "10": {"class_type": "CLIPTextEncode", "inputs": {"text": "", "clip": ["5", 1]}},
    "11": {"class_type": "CLIPTextEncode", "inputs": {"text": "", "clip": ["5", 1]}},
    "20": {"class_type": "EmptyLatentImage", "inputs": {"width": 896, "height": 1152, "batch_size": 1}},
    "21": {"class_type": "KSampler", "inputs": {"seed": 0, "steps": 8, "cfg": 2.0, "sampler_name": "euler", "scheduler": "sgm_uniform", "denoise": 1.0, "model": ["5", 0], "positive": ["10", 0], "negative": ["11", 0], "latent_image": ["20", 0]}},
    "30": {"class_type": "VAEDecode", "inputs": {"samples": ["21", 0], "vae": ["3", 0]}},
    "40": {"class_type": "SaveImage", "inputs": {"images": ["30", 0], "filename_prefix": "vimax"}},
}


class ImageGeneratorComfyUI:
    """Generate images via i9 ComfyUI API (Z-Image Turbo, $0)."""

    def __init__(
        self,
        comfyui_url: str = "http://100.114.67.45:8188",
        width: int = 896,
        height: int = 1152,
        rate_limiter: Optional[RateLimiter] = None,
    ):
        self.url = comfyui_url.rstrip("/")
        self.width = width
        self.height = height
        self.rate_limiter = rate_limiter

    async def generate_single_image(
        self,
        prompt: str,
        reference_image_paths: List[str] = [],
        aspect_ratio: Optional[str] = None,
        **kwargs,
    ) -> ImageOutput:
        if self.rate_limiter:
            await self.rate_limiter.acquire()

        w, h = self._resolve_dimensions(aspect_ratio)
        seed = kwargs.get("seed", random.randint(0, 2**32 - 1))

        workflow = json.loads(json.dumps(PORTRAIT_WORKFLOW))
        workflow["10"]["inputs"]["text"] = prompt
        workflow["20"]["inputs"]["width"] = w
        workflow["20"]["inputs"]["height"] = h
        workflow["21"]["inputs"]["seed"] = seed

        logging.info(f"ComfyUI: generating image {w}x{h}, seed={seed}")

        prompt_id = await self._queue(workflow)
        output_data = await self._poll(prompt_id)
        image = await self._download_image(output_data)

        return ImageOutput(fmt="pil", ext="png", data=image)

    def _resolve_dimensions(self, aspect_ratio: Optional[str]) -> tuple:
        if not aspect_ratio:
            return self.width, self.height
        ratios = {"16:9": (1152, 640), "9:16": (640, 1152), "1:1": (1024, 1024), "4:3": (1024, 768), "3:4": (768, 1024)}
        return ratios.get(aspect_ratio, (self.width, self.height))

    async def _queue(self, workflow: dict) -> str:
        payload = json.dumps({"prompt": workflow}).encode()
        req = urllib.request.Request(f"{self.url}/prompt", data=payload, headers={"Content-Type": "application/json"})
        loop = asyncio.get_event_loop()
        resp = await loop.run_in_executor(None, lambda: urllib.request.urlopen(req, timeout=30))
        data = json.loads(resp.read())
        return data["prompt_id"]

    async def _poll(self, prompt_id: str, timeout: int = 300) -> dict:
        loop = asyncio.get_event_loop()
        for _ in range(timeout // 2):
            await asyncio.sleep(2)
            resp = await loop.run_in_executor(None, lambda: urllib.request.urlopen(f"{self.url}/history/{prompt_id}", timeout=10))
            history = json.loads(resp.read())
            if prompt_id in history:
                outputs = history[prompt_id].get("outputs", {})
                for node_id, node_out in outputs.items():
                    if "images" in node_out:
                        return node_out["images"][0]
        raise TimeoutError(f"ComfyUI generation timed out after {timeout}s")

    async def _download_image(self, image_data: dict) -> Image.Image:
        filename = image_data["filename"]
        subfolder = image_data.get("subfolder", "")
        params = urllib.parse.urlencode({"filename": filename, "subfolder": subfolder, "type": "output"})
        url = f"{self.url}/view?{params}"
        loop = asyncio.get_event_loop()
        resp = await loop.run_in_executor(None, lambda: urllib.request.urlopen(url, timeout=30))
        return Image.open(BytesIO(resp.read()))
