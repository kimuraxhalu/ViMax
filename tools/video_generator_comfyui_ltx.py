"""ComfyUI LTX 13B video generator for ViMax pipeline.

Connects to i9 ComfyUI API to generate image-to-video locally ($0).
Implements the VideoGenerator protocol from tools/protocols.py.
"""

import asyncio
import json
import logging
import os
import random
import tempfile
import urllib.request
import urllib.parse
from typing import List, Optional

from interfaces.video_output import VideoOutput
from utils.rate_limiter import RateLimiter

I2V_WORKFLOW = {
    "1": {"class_type": "UNETLoader", "inputs": {"unet_name": "ltxv-13b-0.9.7-dev_fp8_e4m3fn.safetensors", "weight_dtype": "fp8_e4m3fn"}},
    "2": {"class_type": "CLIPLoader", "inputs": {"clip_name": "t5xxl_fp8_e4m3fn.safetensors", "type": "ltxv"}},
    "3": {"class_type": "VAELoader", "inputs": {"vae_name": "LTX23_video_vae_bf16.safetensors"}},
    "4": {"class_type": "LoadImage", "inputs": {"image": ""}},
    "16": {"class_type": "ImageScale", "inputs": {"image": ["4", 0], "upscale_method": "lanczos", "width": 768, "height": 512, "crop": "center"}},
    "5": {"class_type": "CLIPTextEncode", "inputs": {"text": "", "clip": ["2", 0]}},
    "6": {"class_type": "CLIPTextEncode", "inputs": {"text": "", "clip": ["2", 0]}},
    "7": {"class_type": "LTXVConditioning", "inputs": {"positive": ["5", 0], "negative": ["6", 0], "frame_rate": 24.0}},
    "8": {"class_type": "LTXVImgToVideo", "inputs": {"positive": ["7", 0], "negative": ["7", 1], "vae": ["3", 0], "image": ["16", 0], "width": 768, "height": 512, "length": 97, "batch_size": 1, "strength": 1.0}},
    "9": {"class_type": "LTXVScheduler", "inputs": {"steps": 30, "max_shift": 2.05, "base_shift": 0.95, "stretch": True, "terminal": 0.1, "latent": ["8", 2]}},
    "10": {"class_type": "KSamplerSelect", "inputs": {"sampler_name": "euler"}},
    "11": {"class_type": "SamplerCustomAdvanced", "inputs": {"noise": ["12", 0], "guider": ["13", 0], "sampler": ["10", 0], "sigmas": ["9", 0], "latent_image": ["8", 2]}},
    "12": {"class_type": "RandomNoise", "inputs": {"noise_seed": 0}},
    "13": {"class_type": "BasicGuider", "inputs": {"model": ["1", 0], "conditioning": ["8", 0]}},
    "14": {"class_type": "VAEDecode", "inputs": {"samples": ["11", 0], "vae": ["3", 0]}},
    "15": {"class_type": "SaveAnimatedWEBP", "inputs": {"images": ["14", 0], "filename_prefix": "vimax_video", "fps": 24, "lossless": False, "quality": 90, "method": "default"}},
}


class VideoGeneratorComfyUILTX:
    """Generate video via i9 ComfyUI LTX 13B (image-to-video, $0)."""

    def __init__(
        self,
        comfyui_url: str = "http://100.114.67.45:8188",
        width: int = 768,
        height: int = 512,
        length: int = 97,
        rate_limiter: Optional[RateLimiter] = None,
    ):
        self.url = comfyui_url.rstrip("/")
        self.width = width
        self.height = height
        self.length = length
        self.rate_limiter = rate_limiter

    async def generate_single_video(
        self,
        prompt: str,
        reference_image_paths: List[str] = [],
        **kwargs,
    ) -> VideoOutput:
        if self.rate_limiter:
            await self.rate_limiter.acquire()

        seed = kwargs.get("seed", random.randint(0, 2**32 - 1))

        # Upload reference image if provided
        if reference_image_paths:
            image_name = await self._upload_image(reference_image_paths[0])
        else:
            image_name = "shia_face_05.png"  # fallback default

        workflow = json.loads(json.dumps(I2V_WORKFLOW))
        workflow["4"]["inputs"]["image"] = image_name
        workflow["5"]["inputs"]["text"] = prompt
        workflow["12"]["inputs"]["noise_seed"] = seed
        workflow["16"]["inputs"]["width"] = self.width
        workflow["16"]["inputs"]["height"] = self.height
        workflow["8"]["inputs"]["width"] = self.width
        workflow["8"]["inputs"]["height"] = self.height
        workflow["8"]["inputs"]["length"] = self.length

        logging.info(f"ComfyUI LTX: generating video {self.width}x{self.height}, {self.length} frames, seed={seed}")

        prompt_id = await self._queue(workflow)
        output_data = await self._poll(prompt_id, timeout=600)
        video_bytes = await self._download_video(output_data)

        return VideoOutput(fmt="bytes", ext="webp", data=video_bytes)

    async def _upload_image(self, image_path: str) -> str:
        """Upload image to ComfyUI input folder, return filename."""
        filename = os.path.basename(image_path)
        with open(image_path, "rb") as f:
            image_data = f.read()

        boundary = "----ViMaxBoundary"
        body = (
            f"--{boundary}\r\n"
            f'Content-Disposition: form-data; name="image"; filename="{filename}"\r\n'
            f"Content-Type: image/png\r\n\r\n"
        ).encode() + image_data + f"\r\n--{boundary}--\r\n".encode()

        req = urllib.request.Request(
            f"{self.url}/upload/image",
            data=body,
            headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
            method="POST",
        )
        loop = asyncio.get_event_loop()
        resp = await loop.run_in_executor(None, lambda: urllib.request.urlopen(req, timeout=30))
        data = json.loads(resp.read())
        return data.get("name", filename)

    async def _queue(self, workflow: dict) -> str:
        payload = json.dumps({"prompt": workflow}).encode()
        req = urllib.request.Request(f"{self.url}/prompt", data=payload, headers={"Content-Type": "application/json"})
        loop = asyncio.get_event_loop()
        resp = await loop.run_in_executor(None, lambda: urllib.request.urlopen(req, timeout=30))
        data = json.loads(resp.read())
        return data["prompt_id"]

    async def _poll(self, prompt_id: str, timeout: int = 600) -> dict:
        loop = asyncio.get_event_loop()
        for _ in range(timeout // 5):
            await asyncio.sleep(5)
            resp = await loop.run_in_executor(None, lambda: urllib.request.urlopen(f"{self.url}/history/{prompt_id}", timeout=10))
            history = json.loads(resp.read())
            if prompt_id in history:
                outputs = history[prompt_id].get("outputs", {})
                for node_id, node_out in outputs.items():
                    if "images" in node_out:
                        return node_out["images"][0]
        raise TimeoutError(f"ComfyUI LTX generation timed out after {timeout}s")

    async def _download_video(self, image_data: dict) -> bytes:
        filename = image_data["filename"]
        subfolder = image_data.get("subfolder", "")
        params = urllib.parse.urlencode({"filename": filename, "subfolder": subfolder, "type": "output"})
        url = f"{self.url}/view?{params}"
        loop = asyncio.get_event_loop()
        resp = await loop.run_in_executor(None, lambda: urllib.request.urlopen(url, timeout=60))
        return resp.read()
