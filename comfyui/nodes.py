"""ComfyUI custom nodes that call remote AI workers over AnyServe.

Drop this directory into ``ComfyUI/custom_nodes/comfyui-anyserve`` and install
``anyserve-runtime`` into ComfyUI's Python environment. Nodes included:

- ``AnyServeConfig``: optional endpoint, token, timeout, and placement config.
- ``AnyServeOpenAIChat``: prompt expansion or VLM captioning via ``openai/chat/<model>``.
- ``AnyServeYOLODetect``: remote object detection + inpainting ``MASK`` via ``yolo/<model>``.
- ``AnyServeInvoke``: call any AnyServe worker interface with text or image payloads.
"""
from __future__ import annotations

import base64
import io
import json
from typing import Any

import numpy as np
import torch
from PIL import Image, ImageDraw

from anyserve import AnyserveClient, ExecutionOptions

DEFAULT_CONFIG: dict[str, Any] = {
    "endpoint": "",
    "token": "",
    "namespace": "",
    "timeout_ms": 120_000,
    "required_attributes": {},
    "preferred_attributes": {},
    "priority": 0,
}


def _parse_kv(raw: str) -> dict[str, str]:
    attrs: dict[str, str] = {}
    for part in (raw or "").split(","):
        part = part.strip()
        if not part:
            continue
        if "=" not in part:
            raise ValueError(f"Invalid attribute {part!r}: expected key=value")
        key, value = part.split("=", 1)
        attrs[key.strip()] = value.strip()
    return attrs


def _make_client_and_options(
    config: dict[str, Any] | None,
    *,
    metadata: dict[str, str] | None = None,
    max_attempts: int = 0,
) -> tuple[AnyserveClient, ExecutionOptions]:
    cfg = {**DEFAULT_CONFIG, **(config or {})}
    client = AnyserveClient(
        endpoint=cfg["endpoint"] or None,
        token=cfg["token"] or None,
        namespace=cfg["namespace"] or None,
    )
    options = ExecutionOptions(
        total_timeout_ms=int(cfg["timeout_ms"]),
        required_attributes=dict(cfg["required_attributes"]),
        preferred_attributes=dict(cfg["preferred_attributes"]),
        priority=int(cfg["priority"]),
        max_attempts=max_attempts,
        metadata=dict(metadata or {}),
    )
    return client, options


def _tensor_frame_to_pil(frame: torch.Tensor) -> Image.Image:
    array = np.clip(frame.detach().cpu().numpy() * 255.0, 0, 255).astype(np.uint8)
    if array.ndim == 2:
        return Image.fromarray(array, mode="L").convert("RGB")
    if array.shape[-1] == 4:
        return Image.fromarray(array, mode="RGBA").convert("RGB")
    return Image.fromarray(array, mode="RGB")


def _pil_to_tensor_frame(img: Image.Image) -> torch.Tensor:
    rgb = img.convert("RGB")
    array = np.asarray(rgb, dtype=np.float32) / 255.0
    return torch.from_numpy(array)


def _encode_pil(img: Image.Image, fmt: str = "JPEG", quality: int = 90) -> bytes:
    buf = io.BytesIO()
    if fmt.upper() == "PNG":
        img.save(buf, format="PNG")
    else:
        img.convert("RGB").save(buf, format="JPEG", quality=quality)
    return buf.getvalue()


class AnyServeConfig:
    """Configure endpoint, auth token fallback, timeouts, and placement attributes."""

    CATEGORY = "AnyServe"
    RETURN_TYPES = ("ANYSERVE_CONFIG",)
    RETURN_NAMES = ("config",)
    FUNCTION = "build"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "endpoint": (
                    "STRING",
                    {"default": "http://127.0.0.1:50052", "multiline": False},
                ),
            },
            "optional": {
                "token": ("STRING", {"default": "", "multiline": False}),
                "namespace": ("STRING", {"default": "", "multiline": False}),
                "timeout_ms": (
                    "INT",
                    {"default": 120_000, "min": 1_000, "max": 1_800_000, "step": 1_000},
                ),
                "required_attributes": ("STRING", {"default": "", "multiline": False}),
                "preferred_attributes": ("STRING", {"default": "", "multiline": False}),
                "priority": ("INT", {"default": 0, "min": 0, "max": 3, "step": 1}),
            },
        }

    def build(
        self,
        endpoint: str,
        token: str = "",
        namespace: str = "",
        timeout_ms: int = 120_000,
        required_attributes: str = "",
        preferred_attributes: str = "",
        priority: int = 0,
    ):
        return (
            {
                "endpoint": endpoint.strip(),
                "token": token.strip(),
                "namespace": namespace.strip(),
                "timeout_ms": int(timeout_ms),
                "required_attributes": _parse_kv(required_attributes),
                "preferred_attributes": _parse_kv(preferred_attributes),
                "priority": int(priority),
            },
        )


class AnyServeOpenAIChat:
    """Call an OpenAI-compatible AnyServe worker (openai/chat/<model>) for prompt expansion or VLM captioning."""

    CATEGORY = "AnyServe"
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("text",)
    FUNCTION = "generate"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("STRING", {"default": "Qwen/Qwen2.5-0.5B-Instruct", "multiline": False}),
                "prompt": (
                    "STRING",
                    {
                        "default": "Expand into a vivid, detailed image generation prompt: a rainy neon alley in Tokyo",
                        "multiline": True,
                    },
                ),
                "system_prompt": (
                    "STRING",
                    {
                        "default": "You are a concise image-prompt assistant. Reply with only the prompt text.",
                        "multiline": True,
                    },
                ),
                "max_tokens": ("INT", {"default": 256, "min": 1, "max": 8192, "step": 1}),
                "temperature": ("FLOAT", {"default": 0.7, "min": 0.0, "max": 2.0, "step": 0.05}),
            },
            "optional": {
                "image": ("IMAGE",),
                "config": ("ANYSERVE_CONFIG",),
            },
        }

    def generate(
        self,
        model: str,
        prompt: str,
        system_prompt: str,
        max_tokens: int,
        temperature: float,
        image: torch.Tensor | None = None,
        config: dict[str, Any] | None = None,
    ):
        client, options = _make_client_and_options(config)
        messages: list[dict[str, Any]] = []
        if system_prompt.strip():
            messages.append({"role": "system", "content": system_prompt.strip()})

        if image is not None:
            pil = _tensor_frame_to_pil(image[0])
            data_url = "data:image/jpeg;base64," + base64.b64encode(_encode_pil(pil, "JPEG", 90)).decode()
            user_content: Any = [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": data_url}},
            ]
        else:
            user_content = prompt
        messages.append({"role": "user", "content": user_content})

        body = {
            "model": model,
            "messages": messages,
            "max_tokens": int(max_tokens),
            "temperature": float(temperature),
            "stream": False,
        }
        raw = client.call(f"openai/chat/{model}", json.dumps(body).encode(), options=options)
        reply = json.loads(raw)
        choices = reply.get("choices") or []
        if not choices:
            raise RuntimeError(f"openai/chat/{model} returned no choices: {reply}")
        text = choices[0].get("message", {}).get("content") or ""
        return (text.strip(),)


class AnyServeYOLODetect:
    """Run remote YOLO detection via AnyServe (yolo/<model>) and output annotated IMAGE + inpainting MASK."""

    CATEGORY = "AnyServe"
    RETURN_TYPES = ("IMAGE", "MASK", "STRING", "INT")
    RETURN_NAMES = ("annotated_image", "mask", "detections_json", "count")
    FUNCTION = "detect"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "model": ("STRING", {"default": "yolo11n", "multiline": False}),
                "min_confidence": ("FLOAT", {"default": 0.25, "min": 0.01, "max": 1.0, "step": 0.01}),
                "labels": ("STRING", {"default": "", "multiline": False}),
                "mask_padding": ("INT", {"default": 8, "min": 0, "max": 256, "step": 1}),
                "jpeg_quality": ("INT", {"default": 90, "min": 30, "max": 100, "step": 1}),
            },
            "optional": {
                "config": ("ANYSERVE_CONFIG",),
            },
        }

    def detect(
        self,
        image: torch.Tensor,
        model: str,
        min_confidence: float,
        labels: str,
        mask_padding: int,
        jpeg_quality: int,
        config: dict[str, Any] | None = None,
    ):
        metadata = {"conf": str(min_confidence)}
        if labels.strip():
            metadata["labels"] = labels.strip()
        client, options = _make_client_and_options(config, metadata=metadata)
        interface = f"yolo/{model}"
        allowed = {x.strip().lower() for x in labels.split(",") if x.strip()} if labels.strip() else None

        annotated_frames: list[torch.Tensor] = []
        masks: list[torch.Tensor] = []
        all_results: list[dict[str, Any]] = []
        total_count = 0

        for idx in range(image.shape[0]):
            pil = _tensor_frame_to_pil(image[idx])
            width, height = pil.size
            payload = _encode_pil(pil, "JPEG", quality=int(jpeg_quality))
            reply = json.loads(client.call(interface, payload, options=options))
            detections = [
                det
                for det in reply.get("detections", [])
                if float(det.get("confidence", 0.0)) >= min_confidence
                and (allowed is None or str(det.get("label", "")).lower() in allowed)
            ]
            total_count += len(detections)
            all_results.append(
                {
                    "frame": idx,
                    "detections": detections,
                    "inference_ms": reply.get("inference_ms"),
                }
            )

            canvas = pil.copy()
            draw = ImageDraw.Draw(canvas)
            mask_np = np.zeros((height, width), dtype=np.float32)
            pad = int(mask_padding)
            for det in detections:
                x1, y1, x2, y2 = det["box"]
                ix1 = max(0, int(round(x1)))
                iy1 = max(0, int(round(y1)))
                ix2 = min(width - 1, int(round(x2)))
                iy2 = min(height - 1, int(round(y2)))
                draw.rectangle([ix1, iy1, ix2, iy2], outline=(0, 220, 80), width=3)
                caption = f"{det['label']} {det['confidence']:.2f}"
                draw.text((ix1 + 4, max(0, iy1 - 14)), caption, fill=(0, 220, 80))

                mx1 = max(0, ix1 - pad)
                my1 = max(0, iy1 - pad)
                mx2 = min(width, ix2 + pad)
                my2 = min(height, iy2 + pad)
                mask_np[my1:my2, mx1:mx2] = 1.0

            annotated_frames.append(_pil_to_tensor_frame(canvas))
            masks.append(torch.from_numpy(mask_np))

        out_images = torch.stack(annotated_frames, dim=0)
        out_masks = torch.stack(masks, dim=0)
        summary = all_results[0] if len(all_results) == 1 else {"frames": all_results}
        return (out_images, out_masks, json.dumps(summary, indent=2), total_count)


class AnyServeInvoke:
    """Universal bridge node: call any AnyServe worker interface with text or image input."""

    CATEGORY = "AnyServe"
    RETURN_TYPES = ("STRING", "IMAGE")
    RETURN_NAMES = ("text", "image")
    FUNCTION = "invoke"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "interface": ("STRING", {"default": "demo.echo.v1", "multiline": False}),
                "payload_mode": (["text", "image_png", "image_jpeg"],),
                "output_mode": (["text", "image"],),
                "payload": ("STRING", {"default": "hello from ComfyUI", "multiline": True}),
            },
            "optional": {
                "image": ("IMAGE",),
                "config": ("ANYSERVE_CONFIG",),
            },
        }

    def invoke(
        self,
        interface: str,
        payload_mode: str,
        output_mode: str,
        payload: str,
        image: torch.Tensor | None = None,
        config: dict[str, Any] | None = None,
    ):
        client, options = _make_client_and_options(config)
        if payload_mode == "text":
            raw_in = payload.encode("utf-8")
        else:
            if image is None:
                raise ValueError(f"payload_mode={payload_mode!r} requires an input image")
            pil = _tensor_frame_to_pil(image[0])
            raw_in = _encode_pil(pil, "PNG" if payload_mode == "image_png" else "JPEG", quality=92)

        raw_out = client.call(interface.strip(), raw_in, options=options)

        if output_mode == "image":
            out_pil = Image.open(io.BytesIO(raw_out)).convert("RGB")
            out_tensor = _pil_to_tensor_frame(out_pil).unsqueeze(0)
            info = json.dumps({"interface": interface, "width": out_pil.width, "height": out_pil.height, "bytes": len(raw_out)})
            return (info, out_tensor)

        out_text = raw_out.decode("utf-8", errors="replace")
        passthrough = image if image is not None else torch.zeros((1, 1, 1, 3), dtype=torch.float32)
        return (out_text, passthrough)


NODE_CLASS_MAPPINGS = {
    "AnyServeConfig": AnyServeConfig,
    "AnyServeOpenAIChat": AnyServeOpenAIChat,
    "AnyServeYOLODetect": AnyServeYOLODetect,
    "AnyServeInvoke": AnyServeInvoke,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "AnyServeConfig": "AnyServe Config",
    "AnyServeOpenAIChat": "AnyServe OpenAI Chat / VLM",
    "AnyServeYOLODetect": "AnyServe YOLO Detect & Mask",
    "AnyServeInvoke": "AnyServe Invoke (Generic)",
}
