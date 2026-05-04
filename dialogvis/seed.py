"""Handles seed images: real image files or text descriptions."""
from __future__ import annotations

import base64
import mimetypes
from pathlib import Path


class SeedImageError(Exception):
    pass


class SeedImage:
    """
    Wraps a seed image that is either:
      - A filesystem path to a JPEG/PNG/DICOM file
      - A plain text description

    Call `as_llm_content()` to get the list of content blocks suitable for the
    Anthropic or OpenAI vision message format.
    """

    def __init__(self, source: str | Path):
        self._source = str(source)
        self._path: Path | None = None
        self._text: str | None = None

        p = Path(source)
        if p.exists() and p.is_file():
            self._path = p
        else:
            # Treat as a text description
            self._text = self._source

    # ------------------------------------------------------------------
    @property
    def is_file(self) -> bool:
        return self._path is not None

    @property
    def description_hint(self) -> str:
        """Human-readable hint for prompts when seed is text-based."""
        if self._text:
            return self._text
        return f"Image file: {self._path}"

    # ------------------------------------------------------------------
    def as_anthropic_content(self) -> list[dict]:
        """Content blocks for Anthropic messages API."""
        if self._text:
            return [{"type": "text", "text": f"[Seed image description] {self._text}"}]

        data, media_type = self._load_image_bytes()
        b64 = base64.standard_b64encode(data).decode()
        return [
            {
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": media_type,
                    "data": b64,
                },
            },
            {"type": "text", "text": "(The seed image is provided above.)"},
        ]

    def as_openai_content(self) -> list[dict]:
        """Content blocks for OpenAI / vLLM vision messages."""
        if self._text:
            return [{"type": "text", "text": f"[Seed image description] {self._text}"}]

        data, media_type = self._load_image_bytes()
        b64 = base64.standard_b64encode(data).decode()
        url = f"data:{media_type};base64,{b64}"
        return [
            {"type": "image_url", "image_url": {"url": url}},
            {"type": "text", "text": "(The seed image is provided above.)"},
        ]

    # ------------------------------------------------------------------
    def _load_image_bytes(self) -> tuple[bytes, str]:
        """Returns (raw_bytes, mime_type), converting DICOM → PNG if needed."""
        assert self._path is not None
        suffix = self._path.suffix.lower()

        if suffix in (".dcm", ".dicom"):
            return self._load_dicom()

        media_type = _suffix_to_mime(suffix)
        if media_type not in ("image/jpeg", "image/png", "image/gif", "image/webp"):
            raise SeedImageError(
                f"Unsupported image format '{suffix}'. "
                "Use JPEG, PNG, WebP, GIF, or DICOM (requires pydicom)."
            )
        return self._path.read_bytes(), media_type

    def _load_dicom(self) -> tuple[bytes, str]:
        """Convert a DICOM file to PNG bytes using pydicom + pillow."""
        try:
            import io

            import numpy as np
            import pydicom
            from PIL import Image
        except ImportError as exc:
            raise SeedImageError(
                "DICOM support requires pydicom, numpy, and pillow. "
                "Install with: pip install pydicom numpy pillow"
            ) from exc

        ds = pydicom.dcmread(str(self._path))
        arr = ds.pixel_array.astype(np.float32)

        # Apply window/level from DICOM tags if present, else auto-normalize
        if hasattr(ds, "WindowCenter") and hasattr(ds, "WindowWidth"):
            wc = float(ds.WindowCenter) if not hasattr(ds.WindowCenter, "__len__") else float(ds.WindowCenter[0])
            ww = float(ds.WindowWidth) if not hasattr(ds.WindowWidth, "__len__") else float(ds.WindowWidth[0])
            lo, hi = wc - ww / 2, wc + ww / 2
        else:
            lo, hi = arr.min(), arr.max()

        arr = np.clip((arr - lo) / max(hi - lo, 1e-6) * 255, 0, 255).astype(np.uint8)
        img = Image.fromarray(arr).convert("RGB")
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        return buf.getvalue(), "image/png"


def _suffix_to_mime(suffix: str) -> str:
    mapping = {
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".png": "image/png",
        ".webp": "image/webp",
        ".gif": "image/gif",
    }
    return mapping.get(suffix, mimetypes.guess_type(f"x{suffix}")[0] or "image/jpeg")
