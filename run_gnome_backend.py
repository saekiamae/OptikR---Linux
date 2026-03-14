"""
GNOME on-screen translator backend.

This script runs a tiny local HTTP service that receives screenshot paths or
base64 image data, runs OCR + translation using the existing plugin system,
and returns text results for a GNOME Shell extension frontend.
"""

from __future__ import annotations

import argparse
import base64
import json
import logging
import os
import sys
import tempfile
import time
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.models import CaptureRegion, Frame, Rectangle  # noqa: E402
from app.ocr.ocr_engine_interface import OCRProcessingOptions  # noqa: E402
from app.ocr.ocr_plugin_manager import OCRPluginManager  # noqa: E402
from app.text_translation.translation_engine_interface import (  # noqa: E402
    TranslationOptions,
)
from app.text_translation.translation_plugin_manager import (  # noqa: E402
    TranslationPluginManager,
)


LOGGER = logging.getLogger("gnome.backend")


@dataclass
class LoadedEngineState:
    ocr: set[str]
    translation: set[str]


class GNOMEPipelineService:
    """Reusable OCR + translation service state."""

    def __init__(self) -> None:
        self.ocr_plugin_manager = OCRPluginManager(
            plugin_directories=[str(PROJECT_ROOT / "plugins" / "stages" / "ocr")]
        )
        self.translation_plugin_manager = TranslationPluginManager(
            plugin_directories=[str(PROJECT_ROOT / "plugins" / "stages" / "translation")]
        )
        self.state = LoadedEngineState(ocr=set(), translation=set())
        self.discovered_ocr: set[str] = set()
        self.discovered_translation: set[str] = set()
        self._discover_plugins()

    def _discover_plugins(self) -> None:
        ocr_plugins = self.ocr_plugin_manager.discover_plugins()
        self.discovered_ocr = {plugin.name for plugin in ocr_plugins}
        translation_plugins = self.translation_plugin_manager.discover_plugins()
        self.discovered_translation = {plugin.name for plugin in translation_plugins}
        LOGGER.info("Discovered OCR plugins: %s", sorted(self.discovered_ocr))
        LOGGER.info(
            "Discovered translation plugins: %s",
            sorted(self.discovered_translation),
        )

    def _load_ocr_engine(self, engine_name: str, language: str) -> None:
        if engine_name in self.state.ocr:
            return
        if engine_name not in self.discovered_ocr:
            raise RuntimeError(f"OCR engine '{engine_name}' not discovered")
        ok = self.ocr_plugin_manager.load_plugin(engine_name, {"language": language})
        if not ok:
            raise RuntimeError(f"Failed to load OCR engine '{engine_name}'")
        self.state.ocr.add(engine_name)

    def _load_translation_engine(self, engine_name: str) -> None:
        if engine_name in self.state.translation:
            return
        if engine_name not in self.discovered_translation:
            raise RuntimeError(f"Translation engine '{engine_name}' not discovered")
        ok = self.translation_plugin_manager.load_plugin(engine_name, {})
        if not ok:
            raise RuntimeError(f"Failed to load translation engine '{engine_name}'")
        self.state.translation.add(engine_name)

    def _read_image_as_bgr_frame(self, image_path: Path) -> Frame:
        if not image_path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")

        image = Image.open(image_path).convert("RGB")
        rgb = np.array(image)
        bgr = rgb[:, :, ::-1]  # OCR engines in this project expect BGR frames.
        h, w = bgr.shape[:2]
        region = CaptureRegion(rectangle=Rectangle(0, 0, w, h), monitor_id=0)
        return Frame(data=bgr, timestamp=time.time(), source_region=region)

    def _save_temp_image(self, image_b64: str) -> Path:
        payload = base64.b64decode(image_b64)
        temp_dir = Path(tempfile.gettempdir())
        out_path = temp_dir / f"optikr_capture_{int(time.time() * 1000)}.png"
        out_path.write_bytes(payload)
        return out_path

    def process(
        self,
        *,
        image_path: str | None,
        image_base64: str | None,
        source_lang: str,
        target_lang: str,
        ocr_engine: str,
        translation_engine: str,
        min_confidence: float,
    ) -> dict[str, Any]:
        if not image_path and not image_base64:
            raise ValueError("Either 'image_path' or 'image_base64' must be provided")

        temp_created: Path | None = None
        try:
            if image_base64:
                temp_created = self._save_temp_image(image_base64)
                resolved_image_path = temp_created
            else:
                resolved_image_path = Path(image_path).expanduser().resolve()  # type: ignore[arg-type]

            self._load_ocr_engine(ocr_engine, source_lang)
            self._load_translation_engine(translation_engine)

            frame = self._read_image_as_bgr_frame(resolved_image_path)
            ocr = self.ocr_plugin_manager.get_engine(ocr_engine)
            if ocr is None:
                raise RuntimeError(f"OCR engine '{ocr_engine}' loaded but unavailable")
            ocr.set_language(source_lang)
            text_blocks = ocr.extract_text(
                frame,
                OCRProcessingOptions(
                    language=source_lang,
                    confidence_threshold=max(0.0, min(1.0, min_confidence)),
                    preprocessing_enabled=True,
                ),
            )

            filtered = [tb for tb in text_blocks if tb.text.strip() and tb.confidence >= min_confidence]
            original_text = "\n".join(tb.text for tb in filtered).strip()

            translated_text = ""
            if original_text:
                engine = self.translation_plugin_manager.get_engine(translation_engine)
                if engine is None:
                    raise RuntimeError(
                        f"Translation engine '{translation_engine}' loaded but unavailable"
                    )
                result = engine.translate_text(
                    original_text,
                    "auto" if source_lang.lower() == "auto" else source_lang,
                    target_lang,
                    TranslationOptions(),
                )
                translated_text = result.translated_text.strip()

            return {
                "success": True,
                "ocr_engine": ocr_engine,
                "translation_engine": translation_engine,
                "source_language": source_lang,
                "target_language": target_lang,
                "text_blocks": [
                    {
                        "text": tb.text,
                        "confidence": tb.confidence,
                        "bbox": [
                            tb.position.x,
                            tb.position.y,
                            tb.position.width,
                            tb.position.height,
                        ],
                    }
                    for tb in filtered
                ],
                "original_text": original_text,
                "translated_text": translated_text,
            }
        finally:
            if temp_created and temp_created.exists():
                try:
                    temp_created.unlink(missing_ok=True)
                except OSError:
                    pass


class BackendHandler(BaseHTTPRequestHandler):
    """HTTP API for GNOME extension."""

    service: GNOMEPipelineService | None = None

    def _respond(self, status: int, payload: dict[str, Any]) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _read_json(self) -> dict[str, Any]:
        length = int(self.headers.get("Content-Length", "0"))
        if length <= 0:
            return {}
        raw = self.rfile.read(length)
        return json.loads(raw.decode("utf-8"))

    def do_GET(self) -> None:  # noqa: N802
        if self.path == "/health":
            assert self.service is not None
            self._respond(
                200,
                {
                    "status": "ok",
                    "ocr_plugins": sorted(self.service.discovered_ocr),
                    "translation_plugins": sorted(self.service.discovered_translation),
                },
            )
            return
        self._respond(404, {"success": False, "error": "Not found"})

    def do_POST(self) -> None:  # noqa: N802
        if self.path != "/ocr-translate":
            self._respond(404, {"success": False, "error": "Not found"})
            return

        try:
            payload = self._read_json()
            assert self.service is not None
            result = self.service.process(
                image_path=payload.get("image_path"),
                image_base64=payload.get("image_base64"),
                source_lang=payload.get("source_lang", "auto"),
                target_lang=payload.get("target_lang", "en"),
                ocr_engine=payload.get("ocr_engine", "tesseract"),
                translation_engine=payload.get("translation_engine", "google_free"),
                min_confidence=float(payload.get("min_confidence", 0.25)),
            )
            self._respond(200, result)
        except Exception as exc:  # pylint: disable=broad-except
            LOGGER.exception("Request processing failed")
            self._respond(500, {"success": False, "error": str(exc)})

    def log_message(self, fmt: str, *args: Any) -> None:
        LOGGER.info("%s - %s", self.address_string(), fmt % args)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="GNOME OCR/translation backend")
    parser.add_argument("--host", default="127.0.0.1", help="Bind host")
    parser.add_argument("--port", type=int, default=8765, help="Bind port")
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    os.chdir(PROJECT_ROOT)
    service = GNOMEPipelineService()
    BackendHandler.service = service

    server = ThreadingHTTPServer((args.host, args.port), BackendHandler)
    LOGGER.info("GNOME backend listening on http://%s:%s", args.host, args.port)

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        LOGGER.info("Shutting down backend...")
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
