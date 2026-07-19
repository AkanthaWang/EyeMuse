from __future__ import annotations

import os
from pathlib import Path
import tempfile
from time import time
from typing import List, Optional

from .common import FaceDetectionResult, FaceRegion


def _resolve_yolo_face_model_path(model_path: Optional[str] = None) -> Path:
    if model_path:
        resolved = Path(model_path).expanduser().resolve()
        if not resolved.exists():
            raise FileNotFoundError(f"YOLO face model not found: {resolved}")
        return resolved

    weights_dir = Path(__file__).resolve().parents[2] / "weights"
    candidates = (
        "yolov11n-face.pt",
        "yolov8n-face.pt",
        "yolov8n-face.onnx",
        "yolov11n-face.onnx",
        "yolov11s-face.pt",
        "yolov8s-face.pt",
        "yolo-face.onnx",
        "face-yolo.onnx",
    )
    for candidate in candidates:
        resolved = weights_dir / candidate
        if resolved.exists():
            return resolved

    expected_paths = ", ".join(str(weights_dir / candidate) for candidate in candidates)
    raise FileNotFoundError(
        "YOLO face model not found. Place a pretrained YOLO face model at one of: "
        f"{expected_paths}"
    )


def _prepare_yolo_config_dir() -> Path:
    config_dir = Path(tempfile.gettempdir()) / "eyemuse-ultralytics"
    config_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("YOLO_CONFIG_DIR", str(config_dir))
    return config_dir


def _env_flag(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _resolve_yolo_device(torch_module) -> tuple[str, bool]:
    requested = os.environ.get("EYEMUSE_YOLO_DEVICE", "cpu").strip().lower()
    if requested == "auto":
        device = "cuda:0" if torch_module.cuda.is_available() else "cpu"
    elif requested.startswith("cuda"):
        device = requested if torch_module.cuda.is_available() else "cpu"
    else:
        device = "cpu"

    use_half = device.startswith("cuda") and _env_flag("EYEMUSE_YOLO_HALF", False)
    return device, use_half


class YOLOFaceDetector:
    def __init__(
        self,
        *,
        model_path: Optional[str] = None,
        max_num_faces: int = 1,
        min_detection_confidence: float = 0.4,
        nms_threshold: float = 0.45,
    ) -> None:
        try:
            import cv2
            import torch

            _prepare_yolo_config_dir()
            from ultralytics import YOLO
        except Exception as exc:  # pragma: no cover - runtime dependency guard
            raise RuntimeError(
                "YOLO face detection unavailable. Check ultralytics, torch, and opencv-python."
            ) from exc

        self._cv2 = cv2
        self._torch = torch
        self._backend_name = "yolo-ultralytics"
        self._model_path = _resolve_yolo_face_model_path(model_path)
        self._max_num_faces = max_num_faces
        self._min_detection_confidence = min_detection_confidence
        self._nms_threshold = nms_threshold
        self._device, self._use_half = _resolve_yolo_device(torch)

        try:
            self._model = YOLO(str(self._model_path))
            self._model.to(self._device)
        except Exception as exc:
            raise RuntimeError(f"Unable to load YOLO face model: {self._model_path.name}") from exc

    @property
    def backend_name(self) -> str:
        return self._backend_name

    @property
    def model_name(self) -> str:
        return self._model_path.name

    def detect(self, frame_bgr) -> FaceDetectionResult:
        if frame_bgr is None:
            raise ValueError("frame_bgr must not be None")

        frame_height, frame_width = frame_bgr.shape[:2]
        try:
            results = self._model.predict(
                source=frame_bgr,
                conf=self._min_detection_confidence,
                iou=self._nms_threshold,
                verbose=False,
                device=self._device,
                half=self._use_half,
            )
        except Exception as exc:
            raise RuntimeError(
                "YOLO inference failed. Your torch/ultralytics stack cannot use NumPy in the current environment."
            ) from exc

        if not results:
            regions: List[FaceRegion] = []
        else:
            result = results[0]
            boxes = getattr(result, "boxes", None)
            if boxes is None or len(boxes) == 0:
                regions = []
            else:
                xyxy = boxes.xyxy.int().cpu().tolist()
                confs = boxes.conf.cpu().tolist() if boxes.conf is not None else []
                regions = []
                for index, box in enumerate(xyxy[: self._max_num_faces]):
                    x1, y1, x2, y2 = [int(value) for value in box]
                    x1 = max(0, min(frame_width - 1, x1))
                    y1 = max(0, min(frame_height - 1, y1))
                    x2 = max(0, min(frame_width - 1, x2))
                    y2 = max(0, min(frame_height - 1, y2))
                    if x2 <= x1 or y2 <= y1:
                        continue
                    confidence = float(confs[index]) if index < len(confs) else self._min_detection_confidence
                    regions.append(
                        FaceRegion(
                            x=x1,
                            y=y1,
                            width=x2 - x1,
                            height=y2 - y1,
                            confidence=round(confidence, 3),
                        )
                    )

        return FaceDetectionResult(
            regions=regions,
            frame_width=frame_width,
            frame_height=frame_height,
            timestamp=time(),
        )

    def annotate(self, frame_bgr, result: FaceDetectionResult):
        annotated = frame_bgr.copy()
        for index, region in enumerate(result.regions, start=1):
            top_left = (region.x, region.y)
            bottom_right = (region.x + region.width, region.y + region.height)
            self._cv2.rectangle(annotated, top_left, bottom_right, (0, 255, 0), 2)
            label = f"Face {index} {region.confidence:.2f}"
            label_y = region.y - 10 if region.y > 20 else region.y + 20
            self._cv2.putText(
                annotated,
                label,
                (region.x, label_y),
                self._cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2,
                self._cv2.LINE_AA,
            )
        return annotated

    def close(self) -> None:
        del self._model
