from __future__ import annotations

from pathlib import Path
from time import time
from typing import Dict, List, Optional, Sequence, Tuple

from .common import (
    FaceAnalysisResult,
    FaceRegion,
    ZEROED_SIGNALS,
    average_signal_samples,
    compute_stress,
    crop_frame_to_region,
    estimate_fatigue,
    extract_signals,
    landmarks_to_region,
)


def _resolve_model_path(model_path: Optional[str] = None) -> Path:
    if model_path:
        return Path(model_path).expanduser().resolve()
    return Path(__file__).resolve().parents[2] / "weights" / "face_landmarker_v2_with_blendshapes.task"


class MediaPipeFaceAnalyzer:
    def __init__(
        self,
        *,
        model_path: Optional[str] = None,
        calibration_seconds: float = 12.0,
        min_calibration_samples: int = 3,
        max_num_faces: int = 1,
        min_face_detection_confidence: float = 0.5,
        min_face_presence_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
    ) -> None:
        try:
            import cv2
            from mediapipe import Image, ImageFormat
            from mediapipe.tasks.python import BaseOptions, vision
        except Exception as exc:  # pragma: no cover - runtime dependency guard
            raise RuntimeError(
                "MediaPipe face analysis unavailable. Check mediapipe/tensorflow/numpy compatibility."
            ) from exc

        self._cv2 = cv2
        self._Image = Image
        self._ImageFormat = ImageFormat
        self._vision = vision
        self._BaseOptions = BaseOptions
        self._model_path = _resolve_model_path(model_path)
        self._max_num_faces = max_num_faces
        if not self._model_path.exists():
            raise FileNotFoundError(f"FaceLandmarker model not found: {self._model_path}")

        self._calibration_seconds = calibration_seconds
        self._min_calibration_samples = min_calibration_samples
        self._calibration_start = time()
        self._calibration_samples: List[Dict[str, float]] = []
        self._baseline_signals: Optional[Dict[str, float]] = None
        self._last_analysis: Optional[FaceAnalysisResult] = None

        options = vision.FaceLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(self._model_path)),
            output_face_blendshapes=True,
            output_facial_transformation_matrixes=False,
            num_faces=max_num_faces,
            min_face_detection_confidence=min_face_detection_confidence,
            min_face_presence_confidence=min_face_presence_confidence,
            min_tracking_confidence=min_tracking_confidence,
            running_mode=vision.RunningMode.IMAGE,
        )
        self._landmarker = vision.FaceLandmarker.create_from_options(options)

    def _detect_landmarks(self, frame_bgr):
        frame_rgb = self._cv2.cvtColor(frame_bgr, self._cv2.COLOR_BGR2RGB)
        mp_image = self._Image(image_format=self._ImageFormat.SRGB, data=frame_rgb)
        return self._landmarker.detect(mp_image)

    def _collect_full_frame_regions(self, frame_bgr) -> Tuple[List[FaceRegion], List]:
        frame_height, frame_width = frame_bgr.shape[:2]
        result = self._detect_landmarks(frame_bgr)
        regions: List[FaceRegion] = []
        for landmarks in result.face_landmarks or []:
            region = landmarks_to_region(landmarks, frame_width, frame_height)
            if region is not None:
                regions.append(region)
        return regions, list(result.face_blendshapes or [])

    def _collect_guided_regions(
        self,
        frame_bgr,
        candidate_regions: Sequence[FaceRegion],
    ) -> Tuple[List[FaceRegion], List]:
        frame_height, frame_width = frame_bgr.shape[:2]
        regions: List[FaceRegion] = []
        blendshapes = []

        for candidate in candidate_regions[: self._max_num_faces]:
            cropped_frame, offset_x, offset_y = crop_frame_to_region(frame_bgr, candidate)
            if cropped_frame is None:
                continue

            result = self._detect_landmarks(cropped_frame)
            local_landmarks = result.face_landmarks or []
            if not local_landmarks:
                continue

            for landmarks in local_landmarks[:1]:
                region = landmarks_to_region(
                    landmarks,
                    frame_width,
                    frame_height,
                    source_width=cropped_frame.shape[1],
                    source_height=cropped_frame.shape[0],
                    offset_x=offset_x,
                    offset_y=offset_y,
                )
                if region is not None:
                    regions.append(region)
            blendshapes = list(result.face_blendshapes or [])
            if regions:
                break

        return regions, blendshapes

    def detect(
        self,
        frame_bgr,
        candidate_regions: Optional[Sequence[FaceRegion]] = None,
    ) -> FaceAnalysisResult:
        if frame_bgr is None:
            raise ValueError("frame_bgr must not be None")

        frame_height, frame_width = frame_bgr.shape[:2]
        regions: List[FaceRegion] = []
        face_blendshapes: List = []

        if candidate_regions:
            regions, face_blendshapes = self._collect_guided_regions(frame_bgr, candidate_regions)

        if not regions:
            regions, face_blendshapes = self._collect_full_frame_regions(frame_bgr)

        calibration_state = "waiting"
        calibration_progress = 0.0
        raw_signals = dict(ZEROED_SIGNALS)
        stress_score = 0
        fatigue_score = 0
        signals = dict(ZEROED_SIGNALS)
        dominant = "none"
        face_detected = bool(regions)

        if face_detected and face_blendshapes:
            raw_signals = extract_signals(face_blendshapes[0])

            if self._baseline_signals is None:
                self._calibration_samples.append(raw_signals)
                elapsed = time() - self._calibration_start
                calibration_progress = min(1.0, elapsed / self._calibration_seconds)
                if elapsed >= self._calibration_seconds and len(self._calibration_samples) >= self._min_calibration_samples:
                    self._baseline_signals = average_signal_samples(self._calibration_samples)
                    calibration_state = "ready"
                    calibration_progress = 1.0
                    stress_score, signals, dominant = compute_stress(raw_signals, self._baseline_signals)
                    fatigue_score = estimate_fatigue(stress_score, signals)
                else:
                    calibration_state = "calibrating"
            else:
                calibration_state = "ready"
                calibration_progress = 1.0
                stress_score, signals, dominant = compute_stress(raw_signals, self._baseline_signals)
                fatigue_score = estimate_fatigue(stress_score, signals)
        else:
            if self._baseline_signals is None:
                self._calibration_start = time()
                self._calibration_samples.clear()

        analysis = FaceAnalysisResult(
            regions=regions,
            frame_width=frame_width,
            frame_height=frame_height,
            timestamp=time(),
            face_detected=face_detected,
            stress_score=stress_score,
            fatigue_score=fatigue_score,
            dominant_signal=dominant,
            signals=signals,
            raw_signals=raw_signals,
            calibration_state=calibration_state,
            calibration_progress=calibration_progress,
            baseline=self._baseline_signals,
        )
        self._last_analysis = analysis
        return analysis

    def annotate(self, frame_bgr, result: FaceAnalysisResult):
        annotated = frame_bgr.copy()
        for index, region in enumerate(result.regions, start=1):
            top_left = (region.x, region.y)
            bottom_right = (region.x + region.width, region.y + region.height)
            color = (0, 255, 0)
            if result.calibration_state == "calibrating":
                color = (0, 215, 255)
            elif result.stress_score >= 80:
                color = (70, 70, 255)
            elif result.stress_score >= 60:
                color = (60, 180, 255)

            self._cv2.rectangle(annotated, top_left, bottom_right, color, 2)

            if result.calibration_state == "calibrating":
                label = f"Face {index} | Calibrating {int(result.calibration_progress * 100)}%"
            else:
                label = f"Face {index} | Stress {result.stress_score:02d} | Fatigue {result.fatigue_score:02d}"
                if result.dominant_signal != "none":
                    label += f" | {result.dominant_signal}"

            label_y = region.y - 10 if region.y > 22 else region.y + 20
            text_size, baseline = self._cv2.getTextSize(label, self._cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
            background_top = max(0, label_y - text_size[1] - baseline - 4)
            background_bottom = min(frame_bgr.shape[0] - 1, label_y + 6)
            background_right = min(frame_bgr.shape[1] - 1, region.x + text_size[0] + 10)
            self._cv2.rectangle(
                annotated,
                (region.x, background_top),
                (background_right, background_bottom),
                color,
                -1,
            )
            self._cv2.putText(
                annotated,
                label,
                (region.x + 5, label_y),
                self._cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (0, 0, 0),
                2,
                self._cv2.LINE_AA,
            )
        return annotated

    def close(self) -> None:
        self._landmarker.close()
