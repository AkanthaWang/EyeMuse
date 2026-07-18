from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from time import time
from typing import Dict, List, Optional, Sequence

from ..rppg import POSRPPGProcessor, extract_forehead_rgb
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


_LANDMARK_CONTOURS: tuple[tuple[int, ...], ...] = (
    (10, 109, 67, 103, 54, 21, 162, 127, 234, 93, 132, 58, 172, 136, 150, 149, 176, 148, 152, 377, 400, 378, 379, 365, 397, 288, 361, 323, 454, 356, 389, 251, 284, 332, 297, 338, 10),
    (70, 63, 105, 66, 107),
    (336, 296, 334, 293, 300),
    (33, 160, 158, 133, 153, 144, 33),
    (362, 385, 387, 263, 373, 380, 362),
    (168, 6, 197, 195, 5, 4),
    (61, 185, 40, 39, 37, 0, 267, 269, 270, 409, 291),
    (78, 95, 88, 178, 87, 14, 317, 402, 318, 324, 308),
)


def _resolve_model_path(model_path: Optional[str] = None) -> Path:
    if model_path:
        return Path(model_path).expanduser().resolve()
    return Path(__file__).resolve().parents[2] / "weights" / "face_landmarker_v2_with_blendshapes.task"


@dataclass(frozen=True)
class _FaceObservation:
    region: FaceRegion
    landmarks: Sequence
    blendshapes: Optional[Sequence]


@dataclass(frozen=True)
class _NormalizedLandmark:
    x: float
    y: float


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
        rppg_fps: int = 30,
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
        self._last_observations: List[_FaceObservation] = []
        self._rppg_processor = POSRPPGProcessor(fps=rppg_fps)
        self._rppg_missing_frames = 0
        self._rppg_reset_after_missing_frames = max(5, int(rppg_fps * 1.5))

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

    def _normalize_landmarks_to_frame(
        self,
        landmarks: Sequence,
        *,
        frame_width: int,
        frame_height: int,
        source_width: int,
        source_height: int,
        offset_x: int = 0,
        offset_y: int = 0,
    ) -> list[_NormalizedLandmark]:
        normalized_landmarks: list[_NormalizedLandmark] = []
        for landmark in landmarks:
            x = (offset_x + (landmark.x * source_width)) / float(frame_width)
            y = (offset_y + (landmark.y * source_height)) / float(frame_height)
            normalized_landmarks.append(_NormalizedLandmark(x=x, y=y))
        return normalized_landmarks

    def _collect_full_frame_observations(self, frame_bgr) -> List[_FaceObservation]:
        frame_height, frame_width = frame_bgr.shape[:2]
        result = self._detect_landmarks(frame_bgr)
        observations: List[_FaceObservation] = []
        blendshape_sets = list(result.face_blendshapes or [])
        for index, landmarks in enumerate(result.face_landmarks or []):
            region = landmarks_to_region(landmarks, frame_width, frame_height)
            if region is not None:
                observations.append(
                    _FaceObservation(
                        region=region,
                        landmarks=landmarks,
                        blendshapes=blendshape_sets[index] if index < len(blendshape_sets) else None,
                    )
                )
        return observations

    def _collect_guided_observations(
        self,
        frame_bgr,
        candidate_regions: Sequence[FaceRegion],
    ) -> List[_FaceObservation]:
        frame_height, frame_width = frame_bgr.shape[:2]
        observations: List[_FaceObservation] = []

        for candidate in candidate_regions[: self._max_num_faces]:
            cropped_frame, offset_x, offset_y = crop_frame_to_region(frame_bgr, candidate)
            if cropped_frame is None:
                continue

            result = self._detect_landmarks(cropped_frame)
            local_landmarks = result.face_landmarks or []
            if not local_landmarks:
                continue

            blendshape_sets = list(result.face_blendshapes or [])
            for index, landmarks in enumerate(local_landmarks[:1]):
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
                    observations.append(
                        _FaceObservation(
                            region=region,
                            landmarks=self._normalize_landmarks_to_frame(
                                landmarks,
                                frame_width=frame_width,
                                frame_height=frame_height,
                                source_width=cropped_frame.shape[1],
                                source_height=cropped_frame.shape[0],
                                offset_x=offset_x,
                                offset_y=offset_y,
                            ),
                            blendshapes=blendshape_sets[index] if index < len(blendshape_sets) else None,
                        )
                    )
            if observations:
                break

        return observations

    def _track_rppg(self, frame_bgr, landmarks: Optional[Sequence]):
        if landmarks is None:
            self._rppg_missing_frames += 1
            if self._rppg_missing_frames >= self._rppg_reset_after_missing_frames:
                self._rppg_processor.clear()
            return None, self._rppg_processor.get_progress()

        rgb_sample = extract_forehead_rgb(frame_bgr, landmarks)
        if rgb_sample is None:
            self._rppg_missing_frames += 1
            if self._rppg_missing_frames >= self._rppg_reset_after_missing_frames:
                self._rppg_processor.clear()
            return None, self._rppg_processor.get_progress()

        self._rppg_missing_frames = 0
        self._rppg_processor.add_sample(*rgb_sample)
        return self._rppg_processor.process(), self._rppg_processor.get_progress()

    def _landmark_to_point(self, landmark, frame_width: int, frame_height: int) -> tuple[int, int]:
        x = int(max(0, min(frame_width - 1, landmark.x * frame_width)))
        y = int(max(0, min(frame_height - 1, landmark.y * frame_height)))
        return x, y

    def _draw_landmarks(self, frame_bgr, landmarks: Sequence, color: tuple[int, int, int]) -> None:
        frame_height, frame_width = frame_bgr.shape[:2]
        points = [self._landmark_to_point(landmark, frame_width, frame_height) for landmark in landmarks]
        if not points:
            return

        line_color = tuple(max(0, min(255, channel - 35)) for channel in color)
        for contour in _LANDMARK_CONTOURS:
            contour_points = []
            for index in contour:
                if 0 <= index < len(points):
                    contour_points.append(points[index])
            if len(contour_points) >= 2:
                for start, end in zip(contour_points, contour_points[1:]):
                    self._cv2.line(frame_bgr, start, end, line_color, 1, self._cv2.LINE_AA)

        for point in points:
            self._cv2.circle(frame_bgr, point, 1, color, -1, self._cv2.LINE_AA)

    def detect(
        self,
        frame_bgr,
        candidate_regions: Optional[Sequence[FaceRegion]] = None,
    ) -> FaceAnalysisResult:
        if frame_bgr is None:
            raise ValueError("frame_bgr must not be None")

        frame_height, frame_width = frame_bgr.shape[:2]
        observations: List[_FaceObservation] = []

        if candidate_regions:
            observations = self._collect_guided_observations(frame_bgr, candidate_regions)

        if not observations:
            observations = self._collect_full_frame_observations(frame_bgr)
        self._last_observations = observations

        regions = [observation.region for observation in observations]
        primary_observation = observations[0] if observations else None
        primary_blendshapes = list(primary_observation.blendshapes or []) if primary_observation is not None else []

        calibration_state = "waiting"
        calibration_progress = 0.0
        raw_signals = dict(ZEROED_SIGNALS)
        stress_score = 0
        fatigue_score = 0
        signals = dict(ZEROED_SIGNALS)
        dominant = "none"
        face_detected = bool(regions)
        rppg_result = None
        rppg_progress = self._rppg_processor.get_progress()

        if primary_observation is not None:
            rppg_result, rppg_progress = self._track_rppg(frame_bgr, primary_observation.landmarks)
        else:
            rppg_result, rppg_progress = self._track_rppg(frame_bgr, None)

        if face_detected and primary_blendshapes:
            raw_signals = extract_signals(primary_blendshapes)

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
            heart_rate=None if rppg_result is None else rppg_result.heart_rate,
            respiration_rate=None if rppg_result is None else rppg_result.respiration_rate,
            hrv=None if rppg_result is None else rppg_result.hrv,
            snr=None if rppg_result is None else rppg_result.snr,
            rppg_progress=rppg_progress,
            rppg_signal=[] if rppg_result is None else rppg_result.bvp_signal,
        )
        self._last_analysis = analysis
        return analysis

    def annotate(self, frame_bgr, result: FaceAnalysisResult):
        annotated = frame_bgr.copy()
        for observation in self._last_observations:
            self._draw_landmarks(annotated, observation.landmarks, (255, 220, 90))

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
                if result.heart_rate is not None:
                    label += f" | HR {int(round(result.heart_rate))}"

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
        self._last_observations = []
        self._rppg_processor.clear()
        self._landmarker.close()
