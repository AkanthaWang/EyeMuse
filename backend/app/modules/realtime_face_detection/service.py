from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from time import time
from typing import Dict, List, Optional, Sequence, Tuple


@dataclass(frozen=True)
class FaceRegion:
    x: int
    y: int
    width: int
    height: int
    confidence: float


@dataclass(frozen=True)
class FaceDetectionResult:
    regions: List[FaceRegion]
    frame_width: int
    frame_height: int
    timestamp: float


@dataclass(frozen=True)
class FaceAnalysisResult:
    regions: List[FaceRegion]
    frame_width: int
    frame_height: int
    timestamp: float
    face_detected: bool
    stress_score: int
    fatigue_score: int
    dominant_signal: str
    signals: Dict[str, float]
    raw_signals: Dict[str, float]
    calibration_state: str
    calibration_progress: float
    baseline: Optional[Dict[str, float]]


ZEROED_SIGNALS = {
    "brow_furrow": 0.0,
    "lip_press": 0.0,
    "eye_squint": 0.0,
    "expression_freeze": 0.0,
}

SIGNAL_DEADBANDS = {
    "brow_furrow": 0.03,
    "lip_press": 0.02,
    "eye_squint": 0.03,
    "expression_freeze": 0.04,
}

SIGNAL_RANGES = {
    "brow_furrow": 0.14,
    "lip_press": 0.10,
    "eye_squint": 0.14,
    "expression_freeze": 0.14,
}

SIGNAL_WEIGHTS = {
    "brow_furrow": 0.35,
    "lip_press": 0.30,
    "eye_squint": 0.20,
    "expression_freeze": 0.15,
}


def _resolve_model_path(model_path: Optional[str] = None) -> Path:
    if model_path:
        return Path(model_path).expanduser().resolve()
    return Path(__file__).resolve().parents[2] / "weights" / "face_landmarker_v2_with_blendshapes.task"


def _average_signal_samples(samples: Sequence[Dict[str, float]]) -> Dict[str, float]:
    if not samples:
        return dict(ZEROED_SIGNALS)
    return {
        key: round(sum(sample.get(key, 0.0) for sample in samples) / len(samples), 3)
        for key in ZEROED_SIGNALS
    }


def _dominant_signal(signals: Dict[str, float]) -> str:
    if not signals:
        return "none"
    dominant = max(signals, key=signals.get)
    if signals[dominant] < 0.3:
        return "none"
    return dominant


def _compute_stress(signals: Dict[str, float], baseline: Optional[Dict[str, float]] = None) -> Tuple[int, Dict[str, float], str]:
    if baseline:
        adjusted: Dict[str, float] = {}
        for key, raw in signals.items():
            delta = max(0.0, raw - baseline.get(key, 0.0) - SIGNAL_DEADBANDS[key])
            normalized = min(1.0, delta / SIGNAL_RANGES[key])
            adjusted[key] = normalized ** 0.65

        active_count = sum(1 for value in adjusted.values() if value >= 0.25)
        weighted = sum(SIGNAL_WEIGHTS[key] * adjusted[key] for key in SIGNAL_WEIGHTS)
        max_signal = max(adjusted.values()) if adjusted else 0.0

        if active_count <= 1:
            score_raw = max(weighted, max_signal * 0.60)
        else:
            score_raw = max(weighted * 1.15, max_signal * 0.85)

        stress_score = int(100 * min(1.0, score_raw))
        return min(100, max(0, stress_score)), {key: round(value, 2) for key, value in adjusted.items()}, _dominant_signal(adjusted)

    weighted = sum(SIGNAL_WEIGHTS[key] * signals[key] for key in SIGNAL_WEIGHTS)
    max_signal = max(signals.values()) if signals else 0.0
    stress_score = int(100 * max(weighted, max_signal * 0.45))
    stress_score = min(100, max(0, stress_score))
    return stress_score, dict(signals), _dominant_signal(signals)


def _estimate_fatigue(stress_score: int, signals: Dict[str, float]) -> int:
    eye = signals.get("eye_squint", 0.0)
    freeze = signals.get("expression_freeze", 0.0)
    lip = signals.get("lip_press", 0.0)
    fatigue_raw = (stress_score / 100.0) * 0.45 + eye * 0.35 + freeze * 0.15 + lip * 0.05
    return int(round(min(100.0, max(0.0, fatigue_raw * 100.0))))


def _extract_signals(blendshapes) -> Dict[str, float]:
    category_map = {bs.category_name: index for index, bs in enumerate(blendshapes)}

    def get_score(name: str) -> float:
        index = category_map.get(name)
        if index is None or index >= len(blendshapes):
            return 0.0
        return float(blendshapes[index].score)

    brow_down_l = get_score("browDownLeft")
    brow_down_r = get_score("browDownRight")
    brow_inner_up = get_score("browInnerUp")
    mouth_press_l = get_score("mouthPressLeft")
    mouth_press_r = get_score("mouthPressRight")
    mouth_stretch_l = get_score("mouthStretchLeft")
    mouth_stretch_r = get_score("mouthStretchRight")
    eye_squint_l = get_score("eyeSquintLeft")
    eye_squint_r = get_score("eyeSquintRight")
    jaw_open = get_score("jawOpen")
    jaw_clench_val = get_score("jawForward")
    nose_sneer_l = get_score("noseSneerLeft")
    nose_sneer_r = get_score("noseSneerRight")

    brow_raw = max((brow_down_l + brow_down_r) / 2.0, brow_inner_up * 0.8)
    lip_raw = (mouth_press_l + mouth_press_r) / 2.0
    stretch_raw = (mouth_stretch_l + mouth_stretch_r) / 2.0
    lip_tension = lip_raw + stretch_raw * 0.5 + jaw_clench_val * 0.5
    eye_raw = (eye_squint_l + eye_squint_r) / 2.0
    sneer_raw = (nose_sneer_l + nose_sneer_r) / 2.0
    eye_tension = eye_raw + sneer_raw * 0.3

    jaw_shut = max(0.0, 1.0 - jaw_open * 10.0) if jaw_open < 0.05 else 0.0
    avg_tension = (brow_raw + lip_tension + eye_tension) / 3.0
    expression_freeze = jaw_shut * 0.2 + avg_tension * 0.8

    return {
        "brow_furrow": round(min(1.0, brow_raw), 3),
        "lip_press": round(min(1.0, lip_tension), 3),
        "eye_squint": round(min(1.0, eye_tension), 3),
        "expression_freeze": round(min(1.0, expression_freeze), 3),
    }


def _landmarks_to_region(landmarks, frame_width: int, frame_height: int) -> Optional[FaceRegion]:
    xs = [landmark.x for landmark in landmarks]
    ys = [landmark.y for landmark in landmarks]
    if not xs or not ys:
        return None

    x_min = max(0, int(min(xs) * frame_width))
    x_max = min(frame_width, int(max(xs) * frame_width))
    y_min = max(0, int(min(ys) * frame_height))
    y_max = min(frame_height, int(max(ys) * frame_height))

    width = x_max - x_min
    height = y_max - y_min
    if width <= 0 or height <= 0:
        return None

    pad_x = max(4, int(width * 0.12))
    pad_y = max(4, int(height * 0.16))
    x = max(0, x_min - pad_x)
    y = max(0, y_min - pad_y)
    width = min(frame_width - x, width + pad_x * 2)
    height = min(frame_height - y, height + pad_y * 2)
    if width <= 0 or height <= 0:
        return None

    return FaceRegion(x=x, y=y, width=width, height=height, confidence=1.0)


class MediaPipeFaceDetector:
    def __init__(
        self,
        *,
        model_selection: int = 0,
        max_num_faces: int = 1,
        min_detection_confidence: float = 0.5,
    ) -> None:
        try:
            import cv2
            import mediapipe as mp
        except ImportError as exc:  # pragma: no cover - runtime dependency guard
            raise ImportError(
                "MediaPipe face detection requires `opencv-python` and `mediapipe`."
            ) from exc

        self._cv2 = cv2
        self._face_detection = mp.solutions.face_detection.FaceDetection(
            model_selection=model_selection,
            max_num_faces=max_num_faces,
            min_detection_confidence=min_detection_confidence,
        )

    def detect(self, frame_bgr) -> FaceDetectionResult:
        if frame_bgr is None:
            raise ValueError("frame_bgr must not be None")

        frame_height, frame_width = frame_bgr.shape[:2]
        frame_rgb = self._cv2.cvtColor(frame_bgr, self._cv2.COLOR_BGR2RGB)
        detection_result = self._face_detection.process(frame_rgb)

        regions: List[FaceRegion] = []
        for detection in detection_result.detections or []:
            bbox = detection.location_data.relative_bounding_box
            x = max(0, int(bbox.xmin * frame_width))
            y = max(0, int(bbox.ymin * frame_height))
            width = min(frame_width - x, int(bbox.width * frame_width))
            height = min(frame_height - y, int(bbox.height * frame_height))

            if width <= 0 or height <= 0:
                continue

            confidence = float(detection.score[0]) if detection.score else 0.0
            regions.append(
                FaceRegion(
                    x=x,
                    y=y,
                    width=width,
                    height=height,
                    confidence=confidence,
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
        self._face_detection.close()


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
        except ImportError as exc:  # pragma: no cover - runtime dependency guard
            raise ImportError(
                "MediaPipe face analysis requires `opencv-python` and `mediapipe`."
            ) from exc

        self._cv2 = cv2
        self._Image = Image
        self._ImageFormat = ImageFormat
        self._vision = vision
        self._BaseOptions = BaseOptions
        self._model_path = _resolve_model_path(model_path)
        if not self._model_path.exists():
            raise FileNotFoundError(
                f"FaceLandmarker model not found: {self._model_path}"
            )

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

    def detect(self, frame_bgr) -> FaceAnalysisResult:
        if frame_bgr is None:
            raise ValueError("frame_bgr must not be None")

        frame_height, frame_width = frame_bgr.shape[:2]
        frame_rgb = self._cv2.cvtColor(frame_bgr, self._cv2.COLOR_BGR2RGB)
        mp_image = self._Image(image_format=self._ImageFormat.SRGB, data=frame_rgb)
        result = self._landmarker.detect(mp_image)

        regions: List[FaceRegion] = []
        face_landmarks = result.face_landmarks or []
        face_blendshapes = result.face_blendshapes or []

        for landmarks in face_landmarks:
            region = _landmarks_to_region(landmarks, frame_width, frame_height)
            if region is not None:
                regions.append(region)

        calibration_state = "waiting"
        calibration_progress = 0.0
        raw_signals = dict(ZEROED_SIGNALS)
        stress_score = 0
        fatigue_score = 0
        signals = dict(ZEROED_SIGNALS)
        dominant_signal = "none"
        face_detected = bool(regions)

        if face_detected and face_blendshapes:
            raw_signals = _extract_signals(face_blendshapes[0])

            if self._baseline_signals is None:
                self._calibration_samples.append(raw_signals)
                elapsed = time() - self._calibration_start
                calibration_progress = min(1.0, elapsed / self._calibration_seconds)
                if elapsed >= self._calibration_seconds and len(self._calibration_samples) >= self._min_calibration_samples:
                    self._baseline_signals = _average_signal_samples(self._calibration_samples)
                    calibration_state = "ready"
                    calibration_progress = 1.0
                    stress_score, signals, dominant_signal = _compute_stress(raw_signals, self._baseline_signals)
                    fatigue_score = _estimate_fatigue(stress_score, signals)
                else:
                    calibration_state = "calibrating"
                    signals = dict(ZEROED_SIGNALS)
                    stress_score = 0
                    fatigue_score = 0
            else:
                calibration_state = "ready"
                calibration_progress = 1.0
                stress_score, signals, dominant_signal = _compute_stress(raw_signals, self._baseline_signals)
                fatigue_score = _estimate_fatigue(stress_score, signals)
        else:
            if self._baseline_signals is None:
                self._calibration_start = time()
                self._calibration_samples.clear()
            if face_detected:
                calibration_state = "calibrating"
            else:
                calibration_state = "waiting"

        analysis = FaceAnalysisResult(
            regions=regions,
            frame_width=frame_width,
            frame_height=frame_height,
            timestamp=time(),
            face_detected=face_detected,
            stress_score=stress_score,
            fatigue_score=fatigue_score,
            dominant_signal=dominant_signal,
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
                label = (
                    f"Face {index} | Stress {result.stress_score:02d} | "
                    f"Fatigue {result.fatigue_score:02d}"
                )
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


def preview_camera(
    *,
    camera_index: int = 0,
    window_name: str = "EyeMuse Face Detection",
    min_detection_confidence: float = 0.5,
) -> None:
    try:
        import cv2
    except ImportError as exc:  # pragma: no cover - runtime dependency guard
        raise ImportError("preview_camera requires `opencv-python`.") from exc

    detector = None
    analyzer = None
    try:
        analyzer = MediaPipeFaceAnalyzer(
            min_face_detection_confidence=min_detection_confidence,
            min_face_presence_confidence=min_detection_confidence,
            min_tracking_confidence=min_detection_confidence,
        )
    except Exception:
        detector = MediaPipeFaceDetector(min_detection_confidence=min_detection_confidence)

    capture = cv2.VideoCapture(camera_index)

    if not capture.isOpened():
        if analyzer is not None:
            analyzer.close()
        if detector is not None:
            detector.close()
        raise RuntimeError(f"Unable to open camera index {camera_index}.")

    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break

            if analyzer is not None:
                result = analyzer.detect(frame)
                annotated_frame = analyzer.annotate(frame, result)
            else:
                result = detector.detect(frame)
                annotated_frame = detector.annotate(frame, result)

            cv2.imshow(window_name, annotated_frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        capture.release()
        if analyzer is not None:
            analyzer.close()
        if detector is not None:
            detector.close()
        cv2.destroyAllWindows()