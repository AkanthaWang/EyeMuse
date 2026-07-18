from __future__ import annotations

from dataclasses import dataclass
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
    heart_rate: Optional[float]
    respiration_rate: Optional[float]
    hrv: Optional[float]
    snr: Optional[float]
    rppg_progress: float
    rppg_signal: List[float]


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


def average_signal_samples(samples: Sequence[Dict[str, float]]) -> Dict[str, float]:
    if not samples:
        return dict(ZEROED_SIGNALS)
    return {
        key: round(sum(sample.get(key, 0.0) for sample in samples) / len(samples), 3)
        for key in ZEROED_SIGNALS
    }


def dominant_signal(signals: Dict[str, float]) -> str:
    if not signals:
        return "none"
    dominant = max(signals, key=signals.get)
    if signals[dominant] < 0.3:
        return "none"
    return dominant


def compute_stress(
    signals: Dict[str, float],
    baseline: Optional[Dict[str, float]] = None,
) -> Tuple[int, Dict[str, float], str]:
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
        return (
            min(100, max(0, stress_score)),
            {key: round(value, 2) for key, value in adjusted.items()},
            dominant_signal(adjusted),
        )

    weighted = sum(SIGNAL_WEIGHTS[key] * signals[key] for key in SIGNAL_WEIGHTS)
    max_signal = max(signals.values()) if signals else 0.0
    stress_score = int(100 * max(weighted, max_signal * 0.45))
    stress_score = min(100, max(0, stress_score))
    return stress_score, dict(signals), dominant_signal(signals)


def estimate_fatigue(stress_score: int, signals: Dict[str, float]) -> int:
    eye = signals.get("eye_squint", 0.0)
    freeze = signals.get("expression_freeze", 0.0)
    lip = signals.get("lip_press", 0.0)
    fatigue_raw = (stress_score / 100.0) * 0.45 + eye * 0.35 + freeze * 0.15 + lip * 0.05
    return int(round(min(100.0, max(0.0, fatigue_raw * 100.0))))


def extract_signals(blendshapes) -> Dict[str, float]:
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


def landmarks_to_region(
    landmarks,
    frame_width: int,
    frame_height: int,
    *,
    source_width: Optional[int] = None,
    source_height: Optional[int] = None,
    offset_x: int = 0,
    offset_y: int = 0,
) -> Optional[FaceRegion]:
    width_scale = source_width or frame_width
    height_scale = source_height or frame_height
    xs = [offset_x + landmark.x * width_scale for landmark in landmarks]
    ys = [offset_y + landmark.y * height_scale for landmark in landmarks]
    if not xs or not ys:
        return None

    x_min = max(0, int(min(xs)))
    x_max = min(frame_width, int(max(xs)))
    y_min = max(0, int(min(ys)))
    y_max = min(frame_height, int(max(ys)))

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


def crop_frame_to_region(
    frame_bgr,
    region: FaceRegion,
    *,
    padding_ratio: float = 0.18,
):
    frame_height, frame_width = frame_bgr.shape[:2]
    pad_x = int(region.width * padding_ratio)
    pad_y = int(region.height * padding_ratio)
    x1 = max(0, region.x - pad_x)
    y1 = max(0, region.y - pad_y)
    x2 = min(frame_width, region.x + region.width + pad_x)
    y2 = min(frame_height, region.y + region.height + pad_y)
    if x2 <= x1 or y2 <= y1:
        return None, 0, 0
    return frame_bgr[y1:y2, x1:x2].copy(), x1, y1
