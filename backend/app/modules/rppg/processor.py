from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from math import log10
from typing import Optional, Sequence

import numpy as np


_FOREHEAD_CENTER_INDEX = 10
_LEFT_TEMPLE_INDEX = 109
_RIGHT_TEMPLE_INDEX = 338


@dataclass(frozen=True)
class RPPGResult:
    heart_rate: Optional[float]
    respiration_rate: Optional[float]
    hrv: Optional[float]
    snr: Optional[float]
    bvp_signal: list[float]


def extract_forehead_rgb(
    frame_bgr,
    landmarks: Sequence,
    *,
    roi_scale: float = 0.2,
    min_roi_size: int = 10,
) -> Optional[tuple[float, float, float]]:
    if frame_bgr is None or len(landmarks) <= _RIGHT_TEMPLE_INDEX:
        return None

    frame_height, frame_width = frame_bgr.shape[:2]
    center = landmarks[_FOREHEAD_CENTER_INDEX]
    left_temple = landmarks[_LEFT_TEMPLE_INDEX]
    right_temple = landmarks[_RIGHT_TEMPLE_INDEX]

    center_x = int(center.x * frame_width)
    center_y = int(center.y * frame_height)
    face_width = abs(right_temple.x - left_temple.x) * frame_width
    roi_size = max(min_roi_size, int(face_width * roi_scale))
    half_size = roi_size // 2

    start_x = max(0, min(frame_width - 1, center_x - half_size))
    start_y = max(0, min(frame_height - 1, center_y - half_size))
    end_x = max(0, min(frame_width, center_x + half_size))
    end_y = max(0, min(frame_height, center_y + half_size))
    if end_x <= start_x or end_y <= start_y:
        return None

    roi = frame_bgr[start_y:end_y, start_x:end_x]
    if roi.size == 0:
        return None

    mean_bgr = roi.reshape(-1, 3).mean(axis=0)
    blue, green, red = mean_bgr.tolist()
    return float(red), float(green), float(blue)


class POSRPPGProcessor:
    def __init__(
        self,
        *,
        fps: int = 30,
        buffer_seconds: int = 30,
        min_buffer_seconds: int = 5,
    ) -> None:
        self._fps = fps
        self._buffer_size = fps * buffer_seconds
        self._min_buffer_size = fps * min_buffer_seconds
        self._rgb_buffer: deque[tuple[float, float, float]] = deque(maxlen=self._buffer_size)

    def add_sample(self, red: float, green: float, blue: float) -> None:
        self._rgb_buffer.append((red, green, blue))

    def clear(self) -> None:
        self._rgb_buffer.clear()

    def get_progress(self) -> float:
        if self._buffer_size <= 0:
            return 0.0
        return min(1.0, len(self._rgb_buffer) / float(self._buffer_size))

    def process(self) -> Optional[RPPGResult]:
        if len(self._rgb_buffer) < self._min_buffer_size:
            return None

        rgb = np.asarray(self._rgb_buffer, dtype=np.float64)
        channel_mean = np.maximum(rgb.mean(axis=0), 1e-6)
        normalized = rgb / channel_mean
        bvp = self._pos_algorithm(normalized)

        hr_signal = self._bandpass_filter(bvp, low_freq=0.7, high_freq=4.0)
        heart_rate, rr_intervals = self._calculate_heart_rate_and_rr(hr_signal)
        hrv = float(np.std(rr_intervals) * 1000.0) if len(rr_intervals) >= 5 else None

        respiration_signal = self._bandpass_filter(bvp, low_freq=0.15, high_freq=0.5)
        respiration_rate = self._calculate_respiration_rate(respiration_signal)
        snr = self._estimate_snr(bvp, hr_signal)

        return RPPGResult(
            heart_rate=heart_rate,
            respiration_rate=respiration_rate,
            hrv=hrv,
            snr=snr,
            bvp_signal=hr_signal[-300:].astype(float).tolist(),
        )

    def _pos_algorithm(self, normalized_rgb: np.ndarray) -> np.ndarray:
        green = normalized_rgb[:, 1]
        blue = normalized_rgb[:, 2]
        red = normalized_rgb[:, 0]

        s1 = green - blue
        s2 = green + blue - (2.0 * red)
        alpha = float(np.std(s1) / (np.std(s2) + 1e-8))
        return s1 + alpha * s2

    def _bandpass_filter(self, signal: np.ndarray, *, low_freq: float, high_freq: float) -> np.ndarray:
        low_pass_window = max(1, int(self._fps / max(high_freq, 1e-6)))
        smoothed = self._moving_average(signal, low_pass_window)

        trend_window = max(1, int(self._fps / max(low_freq, 1e-6)))
        trend = self._moving_average(smoothed, trend_window)
        return smoothed - trend

    def _moving_average(self, signal: np.ndarray, window_size: int) -> np.ndarray:
        if window_size <= 1 or signal.size == 0:
            return signal.copy()

        kernel = np.ones(window_size, dtype=np.float64) / float(window_size)
        padded = np.pad(signal, (window_size // 2, window_size - 1 - window_size // 2), mode="edge")
        return np.convolve(padded, kernel, mode="valid")

    def _calculate_heart_rate_and_rr(self, signal: np.ndarray) -> tuple[Optional[float], list[float]]:
        min_distance = max(1, int(self._fps * 0.4))
        peaks = self._find_peaks(signal, min_distance=min_distance)
        if len(peaks) < 2:
            return None, []

        rr_intervals = np.diff(peaks) / float(self._fps)
        valid_rr = [float(interval) for interval in rr_intervals if 0.4 <= interval <= 1.5]
        if not valid_rr:
            return None, []

        mean_rr = sum(valid_rr) / len(valid_rr)
        return 60.0 / mean_rr, valid_rr

    def _calculate_respiration_rate(self, signal: np.ndarray) -> Optional[float]:
        min_distance = max(1, int(self._fps * 1.5))
        peaks = self._find_peaks(signal, min_distance=min_distance)
        if len(peaks) < 2:
            return None

        intervals = np.diff(peaks) / float(self._fps)
        valid_intervals = [float(interval) for interval in intervals if 2.0 <= interval <= 10.0]
        if not valid_intervals:
            return None

        mean_interval = sum(valid_intervals) / len(valid_intervals)
        return 60.0 / mean_interval

    def _find_peaks(self, signal: np.ndarray, *, min_distance: int) -> list[int]:
        peaks: list[int] = []
        for index in range(1, len(signal) - 1):
            if signal[index] <= signal[index - 1] or signal[index] <= signal[index + 1]:
                continue
            if not peaks or index - peaks[-1] >= min_distance:
                peaks.append(index)
                continue
            if signal[index] > signal[peaks[-1]]:
                peaks[-1] = index
        return peaks

    def _estimate_snr(self, raw_signal: np.ndarray, filtered_signal: np.ndarray) -> Optional[float]:
        if raw_signal.size == 0 or filtered_signal.size == 0:
            return None

        signal_power = float(np.var(filtered_signal))
        noise = raw_signal[-filtered_signal.size :] - filtered_signal
        noise_power = float(np.var(noise))
        if signal_power <= 0.0 or noise_power <= 0.0:
            return None
        return 10.0 * log10(signal_power / noise_power)
