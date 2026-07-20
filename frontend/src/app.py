from __future__ import annotations
from collections import deque
import faulthandler
from pathlib import Path
from dataclasses import dataclass
from enum import Enum
from html import escape
import json
import math
import os
import threading
import time
from typing import Optional
import sys
import traceback

import cv2
from PySide6.QtCore import QDate, QDateTime, QEvent, QObject, QPoint, QRectF, QThread, QTimer, Qt, QUrl, Signal, Property, QSize, Slot
from PySide6.QtGui import QColor, QFont, QFontDatabase, QImage, QMovie, QPainter, QPixmap, QPalette
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QDateEdit,
    QFileDialog,
    QFrame,
    QGraphicsDropShadowEffect,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMenu,
    QProgressBar,
    QPushButton,
    QSizePolicy,
    QStyle,
    QStackedWidget,
    QTextBrowser,
    QVBoxLayout,
    QWidget,
)

from ui import (
    apply_modern_theme,
    build_companion_chat_html,
    build_dashboard_chart_html,
    build_dashboard_fallback_html,
    build_main_conversation_html,
)


try:
    from backend.app.modules.realtime_face_detection.service import MediaPipeFaceAnalyzer, YOLOFaceDetector
except Exception:  # pragma: no cover - optional runtime dependency path
    MediaPipeFaceAnalyzer = None
    YOLOFaceDetector = None

try:
    from backend.app.modules.llm import LLMClient, LLMClientError
except Exception:  # pragma: no cover - optional runtime dependency path
    LLMClient = None
    LLMClientError = RuntimeError

try:
    from backend.app.modules.dashboard_data import DashboardRepository, RealtimeSnapshot
except Exception:  # pragma: no cover - optional runtime dependency path
    DashboardRepository = None
    RealtimeSnapshot = None

try:
    from PySide6.QtWebEngineWidgets import QWebEngineView
except Exception:  # pragma: no cover - optional runtime dependency path
    QWebEngineView = None

try:
    from pynput import keyboard as pynput_keyboard
    from pynput import mouse as pynput_mouse
except Exception:  # pragma: no cover - optional runtime dependency path
    pynput_keyboard = None
    pynput_mouse = None


class PetMood(str, Enum):
    idle = "idle"
    hover = "hover"
    listening = "listening"
    thinking = "thinking"
    responding = "responding"
    alert = "alert"
    offline = "offline"


@dataclass
class ConversationItem:
    role: str
    text: str
    timestamp: str


@dataclass(frozen=True)
class MonitoringMetricSample:
    captured_at: float
    stress_score: float
    fatigue_score: float
    heart_rate: Optional[float]
    respiration_rate: Optional[float]
    hrv: Optional[float]


_MONITORING_WINDOW_SECONDS = 12.0
_MONITORING_MIN_SECONDS = 4.0
_COMPANION_SIGNAL_SECONDS = 10.0
_COMPANION_SIGNAL_MIN_SECONDS = 8.0
_COMPANION_SIGNAL_RATIO = 0.75
_COMPANION_RISK_CONFIRM_SECONDS = 6.0
_COMPANION_FOCUS_CONFIRM_SECONDS = 8.0
_COMPANION_RECOVERY_CONFIRM_SECONDS = 18.0
_COMPANION_BUBBLE_DISPLAY_MS = 9000
_ACTIVITY_PERIOD_SECONDS = 30.0
_ACTIVITY_BASELINE_PERIODS = 10
_ACTIVITY_SWITCH_WINDOW_SECONDS = 2.0


def _env_flag(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _log_dir() -> Path:
    path = Path(__file__).resolve().parents[2] / "data" / "logs"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _enable_crash_logging() -> None:
    log_path = _log_dir() / "frontend_crash.log"
    log_file = log_path.open("a", encoding="utf-8")
    faulthandler.enable(log_file, all_threads=True)

    def _handle_exception(exc_type, exc_value, exc_tb) -> None:
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        log_file.write(f"\n[{timestamp}] Unhandled exception\n")
        traceback.print_exception(exc_type, exc_value, exc_tb, file=log_file)
        log_file.flush()
        traceback.print_exception(exc_type, exc_value, exc_tb)

    sys.excepthook = _handle_exception


def _make_dark_background_transparent(
    pixmap: QPixmap,
    *,
    threshold: int = 28,
) -> QPixmap:
    image = pixmap.toImage().convertToFormat(QImage.Format_ARGB32)
    for y in range(image.height()):
        for x in range(image.width()):
            color = image.pixelColor(x, y)
            if (
                color.red() <= threshold
                and color.green() <= threshold
                and color.blue() <= threshold
            ):
                color.setAlpha(0)
                image.setPixelColor(x, y, color)
    return QPixmap.fromImage(image)


class CameraWorker(QObject):
    frame_ready = Signal(QImage)
    status_changed = Signal(str)
    face_count_changed = Signal(int)
    analysis_changed = Signal(object)
    open_failed = Signal(str)
    finished = Signal()

    def __init__(self, camera_index: int = 0) -> None:
        super().__init__()
        self._camera_index = camera_index
        self._capture: Optional[cv2.VideoCapture] = None
        self._timer = QTimer(self)
        self._timer.setInterval(100)
        self._timer.timeout.connect(self._read_frame)
        self._analyzer = None
        self._detector = None
        self._stopped = True

    @Slot()
    def start(self) -> None:
        self._stopped = False
        if not self._open_capture():
            self._stopped = True
            self.finished.emit()
            return

        if YOLOFaceDetector is not None and _env_flag("EYEMUSE_ENABLE_YOLO", False):
            try:
                self._detector = YOLOFaceDetector()
            except Exception as exc:
                self._detector = None
                self.status_changed.emit(f"摄像头已开启，YOLO 面部检测不可用：{exc}")
            else:
                backend_name = getattr(self._detector, "backend_name", "unknown")
                model_name = getattr(self._detector, "model_name", "unknown")
                self.status_changed.emit(f"摄像头已开启，YOLO 面部检测已连接（{backend_name} / {model_name}）")

        if MediaPipeFaceAnalyzer is not None:
            try:
                self._analyzer = MediaPipeFaceAnalyzer()
            except Exception as exc:
                self._analyzer = None
                self.status_changed.emit(f"摄像头已开启，压力分析不可用：{exc}")
            else:
                self.status_changed.emit("摄像头已开启，压力分析已连接")

        if self._detector is None and self._analyzer is None:
            self.status_changed.emit("摄像头已开启，但 YOLO 面部检测和压力分析都不可用。")

        self._timer.start()

    @Slot()
    def stop(self) -> None:
        if self._stopped:
            return
        self._stopped = True
        self._timer.stop()
        if self._analyzer is not None and hasattr(self._analyzer, "close"):
            self._analyzer.close()
        self._analyzer = None
        if self._detector is not None and hasattr(self._detector, "close"):
            self._detector.close()
        self._detector = None
        self._cleanup_capture()
        self.finished.emit()

    def _cleanup_capture(self) -> None:
        if self._capture is not None:
            self._capture.release()
            self._capture = None

    def _open_capture(self) -> bool:
        backends: list[tuple[str, Optional[int]]] = [("default", None)]
        if sys.platform.startswith("win"):
            for backend_name in ("CAP_DSHOW", "CAP_MSMF"):
                backend = getattr(cv2, backend_name, None)
                if backend is not None:
                    backends.append((backend_name, backend))

        attempted_backends: list[str] = []
        for backend_name, backend in backends:
            attempted_backends.append(backend_name)
            capture = cv2.VideoCapture(self._camera_index) if backend is None else cv2.VideoCapture(self._camera_index, backend)
            if capture.isOpened():
                self._capture = capture
                return True
            capture.release()

        backend_text = ", ".join(attempted_backends)
        message = f"无法打开摄像头 {self._camera_index}（已尝试后端：{backend_text}）"
        self.status_changed.emit(message)
        self.open_failed.emit(message)
        self._cleanup_capture()
        return False

    def _read_frame(self) -> None:
        if self._capture is None:
            return

        ok, frame = self._capture.read()
        if not ok:
            self.status_changed.emit("摄像头帧读取失败")
            self.stop()
            return

        face_count = 0
        annotated_frame = frame
        emitted_analysis = False
        detection_result = None

        if self._detector is not None:
            try:
                detection_result = self._detector.detect(frame)
                face_count = len(detection_result.regions)
                annotated_frame = self._detector.annotate(frame, detection_result)
                if self._analyzer is None:
                    self.analysis_changed.emit(
                        {
                            "face_count": face_count,
                            "stress_score": 0,
                            "fatigue_score": 0,
                            "heart_rate": None,
                            "respiration_rate": None,
                            "hrv": None,
                            "snr": None,
                            "rppg_progress": 0.0,
                            "dominant_signal": "none",
                            "calibration_state": "unavailable",
                            "calibration_progress": 0.0,
                            "face_detected": bool(face_count),
                        }
                    )
                    emitted_analysis = True
            except Exception as exc:
                self.status_changed.emit(f"YOLO 面部检测异常：{exc}")
                if self._detector is not None and hasattr(self._detector, "close"):
                    self._detector.close()
                self._detector = None

        if self._analyzer is not None:
            try:
                analysis_result = self._analyzer.detect(
                    frame,
                    detection_result.regions if detection_result is not None else None,
                )
                if analysis_result.face_detected:
                    face_count = len(analysis_result.regions)
                    annotated_frame = self._analyzer.annotate(frame, analysis_result)
                elif self._detector is None:
                    annotated_frame = self._analyzer.annotate(frame, analysis_result)
                self.analysis_changed.emit(
                    {
                        "face_count": face_count,
                        "stress_score": analysis_result.stress_score,
                        "fatigue_score": analysis_result.fatigue_score,
                        "heart_rate": analysis_result.heart_rate,
                        "respiration_rate": analysis_result.respiration_rate,
                        "hrv": analysis_result.hrv,
                        "snr": analysis_result.snr,
                        "rppg_progress": analysis_result.rppg_progress,
                        "dominant_signal": analysis_result.dominant_signal,
                        "calibration_state": analysis_result.calibration_state,
                        "calibration_progress": analysis_result.calibration_progress,
                        "face_detected": analysis_result.face_detected,
                    }
                )
                emitted_analysis = True
            except Exception as exc:
                self.status_changed.emit(f"压力分析异常：{exc}")
                if self._analyzer is not None and hasattr(self._analyzer, "close"):
                    self._analyzer.close()
                self._analyzer = None

        if not emitted_analysis:
            self.analysis_changed.emit(
                {
                    "face_count": face_count,
                    "stress_score": 0,
                    "fatigue_score": 0,
                    "heart_rate": None,
                    "respiration_rate": None,
                    "hrv": None,
                    "snr": None,
                    "rppg_progress": 0.0,
                    "dominant_signal": "none",
                    "calibration_state": "unavailable",
                    "calibration_progress": 0.0,
                    "face_detected": bool(face_count),
                }
            )

        self.face_count_changed.emit(face_count)
        rgb_frame = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
        height, width, channels = rgb_frame.shape
        bytes_per_line = channels * width
        image = QImage(rgb_frame.data, width, height, bytes_per_line, QImage.Format_RGB888)
        self.frame_ready.emit(image.copy())


class LLMStreamWorker(QObject):
    chunk_received = Signal(str)
    completed = Signal(str)
    failed = Signal(str)

    def __init__(
        self,
        *,
        llm_client,
        user_text: str,
        conversation_items: list[dict[str, str]],
        context_summary: str,
    ) -> None:
        super().__init__()
        self._llm_client = llm_client
        self._user_text = user_text
        self._conversation_items = conversation_items
        self._context_summary = context_summary

    @Slot()
    def run(self) -> None:
        try:
            chunks: list[str] = []
            for chunk in self._llm_client.stream_reply(
                user_text=self._user_text,
                conversation_items=self._conversation_items,
                context_summary=self._context_summary,
            ):
                if not chunk:
                    continue
                chunks.append(chunk)
                self.chunk_received.emit(chunk)
            final_text = "".join(chunks).strip()
            if not final_text:
                raise LLMClientError("LLM stream did not include assistant content.")
            self.completed.emit(final_text)
        except Exception as exc:
            self.failed.emit(str(exc))

class PetAvatar(QLabel):
    def __init__(
        self,
        parent: Optional[QWidget] = None,
        *,
        max_movie_size: Optional[QSize] = None,
        minimum_size: Optional[QSize] = None,
        show_background: bool = True,
        render_scale: float = 0.86,
    ) -> None:
        super().__init__(parent)
        self._mood = PetMood.idle
        self._hover_once_active = False
        self._last_frame_number = -1
        self._movie: Optional[QMovie] = None
        self._max_movie_size = max_movie_size or QSize(320, 320)
        self._render_scale = max(0.5, min(1.0, render_scale))
        self.setMinimumSize(minimum_size or self._max_movie_size)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.setAlignment(Qt.AlignHCenter | Qt.AlignTop)

        self._asset_dir = Path(__file__).resolve().parents[1] / "assets"
        self._mood_assets = {
            PetMood.idle: "idle.gif",
            PetMood.hover: "hover.gif",
            PetMood.listening: "listening.gif",
            PetMood.thinking: "thinking.gif",
            PetMood.responding: "responding.gif",
            PetMood.alert: "alert.gif",
            PetMood.offline: "offline.gif",
        }
        self._current_asset_name: Optional[str] = None

        if show_background:
            self.setStyleSheet(
                "background: rgba(233, 246, 255, 0.36);"
                "border: 1px solid rgba(255, 255, 255, 0.72);"
                "border-radius: 24px;"
            )
        else:
            self.setStyleSheet("background: transparent; border: none;")
        self.setMood(PetMood.idle)

    def _target_render_size(self) -> QSize:
        available_size = QSize(
            max(1, int(self.width() * self._render_scale)),
            max(1, int(self.height() * self._render_scale)),
        )
        available_size.setWidth(min(available_size.width(), self._max_movie_size.width()))
        available_size.setHeight(min(available_size.height(), self._max_movie_size.height()))
        return available_size

    def _render_current_frame(self) -> None:
        if self._movie is None:
            return

        current_pixmap = self._movie.currentPixmap()
        if current_pixmap.isNull():
            return

        scaled_pixmap = current_pixmap.scaled(
            self._target_render_size(),
            Qt.KeepAspectRatio,
            Qt.SmoothTransformation,
        )
        self.setPixmap(scaled_pixmap)

    def _handle_movie_frame_changed(self, _frame_number: int) -> None:
        self._render_current_frame()
        if self._hover_once_active and self._current_asset_name == self._mood_assets[PetMood.hover]:
            if self._last_frame_number >= 0 and _frame_number < self._last_frame_number:
                self._hover_once_active = False
                self._last_frame_number = -1
                self._apply_visual_mood()
                return
            self._last_frame_number = _frame_number

    def setMood(self, mood: PetMood) -> None:
        if self._mood == mood:
            self._apply_visual_mood()
            return

        self._mood = mood
        self._apply_visual_mood()

    def playHoverOnce(self) -> None:
        hover_asset = self._asset_dir / self._mood_assets[PetMood.hover]
        if not hover_asset.exists():
            return

        self._hover_once_active = True
        self._last_frame_number = -1
        self._apply_visual_mood()

    def _effective_mood(self) -> PetMood:
        if self._hover_once_active:
            return PetMood.hover
        return self._mood

    def _apply_visual_mood(self) -> None:
        mood = self._effective_mood()
        file_name = self._mood_assets.get(mood, "idle.gif")
        if self._current_asset_name == file_name and self._movie is not None:
            return

        self._current_asset_name = file_name
        asset_path = self._asset_dir / file_name

        if not asset_path.exists():
            fallback_path = self._asset_dir / "idle.gif"
            if fallback_path.exists():
                asset_path = fallback_path
            else:
                if self._movie is not None:
                    self._movie.stop()
                    try:
                        self._movie.frameChanged.disconnect(self._handle_movie_frame_changed)
                    except (TypeError, RuntimeError):
                        pass
                    self._movie = None
                    self.clear()
                self.setText(f"缺少素材: {file_name}")
                return

        if self._movie is not None:
            self._movie.stop()
            try:
                self._movie.frameChanged.disconnect(self._handle_movie_frame_changed)
            except (TypeError, RuntimeError):
                pass

        self.clear()
        self._movie = QMovie(str(asset_path))
        self._movie.setCacheMode(QMovie.CacheAll)
        self._last_frame_number = -1
        self._movie.frameChanged.connect(self._handle_movie_frame_changed)
        self._movie.start()
        self._movie.jumpToFrame(0)
        self._render_current_frame()
        self.setText("")

    def resizeEvent(self, event) -> None:  # noqa: N802
        super().resizeEvent(event)
        self._render_current_frame()


class CompanionPetWindow(QWidget):
    toolbar_action_requested = Signal(str)
    chat_submitted = Signal(str)
    rest_requested = Signal(int)

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setWindowFlags(Qt.FramelessWindowHint | Qt.Tool | Qt.WindowStaysOnTopHint)
        self.setAttribute(Qt.WA_TranslucentBackground, True)
        self.setObjectName("CompanionPetWindow")
        self.setToolTip("左键拖动，鼠标中键展开下方工具栏")
        self._compact_size = QSize(280, 314)
        self._expanded_size = QSize(280, 344)
        self._chat_expanded_size = QSize(300, 482)
        self._rest_controls_height = 38
        self._bubble_extra_height = 0
        self.setFixedSize(self._compact_size)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(1)
        self.bubble_label = QLabel("陪伴模式已开启")
        self.bubble_label.setWordWrap(True)
        self.bubble_label.setAlignment(Qt.AlignCenter)
        self.bubble_label.setMinimumHeight(52)
        self.bubble_label.setStyleSheet(
            "background: qlineargradient(x1:0, y1:0, x2:1, y2:1,"
            " stop:0 rgba(229, 243, 255, 0.86), stop:1 rgba(197, 223, 255, 0.74));"
            "color: #3b679e;"
            "border: 1px solid rgba(255, 255, 255, 0.82);"
            "border-radius: 18px;"
            "padding: 10px 14px;"
            "font-size: 12px;"
            "font-weight: 600;"
        )
        self.bubble_label.hide()
        layout.addWidget(self.bubble_label)

        self.marquee_label = QLabel("")
        self.marquee_label.setAlignment(Qt.AlignCenter)
        self.marquee_label.setStyleSheet(
            "color: #4a74aa;"
            "background: rgba(229, 243, 255, 0.58);"
            "border: 1px solid rgba(255, 255, 255, 0.80);"
            "border-radius: 12px;"
            "padding: 5px 12px;"
            "font-size: 11px;"
            "font-weight: 700;"
        )
        self.marquee_label.hide()
        layout.addWidget(self.marquee_label)

        self.rest_frame = QFrame()
        self.rest_frame.setStyleSheet(
            "background: rgba(255, 246, 224, 0.86);"
            "border: 1px solid rgba(255, 255, 255, 0.88);"
            "border-radius: 13px;"
        )
        rest_layout = QHBoxLayout(self.rest_frame)
        rest_layout.setContentsMargins(7, 4, 7, 4)
        rest_layout.setSpacing(6)
        self.rest_duration_combo = QComboBox()
        for minutes in (1, 3, 5, 10):
            self.rest_duration_combo.addItem(f"休息 {minutes} 分钟", minutes)
        self.rest_duration_combo.setCurrentIndex(2)
        self.rest_duration_combo.setStyleSheet(
            "QComboBox {"
            "background: rgba(255, 255, 255, 0.72);"
            "border: 1px solid rgba(191, 168, 119, 0.24);"
            "border-radius: 9px;"
            "padding: 4px 8px;"
            "color: #7b6a4b;"
            "font-size: 11px;"
            "}"
        )
        self.rest_start_button = QPushButton("开始休息")
        self.rest_start_button.setCursor(Qt.PointingHandCursor)
        self.rest_start_button.setStyleSheet(
            "QPushButton {"
            "background: #e5a84b;"
            "border: none;"
            "border-radius: 9px;"
            "padding: 5px 10px;"
            "color: #fffdf7;"
            "font-size: 11px;"
            "font-weight: 700;"
            "}"
            "QPushButton:hover { background: #d69535; }"
            "QPushButton:disabled { background: #d8c7a8; color: #fffaf0; }"
        )
        self.rest_start_button.clicked.connect(self._emit_rest_requested)
        self.rest_close_button = QPushButton("×")
        self.rest_close_button.setToolTip("暂不休息")
        self.rest_close_button.setAccessibleName("关闭休息选择")
        self.rest_close_button.setCursor(Qt.PointingHandCursor)
        self.rest_close_button.setFixedSize(24, 24)
        self.rest_close_button.setStyleSheet(
            "QPushButton {"
            "background: rgba(139, 112, 69, 0.10);"
            "border: 1px solid rgba(139, 112, 69, 0.16);"
            "border-radius: 12px;"
            "color: #927c59;"
            "font-size: 16px;"
            "font-weight: 700;"
            "padding: 0;"
            "}"
            "QPushButton:hover {"
            "background: rgba(139, 112, 69, 0.20);"
            "color: #6f5a39;"
            "}"
        )
        self.rest_close_button.clicked.connect(self._dismiss_rest_prompt)
        rest_layout.addWidget(self.rest_duration_combo, 1)
        rest_layout.addWidget(self.rest_start_button)
        rest_layout.addWidget(self.rest_close_button)
        self.rest_frame.hide()
        layout.addWidget(self.rest_frame)

        self._bubble_timer = QTimer(self)
        self._bubble_timer.setSingleShot(True)
        self._bubble_timer.timeout.connect(self._hide_bubble)
        self._rest_active = False
        self._rest_mode_available = False
        self._rest_prompt_dismissed = False

        self.avatar = PetAvatar(
            max_movie_size=QSize(220, 220),
            minimum_size=QSize(220, 220),
            show_background=False,
            render_scale=1.0,
        )
        self.avatar.setFixedSize(220, 220)
        self.avatar.setAttribute(Qt.WA_TransparentForMouseEvents, True)
        layout.addWidget(self.avatar, 0, Qt.AlignHCenter)

        self.chat_frame = QFrame()
        self.chat_frame.setStyleSheet(
            "background: qlineargradient(x1:0, y1:0, x2:1, y2:1,"
            " stop:0 rgba(233, 246, 255, 0.70), stop:1 rgba(198, 224, 255, 0.60));"
            "border: 1px solid rgba(255, 255, 255, 0.84);"
            "border-radius: 18px;"
        )
        chat_layout = QVBoxLayout(self.chat_frame)
        chat_layout.setContentsMargins(10, 9, 10, 10)
        chat_layout.setSpacing(6)
        self.chat_hint_label = QLabel("发送后，EyeMuse 会在上方气泡回复")
        self.chat_hint_label.setAlignment(Qt.AlignCenter)
        self.chat_hint_label.setStyleSheet(
            "color: #6e8db8;"
            "font-size: 11px;"
            "font-weight: 700;"
        )
        self.chat_view = QTextBrowser()
        self.chat_view.setMinimumHeight(82)
        self.chat_view.setMaximumHeight(120)
        self.chat_view.setStyleSheet(
            "background: transparent;"
            "border: none;"
            "padding: 2px 2px 0 2px;"
            "color: #3b679e;"
            "font-size: 12px;"
        )
        self.chat_view.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.chat_view.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.chat_view.setHtml("<p style='color:#94a3b8; text-align:center;'>输入消息后，回复会显示在上方气泡中。</p>")
        self.chat_input = QLineEdit()
        self.chat_input.setPlaceholderText("直接输入文字，和 EyeMuse 对话")
        self.chat_input.setStyleSheet(
            "background: rgba(214, 234, 255, 0.62);"
            "border: 1px solid rgba(255, 255, 255, 0.84);"
            "border-radius: 14px;"
            "padding: 8px 12px;"
            "color: #355e96;"
            "font-size: 12px;"
        )
        self.chat_input.returnPressed.connect(self._emit_chat_submit)
        self.chat_send_button = QPushButton("说")
        self.chat_send_button.setCursor(Qt.PointingHandCursor)
        self.chat_send_button.setMinimumHeight(30)
        self.chat_send_button.setStyleSheet(
            "QPushButton {"
            "background: qlineargradient(x1:0, y1:0, x2:0, y2:1,"
            " stop:0 rgba(124, 149, 232, 0.98), stop:1 rgba(80, 107, 195, 0.96));"
            "border: 1px solid rgba(154, 180, 245, 0.30);"
            "border-radius: 14px;"
            "color: #164a83;"
            "font-size: 12px;"
            "font-weight: 700;"
            "padding: 6px 14px;"
            "}"
            "QPushButton:hover {"
            "background: qlineargradient(x1:0, y1:0, x2:0, y2:1,"
            " stop:0 rgba(140, 167, 245, 0.98), stop:1 rgba(102, 129, 224, 0.98));"
            "}"
        )
        self.chat_send_button.clicked.connect(self._emit_chat_submit)
        chat_input_row = QHBoxLayout()
        chat_input_row.setContentsMargins(0, 0, 0, 0)
        chat_input_row.setSpacing(6)
        chat_input_row.addWidget(self.chat_input, 1)
        chat_input_row.addWidget(self.chat_send_button)
        chat_layout.addWidget(self.chat_hint_label)
        chat_layout.addWidget(self.chat_view)
        chat_layout.addLayout(chat_input_row)
        self.chat_frame.hide()
        layout.addWidget(self.chat_frame)

        self.toolbar_frame = QFrame()
        self.toolbar_frame.setStyleSheet(
            "background: qlineargradient(x1:0, y1:0, x2:1, y2:1,"
            " stop:0 rgba(233, 246, 255, 0.68), stop:1 rgba(198, 224, 255, 0.58));"
            "border: 1px solid rgba(255, 255, 255, 0.82);"
            "border-radius: 14px;"
            "padding: 1px;"
        )
        self.toolbar_frame.setFixedHeight(62)
        toolbar_layout = QVBoxLayout(self.toolbar_frame)
        toolbar_layout.setContentsMargins(8, 4, 8, 4)
        toolbar_layout.setSpacing(2)
        primary_toolbar_row = QHBoxLayout()
        primary_toolbar_row.setContentsMargins(0, 0, 0, 0)
        primary_toolbar_row.setSpacing(4)
        secondary_toolbar_row = QHBoxLayout()
        secondary_toolbar_row.setContentsMargins(0, 0, 0, 0)
        secondary_toolbar_row.setSpacing(4)
        toolbar_button_style = (
            "QPushButton {"
            "background: rgba(228, 242, 255, 0.52);"
            "border: 1px solid rgba(255, 255, 255, 0.80);"
            "border-radius: 10px;"
            "color: #4470a7;"
            "font-size: 11px;"
            "font-weight: 700;"
            "padding: 4px 8px;"
            "}"
            "QPushButton:hover {"
            "background: rgba(206, 229, 255, 0.72);"
            "border: 1px solid rgba(255, 255, 255, 0.88);"
            "color: #2f5c98;"
            "}"
        )
        self._toolbar_buttons: dict[str, QPushButton] = {}
        for action_key, label in (
            ("home", "首页"),
            ("chat", "对话"),
            ("camera", "摄像头"),
        ):
            button = QPushButton(label)
            button.setCursor(Qt.PointingHandCursor)
            button.setMinimumHeight(22)
            button.setStyleSheet(toolbar_button_style)
            if action_key == "chat":
                button.clicked.connect(self.toggle_chat_panel)
            else:
                button.clicked.connect(lambda _checked=False, key=action_key: self.toolbar_action_requested.emit(key))
            primary_toolbar_row.addWidget(button, 1)
            self._toolbar_buttons[action_key] = button
        for action_key, label in (
            ("dashboard", "可视化分析"),
            ("report", "健康报告"),
            ("exit", "退出"),
        ):
            button = QPushButton(label)
            button.setCursor(Qt.PointingHandCursor)
            button.setMinimumHeight(24)
            button.setStyleSheet(toolbar_button_style)
            button.clicked.connect(lambda _checked=False, key=action_key: self.toolbar_action_requested.emit(key))
            secondary_toolbar_row.addWidget(button, 1)
            self._toolbar_buttons[action_key] = button
        toolbar_layout.addLayout(primary_toolbar_row)
        toolbar_layout.addLayout(secondary_toolbar_row)
        self.toolbar_frame.hide()
        layout.addWidget(self.toolbar_frame)

        self._drag_offset: Optional[QPoint] = None
        self._press_global_pos: Optional[QPoint] = None
        self._press_local_pos: Optional[QPoint] = None
        self._current_mode = "idle"

    def setMood(self, mood: PetMood) -> None:
        self.avatar.setMood(mood)

    def _is_avatar_hit(self, position: QPoint) -> bool:
        return self.avatar.geometry().contains(position)

    def set_companion_feedback(
        self,
        *,
        mode_key: str,
        bubble_text: str,
        marquee_text: str = "",
        show_bubble: bool = True,
        auto_hide_ms: int = 0,
    ) -> None:
        previous_mode = self._current_mode
        self._current_mode = mode_key
        normalized_text = bubble_text.strip()
        if show_bubble and normalized_text:
            display_text = normalized_text
            if len(display_text) > 260:
                display_text = f"{display_text[:257]}..."
            self.bubble_label.setText(display_text)
            self.bubble_label.setToolTip(normalized_text if display_text != normalized_text else "")
            self.bubble_label.show()
            if auto_hide_ms > 0:
                self._bubble_timer.start(auto_hide_ms)
            else:
                self._bubble_timer.stop()
        elif mode_key != previous_mode:
            self._bubble_timer.stop()
            self.bubble_label.hide()
            self.bubble_label.setToolTip("")
            self._bubble_extra_height = 0

        status_text = marquee_text.strip()
        if status_text:
            self.marquee_label.setText(status_text)
            self.marquee_label.show()
        else:
            self.marquee_label.hide()
        self.set_rest_mode_available(mode_key == "fatigue")

        bubble_styles = {
            "focus": (
                "background: rgba(227, 242, 255, 0.84);"
                "color: #3b679e;"
                "border: 1px solid rgba(255, 255, 255, 0.84);"
            ),
            "fatigue": (
                "background: rgba(255, 241, 219, 0.82);"
                "color: #7b6a4b;"
                "border: 1px solid rgba(255, 255, 255, 0.84);"
            ),
            "soothe": (
                "background: rgba(230, 233, 255, 0.82);"
                "color: #5d6fa7;"
                "border: 1px solid rgba(255, 255, 255, 0.84);"
            ),
        }
        bubble_style = bubble_styles.get(
            mode_key,
            "background: rgba(229, 243, 255, 0.84);"
            "color: #3b679e;"
            "border: 1px solid rgba(255, 255, 255, 0.84);",
        )
        self.bubble_label.setStyleSheet(
            bubble_style
            + "border-radius: 18px;"
            + "padding: 10px 14px;"
            + "font-size: 12px;"
            + "font-weight: 600;"
        )
        if show_bubble and normalized_text:
            content_height = self.bubble_label.heightForWidth(max(180, self.width() - 32))
            self._bubble_extra_height = max(0, min(220, content_height - 52))
        self._apply_window_size()

    def _toggle_toolbar(self) -> None:
        visible = not self.toolbar_frame.isVisible()
        self.toolbar_frame.setVisible(visible)
        if not visible:
            self.chat_frame.hide()
        self._apply_window_size()

    def _apply_window_size(self) -> None:
        if self.chat_frame.isVisible():
            target_size = QSize(self._chat_expanded_size)
        elif self.toolbar_frame.isVisible():
            target_size = QSize(self._expanded_size)
        else:
            target_size = QSize(self._compact_size)
        if self.rest_frame.isVisible():
            target_size.setHeight(target_size.height() + self._rest_controls_height)
        target_size.setHeight(target_size.height() + self._bubble_extra_height)
        self.setFixedSize(target_size)

    def _hide_bubble(self) -> None:
        self.bubble_label.hide()
        self.bubble_label.setToolTip("")
        self._bubble_extra_height = 0
        self._apply_window_size()

    def _emit_rest_requested(self) -> None:
        if self._rest_active:
            return
        duration_minutes = int(self.rest_duration_combo.currentData() or 5)
        self.rest_requested.emit(duration_minutes)

    def _dismiss_rest_prompt(self) -> None:
        if self._rest_active:
            return
        self._rest_prompt_dismissed = True
        self.rest_frame.hide()
        self._apply_window_size()

    def set_rest_mode_available(self, available: bool) -> None:
        if not available or (available and not self._rest_mode_available):
            self._rest_prompt_dismissed = False
        self._rest_mode_available = available
        self.rest_close_button.setVisible(available and not self._rest_active)
        self.rest_frame.setVisible(
            self._rest_active or (available and not self._rest_prompt_dismissed)
        )
        self._apply_window_size()

    def set_rest_progress(self, active: bool, remaining_seconds: int = 0) -> None:
        self._rest_active = active
        self.rest_duration_combo.setEnabled(not active)
        self.rest_start_button.setEnabled(not active)
        self.rest_close_button.setVisible(
            not active and self._rest_mode_available and not self._rest_prompt_dismissed
        )
        if active:
            minutes, seconds = divmod(max(0, remaining_seconds), 60)
            self.rest_start_button.setText(f"休息中 {minutes:02d}:{seconds:02d}")
        else:
            self.rest_start_button.setText("开始休息")
        self.rest_frame.setVisible(
            active or (self._rest_mode_available and not self._rest_prompt_dismissed)
        )
        self._apply_window_size()

    def set_camera_enabled(self, enabled: bool) -> None:
        camera_button = self._toolbar_buttons.get("camera")
        if camera_button is None:
            return
        camera_button.setText("关摄像头" if enabled else "开摄像头")

    def set_chat_history_html(self, html: str) -> None:
        self.chat_view.setHtml(html)
        self.chat_view.verticalScrollBar().setValue(self.chat_view.verticalScrollBar().maximum())

    def set_chat_busy(self, busy: bool) -> None:
        self.chat_input.setEnabled(not busy)
        self.chat_send_button.setEnabled(not busy)

    def clear_chat_input(self) -> None:
        self.chat_input.clear()

    def toggle_chat_panel(self) -> None:
        self.toolbar_frame.show()
        self.chat_frame.setVisible(not self.chat_frame.isVisible())
        self._apply_window_size()
        if self.chat_frame.isVisible():
            self.chat_input.setFocus()

    def _emit_chat_submit(self) -> None:
        text = self.chat_input.text().strip()
        if not text:
            return
        self.chat_submitted.emit(text)

    def mousePressEvent(self, event) -> None:  # noqa: N802
        if event.button() == Qt.LeftButton:
            self._press_global_pos = event.globalPosition().toPoint()
            self._press_local_pos = event.position().toPoint()
            self._drag_offset = None
            event.accept()
            return
        if event.button() == Qt.MiddleButton:
            self._toggle_toolbar()
            event.accept()
            return
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event) -> None:  # noqa: N802
        if event.buttons() & Qt.LeftButton and self._press_global_pos is not None:
            if self._drag_offset is None:
                movement = event.globalPosition().toPoint() - self._press_global_pos
                if movement.manhattanLength() < QApplication.startDragDistance():
                    event.accept()
                    return
                self._drag_offset = self._press_global_pos - self.frameGeometry().topLeft()
            if self._drag_offset is not None:
                self.move(event.globalPosition().toPoint() - self._drag_offset)
                event.accept()
                return
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event) -> None:  # noqa: N802
        self._drag_offset = None
        self._press_global_pos = None
        self._press_local_pos = None
        super().mouseReleaseEvent(event)

    def mouseDoubleClickEvent(self, event) -> None:  # noqa: N802
        if event.button() == Qt.LeftButton and self._is_avatar_hit(event.position().toPoint()):
            self._drag_offset = None
            self._press_global_pos = None
            self._press_local_pos = None
            self.avatar.playHoverOnce()
            event.accept()
            return
        super().mouseDoubleClickEvent(event)


def _clamp_unit(value: float) -> float:
    return max(0.0, min(1.0, value))


def _behavior_baseline(history: list[float]) -> Optional[float]:
    if len(history) < _ACTIVITY_BASELINE_PERIODS:
        return None
    recent = history[-4:]
    older = history[-10:-4]
    recent_avg = sum(recent) / max(1, len(recent))
    older_avg = sum(older) / max(1, len(older))
    return 0.4 * recent_avg + 0.6 * older_avg


def _format_behavior_state(state: str) -> str:
    mapping = {
        "warming": "基线积累中",
        "steady": "平稳",
        "anxious": "高频切换",
        "fatigued": "操作迟缓",
        "slowed": "节奏放缓",
        "unavailable": "未启用",
    }
    return mapping.get(state, "平稳")


class ActivityMonitor(QObject):
    snapshot_changed = Signal(object)
    status_changed = Signal(str)

    def __init__(self, parent: Optional[QObject] = None) -> None:
        super().__init__(parent)
        self._timer = QTimer(self)
        self._timer.setInterval(1000)
        self._timer.timeout.connect(self._tick)
        self._lock = threading.Lock()
        self._keyboard_listener = None
        self._mouse_listener = None
        self._period_started_at: Optional[float] = None
        self._key_count = 0
        self._keyboard_active_seconds: set[int] = set()
        self._mouse_active_seconds: set[int] = set()
        self._mouse_distance = 0.0
        self._mouse_click_count = 0
        self._mouse_scroll_count = 0
        self._modality_switches = 0
        self._last_mouse_position: Optional[tuple[int, int]] = None
        self._last_modality: Optional[str] = None
        self._last_modality_at: Optional[float] = None
        self._keyboard_history: deque[float] = deque(maxlen=_ACTIVITY_BASELINE_PERIODS)
        self._mouse_history: deque[float] = deque(maxlen=_ACTIVITY_BASELINE_PERIODS)
        self._available = pynput_keyboard is not None and pynput_mouse is not None

    def start(self) -> None:
        if self._timer.isActive():
            return
        if not self._available:
            message = "键鼠行为监测未启用，安装 pynput 后可用。"
            self.status_changed.emit(message)
            self.snapshot_changed.emit(self._build_snapshot(status=message))
            return
        try:
            self._keyboard_listener = pynput_keyboard.Listener(on_press=self._on_key_press)
            self._mouse_listener = pynput_mouse.Listener(
                on_move=self._on_mouse_move,
                on_click=self._on_mouse_click,
                on_scroll=self._on_mouse_scroll,
            )
            self._keyboard_listener.start()
            self._mouse_listener.start()
        except Exception as exc:
            self._keyboard_listener = None
            self._mouse_listener = None
            message = f"键鼠行为监测启动失败：{exc}"
            self.status_changed.emit(message)
            self.snapshot_changed.emit(self._build_snapshot(status=message))
            return

        now = time.monotonic()
        with self._lock:
            self._period_started_at = now
            self._reset_period_locked()
        self._timer.start()
        message = "键鼠行为监测已启动，按 30 秒周期分析活跃度。"
        self.status_changed.emit(message)
        self.snapshot_changed.emit(self._build_snapshot(status=message))

    def stop(self) -> None:
        self._timer.stop()
        for listener in (self._keyboard_listener, self._mouse_listener):
            if listener is not None:
                try:
                    listener.stop()
                except Exception:
                    pass
        self._keyboard_listener = None
        self._mouse_listener = None
        with self._lock:
            self._period_started_at = None
            self._reset_period_locked()

    def _reset_period_locked(self) -> None:
        self._key_count = 0
        self._keyboard_active_seconds.clear()
        self._mouse_active_seconds.clear()
        self._mouse_distance = 0.0
        self._mouse_click_count = 0
        self._mouse_scroll_count = 0
        self._modality_switches = 0
        self._last_mouse_position = None
        self._last_modality = None
        self._last_modality_at = None

    def _tick(self) -> None:
        with self._lock:
            started_at = self._period_started_at
        if started_at is None:
            return
        now = time.monotonic()
        if now - started_at < _ACTIVITY_PERIOD_SECONDS:
            return
        self._complete_period(now)

    def _record_second_locked(self, bucket: set[int], timestamp: float) -> None:
        if self._period_started_at is None:
            return
        elapsed = int(timestamp - self._period_started_at)
        if 0 <= elapsed < int(_ACTIVITY_PERIOD_SECONDS):
            bucket.add(elapsed)

    def _record_modality_locked(self, modality: str, timestamp: float) -> None:
        if (
            self._last_modality is not None
            and self._last_modality != modality
            and self._last_modality_at is not None
            and timestamp - self._last_modality_at <= _ACTIVITY_SWITCH_WINDOW_SECONDS
        ):
            self._modality_switches += 1
        self._last_modality = modality
        self._last_modality_at = timestamp

    def _on_key_press(self, key) -> None:  # pragma: no cover - callback executed by listener thread
        del key
        timestamp = time.monotonic()
        with self._lock:
            if self._period_started_at is None:
                return
            self._key_count += 1
            self._record_second_locked(self._keyboard_active_seconds, timestamp)
            self._record_modality_locked("keyboard", timestamp)

    def _on_mouse_move(self, x: int, y: int) -> None:  # pragma: no cover - callback executed by listener thread
        timestamp = time.monotonic()
        with self._lock:
            if self._period_started_at is None:
                return
            if self._last_mouse_position is not None:
                last_x, last_y = self._last_mouse_position
                self._mouse_distance += math.hypot(x - last_x, y - last_y)
            self._last_mouse_position = (x, y)
            self._record_second_locked(self._mouse_active_seconds, timestamp)
            self._record_modality_locked("mouse", timestamp)

    def _on_mouse_click(self, x: int, y: int, button, pressed: bool) -> None:  # pragma: no cover - callback executed by listener thread
        del x, y, button
        if not pressed:
            return
        timestamp = time.monotonic()
        with self._lock:
            if self._period_started_at is None:
                return
            self._mouse_click_count += 1
            self._record_second_locked(self._mouse_active_seconds, timestamp)
            self._record_modality_locked("mouse", timestamp)

    def _on_mouse_scroll(self, x: int, y: int, dx: int, dy: int) -> None:  # pragma: no cover - callback executed by listener thread
        del x, y
        timestamp = time.monotonic()
        with self._lock:
            if self._period_started_at is None:
                return
            self._mouse_scroll_count += 1
            self._mouse_distance += (abs(dx) + abs(dy)) * 120.0
            self._record_second_locked(self._mouse_active_seconds, timestamp)
            self._record_modality_locked("mouse", timestamp)

    def _build_snapshot(
        self,
        *,
        status: str,
        key_rate_per_min: float = 0.0,
        keyboard_active_seconds: int = 0,
        keyboard_activity: float = 0.0,
        keyboard_baseline: Optional[float] = None,
        keyboard_declined: bool = False,
        mouse_distance: float = 0.0,
        mouse_active_seconds: int = 0,
        mouse_activity: float = 0.0,
        mouse_baseline: Optional[float] = None,
        mouse_declined: bool = False,
        mouse_click_count: int = 0,
        mouse_scroll_count: int = 0,
        modality_switches: int = 0,
        behavior_state: str = "warming",
    ) -> dict[str, object]:
        periods_collected = max(len(self._keyboard_history), len(self._mouse_history))
        baseline_ready = keyboard_baseline is not None and mouse_baseline is not None
        behavior_label = _format_behavior_state(behavior_state)
        if behavior_state == "anxious":
            insight = (
                f"近 30 秒键鼠切换 {modality_switches} 次，"
                f"键盘活跃 {keyboard_activity:.2f}，鼠标活跃 {mouse_activity:.2f}。"
            )
        elif behavior_state == "fatigued":
            insight = (
                f"近 30 秒键盘/鼠标活跃度较基线下降，"
                f"键盘 {keyboard_activity:.2f}，鼠标 {mouse_activity:.2f}。"
            )
        elif behavior_state == "slowed":
            insight = (
                f"近 30 秒存在节奏放缓，键盘 {keyboard_activity:.2f}，"
                f"鼠标 {mouse_activity:.2f}。"
            )
        elif baseline_ready:
            insight = (
                f"键盘 {keyboard_activity:.2f} / 鼠标 {mouse_activity:.2f}，"
                f"行为节奏总体平稳。"
            )
        else:
            insight = f"基线积累中（已完成 {periods_collected}/{_ACTIVITY_BASELINE_PERIODS} 个周期）。"
        return {
            "available": self._available,
            "enabled": self._timer.isActive() and self._available,
            "status": status,
            "period_seconds": int(_ACTIVITY_PERIOD_SECONDS),
            "periods_collected": periods_collected,
            "baseline_ready": baseline_ready,
            "key_rate_per_min": round(key_rate_per_min, 1),
            "keyboard_active_seconds": keyboard_active_seconds,
            "keyboard_activity": round(keyboard_activity, 3),
            "keyboard_baseline": round(keyboard_baseline, 3) if keyboard_baseline is not None else None,
            "keyboard_declined": keyboard_declined,
            "mouse_distance": round(mouse_distance, 1),
            "mouse_active_seconds": mouse_active_seconds,
            "mouse_activity": round(mouse_activity, 3),
            "mouse_baseline": round(mouse_baseline, 3) if mouse_baseline is not None else None,
            "mouse_declined": mouse_declined,
            "mouse_click_count": mouse_click_count,
            "mouse_scroll_count": mouse_scroll_count,
            "modality_switches": modality_switches,
            "behavior_state": behavior_state,
            "behavior_label": behavior_label,
            "insight": insight,
        }

    def _complete_period(self, now: float) -> None:
        with self._lock:
            started_at = self._period_started_at
            if started_at is None:
                return
            elapsed = max(1.0, min(_ACTIVITY_PERIOD_SECONDS, now - started_at))
            key_count = self._key_count
            keyboard_active_seconds = min(int(_ACTIVITY_PERIOD_SECONDS), len(self._keyboard_active_seconds))
            mouse_active_seconds = min(int(_ACTIVITY_PERIOD_SECONDS), len(self._mouse_active_seconds))
            mouse_distance = self._mouse_distance
            mouse_click_count = self._mouse_click_count
            mouse_scroll_count = self._mouse_scroll_count
            modality_switches = self._modality_switches
            self._period_started_at = now
            self._reset_period_locked()

        key_rate_per_min = key_count * 60.0 / elapsed
        keyboard_activity = 0.7 * _clamp_unit(key_rate_per_min / 200.0) + 0.3 * (keyboard_active_seconds / _ACTIVITY_PERIOD_SECONDS)
        mouse_activity = 0.5 * _clamp_unit(mouse_distance / 6000.0) + 0.5 * (mouse_active_seconds / _ACTIVITY_PERIOD_SECONDS)
        keyboard_baseline = _behavior_baseline(list(self._keyboard_history))
        mouse_baseline = _behavior_baseline(list(self._mouse_history))
        keyboard_declined = keyboard_baseline is not None and keyboard_activity <= keyboard_baseline * 0.8
        mouse_declined = mouse_baseline is not None and mouse_activity <= mouse_baseline * 0.8

        if keyboard_declined and mouse_declined:
            behavior_state = "fatigued"
        elif modality_switches >= 10 and keyboard_activity >= 0.55 and mouse_activity >= 0.45:
            behavior_state = "anxious"
        elif keyboard_declined or mouse_declined:
            behavior_state = "slowed"
        elif keyboard_baseline is None or mouse_baseline is None:
            behavior_state = "warming"
        else:
            behavior_state = "steady"

        self._keyboard_history.append(keyboard_activity)
        self._mouse_history.append(mouse_activity)

        summary = self._build_snapshot(
            status="键鼠行为周期已更新",
            key_rate_per_min=key_rate_per_min,
            keyboard_active_seconds=keyboard_active_seconds,
            keyboard_activity=keyboard_activity,
            keyboard_baseline=keyboard_baseline,
            keyboard_declined=keyboard_declined,
            mouse_distance=mouse_distance,
            mouse_active_seconds=mouse_active_seconds,
            mouse_activity=mouse_activity,
            mouse_baseline=mouse_baseline,
            mouse_declined=mouse_declined,
            mouse_click_count=mouse_click_count,
            mouse_scroll_count=mouse_scroll_count,
            modality_switches=modality_switches,
            behavior_state=behavior_state,
        )
        self.snapshot_changed.emit(summary)
        self.status_changed.emit(str(summary["insight"]))

class StatCard(QFrame):
    def __init__(self, title: str, value: str, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setObjectName("StatCard")
        glow = QGraphicsDropShadowEffect(self)
        glow.setBlurRadius(30)
        glow.setOffset(0, 8)
        glow.setColor(QColor(0, 0, 0, 88))
        self.setGraphicsEffect(glow)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 14, 16, 14)
        layout.setSpacing(6)
        self.title_label = QLabel(title)
        self.title_label.setObjectName("CardTitle")
        self.value_label = QLabel(value)
        self.value_label.setObjectName("CardValue")
        self.value_label.setWordWrap(True)
        layout.addWidget(self.title_label)
        layout.addWidget(self.value_label)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)

    def setValue(self, value: str) -> None:
        self.value_label.setText(value)


class EyeMuseWindow(QMainWindow):
    camera_stop_requested = Signal()

    def __init__(self) -> None:
        super().__init__()
        self.setWindowFlags(Qt.Window | Qt.FramelessWindowHint)
        self.setWindowTitle("EyeMuse")
        self.resize(1420, 920)
        self.setMinimumSize(1180, 760)
        self._dragging_window = False
        self._drag_offset = QPoint()

        self._camera_worker: Optional[CameraWorker] = None
        self._camera_thread: Optional[QThread] = None
        self._llm_thread: Optional[QThread] = None
        self._llm_worker: Optional[LLMStreamWorker] = None
        self._conversation: list[ConversationItem] = []
        self._local_camera_enabled = False
        self._face_count = 0
        self._stress_score = 0
        self._fatigue_score = 0
        self._dominant_signal = "none"
        self._calibration_state = "waiting"
        self._heart_rate: Optional[float] = None
        self._respiration_rate: Optional[float] = None
        self._hrv: Optional[float] = None
        self._monitoring_samples: deque[MonitoringMetricSample] = deque()
        self._activity_monitor: Optional[ActivityMonitor] = None
        self._behavior_summary = self._default_behavior_summary()
        self._llm_client = self._create_llm_client()
        self._dashboard_repository = self._create_dashboard_repository()
        self._dashboard_period = "day"
        self._dashboard_custom_range = (
            QDate.currentDate().addDays(-7).toPython(),
            QDate.currentDate().addDays(-1).toPython(),
        )
        self._dashboard_payload_key = ""
        self._dashboard_chart_shell_loaded = False
        self._dashboard_chart_page_ready = False
        self._pending_dashboard_payload: Optional[dict] = None
        self._report_custom_range = (
            QDate.currentDate().addDays(-7).toPython(),
            QDate.currentDate().addDays(-1).toPython(),
        )
        self._report_custom_mode = False
        self._latest_daily_report_md = ""
        self._latest_weekly_report_md = ""
        self._latest_daily_report_html = ""
        self._latest_weekly_report_html = ""
        self._report_snapshot_cache: dict[str, str] = {}
        self._report_payload_key = ""
        self._streaming_reply_index: Optional[int] = None
        self._streaming_user_text: str = ""
        self._streaming_from_companion = False
        self._current_pet_mood = PetMood.idle
        self._current_pet_hint = "等待用户输入，或开启摄像头观察状态变化。"
        self._current_companion_mode = "idle"
        self._companion_mode_since = time.monotonic()
        self._companion_candidate_mode = "idle"
        self._companion_candidate_since = self._companion_mode_since
        self._companion_window: Optional[CompanionPetWindow] = None
        self._rest_duration_seconds = 0
        self._rest_started_at = 0.0
        self._rest_timer = QTimer(self)
        self._rest_timer.setInterval(1000)
        self._rest_timer.timeout.connect(self._update_rest_countdown)
        self._dashboard_refresh_timer = QTimer(self)
        self._dashboard_refresh_timer.setSingleShot(True)
        self._dashboard_refresh_timer.setInterval(2000)
        self._dashboard_refresh_timer.timeout.connect(self._refresh_dashboard_page)
        self._report_refresh_timer = QTimer(self)
        self._report_refresh_timer.setSingleShot(True)
        self._report_refresh_timer.setInterval(10000)
        self._report_refresh_timer.timeout.connect(self._refresh_report_page)
        self._repository_sync_timer = QTimer(self)
        self._repository_sync_timer.setInterval(10000)
        self._repository_sync_timer.timeout.connect(self._sync_dashboard_repository)

        self._build_ui()
        self._update_companion_controls()
        self._update_window_control_buttons()
        self._start_activity_monitor()
        self._apply_theme()
        self._sync_dashboard_repository()
        self._refresh_dashboard_page()
        self._refresh_report_page()
        self._repository_sync_timer.start()
        self._append_system_message("EyeMuse 前端原型已就绪，输入文本或打开摄像头开始交互。")
        QTimer.singleShot(0, self._enter_default_companion_mode)

    @staticmethod
    def _create_llm_client():
        if LLMClient is None:
            return None
        try:
            return LLMClient()
        except Exception:
            return None

    @staticmethod
    def _create_dashboard_repository():
        if DashboardRepository is None:
            return None
        try:
            return DashboardRepository()
        except Exception:
            return None

    @staticmethod
    def _default_behavior_summary() -> dict[str, object]:
        available = pynput_keyboard is not None and pynput_mouse is not None
        status = "键鼠行为监测未启动。" if available else "键鼠行为监测未启用，安装 pynput 后可用。"
        return {
            "available": available,
            "enabled": False,
            "status": status,
            "period_seconds": int(_ACTIVITY_PERIOD_SECONDS),
            "periods_collected": 0,
            "baseline_ready": False,
            "key_rate_per_min": 0.0,
            "keyboard_active_seconds": 0,
            "keyboard_activity": 0.0,
            "keyboard_baseline": None,
            "keyboard_declined": False,
            "mouse_distance": 0.0,
            "mouse_active_seconds": 0,
            "mouse_activity": 0.0,
            "mouse_baseline": None,
            "mouse_declined": False,
            "mouse_click_count": 0,
            "mouse_scroll_count": 0,
            "modality_switches": 0,
            "behavior_state": "warming" if available else "unavailable",
            "behavior_label": _format_behavior_state("warming" if available else "unavailable"),
            "insight": "等待键鼠行为基线积累。" if available else "未安装 pynput。",
        }

    def _start_activity_monitor(self) -> None:
        if self._activity_monitor is not None:
            return
        self._activity_monitor = ActivityMonitor(self)
        self._activity_monitor.snapshot_changed.connect(self._update_behavior_summary)
        self._activity_monitor.status_changed.connect(self._update_behavior_status)
        self._activity_monitor.start()

    def _update_behavior_status(self, message: str) -> None:
        del message

    def _behavior_hint_fragment(self) -> str:
        state = str(self._behavior_summary.get("behavior_state", "warming"))
        if state in {"warming", "unavailable"}:
            return str(self._behavior_summary.get("insight", ""))
        keyboard_activity = float(self._behavior_summary.get("keyboard_activity", 0.0) or 0.0)
        mouse_activity = float(self._behavior_summary.get("mouse_activity", 0.0) or 0.0)
        if state == "anxious":
            switches = int(self._behavior_summary.get("modality_switches", 0) or 0)
            return (
                f"键鼠 30 秒行为显示高频切换（切换 {switches} 次，"
                f"键盘 {keyboard_activity:.2f}，鼠标 {mouse_activity:.2f}）。"
            )
        if state == "fatigued":
            return (
                f"键鼠 30 秒行为显示操作迟缓（键盘 {keyboard_activity:.2f}，"
                f"鼠标 {mouse_activity:.2f}）。"
            )
        if state == "slowed":
            return (
                f"键鼠活跃较近期基线略有下降（键盘 {keyboard_activity:.2f}，"
                f"鼠标 {mouse_activity:.2f}）。"
            )
        return (
            f"键盘 {keyboard_activity:.2f} / 鼠标 {mouse_activity:.2f}，"
            "行为节奏平稳。"
        )

    def _apply_behavior_signal_to_mood(self, mood: PetMood, hint: str) -> tuple[PetMood, str]:
        if not bool(self._behavior_summary.get("available")):
            return mood, hint
        state = str(self._behavior_summary.get("behavior_state", "warming"))
        if state in {"warming", "steady", "unavailable"}:
            if state == "steady" and mood == PetMood.offline:
                return PetMood.idle, self._behavior_hint_fragment()
            return mood, hint

        behavior_hint = self._behavior_hint_fragment()
        merged_hint = f"{behavior_hint} {hint}".strip() if hint else behavior_hint
        if state == "fatigued":
            if mood in {PetMood.offline, PetMood.idle, PetMood.listening, PetMood.responding}:
                return PetMood.thinking if self._fatigue_score < 55 else PetMood.alert, merged_hint
            if mood == PetMood.thinking and self._fatigue_score >= 60:
                return PetMood.alert, merged_hint
            return mood, merged_hint
        if state == "anxious":
            if mood in {PetMood.offline, PetMood.idle, PetMood.listening}:
                return PetMood.thinking, merged_hint
            return mood, merged_hint
        if mood == PetMood.offline:
            return PetMood.idle, behavior_hint
        return mood, merged_hint

    def _maybe_apply_behavior_only_mood(self) -> None:
        if self._local_camera_enabled and self._face_count > 0 and self._calibration_state in {"calibrating", "ready"}:
            return
        state = str(self._behavior_summary.get("behavior_state", "warming"))
        if state == "warming":
            return
        if not bool(self._behavior_summary.get("available")):
            return
        base_mood = PetMood.offline if not self._local_camera_enabled else PetMood.idle
        base_hint = "摄像头未提供稳定信号，当前以键鼠行为监测为辅助参考。"
        mood, hint = self._apply_behavior_signal_to_mood(base_mood, base_hint)
        self._set_mood(mood, hint)

    def _update_behavior_summary(self, payload: object) -> None:
        if not isinstance(payload, dict):
            return
        self._behavior_summary = {**self._default_behavior_summary(), **payload}
        keyboard_activity = float(self._behavior_summary.get("keyboard_activity", 0.0) or 0.0)
        mouse_activity = float(self._behavior_summary.get("mouse_activity", 0.0) or 0.0)
        key_rate = float(self._behavior_summary.get("key_rate_per_min", 0.0) or 0.0)
        mouse_distance = float(self._behavior_summary.get("mouse_distance", 0.0) or 0.0)
        behavior_label = str(self._behavior_summary.get("behavior_label", "基线积累中"))
        periods_collected = int(self._behavior_summary.get("periods_collected", 0) or 0)

        if hasattr(self, "keyboard_card"):
            if not self._behavior_summary.get("available"):
                self.keyboard_card.setValue("未启用")
            elif self._behavior_summary.get("baseline_ready"):
                decline_suffix = " ↓" if self._behavior_summary.get("keyboard_declined") else ""
                self.keyboard_card.setValue(f"{keyboard_activity:.2f}{decline_suffix} | {key_rate:.0f} keys/min")
            else:
                self.keyboard_card.setValue(f"基线 {periods_collected}/{_ACTIVITY_BASELINE_PERIODS}")
        if hasattr(self, "mouse_card"):
            if not self._behavior_summary.get("available"):
                self.mouse_card.setValue("未启用")
            elif self._behavior_summary.get("baseline_ready"):
                decline_suffix = " ↓" if self._behavior_summary.get("mouse_declined") else ""
                self.mouse_card.setValue(f"{mouse_activity:.2f}{decline_suffix} | {mouse_distance:.0f}px")
            else:
                self.mouse_card.setValue(f"基线 {periods_collected}/{_ACTIVITY_BASELINE_PERIODS}")
        if hasattr(self, "behavior_card"):
            if not self._behavior_summary.get("available"):
                self.behavior_card.setValue("未安装 pynput，键鼠监测未启用")
            elif not self._behavior_summary.get("baseline_ready"):
                self.behavior_card.setValue(f"基线积累中（{periods_collected}/{_ACTIVITY_BASELINE_PERIODS}）")
            else:
                self.behavior_card.setValue(f"{behavior_label} · 键{keyboard_activity:.2f} 鼠{mouse_activity:.2f}")
        self._refresh_companion_feedback()
        self._maybe_apply_behavior_only_mood()
        self._schedule_analytics_refresh()

    def _build_ui(self) -> None:
        central = QWidget(self)
        self.setCentralWidget(central)

        root_layout = QVBoxLayout(central)
        root_layout.setContentsMargins(0, 0, 0, 0)
        root_layout.setSpacing(0)

        root_layout.addWidget(self._build_nav_bar())

        self.page_stack = QStackedWidget()
        self.page_stack.setObjectName("PageStack")
        root_layout.addWidget(self.page_stack, 1)

        self.home_page = self._build_home_page()
        self.dashboard_page = self._build_dashboard_page()
        self.report_page = self._build_report_page()

        self.page_stack.addWidget(self.home_page)
        self.page_stack.addWidget(self.dashboard_page)
        self.page_stack.addWidget(self.report_page)
        self._switch_page("home", show_status=False)

        self.statusBar().showMessage("本地优先，启动后自动进入陪伴桌宠模式")

    def _build_nav_bar(self) -> QFrame:
        frame = QFrame()
        self.nav_bar_frame = frame
        frame.setObjectName("NavBar")
        frame.installEventFilter(self)
        layout = QHBoxLayout(frame)
        layout.setContentsMargins(8, 1, 8, 1)
        layout.setSpacing(0)

        logo_path = Path(__file__).resolve().parents[1] / "assets" / "logo.png"
        self.nav_logo_label = QLabel()
        self.nav_logo_label.setObjectName("NavLogo")
        self.nav_logo_label.setFixedWidth(320)
        self.nav_logo_label.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)
        self.nav_logo_label.installEventFilter(self)
        logo_pixmap = QPixmap(str(logo_path))
        if not logo_pixmap.isNull():
            logo_pixmap = _make_dark_background_transparent(logo_pixmap)
            self.nav_logo_label.setPixmap(logo_pixmap.scaledToHeight(64, Qt.SmoothTransformation))
        else:
            self.nav_logo_label.setText("EyeMuse")

        self.nav_right_controls = QWidget()
        self.nav_right_controls.setObjectName("WindowControlGroup")
        self.nav_right_controls.setFixedWidth(320)
        right_controls_layout = QHBoxLayout(self.nav_right_controls)
        right_controls_layout.setContentsMargins(8, 5, 8, 5)
        right_controls_layout.setSpacing(6)
        right_controls_layout.addStretch(1)

        self.window_minimize_button = QPushButton()
        self.window_maximize_button = QPushButton()
        self.window_close_button = QPushButton()
        for button in (self.window_minimize_button, self.window_maximize_button, self.window_close_button):
            button.setObjectName("WindowControlButton")
            button.setCursor(Qt.PointingHandCursor)
            button.setFixedSize(30, 30)
            button.setFocusPolicy(Qt.NoFocus)
        self.window_minimize_button.setIcon(self.style().standardIcon(QStyle.StandardPixmap.SP_TitleBarMinButton))
        self.window_maximize_button.setIcon(self.style().standardIcon(QStyle.StandardPixmap.SP_TitleBarMaxButton))
        self.window_close_button.setIcon(self.style().standardIcon(QStyle.StandardPixmap.SP_TitleBarCloseButton))
        self.window_minimize_button.setIconSize(QSize(14, 14))
        self.window_maximize_button.setIconSize(QSize(14, 14))
        self.window_close_button.setIconSize(QSize(14, 14))
        self.window_minimize_button.setToolTip("最小化")
        self.window_maximize_button.setToolTip("最大化 / 还原")
        self.window_close_button.setToolTip("关闭")
        self.window_close_button.setObjectName("WindowCloseButton")

        self.window_minimize_button.clicked.connect(self.showMinimized)
        self.window_maximize_button.clicked.connect(self._toggle_window_max_restore)
        self.window_close_button.clicked.connect(self.close)

        right_controls_layout.addWidget(self.window_minimize_button)
        right_controls_layout.addWidget(self.window_maximize_button)
        right_controls_layout.addWidget(self.window_close_button)

        self.home_nav_button = QPushButton("主页面")
        self.dashboard_nav_button = QPushButton("可视化分析")
        self.report_nav_button = QPushButton("健康报告")

        for button in (self.dashboard_nav_button, self.home_nav_button, self.report_nav_button):
            button.setObjectName("NavButton")
            button.setCursor(Qt.PointingHandCursor)
            button.setCheckable(True)

        self.home_nav_button.clicked.connect(lambda: self._switch_page("home"))
        self.dashboard_nav_button.clicked.connect(lambda: self._switch_page("dashboard"))
        self.report_nav_button.clicked.connect(lambda: self._switch_page("report"))
        self.home_nav_button.setChecked(True)

        layout.addWidget(self.nav_logo_label, 0, Qt.AlignLeft | Qt.AlignVCenter)
        layout.addStretch(1)
        layout.addWidget(self.dashboard_nav_button, 0, Qt.AlignVCenter)
        layout.addSpacing(14)
        layout.addWidget(self.home_nav_button, 0, Qt.AlignCenter)
        layout.addSpacing(14)
        layout.addWidget(self.report_nav_button, 0, Qt.AlignVCenter)
        layout.addStretch(1)
        layout.addWidget(self.nav_right_controls, 0, Qt.AlignRight | Qt.AlignVCenter)
        return frame

    def _toggle_window_max_restore(self) -> None:
        if self.isMaximized():
            self.showNormal()
        else:
            self.showMaximized()
        self._update_window_control_buttons()

    def _update_window_control_buttons(self) -> None:
        if hasattr(self, "window_maximize_button"):
            icon = (
                QStyle.StandardPixmap.SP_TitleBarNormalButton
                if self.isMaximized()
                else QStyle.StandardPixmap.SP_TitleBarMaxButton
            )
            self.window_maximize_button.setIcon(self.style().standardIcon(icon))

    def changeEvent(self, event) -> None:  # noqa: N802
        if event.type() == QEvent.Type.WindowStateChange:
            self._update_window_control_buttons()
        super().changeEvent(event)

    def eventFilter(self, watched, event) -> bool:
        if watched in {getattr(self, "nav_logo_label", None), getattr(self, "nav_bar_frame", None)}:
            if event.type() == QEvent.Type.MouseButtonPress and event.button() == Qt.LeftButton:
                self._dragging_window = True
                self._drag_offset = event.globalPosition().toPoint() - self.frameGeometry().topLeft()
                return True
            if event.type() == QEvent.Type.MouseMove and self._dragging_window and event.buttons() & Qt.LeftButton:
                if self.isMaximized():
                    self.showNormal()
                    self._update_window_control_buttons()
                    self._drag_offset = QPoint(self.width() // 2, 24)
                self.move(event.globalPosition().toPoint() - self._drag_offset)
                return True
            if event.type() == QEvent.Type.MouseButtonRelease and event.button() == Qt.LeftButton:
                self._dragging_window = False
                return True
            if event.type() == QEvent.Type.MouseButtonDblClick and event.button() == Qt.LeftButton:
                self._toggle_window_max_restore()
                return True
        return super().eventFilter(watched, event)

    def _build_home_page(self) -> QWidget:
        page = QWidget()
        page.setObjectName("Page")
        root_layout = QGridLayout(page)
        root_layout.setContentsMargins(24, 24, 24, 24)
        root_layout.setHorizontalSpacing(24)
        root_layout.setVerticalSpacing(24)

        left_panel = self._build_left_panel()
        middle_panel = self._build_middle_panel()
        right_panel = self._build_right_panel()

        root_layout.addWidget(left_panel, 0, 0)
        root_layout.addWidget(middle_panel, 0, 1)
        root_layout.addWidget(right_panel, 0, 2)
        root_layout.setColumnStretch(0, 6)
        root_layout.setColumnStretch(1, 7)
        root_layout.setColumnStretch(2, 5)
        return page

    def _panel_frame(self) -> QFrame:
        frame = QFrame()
        frame.setObjectName("Panel")
        effect = QGraphicsDropShadowEffect(frame)
        effect.setBlurRadius(34)
        effect.setOffset(0, 10)
        effect.setColor(QColor(0, 0, 0, 96))
        frame.setGraphicsEffect(effect)
        return frame

    def _build_left_panel(self) -> QFrame:
        frame = self._panel_frame()
        layout = QVBoxLayout(frame)
        layout.setContentsMargins(22, 22, 22, 22)
        layout.setSpacing(16)

        title = QLabel("EyeMuse 桌面宠物")
        title.setObjectName("Title")
        subtitle = QLabel("感知、陪伴、提醒，尽量都留在本地")
        subtitle.setObjectName("Subtitle")

        self.avatar = PetAvatar(show_background=False)
        self.avatar.setObjectName("Avatar")
        self.avatar.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Preferred)
        self.avatar.setAlignment(Qt.AlignCenter)

        avatar_container = QWidget()
        avatar_layout = QVBoxLayout(avatar_container)
        avatar_layout.setContentsMargins(0, 0, 0, 0)
        avatar_layout.setSpacing(0)
        avatar_layout.addStretch(1)
        avatar_layout.addWidget(self.avatar, 0, Qt.AlignCenter)
        avatar_layout.addStretch(1)

        status_row = QHBoxLayout()
        status_row.setSpacing(10)
        self.mood_badge = QLabel("idle")
        self.mood_badge.setObjectName("Badge")
        self.privacy_badge = QLabel("本地模式")
        self.privacy_badge.setObjectName("BadgeSecondary")
        self.minimize_companion_button = QPushButton("最小化陪伴桌宠")
        self.minimize_companion_button.setObjectName("CompanionModeButton")
        self.minimize_companion_button.setCursor(Qt.PointingHandCursor)
        self.minimize_companion_button.clicked.connect(self._enter_companion_mode)
        self.keep_companion_checkbox = QCheckBox("显示桌宠")
        self.keep_companion_checkbox.setChecked(False)
        self.keep_companion_checkbox.stateChanged.connect(self._handle_companion_presence_toggle)
        status_row.addWidget(self.mood_badge)
        status_row.addWidget(self.privacy_badge)
        status_row.addWidget(self.minimize_companion_button)
        status_row.addWidget(self.keep_companion_checkbox)
        status_row.addStretch(1)

        self.pet_hint = QLabel("等待用户输入，或开启摄像头观察状态变化。")
        self.pet_hint.setWordWrap(True)
        self.pet_hint.setObjectName("Hint")

        quick_row = QHBoxLayout()
        quick_row.setSpacing(10)
        self.listen_button = QPushButton("进入聆听")
        self.think_button = QPushButton("思考中")
        self.respond_button = QPushButton("回应一下")
        self.listen_button.setObjectName("PrimaryActionButton")
        self.think_button.setObjectName("SecondaryActionButton")
        self.respond_button.setObjectName("SecondaryActionButton")
        for button in (self.listen_button, self.think_button, self.respond_button):
            button.setCursor(Qt.PointingHandCursor)
            quick_row.addWidget(button)

        self.listen_button.clicked.connect(lambda: self._set_mood(PetMood.listening, "我在听，你可以继续说。"))
        self.think_button.clicked.connect(lambda: self._set_mood(PetMood.thinking, "我在整理你刚才说的话。"))
        self.respond_button.clicked.connect(lambda: self._set_mood(PetMood.responding, "准备给你一个更自然的回应。"))

        self.stress_card = StatCard("压力估计", "未开始检测")
        self.fatigue_card = StatCard("疲劳状态", "未开始检测")
        self.camera_card = StatCard("摄像头", "关闭")
        self.analysis_card = StatCard("分析状态", "等待开始")
        self.event_card = StatCard("最近事件", "等待开始")

        layout.addWidget(title)
        layout.addWidget(subtitle)
        layout.addWidget(avatar_container, 1)
        layout.addLayout(status_row)
        layout.addWidget(self.pet_hint)
        layout.addLayout(quick_row)
        layout.addWidget(self.analysis_card)
        layout.addWidget(self.event_card)
        return frame

    def _build_middle_panel(self) -> QFrame:
        frame = self._panel_frame()
        layout = QVBoxLayout(frame)
        layout.setContentsMargins(22, 22, 22, 22)
        layout.setSpacing(16)

        title = QLabel("对话与回应")
        title.setObjectName("Title")
        subtitle = QLabel("文本输入链路已打通，后续可接入 LLM 与多轮上下文")
        subtitle.setObjectName("Subtitle")

        self.conversation_view = QTextBrowser()
        self.conversation_view.setObjectName("ConversationView")

        input_row = QHBoxLayout()
        self.message_input = QLineEdit()
        self.message_input.setPlaceholderText("输入一句话，让 EyeMuse 回应你")
        self.message_input.returnPressed.connect(self._handle_send)
        self.send_button = QPushButton("发送")
        self.send_button.setObjectName("PrimaryActionButton")
        self.send_button.setCursor(Qt.PointingHandCursor)
        self.send_button.clicked.connect(self._handle_send)
        input_row.addWidget(self.message_input, 1)
        input_row.addWidget(self.send_button)

        action_row = QHBoxLayout()
        self.remind_button = QPushButton("提醒我休息")
        self.energy_button = QPushButton("查看状态")
        self.clear_button = QPushButton("清空对话")
        self.remind_button.setObjectName("SecondaryActionButton")
        self.energy_button.setObjectName("GhostActionButton")
        self.clear_button.setObjectName("GhostActionButton")
        for button in (self.remind_button, self.energy_button, self.clear_button):
            button.setCursor(Qt.PointingHandCursor)
            action_row.addWidget(button)

        self.remind_button.clicked.connect(lambda: self._inject_message("user", "提醒我休息一下"))
        self.energy_button.clicked.connect(lambda: self._inject_message("system", self._current_summary()))
        self.clear_button.clicked.connect(self._clear_conversation)

        layout.addWidget(title)
        layout.addWidget(subtitle)
        layout.addWidget(self.conversation_view, 1)
        layout.addLayout(input_row)
        layout.addLayout(action_row)
        return frame

    def _build_right_panel(self) -> QFrame:
        frame = self._panel_frame()
        layout = QVBoxLayout(frame)
        layout.setContentsMargins(22, 22, 22, 22)
        layout.setSpacing(12)

        title = QLabel("感知面板")
        title.setObjectName("Title")

        self.camera_preview = QLabel("摄像头未开启")
        self.camera_preview.setObjectName("CameraPreview")
        self.camera_preview.setAlignment(Qt.AlignCenter)
        self.camera_preview.setMinimumSize(QSize(360, 250))
        self.camera_preview.setScaledContents(False)

        camera_controls = QHBoxLayout()
        camera_controls.setContentsMargins(0, 2, 0, 2)
        camera_controls.setSpacing(8)
        self.camera_toggle = QCheckBox("开启摄像头")
        self.camera_toggle.stateChanged.connect(self._toggle_camera)
        self.camera_status = QLabel("关闭")
        self.camera_status.setObjectName("InlineStatus")
        camera_controls.addWidget(self.camera_toggle)
        camera_controls.addStretch(1)
        camera_controls.addWidget(self.camera_status)

        self.camera_note = QLabel("权限提示、失败提示和降级路径都先保留在界面上。")
        self.camera_note.setWordWrap(True)
        self.camera_note.setObjectName("Hint")
        self.camera_note.setMaximumHeight(38)

        self.face_card = StatCard("面部检测", "0 个面部")
        self.face_card.setMaximumHeight(78)
        self.face_card.setMinimumHeight(68)
        self.face_card.value_label.setWordWrap(False)
        self.face_card.value_label.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)
        self.face_card.value_label.setTextFormat(Qt.PlainText)
        self.face_card.value_label.setStyleSheet("font-size: 13px; color: #355E96;")
        self.mode_card = StatCard("当前模式", "idle")
        self.mode_card.setMaximumHeight(78)
        self.mode_card.setMinimumHeight(68)
        self.mode_card.value_label.setWordWrap(False)
        self.mode_card.value_label.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)
        stress_row = QHBoxLayout()
        stress_row.setSpacing(8)
        stress_row.addWidget(self.stress_card)
        stress_row.addWidget(self.fatigue_card)
        self.heart_rate_card = StatCard("Heart Rate", "-- bpm")
        self.respiration_card = StatCard("Respiration", "-- rpm")
        self.hrv_card = StatCard("HRV", "-- ms")
        for card in (self.stress_card, self.fatigue_card, self.heart_rate_card, self.respiration_card, self.hrv_card):
            card.setMaximumHeight(90)
            card.setMinimumHeight(78)
        self.keyboard_card = StatCard("键盘活跃", "等待基线")
        self.mouse_card = StatCard("鼠标活跃", "等待基线")
        self.behavior_card = StatCard("行为序列", "键鼠行为监测初始化中")
        self.keyboard_card.setMaximumHeight(78)
        self.keyboard_card.setMinimumHeight(68)
        self.mouse_card.setMaximumHeight(78)
        self.mouse_card.setMinimumHeight(68)
        self.behavior_card.setMaximumHeight(66)
        self.behavior_card.setMinimumHeight(58)
        self.behavior_card.value_label.setStyleSheet("font-size: 13px; color: #355E96;")
        self.behavior_card.value_label.setWordWrap(False)
        self.behavior_card.value_label.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)
        overview_row = QHBoxLayout()
        overview_row.setContentsMargins(0, 0, 0, 0)
        overview_row.setSpacing(8)
        overview_row.addWidget(self.face_card)
        overview_row.addWidget(self.mode_card)
        metrics_row = QHBoxLayout()
        metrics_row.setContentsMargins(0, 0, 0, 0)
        metrics_row.setSpacing(8)
        metrics_row.addWidget(self.heart_rate_card)
        metrics_row.addWidget(self.respiration_card)
        metrics_row.addWidget(self.hrv_card)
        behavior_row = QHBoxLayout()
        behavior_row.setContentsMargins(0, 0, 0, 0)
        behavior_row.setSpacing(8)
        behavior_row.addWidget(self.keyboard_card)
        behavior_row.addWidget(self.mouse_card)

        layout.addWidget(title)
        layout.addLayout(camera_controls)
        layout.addWidget(self.camera_preview, 1)
        layout.addWidget(self.camera_note)
        layout.addLayout(overview_row)
        layout.addLayout(stress_row)
        layout.addLayout(metrics_row)
        layout.addLayout(behavior_row)
        layout.addWidget(self.behavior_card)
        return frame

    def _build_dashboard_page(self) -> QWidget:
        page = QWidget()
        page.setObjectName("Page")
        root_layout = QVBoxLayout(page)
        root_layout.setContentsMargins(24, 24, 24, 24)
        root_layout.setSpacing(18)

        filter_row = QHBoxLayout()
        filter_row.setSpacing(18)
        filter_row.addStretch(1)
        segment_frame = QFrame()
        segment_frame.setObjectName("DashboardSegment")
        segment_layout = QHBoxLayout(segment_frame)
        segment_layout.setContentsMargins(6, 6, 6, 6)
        segment_layout.setSpacing(4)
        self.dashboard_day_button = QPushButton("前一天")
        self.dashboard_week_button = QPushButton("前一周")
        self.dashboard_month_button = QPushButton("前一个月")
        self._dashboard_filter_buttons = {
            "day": self.dashboard_day_button,
            "week": self.dashboard_week_button,
            "month": self.dashboard_month_button,
        }
        for period, button in self._dashboard_filter_buttons.items():
            button.setObjectName("DashboardFilterButton")
            button.setCheckable(True)
            button.setCursor(Qt.PointingHandCursor)
            button.clicked.connect(lambda checked=False, value=period: self._set_dashboard_period(value))
            segment_layout.addWidget(button)
        filter_row.addWidget(segment_frame)

        custom_frame = QFrame()
        custom_frame.setObjectName("DashboardDateRange")
        custom_layout = QHBoxLayout(custom_frame)
        custom_layout.setContentsMargins(10, 6, 10, 6)
        custom_layout.setSpacing(10)
        custom_label = QLabel("自定义日期")
        custom_label.setObjectName("DashboardDateLabel")
        self.dashboard_start_date = QDateEdit(QDate.currentDate().addDays(-7))
        self.dashboard_end_date = QDateEdit(QDate.currentDate().addDays(-1))
        for widget in (self.dashboard_start_date, self.dashboard_end_date):
            widget.setCalendarPopup(True)
            widget.setDisplayFormat("yyyy-MM-dd")
            widget.setObjectName("DashboardDateEdit")
        self.dashboard_custom_apply_button = QPushButton("应用")
        self.dashboard_custom_apply_button.setObjectName("DashboardApplyButton")
        self.dashboard_custom_apply_button.setCursor(Qt.PointingHandCursor)
        self.dashboard_custom_apply_button.clicked.connect(self._apply_custom_dashboard_range)
        custom_layout.addWidget(custom_label)
        custom_layout.addWidget(self.dashboard_start_date)
        custom_layout.addWidget(self.dashboard_end_date)
        custom_layout.addWidget(self.dashboard_custom_apply_button)
        filter_row.addWidget(custom_frame)
        root_layout.addLayout(filter_row)

        chart_frame = self._panel_frame()
        chart_layout = QVBoxLayout(chart_frame)
        chart_layout.setContentsMargins(22, 22, 22, 22)
        chart_layout.setSpacing(12)
        if QWebEngineView is not None:
            self.dashboard_chart_view = QWebEngineView()
            self.dashboard_chart_view.setObjectName("ChartView")
            self.dashboard_chart_view.page().setBackgroundColor(QColor("#EAF7FF"))
            self.dashboard_chart_view.loadFinished.connect(self._on_dashboard_chart_shell_loaded)
            chart_layout.addWidget(self.dashboard_chart_view, 1)
            self.dashboard_chart_fallback = None
            self._load_dashboard_chart_shell()
        else:
            self.dashboard_chart_view = None
            self.dashboard_chart_fallback = QTextBrowser()
            self.dashboard_chart_fallback.setObjectName("OverviewPanel")
            chart_layout.addWidget(self.dashboard_chart_fallback, 1)
        root_layout.addWidget(chart_frame, 1)
        return page

    def _build_report_page(self) -> QWidget:
        page = QWidget()
        page.setObjectName("Page")
        root_layout = QVBoxLayout(page)
        root_layout.setContentsMargins(24, 24, 24, 24)
        root_layout.setSpacing(20)

        report_grid = QGridLayout()
        report_grid.setHorizontalSpacing(24)
        report_grid.setVerticalSpacing(24)

        daily_frame = self._panel_frame()
        daily_layout = QVBoxLayout(daily_frame)
        daily_layout.setContentsMargins(20, 20, 20, 20)
        daily_layout.setSpacing(14)
        daily_header = QHBoxLayout()
        daily_header.setSpacing(12)
        daily_title = QLabel("每日健康分析报告")
        daily_title.setObjectName("SectionTitle")
        daily_header.addWidget(daily_title)
        daily_header.addStretch(1)
        self.export_daily_report_button = QPushButton("导出日报 MD")
        self.export_daily_report_button.setObjectName("GhostButton")
        self.export_daily_report_button.setCursor(Qt.PointingHandCursor)
        self.export_daily_report_button.clicked.connect(lambda: self._export_report_markdown("daily"))
        daily_header.addWidget(self.export_daily_report_button)
        self.export_daily_report_image_button = QPushButton("导出日报图片")
        self.export_daily_report_image_button.setObjectName("GhostButton")
        self.export_daily_report_image_button.setCursor(Qt.PointingHandCursor)
        self.export_daily_report_image_button.clicked.connect(lambda: self._export_report_image("daily"))
        daily_header.addWidget(self.export_daily_report_image_button)
        daily_stats_grid = QGridLayout()
        daily_stats_grid.setHorizontalSpacing(12)
        daily_stats_grid.setVerticalSpacing(12)
        self.daily_avg_stress_card = StatCard("当日平均压力指数", "0")
        self.daily_rest_count_card = StatCard("休息活动次数", "0")
        self.daily_focus_index_card = StatCard("专注指数", "0")
        daily_stats_grid.addWidget(self.daily_avg_stress_card, 0, 0)
        daily_stats_grid.addWidget(self.daily_rest_count_card, 0, 1)
        daily_stats_grid.addWidget(self.daily_focus_index_card, 0, 2)
        self.daily_report_view = QTextBrowser()
        self.daily_report_view.setObjectName("ReportBrowser")
        self.daily_report_view.setOpenExternalLinks(False)
        self.daily_report_view.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOn)
        self.daily_report_view.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.daily_report_view.document().setDocumentMargin(0)
        daily_layout.addLayout(daily_header)
        daily_layout.addLayout(daily_stats_grid)
        daily_layout.addWidget(self.daily_report_view, 1)

        weekly_frame = self._panel_frame()
        weekly_layout = QVBoxLayout(weekly_frame)
        weekly_layout.setContentsMargins(20, 20, 20, 20)
        weekly_layout.setSpacing(14)
        weekly_header = QHBoxLayout()
        weekly_header.setSpacing(12)
        self.weekly_report_title = QLabel("每周健康分析报告")
        self.weekly_report_title.setObjectName("SectionTitle")
        weekly_header.addWidget(self.weekly_report_title)
        weekly_header.addStretch(1)
        self.export_weekly_report_button = QPushButton("导出当前 MD")
        self.export_weekly_report_button.setObjectName("GhostButton")
        self.export_weekly_report_button.setCursor(Qt.PointingHandCursor)
        self.export_weekly_report_button.clicked.connect(lambda: self._export_report_markdown("period"))
        weekly_header.addWidget(self.export_weekly_report_button)
        self.export_weekly_report_image_button = QPushButton("导出当前图片")
        self.export_weekly_report_image_button.setObjectName("GhostButton")
        self.export_weekly_report_image_button.setCursor(Qt.PointingHandCursor)
        self.export_weekly_report_image_button.clicked.connect(lambda: self._export_report_image("period"))
        weekly_header.addWidget(self.export_weekly_report_image_button)

        custom_row = QHBoxLayout()
        custom_row.setSpacing(10)
        custom_hint = QLabel("自定义时间分析")
        custom_hint.setObjectName("DashboardDateLabel")
        self.report_start_date = QDateEdit(QDate.currentDate().addDays(-7))
        self.report_end_date = QDateEdit(QDate.currentDate().addDays(-1))
        for widget in (self.report_start_date, self.report_end_date):
            widget.setCalendarPopup(True)
            widget.setDisplayFormat("yyyy-MM-dd")
            widget.setObjectName("DashboardDateEdit")
        self.report_custom_apply_button = QPushButton("生成区间报告")
        self.report_custom_apply_button.setObjectName("DashboardApplyButton")
        self.report_custom_apply_button.setCursor(Qt.PointingHandCursor)
        self.report_custom_apply_button.clicked.connect(self._apply_custom_report_range)
        self.report_reset_button = QPushButton("恢复每周")
        self.report_reset_button.setObjectName("GhostButton")
        self.report_reset_button.setCursor(Qt.PointingHandCursor)
        self.report_reset_button.clicked.connect(self._reset_weekly_report_range)
        custom_row.addWidget(custom_hint)
        custom_row.addWidget(self.report_start_date)
        custom_row.addWidget(self.report_end_date)
        custom_row.addWidget(self.report_custom_apply_button)
        custom_row.addWidget(self.report_reset_button)
        custom_row.addStretch(1)

        self.weekly_report_view = QTextBrowser()
        self.weekly_report_view.setObjectName("ReportBrowser")
        self.weekly_report_view.setOpenExternalLinks(False)
        self.weekly_report_view.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOn)
        self.weekly_report_view.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.weekly_report_view.document().setDocumentMargin(0)
        weekly_layout.addLayout(weekly_header)
        weekly_layout.addLayout(custom_row)
        weekly_layout.addWidget(self.weekly_report_view, 1)

        report_grid.addWidget(daily_frame, 0, 0)
        report_grid.addWidget(weekly_frame, 0, 1)
        report_grid.setColumnStretch(0, 1)
        report_grid.setColumnStretch(1, 1)
        root_layout.addLayout(report_grid, 1)
        return page

    def _switch_page(self, page: str, *, show_status: bool = True) -> None:
        mapping = {"home": 0, "dashboard": 1, "report": 2}
        index = mapping.get(page, 0)
        self.page_stack.setCurrentIndex(index)

        self.home_nav_button.setChecked(page == "home")
        self.dashboard_nav_button.setChecked(page == "dashboard")
        self.report_nav_button.setChecked(page == "report")

        if page == "dashboard":
            self._refresh_dashboard_page()
        elif page == "report":
            self._refresh_report_page()

        self._update_companion_controls()

        if show_status:
            page_name = {"home": "主页面", "dashboard": "可视化分析大屏", "report": "健康报告页面"}.get(page, "主页面")
            self.statusBar().showMessage(f"已切换到{page_name}", 2500)

    def _focus_score(self) -> int:
        score = 100
        score -= int(self._stress_score * 0.4)
        score -= int(self._fatigue_score * 0.45)
        if self._behavior_summary.get("keyboard_declined"):
            score -= 10
        if self._behavior_summary.get("mouse_declined"):
            score -= 10
        behavior_state = str(self._behavior_summary.get("behavior_state", "warming"))
        if behavior_state == "anxious":
            score -= 8
        elif behavior_state == "fatigued":
            score -= 14
        if not self._local_camera_enabled:
            score -= 10
        if self._face_count == 0 and self._local_camera_enabled:
            score -= 15
        return max(0, min(100, score))

    def _emotion_tendency(self) -> str:
        behavior_state = str(self._behavior_summary.get("behavior_state", "warming"))
        if self._fatigue_score >= 80 or behavior_state == "fatigued":
            return "疲惫"
        if self._stress_score >= 80 or behavior_state == "anxious":
            return "焦虑"
        if behavior_state == "slowed":
            return "专注波动"
        if self._stress_score >= 55 or self._fatigue_score >= 55:
            return "专注波动"
        if self._face_count > 0 and self._local_camera_enabled:
            return "专注"
        return "平稳"

    def _record_monitoring_sample(
        self,
        *,
        face_count: int,
        calibration_state: str,
        stress_score: float,
        fatigue_score: float,
        heart_rate: Optional[float],
        respiration_rate: Optional[float],
        hrv: Optional[float],
    ) -> None:
        now = time.monotonic()
        cutoff = now - _MONITORING_WINDOW_SECONDS
        while self._monitoring_samples and self._monitoring_samples[0].captured_at < cutoff:
            self._monitoring_samples.popleft()

        if face_count <= 0 or calibration_state not in {"calibrating", "ready"}:
            return

        self._monitoring_samples.append(
            MonitoringMetricSample(
                captured_at=now,
                stress_score=stress_score,
                fatigue_score=fatigue_score,
                heart_rate=heart_rate,
                respiration_rate=respiration_rate,
                hrv=hrv,
            )
        )

    def _monitoring_averages(self) -> dict[str, Optional[float]]:
        samples = list(self._monitoring_samples)
        if not samples:
            return {
                "sample_count": 0.0,
                "window_seconds": 0.0,
                "stress_score": None,
                "fatigue_score": None,
                "heart_rate": None,
                "respiration_rate": None,
                "hrv": None,
            }

        def average(values: list[Optional[float]]) -> Optional[float]:
            valid = [value for value in values if value is not None]
            if not valid:
                return None
            return sum(valid) / len(valid)

        window_seconds = samples[-1].captured_at - samples[0].captured_at if len(samples) > 1 else 0.0
        return {
            "sample_count": float(len(samples)),
            "window_seconds": window_seconds,
            "stress_score": average([sample.stress_score for sample in samples]),
            "fatigue_score": average([sample.fatigue_score for sample in samples]),
            "heart_rate": average([sample.heart_rate for sample in samples]),
            "respiration_rate": average([sample.respiration_rate for sample in samples]),
            "hrv": average([sample.hrv for sample in samples]),
        }

    def _resolve_monitoring_mood(
        self,
        *,
        face_count: int,
        calibration_state: str,
        averages: dict[str, Optional[float]],
    ) -> tuple[PetMood, str]:
        if not self._local_camera_enabled:
            return PetMood.offline, "摄像头关闭，当前未进行状态分析。"
        if calibration_state == "unavailable":
            return PetMood.offline, "分析链路不可用，当前处于离线监测状态。"
        if face_count <= 0:
            return PetMood.idle, "未检测到稳定人脸，当前保持空闲观察。"

        window_seconds = float(averages.get("window_seconds") or 0.0)
        if calibration_state == "calibrating" or window_seconds < _MONITORING_MIN_SECONDS:
            return PetMood.listening, f"正在积累近 {_MONITORING_WINDOW_SECONDS:.0f} 秒状态均值，继续观察中。"

        stress_avg = averages.get("stress_score")
        fatigue_avg = averages.get("fatigue_score")
        heart_rate_avg = averages.get("heart_rate")
        respiration_avg = averages.get("respiration_rate")
        hrv_avg = averages.get("hrv")

        if stress_avg is None or fatigue_avg is None:
            return PetMood.listening, "监测数据仍在积累，等待均值稳定。"

        def band_level(
            value: Optional[float],
            *,
            low_warn: Optional[float] = None,
            low_alert: Optional[float] = None,
            high_warn: Optional[float] = None,
            high_alert: Optional[float] = None,
        ) -> int:
            if value is None:
                return 0
            if (low_alert is not None and value <= low_alert) or (high_alert is not None and value >= high_alert):
                return 2
            if (low_warn is not None and value <= low_warn) or (high_warn is not None and value >= high_warn):
                return 1
            return 0

        stress_level = band_level(stress_avg, high_warn=58.0, high_alert=75.0)
        fatigue_level = band_level(fatigue_avg, high_warn=55.0, high_alert=72.0)
        heart_rate_level = band_level(heart_rate_avg, low_warn=56.0, low_alert=50.0, high_warn=92.0, high_alert=105.0)
        respiration_level = band_level(respiration_avg, low_warn=11.0, low_alert=9.0, high_warn=19.0, high_alert=24.0)
        hrv_level = band_level(hrv_avg, low_warn=38.0, low_alert=24.0)

        levels = [stress_level, fatigue_level, heart_rate_level, respiration_level, hrv_level]
        alert_count = sum(1 for level in levels if level == 2)
        elevated_count = sum(1 for level in levels if level >= 1)
        risk_points = sum(levels)

        avg_parts = [
            f"压力 {stress_avg:.0f}",
            f"疲劳 {fatigue_avg:.0f}",
        ]
        if heart_rate_avg is not None:
            avg_parts.append(f"HR {heart_rate_avg:.0f} bpm")
        if respiration_avg is not None:
            avg_parts.append(f"Resp {respiration_avg:.0f} rpm")
        if hrv_avg is not None:
            avg_parts.append(f"HRV {hrv_avg:.0f} ms")
        avg_summary = "，".join(avg_parts)

        if alert_count >= 2 or risk_points >= 5 or stress_level == 2 or fatigue_level == 2:
            return PetMood.alert, f"近 {_MONITORING_WINDOW_SECONDS:.0f} 秒均值偏高：{avg_summary}，建议暂停并休息。"

        if elevated_count >= 2 or risk_points >= 2:
            return PetMood.thinking, f"近 {_MONITORING_WINDOW_SECONDS:.0f} 秒均值有波动：{avg_summary}，建议降低任务强度。"

        comfortable = (
            stress_avg <= 35.0
            and fatigue_avg <= 35.0
            and (heart_rate_avg is None or 60.0 <= heart_rate_avg <= 90.0)
            and (respiration_avg is None or 12.0 <= respiration_avg <= 18.0)
            and (hrv_avg is None or hrv_avg >= 45.0)
        )
        if comfortable and heart_rate_avg is not None and respiration_avg is not None and hrv_avg is not None:
            return PetMood.responding, f"近 {_MONITORING_WINDOW_SECONDS:.0f} 秒均值平稳：{avg_summary}，当前处于积极响应状态。"

        return PetMood.idle, f"近 {_MONITORING_WINDOW_SECONDS:.0f} 秒均值总体平稳：{avg_summary}，当前保持空闲观察。"

    def _build_realtime_snapshot(self):
        if RealtimeSnapshot is None:
            return None
        event_text = "等待开始"
        if hasattr(self, "event_card"):
            event_text = self.event_card.value_label.text()
        behavior_hint = self._behavior_hint_fragment()
        if behavior_hint and behavior_hint not in event_text:
            event_text = f"{event_text} | {behavior_hint}"
        return RealtimeSnapshot(
            recorded_at=QDateTime.currentDateTime().toString("yyyy-MM-ddTHH:mm:ss"),
            mood=self.mode_card.value_label.text() if hasattr(self, "mode_card") else "idle",
            emotion=self._emotion_tendency(),
            stress_score=self._stress_score,
            fatigue_score=self._fatigue_score,
            focus_score=self._focus_score(),
            face_count=self._face_count,
            dominant_signal=self._dominant_signal,
            event_text=event_text,
            camera_enabled=self._local_camera_enabled,
            key_rate_per_min=float(self._behavior_summary.get("key_rate_per_min", 0.0) or 0.0),
            keyboard_active_seconds=int(self._behavior_summary.get("keyboard_active_seconds", 0) or 0),
            keyboard_activity=float(self._behavior_summary.get("keyboard_activity", 0.0) or 0.0),
            keyboard_declined=bool(self._behavior_summary.get("keyboard_declined", False)),
            mouse_distance=float(self._behavior_summary.get("mouse_distance", 0.0) or 0.0),
            mouse_active_seconds=int(self._behavior_summary.get("mouse_active_seconds", 0) or 0),
            mouse_activity=float(self._behavior_summary.get("mouse_activity", 0.0) or 0.0),
            mouse_declined=bool(self._behavior_summary.get("mouse_declined", False)),
            modality_switches=int(self._behavior_summary.get("modality_switches", 0) or 0),
            behavior_state=str(self._behavior_summary.get("behavior_state", "warming")),
        )

    def _sync_dashboard_repository(self) -> None:
        if self._dashboard_repository is None:
            return
        snapshot = self._build_realtime_snapshot()
        if snapshot is not None:
            self._dashboard_repository.record_runtime_snapshot(snapshot)

    def _schedule_analytics_refresh(self) -> None:
        if (
            not hasattr(self, "page_stack")
            or not self.isVisible()
            or self.isMinimized()
        ):
            return
        current_page = self.page_stack.currentWidget()
        if current_page is self.dashboard_page:
            if not self._dashboard_refresh_timer.isActive():
                self._dashboard_refresh_timer.start()
        elif current_page is self.report_page:
            if not self._report_refresh_timer.isActive():
                self._report_refresh_timer.start()

    def _set_dashboard_period(self, period: str) -> None:
        if period == self._dashboard_period:
            return
        self._dashboard_period = period
        self._dashboard_payload_key = ""
        self._refresh_dashboard_page()

    def _apply_custom_dashboard_range(self) -> None:
        start_date = self.dashboard_start_date.date().toPython()
        end_date = self.dashboard_end_date.date().toPython()
        if start_date > end_date:
            self.statusBar().showMessage("开始日期不能晚于结束日期", 3200)
            return
        self._dashboard_custom_range = (start_date, end_date)
        self._dashboard_period = "custom"
        self._dashboard_payload_key = ""
        self._refresh_dashboard_page()

    def _load_dashboard_chart_shell(self) -> None:
        if self.dashboard_chart_view is None or self._dashboard_chart_shell_loaded:
            return
        assets_dir = Path(__file__).resolve().parents[1] / "assets"
        self._dashboard_chart_shell_loaded = True
        self._dashboard_chart_page_ready = False
        self.dashboard_chart_view.setHtml(
            self._build_dashboard_chart_html(),
            QUrl.fromLocalFile(f"{assets_dir.as_posix()}/"),
        )

    def _on_dashboard_chart_shell_loaded(self, ok: bool) -> None:
        self._dashboard_chart_page_ready = ok
        if not ok:
            self.statusBar().showMessage("分析大屏加载失败，已回退到当前内容。", 3200)
            if self.dashboard_chart_view is not None:
                self.dashboard_chart_view.setHtml(
                    self._build_dashboard_fallback_html(self._pending_dashboard_payload or {}),
                )
            return
        if self._pending_dashboard_payload is not None:
            payload = self._pending_dashboard_payload
            self._pending_dashboard_payload = None
            self._push_dashboard_payload_to_view(payload)

    def _push_dashboard_payload_to_view(self, payload: dict) -> None:
        if self.dashboard_chart_view is None:
            return
        script = f"window.updateDashboard({json.dumps(payload, ensure_ascii=False)});"
        self.dashboard_chart_view.page().runJavaScript(script)

    def _build_dashboard_chart_html(self) -> str:
        return """
<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta http-equiv="X-UA-Compatible" content="IE=edge">
    <style>
        html, body {
            margin: 0;
            width: 100%;
            height: 100%;
            background: transparent;
            overflow: hidden;
            font-family: "Segoe UI", "Microsoft YaHei", sans-serif;
        }
        body {
            background:
                radial-gradient(circle at 16% 10%, rgba(59, 130, 246, 0.22), transparent 28%),
                radial-gradient(circle at 82% 12%, rgba(34, 211, 238, 0.12), transparent 24%),
                radial-gradient(circle at 50% 100%, rgba(22, 163, 74, 0.08), transparent 28%),
                linear-gradient(160deg, rgba(4, 11, 21, 0.98), rgba(7, 17, 31, 0.98) 42%, rgba(11, 23, 40, 0.96));
        }
        .board {
            width: 100%;
            height: 100%;
            padding: 22px;
            box-sizing: border-box;
            display: grid;
            grid-template-columns: 1.5fr 1.1fr 1.1fr;
            grid-template-rows: 1.2fr 1fr 96px;
            row-gap: 24px;
            column-gap: 20px;
        }
        .metric-card, .chart-card {
            background: linear-gradient(145deg, rgba(9, 20, 38, 0.92), rgba(15, 35, 61, 0.84));
            border: 1px solid rgba(125, 211, 252, 0.14);
            border-radius: 22px;
            box-shadow:
                inset 0 1px 0 rgba(255, 255, 255, 0.04),
                inset 0 0 0 1px rgba(125, 211, 252, 0.04),
                0 24px 50px rgba(2, 6, 23, 0.46);
        }
        .metric-row {
            grid-column: 1 / span 3;
            grid-row: 3;
            display: grid;
            grid-template-columns: repeat(6, minmax(0, 1fr));
            gap: 18px;
        }
        .metric-card {
            padding: 15px 18px;
            display: flex;
            flex-direction: column;
            justify-content: center;
        }
        .metric-label {
            color: #8ba8c6;
            font-size: 12px;
            margin-bottom: 9px;
            letter-spacing: 0.4px;
        }
        .metric-value {
            color: #f8fdff;
            font-size: 30px;
            font-weight: 700;
            letter-spacing: 0.4px;
            text-shadow: 0 0 18px rgba(125, 211, 252, 0.16);
        }
        .metric-sub {
            color: #6dd3ff;
            font-size: 12px;
            margin-top: 6px;
        }
        .chart-card {
            position: relative;
            padding: 18px;
        }
        .chart-title {
            position: absolute;
            left: 22px;
            top: 18px;
            color: #f8fdff;
            font-size: 15px;
            font-weight: 700;
            letter-spacing: 0.3px;
            z-index: 2;
        }
        .chart {
            width: 100%;
            height: 100%;
        }
        .chart-status {
            position: absolute;
            inset: 16px;
            display: flex;
            align-items: center;
            justify-content: center;
            border-radius: 16px;
            background: rgba(4, 12, 24, 0.84);
            color: #d8ebfc;
            font-size: 14px;
            letter-spacing: 0.3px;
            z-index: 5;
            transition: opacity 0.18s ease;
        }
        .chart-status.hidden {
            opacity: 0;
            pointer-events: none;
        }
        .chart-status.error {
            color: #fecaca;
            border: 1px solid rgba(248, 113, 113, 0.30);
        }
        .trend {
            grid-column: 1 / span 2;
            grid-row: 1;
        }
        .donut {
            grid-column: 3;
            grid-row: 1;
        }
        .bar {
            grid-column: 1;
            grid-row: 2;
        }
        .radar {
            grid-column: 2;
            grid-row: 2;
        }
        .mini-line {
            grid-column: 3;
            grid-row: 2;
        }
    </style>
</head>
<body>
    <div class="board">
        <div id="dashboardStatus" class="chart-status">正在准备历史分析大屏...</div>
        <div class="chart-card trend">
            <div class="chart-title">历史趋势折线</div>
            <div id="trendChart" class="chart"></div>
        </div>
        <div class="chart-card donut">
            <div class="chart-title">情绪结构扇形图</div>
            <div id="pieChart" class="chart"></div>
        </div>
        <div class="chart-card bar">
            <div class="chart-title">主信号柱状图</div>
            <div id="barChart" class="chart"></div>
        </div>
        <div class="chart-card radar">
            <div class="chart-title">综合画像雷达图</div>
            <div id="radarChart" class="chart"></div>
        </div>
        <div class="chart-card mini-line">
            <div class="chart-title">近期波动面积图</div>
            <div id="miniLineChart" class="chart"></div>
        </div>
        <div class="metric-row">
            <div class="metric-card">
                <div class="metric-label">历史平均压力</div>
                <div id="avgStressValue" class="metric-value">0</div>
                <div id="avgStressSub" class="metric-sub">等待载入</div>
            </div>
            <div class="metric-card">
                <div class="metric-label">历史平均疲劳</div>
                <div id="avgFatigueValue" class="metric-value">0</div>
                <div id="avgFatigueSub" class="metric-sub">等待载入</div>
            </div>
            <div class="metric-card">
                <div class="metric-label">历史平均专注</div>
                <div id="avgFocusValue" class="metric-value">0</div>
                <div id="avgFocusSub" class="metric-sub">等待载入</div>
            </div>
            <div class="metric-card">
                <div class="metric-label">键盘活跃均值</div>
                <div id="avgKeyboardValue" class="metric-value">0</div>
                <div id="avgKeyboardSub" class="metric-sub">等待载入</div>
            </div>
            <div class="metric-card">
                <div class="metric-label">鼠标活跃均值</div>
                <div id="avgMouseValue" class="metric-value">0</div>
                <div id="avgMouseSub" class="metric-sub">等待载入</div>
            </div>
            <div class="metric-card">
                <div class="metric-label">历史样本规模</div>
                <div id="sampleCountValue" class="metric-value">0</div>
                <div id="sampleCountSub" class="metric-sub">等待载入</div>
            </div>
        </div>
    </div>
    <script>
        const palette = ['#67e8f9', '#38bdf8', '#22c55e', '#a78bfa', '#fb7185', '#14b8a6'];
        const textColor = '#d7e8f9';
        const axisColor = 'rgba(148, 163, 184, 0.18)';
        const tooltipStyle = {
            backgroundColor: 'rgba(4, 12, 24, 0.94)',
            borderColor: 'rgba(125, 211, 252, 0.22)',
            borderWidth: 1,
            textStyle: { color: '#eff9ff' },
            extraCssText: 'box-shadow:0 18px 36px rgba(2,6,23,0.42); border-radius:14px;'
        };
        const commonGrid = { left: 42, right: 22, top: 52, bottom: 28 };
        window.dashboardCharts = {};
        window.dashboardReady = false;
        window.pendingDashboardPayload = null;

        function setStatus(message, isError = false) {
            const node = document.getElementById('dashboardStatus');
            node.textContent = message || '';
            node.classList.toggle('hidden', !message);
            node.classList.toggle('error', !!isError);
        }

        function loadScript(src) {
            return new Promise((resolve, reject) => {
                const script = document.createElement('script');
                script.src = src;
                script.onload = resolve;
                script.onerror = reject;
                document.head.appendChild(script);
            });
        }

        async function ensureECharts() {
            if (window.echarts) {
                return;
            }
            const sources = [
                'vendor/echarts.min.js',
                'https://cdn.jsdelivr.net/npm/echarts@5/dist/echarts.min.js',
                'https://cdn.staticfile.net/echarts/5.5.0/echarts.min.js',
            ];
            for (const source of sources) {
                try {
                    await loadScript(source);
                    if (window.echarts) {
                        return;
                    }
                } catch (error) {
                    console.warn('load echarts failed:', source);
                }
            }
            throw new Error('图表资源加载失败，请检查网络或将 echarts.min.js 放到 frontend/assets/vendor 目录。');
        }

        function variance(series) {
            if (!series || !series.length) {
                return 0;
            }
            const average = series.reduce((sum, item) => sum + item, 0) / series.length;
            return series.reduce((sum, item) => sum + Math.pow(item - average, 2), 0) / series.length;
        }

        function initCharts() {
            if (window.dashboardCharts.trendChart) {
                return;
            }
            window.dashboardCharts.trendChart = echarts.init(document.getElementById('trendChart'), null, { renderer: 'canvas' });
            window.dashboardCharts.pieChart = echarts.init(document.getElementById('pieChart'), null, { renderer: 'canvas' });
            window.dashboardCharts.barChart = echarts.init(document.getElementById('barChart'), null, { renderer: 'canvas' });
            window.dashboardCharts.radarChart = echarts.init(document.getElementById('radarChart'), null, { renderer: 'canvas' });
            window.dashboardCharts.miniLineChart = echarts.init(document.getElementById('miniLineChart'), null, { renderer: 'canvas' });
        }

        function updateMetric(id, value, subtext) {
            document.getElementById(id + 'Value').textContent = value;
            document.getElementById(id + 'Sub').textContent = subtext;
        }

        window.updateDashboard = function (payload) {
            window.pendingDashboardPayload = payload;
            if (!window.dashboardReady || !window.echarts) {
                return;
            }

            const categories = payload.line_categories || [];
            const lineSeries = payload.line_series || {};
            const stressSeries = lineSeries['压力'] || [];
            const fatigueSeries = lineSeries['疲劳'] || [];
            const focusSeries = lineSeries['专注'] || [];
            const emotionDistribution = payload.emotion_distribution || [];
            const signalDistribution = payload.signal_distribution || [];
            const averages = payload.averages || {};
            const recentCategories = categories.slice(-6);
            const recentStress = stressSeries.slice(-6);
            const recentFatigue = fatigueSeries.slice(-6);
            const recentFocus = focusSeries.slice(-6);
            const stabilityScore = Math.max(0, 100 - Math.round(variance(recentFocus) * 0.7));
            const rhythmScore = Math.max(0, 100 - Math.round((variance(recentStress) + variance(recentFatigue)) * 0.25));
            const recoveryScore = Math.max(0, 100 - Math.round(((averages.avg_fatigue || 0) + (averages.avg_stress || 0)) * 0.45));
            const periodLabel = payload.period_label || '历史区间';
            const rangeStart = (payload.range_start || '').replace('T', ' ');
            const rangeEnd = (payload.range_end || '').replace('T', ' ');

            updateMetric('avgStress', averages.avg_stress || 0, periodLabel + '平均压力水平');
            updateMetric('avgFatigue', averages.avg_fatigue || 0, periodLabel + '疲劳波动概览');
            updateMetric('avgFocus', averages.avg_focus || 0, periodLabel + '专注表现均值');
            updateMetric('avgKeyboard', (averages.avg_keyboard_activity || 0).toFixed(2), periodLabel + '键盘活跃均值');
            updateMetric('avgMouse', (averages.avg_mouse_activity || 0).toFixed(2), periodLabel + '鼠标活跃均值');
            updateMetric('sampleCount', payload.sample_count || 0, rangeStart + ' - ' + rangeEnd);

            window.dashboardCharts.trendChart.setOption({
                animation: false,
                color: palette,
                tooltip: { ...tooltipStyle, trigger: 'axis' },
                legend: {
                    top: 12,
                    right: 10,
                    textStyle: { color: textColor },
                },
                grid: commonGrid,
                xAxis: {
                    type: 'category',
                    boundaryGap: false,
                    data: categories,
                    axisLine: { lineStyle: { color: axisColor } },
                    axisLabel: { color: textColor },
                },
                yAxis: {
                    type: 'value',
                    max: 100,
                    splitLine: { lineStyle: { color: axisColor } },
                    axisLabel: { color: textColor },
                },
                series: [
                    {
                        name: '压力',
                        type: 'line',
                        smooth: true,
                        symbol: 'none',
                        lineStyle: { width: 3 },
                        areaStyle: { opacity: 0.14 },
                        data: stressSeries,
                    },
                    {
                        name: '疲劳',
                        type: 'line',
                        smooth: true,
                        symbol: 'none',
                        lineStyle: { width: 3 },
                        areaStyle: { opacity: 0.12 },
                        data: fatigueSeries,
                    },
                    {
                        name: '专注',
                        type: 'line',
                        smooth: true,
                        symbol: 'none',
                        lineStyle: { width: 3 },
                        areaStyle: { opacity: 0.08 },
                        data: focusSeries,
                    },
                ],
            }, true);

            window.dashboardCharts.pieChart.setOption({
                animation: false,
                tooltip: { ...tooltipStyle, trigger: 'item' },
                legend: {
                    bottom: 6,
                    textStyle: { color: textColor },
                },
                color: palette,
                series: [{
                    type: 'pie',
                    radius: ['46%', '72%'],
                    center: ['50%', '48%'],
                    label: {
                        color: textColor,
                        formatter: '{b}\\n{d}%',
                    },
                    itemStyle: {
                        borderColor: '#07111f',
                        borderWidth: 4,
                        borderRadius: 8,
                    },
                    data: emotionDistribution,
                }],
            }, true);

            window.dashboardCharts.barChart.setOption({
                animation: false,
                color: ['#22d3ee'],
                tooltip: { ...tooltipStyle, trigger: 'axis', axisPointer: { type: 'shadow' } },
                grid: commonGrid,
                xAxis: {
                    type: 'category',
                    data: signalDistribution.map(item => item.name),
                    axisLine: { lineStyle: { color: axisColor } },
                    axisLabel: { color: textColor },
                },
                yAxis: {
                    type: 'value',
                    splitLine: { lineStyle: { color: axisColor } },
                    axisLabel: { color: textColor },
                },
                series: [{
                    type: 'bar',
                    barWidth: '42%',
                    data: signalDistribution.map(item => item.value),
                    itemStyle: {
                        borderRadius: [10, 10, 0, 0],
                        shadowBlur: 16,
                        shadowColor: 'rgba(34, 211, 238, 0.22)',
                    },
                }],
            }, true);

            window.dashboardCharts.radarChart.setOption({
                animation: false,
                color: ['#38bdf8'],
                radar: {
                    center: ['50%', '56%'],
                    radius: '62%',
                    splitNumber: 5,
                    indicator: [
                        { name: '稳定度', max: 100 },
                        { name: '恢复力', max: 100 },
                        { name: '专注力', max: 100 },
                        { name: '节奏感', max: 100 },
                        { name: '低压力', max: 100 },
                    ],
                    axisName: { color: textColor },
                    splitLine: { lineStyle: { color: axisColor } },
                    splitArea: { areaStyle: { color: ['rgba(15,23,42,0.4)', 'rgba(15,23,42,0.2)'] } },
                    axisLine: { lineStyle: { color: axisColor } },
                },
                series: [{
                    type: 'radar',
                    areaStyle: { color: 'rgba(56, 189, 248, 0.18)' },
                    lineStyle: { width: 2 },
                    data: [{
                        value: [stabilityScore, recoveryScore, Math.round(averages.avg_focus || 0), rhythmScore, Math.max(0, 100 - Math.round(averages.avg_stress || 0))],
                    }],
                }],
            }, true);

            window.dashboardCharts.miniLineChart.setOption({
                animation: false,
                color: ['#f59e0b', '#38bdf8', '#22c55e'],
                tooltip: { ...tooltipStyle, trigger: 'axis' },
                legend: {
                    top: 12,
                    textStyle: { color: textColor, fontSize: 11 },
                },
                grid: commonGrid,
                xAxis: {
                    type: 'category',
                    boundaryGap: false,
                    data: recentCategories,
                    axisLine: { lineStyle: { color: axisColor } },
                    axisLabel: { color: textColor },
                },
                yAxis: {
                    type: 'value',
                    max: 100,
                    splitLine: { lineStyle: { color: axisColor } },
                    axisLabel: { color: textColor },
                },
                series: [
                    {
                        name: '压力',
                        type: 'line',
                        smooth: true,
                        symbol: 'none',
                        areaStyle: { opacity: 0.16 },
                        data: recentStress,
                    },
                    {
                        name: '疲劳',
                        type: 'line',
                        smooth: true,
                        symbol: 'none',
                        areaStyle: { opacity: 0.12 },
                        data: recentFatigue,
                    },
                    {
                        name: '专注',
                        type: 'line',
                        smooth: true,
                        symbol: 'none',
                        lineStyle: { width: 2 },
                        data: recentFocus,
                    },
                ],
            }, true);

            setStatus('');
        };

        async function bootDashboard() {
            setStatus('正在加载图表资源...');
            try {
                await ensureECharts();
                initCharts();
                window.dashboardReady = true;
                if (window.pendingDashboardPayload) {
                    const latest = window.pendingDashboardPayload;
                    window.pendingDashboardPayload = null;
                    window.updateDashboard(latest);
                } else {
                    setStatus('正在等待历史数据...');
                }
            } catch (error) {
                setStatus(error.message || '图表加载失败');
            }
        }

        window.addEventListener('resize', () => {
            Object.values(window.dashboardCharts).forEach(chart => {
                if (chart) {
                    chart.resize();
                }
            });
        });

        bootDashboard();
    </script>
</body>
</html>
        """

    def _build_dashboard_fallback_html(self, payload: dict) -> str:
        categories = [str(item) for item in payload.get("line_categories", [])]
        line_series = payload.get("line_series", {})
        stress_series = [float(item) for item in line_series.get("压力", [])]
        fatigue_series = [float(item) for item in line_series.get("疲劳", [])]
        focus_series = [float(item) for item in line_series.get("专注", [])]
        emotion_distribution = payload.get("emotion_distribution", [])
        signal_distribution = payload.get("signal_distribution", [])
        averages = payload.get("averages", {})
        range_start = str(payload.get("range_start", "")).replace("T", " ")
        range_end = str(payload.get("range_end", "")).replace("T", " ")
        period_label = escape(str(payload.get("period_label", "历史区间")))

        recent_stress = stress_series[-6:]
        recent_fatigue = fatigue_series[-6:]
        recent_focus = focus_series[-6:]
        sample_count = int(payload.get("sample_count", 0) or 0)
        avg_stress = int(round(float(averages.get("avg_stress", 0) or 0)))
        avg_fatigue = int(round(float(averages.get("avg_fatigue", 0) or 0)))
        avg_focus = int(round(float(averages.get("avg_focus", 0) or 0)))
        avg_keyboard = round(float(averages.get("avg_keyboard_activity", 0) or 0), 2)
        avg_mouse = round(float(averages.get("avg_mouse_activity", 0) or 0), 2)

        def variance(series: list[float]) -> float:
            if not series:
                return 0.0
            average = sum(series) / len(series)
            return sum((item - average) ** 2 for item in series) / len(series)

        stability_score = max(0, min(100, round(100 - variance(recent_focus) * 0.7)))
        rhythm_score = max(0, min(100, round(100 - (variance(recent_stress) + variance(recent_fatigue)) * 0.25)))
        recovery_score = max(0, min(100, round(100 - (avg_fatigue + avg_stress) * 0.45)))

        def make_polyline(
            series: list[float],
            *,
            width: int = 520,
            height: int = 190,
            max_value: float = 100.0,
        ) -> str:
            if not series:
                return ""
            if len(series) == 1:
                y = height - (series[0] / max_value) * height
                return f"0,{y:.1f} {width},{y:.1f}"
            step_x = width / max(1, len(series) - 1)
            points = []
            for index, value in enumerate(series):
                x = step_x * index
                y = height - (max(0.0, min(max_value, value)) / max_value) * height
                points.append(f"{x:.1f},{y:.1f}")
            return " ".join(points)

        def make_grid_lines(width: int = 520, height: int = 190) -> str:
            lines: list[str] = []
            for ratio in (0.2, 0.4, 0.6, 0.8):
                y = height * ratio
                lines.append(
                    f"<line x1='0' y1='{y:.1f}' x2='{width}' y2='{y:.1f}' "
                    "stroke='rgba(148,163,184,0.16)' stroke-width='1' />"
                )
            return "".join(lines)

        def make_distribution_rows(items: list[dict], color: str, total_mode: str) -> str:
            if not items:
                return "<div class='empty'>暂无数据</div>"
            if total_mode == "sum":
                total = sum(max(0, int(item.get("value", 0) or 0)) for item in items) or 1
            else:
                total = max(max(0, int(item.get("value", 0) or 0)) for item in items) or 1
            rows = []
            for item in items:
                name = escape(str(item.get("name", "-")))
                value = max(0, int(item.get("value", 0) or 0))
                percent = round(value / total * 100)
                rows.append(
                    "<div class='dist-row'>"
                    f"<span>{name}</span>"
                    f"<div class='dist-track'><i style='width:{percent}%; background:{color};'></i></div>"
                    f"<b>{value}</b>"
                    "</div>"
                )
            return "".join(rows)

        trend_svg = (
            "<svg viewBox='0 0 520 190' preserveAspectRatio='none'>"
            f"{make_grid_lines()}"
            f"<polyline fill='none' stroke='#38bdf8' stroke-width='3' points='{make_polyline(stress_series)}' />"
            f"<polyline fill='none' stroke='#f59e0b' stroke-width='3' points='{make_polyline(fatigue_series)}' />"
            f"<polyline fill='none' stroke='#22c55e' stroke-width='3' points='{make_polyline(focus_series)}' />"
            "</svg>"
        )
        mini_svg = (
            "<svg viewBox='0 0 320 150' preserveAspectRatio='none'>"
            f"{make_grid_lines(320, 150)}"
            f"<polyline fill='none' stroke='#f59e0b' stroke-width='3' points='{make_polyline(recent_stress, width=320, height=150)}' />"
            f"<polyline fill='none' stroke='#38bdf8' stroke-width='3' points='{make_polyline(recent_fatigue, width=320, height=150)}' />"
            f"<polyline fill='none' stroke='#22c55e' stroke-width='3' points='{make_polyline(recent_focus, width=320, height=150)}' />"
            "</svg>"
        )

        recent_labels = " / ".join(escape(item) for item in categories[-6:]) or "等待历史数据"
        emotion_rows = make_distribution_rows(emotion_distribution, "#a78bfa", "sum")
        signal_rows = make_distribution_rows(signal_distribution, "#22c55e", "max")
        radar_rows = make_distribution_rows(
            [
                {"name": "稳定度", "value": stability_score},
                {"name": "恢复力", "value": recovery_score},
                {"name": "专注力", "value": avg_focus},
                {"name": "节奏感", "value": rhythm_score},
                {"name": "低压力", "value": max(0, 100 - avg_stress)},
            ],
            "#38bdf8",
            "max",
        )

        return f"""
<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
        <style>
            html, body {{
                margin: 0;
                width: 100%;
                height: 100%;
                background:
                    radial-gradient(circle at 16% 10%, rgba(59, 130, 246, 0.22), transparent 28%),
                    radial-gradient(circle at 82% 12%, rgba(34, 211, 238, 0.12), transparent 24%),
                    linear-gradient(160deg, rgba(4, 11, 21, 0.98), rgba(7, 17, 31, 0.98) 42%, rgba(11, 23, 40, 0.96));
                color: #e2e8f0;
                font-family: "Segoe UI", "Microsoft YaHei", sans-serif;
            }}
            .board {{
                height: 100%;
                box-sizing: border-box;
                padding: 22px;
                display: grid;
                grid-template-columns: 1.5fr 1.1fr 1.1fr;
                grid-template-rows: 1.2fr 1fr 96px;
                gap: 20px;
            }}
            .card {{
                position: relative;
                background: linear-gradient(145deg, rgba(9, 20, 38, 0.92), rgba(15, 35, 61, 0.84));
                border: 1px solid rgba(125, 211, 252, 0.14);
                border-radius: 22px;
                box-shadow: inset 0 1px 0 rgba(255,255,255,0.04), 0 24px 50px rgba(2, 6, 23, 0.46);
                overflow: hidden;
            }}
            .title {{
                position: absolute;
                left: 22px;
                top: 18px;
                color: #f8fdff;
                font-size: 15px;
                font-weight: 700;
            }}
        .trend {{ grid-column: 1 / span 2; grid-row: 1; }}
        .donut {{ grid-column: 3; grid-row: 1; }}
        .bar {{ grid-column: 1; grid-row: 2; }}
        .radar {{ grid-column: 2; grid-row: 2; }}
        .mini {{ grid-column: 3; grid-row: 2; }}
        .metrics {{
            grid-column: 1 / span 3;
            grid-row: 3;
            display: grid;
            grid-template-columns: repeat(6, minmax(0, 1fr));
            gap: 18px;
        }}
        .metric {{
            padding: 14px 18px;
        }}
            .metric-label {{
                color: #8ba8c6;
                font-size: 12px;
                margin-bottom: 8px;
            }}
            .metric-value {{
                color: #f8fdff;
                font-size: 30px;
                font-weight: 700;
            }}
            .metric-sub {{
                color: #6dd3ff;
                font-size: 12px;
                margin-top: 6px;
            }}
        .chart-area {{
            position: absolute;
            inset: 56px 18px 18px 18px;
        }}
        .legend {{
            display: flex;
            gap: 14px;
            margin-bottom: 12px;
            color: #cbd5e1;
            font-size: 12px;
        }}
        .legend i {{
            display: inline-block;
            width: 10px;
            height: 10px;
            margin-right: 6px;
            border-radius: 999px;
        }}
        svg {{
            width: 100%;
            height: calc(100% - 28px);
        }}
        .dist-list {{
            display: flex;
            flex-direction: column;
            gap: 12px;
            padding-top: 8px;
        }}
        .dist-row {{
            display: grid;
            grid-template-columns: 64px 1fr 40px;
            gap: 10px;
            align-items: center;
            color: #cbd5e1;
            font-size: 12px;
        }}
        .dist-track {{
            height: 10px;
            border-radius: 999px;
            background: rgba(30, 41, 59, 0.95);
            overflow: hidden;
        }}
        .dist-track i {{
            display: block;
            height: 100%;
            border-radius: 999px;
        }}
        .dist-row b {{
            color: #f8fafc;
            text-align: right;
        }}
        .subnote {{
            color: #94a3b8;
            font-size: 12px;
            line-height: 1.5;
            margin-top: 10px;
        }}
        .empty {{
            color: #94a3b8;
            font-size: 13px;
            padding-top: 16px;
        }}
    </style>
</head>
<body>
    <div class="board">
        <div class="card trend">
            <div class="title">历史趋势折线</div>
            <div class="chart-area">
                <div class="legend">
                    <span><i style="background:#38bdf8;"></i>压力</span>
                    <span><i style="background:#f59e0b;"></i>疲劳</span>
                    <span><i style="background:#22c55e;"></i>专注</span>
                </div>
                {trend_svg}
            </div>
        </div>
        <div class="card donut">
            <div class="title">情绪结构扇形图</div>
            <div class="chart-area">
                <div class="dist-list">{emotion_rows}</div>
                <div class="subnote">{period_label} · {escape(range_start or "待采集")} 到 {escape(range_end or "待采集")}</div>
            </div>
        </div>
        <div class="card bar">
            <div class="title">主信号柱状图</div>
            <div class="chart-area">
                <div class="dist-list">{signal_rows}</div>
            </div>
        </div>
        <div class="card radar">
            <div class="title">综合画像雷达图</div>
            <div class="chart-area">
                <div class="dist-list">{radar_rows}</div>
            </div>
        </div>
        <div class="card mini">
            <div class="title">近期波动面积图</div>
            <div class="chart-area">
                <div class="legend">
                    <span><i style="background:#f59e0b;"></i>压力</span>
                    <span><i style="background:#38bdf8;"></i>疲劳</span>
                    <span><i style="background:#22c55e;"></i>专注</span>
                </div>
                {mini_svg}
                <div class="subnote">最近窗口：{recent_labels}</div>
            </div>
        </div>
        <div class="metrics">
            <div class="card metric">
                <div class="metric-label">历史平均压力</div>
                <div class="metric-value">{avg_stress}</div>
                <div class="metric-sub">{period_label}平均压力水平</div>
            </div>
            <div class="card metric">
                <div class="metric-label">历史平均疲劳</div>
                <div class="metric-value">{avg_fatigue}</div>
                <div class="metric-sub">{period_label}疲劳波动概览</div>
            </div>
            <div class="card metric">
                <div class="metric-label">历史平均专注</div>
                <div class="metric-value">{avg_focus}</div>
                <div class="metric-sub">{period_label}专注表现均值</div>
            </div>
            <div class="card metric">
                <div class="metric-label">键盘活跃均值</div>
                <div class="metric-value">{avg_keyboard}</div>
                <div class="metric-sub">{period_label}键盘活跃均值</div>
            </div>
            <div class="card metric">
                <div class="metric-label">鼠标活跃均值</div>
                <div class="metric-value">{avg_mouse}</div>
                <div class="metric-sub">{period_label}鼠标活跃均值</div>
            </div>
            <div class="card metric">
                <div class="metric-label">历史样本规模</div>
                <div class="metric-value">{sample_count}</div>
                <div class="metric-sub">{escape(range_start or "待采集")} - {escape(range_end or "待采集")}</div>
            </div>
        </div>
    </div>
</body>
</html>
        """

    def _update_dashboard_chart_view(self, payload: dict) -> None:
        if self.dashboard_chart_view is not None:
            self._pending_dashboard_payload = payload
            if not self._dashboard_chart_shell_loaded:
                self._load_dashboard_chart_shell()
                return
            if not self._dashboard_chart_page_ready:
                return
            self._push_dashboard_payload_to_view(payload)
            return
        if self.dashboard_chart_fallback is not None:
            self.dashboard_chart_fallback.setHtml(self._build_dashboard_fallback_html(payload))

    def _refresh_dashboard_page(self) -> None:
        if self._dashboard_refresh_timer.isActive():
            self._dashboard_refresh_timer.stop()
        if self._dashboard_repository is not None:
            if self._dashboard_period == "custom":
                start_date, end_date = self._dashboard_custom_range
                payload = self._dashboard_repository.get_dashboard_payload(
                    "custom",
                    start_date=start_date,
                    end_date=end_date,
                )
            else:
                payload = self._dashboard_repository.get_dashboard_payload(self._dashboard_period)
        else:
            payload = {}

        if hasattr(self, "_dashboard_filter_buttons"):
            for period, button in self._dashboard_filter_buttons.items():
                button.setChecked(period == self._dashboard_period)
        if hasattr(self, "dashboard_custom_apply_button"):
            self.dashboard_custom_apply_button.setProperty("active", self._dashboard_period == "custom")
            self.dashboard_custom_apply_button.style().unpolish(self.dashboard_custom_apply_button)
            self.dashboard_custom_apply_button.style().polish(self.dashboard_custom_apply_button)

        payload_key = json.dumps(payload, ensure_ascii=False, sort_keys=True)
        if self._dashboard_payload_key != payload_key:
            self._dashboard_payload_key = payload_key
            self._update_dashboard_chart_view(payload)

    def _refresh_report_page(self) -> None:
        today = QDate.currentDate().addDays(-1).toPython()
        week_start = QDate.currentDate().addDays(-7).toPython()
        week_end = QDate.currentDate().addDays(-1).toPython()

        if self._dashboard_repository is not None:
            daily_summary = self._dashboard_repository.get_report_summary(start_date=today, end_date=today)
            if self._report_custom_mode:
                custom_start, custom_end = self._report_custom_range
                period_summary = self._dashboard_repository.get_report_summary(start_date=custom_start, end_date=custom_end)
                period_title = "自定义时间健康分析报告"
            else:
                period_summary = self._dashboard_repository.get_report_summary(start_date=week_start, end_date=week_end)
                period_title = "每周健康分析报告"
        else:
            daily_summary = {}
            period_summary = {}
            period_title = "每周健康分析报告"

        if not hasattr(self, "weekly_report_title"):
            return

        report_payload_key = json.dumps(
            {
                "daily": daily_summary,
                "period": period_summary,
                "custom_mode": self._report_custom_mode,
                "period_title": period_title,
                "period_mode": period_mode,
            },
            ensure_ascii=False,
            sort_keys=True,
        )
        if report_payload_key == self._report_payload_key:
            return
        self._report_payload_key = report_payload_key

        self.weekly_report_title.setText(period_title)
        self.report_custom_apply_button.setProperty("active", self._report_custom_mode)
        self.report_custom_apply_button.style().unpolish(self.report_custom_apply_button)
        self.report_custom_apply_button.style().polish(self.report_custom_apply_button)
        self.daily_avg_stress_card.setValue(f"{daily_summary.get('average_stress', 0)}")
        self.daily_rest_count_card.setValue(f"{daily_summary.get('rest_activity_count', 0)} 次")
        self.daily_focus_index_card.setValue(f"{daily_summary.get('average_focus', 0)} / 100")
        self._latest_daily_report_md = self._build_health_report_markdown(
            title="每日健康分析报告",
            summary=daily_summary,
            range_label=f"{today.isoformat()} 至 {today.isoformat()}",
            mode_label="日维度固定模板",
        )
        self._latest_weekly_report_md = self._build_health_report_markdown(
            title=period_title,
            summary=period_summary,
            range_label=f"{period_summary.get('start_date', week_start.isoformat())} 至 {period_summary.get('end_date', week_end.isoformat())}",
            mode_label="周维度重点复盘" if not self._report_custom_mode else "自定义区间复盘",
        )
        self.daily_report_view.setMarkdown(self._latest_daily_report_md)
        self.weekly_report_view.setMarkdown(self._latest_weekly_report_md)
        self._write_report_snapshot("daily_latest.md", self._latest_daily_report_md)
        period_file = "custom_latest.md" if self._report_custom_mode else "weekly_latest.md"
        self._write_report_snapshot(period_file, self._latest_weekly_report_md)

    def _apply_custom_report_range(self) -> None:
        start_date = self.report_start_date.date().toPython()
        end_date = self.report_end_date.date().toPython()
        if start_date > end_date:
            self.statusBar().showMessage("报告开始日期不能晚于结束日期", 3200)
            return
        self._report_custom_range = (start_date, end_date)
        self._report_custom_mode = True
        self._refresh_report_page()

    def _reset_weekly_report_range(self) -> None:
        self._report_custom_mode = False
        self.report_start_date.setDate(QDate.currentDate().addDays(-7))
        self.report_end_date.setDate(QDate.currentDate().addDays(-1))
        self._report_custom_range = (
            self.report_start_date.date().toPython(),
            self.report_end_date.date().toPython(),
        )
        self._refresh_report_page()

    def _report_storage_dir(self) -> Path:
        report_dir = Path(__file__).resolve().parents[2] / "data" / "reports"
        report_dir.mkdir(parents=True, exist_ok=True)
        return report_dir

    def _write_report_snapshot(self, file_name: str, content: str) -> Path:
        path = self._report_storage_dir() / file_name
        if self._report_snapshot_cache.get(file_name) == content and path.exists():
            return path
        path.write_text(content, encoding="utf-8")
        self._report_snapshot_cache[file_name] = content
        return path

    def _export_report_markdown(self, report_type: str) -> None:
        if report_type == "daily":
            content = self._latest_daily_report_md
            default_name = "eyemuse_daily_report.md"
        else:
            content = self._latest_weekly_report_md
            default_name = "eyemuse_period_report.md"
        if not content:
            self.statusBar().showMessage("当前没有可导出的报告内容", 3200)
            return
        default_path = str(self._report_storage_dir() / default_name)
        file_path, _ = QFileDialog.getSaveFileName(
            self,
            "导出 Markdown 报告",
            default_path,
            "Markdown Files (*.md)",
        )
        if not file_path:
            return
        Path(file_path).write_text(content, encoding="utf-8")
        self.statusBar().showMessage(f"报告已导出到 {file_path}", 3200)

    def _build_health_report_markdown(self, *, title: str, summary: dict, range_label: str, mode_label: str) -> str:
        sample_count = summary.get("sample_count", 0)
        avg_stress = summary.get("average_stress", 0)
        avg_fatigue = summary.get("average_fatigue", 0)
        avg_focus = summary.get("average_focus", 0)
        avg_keyboard_activity = summary.get("average_keyboard_activity", 0)
        avg_mouse_activity = summary.get("average_mouse_activity", 0)
        avg_key_rate = summary.get("average_key_rate_per_min", 0)
        avg_mouse_distance = summary.get("average_mouse_distance", 0)
        top_emotion = summary.get("top_emotion", "平稳")
        top_signal = summary.get("top_signal", "稳定")
        top_behavior = _format_behavior_state(str(summary.get("top_behavior_state", "warming")))
        high_stress_count = summary.get("high_stress_count", 0)
        high_fatigue_count = summary.get("high_fatigue_count", 0)
        keyboard_decline_count = summary.get("keyboard_decline_count", 0)
        mouse_decline_count = summary.get("mouse_decline_count", 0)
        high_switch_count = summary.get("high_switch_count", 0)
        peak_stress = summary.get("peak_stress", 0)
        peak_fatigue = summary.get("peak_fatigue", 0)
        lowest_focus = summary.get("lowest_focus", 0)
        events = summary.get("events", [])

        if avg_fatigue >= 75 or high_fatigue_count >= 3 or (keyboard_decline_count >= 2 and mouse_decline_count >= 2):
            risk_level = "高"
            suggestion = "优先降低连续用眼时长，安排 10 分钟离屏休息，并在下一阶段减少高强度任务。"
        elif avg_stress >= 70 or high_stress_count >= 3 or high_switch_count >= 3:
            risk_level = "中高"
            suggestion = "优先做任务拆分与节奏降噪，避免多任务并发，加入呼吸放松或白噪音干预。"
        elif avg_focus >= 72:
            risk_level = "低"
            suggestion = "保持当前节奏，把高价值任务集中放在状态最稳的时间段，避免被低优先级事务打断。"
        else:
            risk_level = "中"
            suggestion = "建议采用 45-10 或 50-10 的学习工作节奏，控制压力峰值，穿插短休息。"

        event_lines = "\n".join(f"- {item}" for item in events) if events else "- 本时段暂无显著事件波动记录。"
        return (
            f"# {title}\n\n"
            f"> 统计区间：{range_label}  \n"
            f"> 分析模式：{mode_label}\n\n"
            "## 划重点\n"
            f"- **主导情绪**：{top_emotion}，当前主要行为信号为 **{top_signal}**，行为序列画像为 **{top_behavior}**。\n"
            f"- **核心风险等级**：{risk_level}，期间共记录 **{sample_count}** 条有效样本。\n"
            f"- **优先建议**：{suggestion}\n\n"
            "## 核心指标表\n\n"
            "| 指标 | 数值 | 解读 |\n"
            "| --- | ---: | --- |\n"
            f"| 平均压力 | {avg_stress} | 压力值越高，越需要减少外界干扰 |\n"
            f"| 平均疲劳 | {avg_fatigue} | 疲劳值越高，越需要休息与节奏调整 |\n"
            f"| 平均专注 | {avg_focus} | 专注值越高，越适合安排核心任务 |\n"
            f"| 压力峰值 | {peak_stress} | 观察是否存在明显高压时段 |\n"
            f"| 疲劳峰值 | {peak_fatigue} | 观察是否出现连续高疲劳趋势 |\n"
            f"| 最低专注 | {lowest_focus} | 用于判断注意力最低谷 |\n"
            f"| 键盘活跃度 | {avg_keyboard_activity} | 结合 key-rate 与活跃秒数的 30 秒周期指标 |\n"
            f"| 鼠标活跃度 | {avg_mouse_activity} | 结合位移距离与活跃秒数的 30 秒周期指标 |\n"
            f"| 平均 key-rate/min | {avg_key_rate} | 键盘每分钟等效活跃频率 |\n"
            f"| 平均鼠标位移 | {avg_mouse_distance} px | 鼠标 30 秒窗口平均位移距离 |\n\n"
            "## 风险观察\n"
            f"- 高压力样本次数：**{high_stress_count}**\n"
            f"- 高疲劳样本次数：**{high_fatigue_count}**\n"
            f"- 键盘活跃下降次数：**{keyboard_decline_count}**\n"
            f"- 鼠标活跃下降次数：**{mouse_decline_count}**\n"
            f"- 高频切换周期次数：**{high_switch_count}**\n"
            f"- 建议重点关注：**{top_emotion} / {top_signal}** 组合出现的时间段\n\n"
            "## 近期事件记录\n"
            f"{event_lines}\n\n"
            "## 重要分析建议\n"
            f"1. **优先级一**：{suggestion}\n"
            "2. **优先级二**：把高价值任务安排在专注度更高的时间段，避免在高疲劳段继续硬撑。\n"
            "3. **优先级三**：若高频切换周期持续增多，建议做任务分块和消息降噪；若键鼠活跃连续下降，建议主动安排短休息。\n"
        )

    def _refresh_report_page(self) -> None:
        if self._report_refresh_timer.isActive():
            self._report_refresh_timer.stop()
        today = QDate.currentDate().addDays(-1).toPython()
        current_date = QDate.currentDate().toPython()
        week_start = QDate.currentDate().addDays(-7).toPython()
        week_end = QDate.currentDate().addDays(-1).toPython()

        if self._dashboard_repository is not None:
            daily_summary = self._dashboard_repository.get_report_summary(start_date=today, end_date=today)
            current_rest_count = self._dashboard_repository.get_completed_rest_count(
                start_date=current_date,
                end_date=current_date,
            )
            if self._report_custom_mode:
                custom_start, custom_end = self._report_custom_range
                period_summary = self._dashboard_repository.get_report_summary(start_date=custom_start, end_date=custom_end)
                period_title = "自定义时间健康分析报告"
                period_mode = "自定义区间长期复盘"
            else:
                period_summary = self._dashboard_repository.get_report_summary(start_date=week_start, end_date=week_end)
                period_title = "情绪健康周报"
                period_mode = "周维度长期价值复盘"
        else:
            daily_summary = {}
            period_summary = {}
            current_rest_count = 0
            period_title = "情绪健康周报"
            period_mode = "周维度长期价值复盘"

        if not hasattr(self, "weekly_report_title"):
            return

        report_payload_key = json.dumps(
            {
                "daily": daily_summary,
                "period": period_summary,
                "custom_mode": self._report_custom_mode,
                "period_title": period_title,
                "period_mode": period_mode,
                "current_rest_count": current_rest_count,
            },
            ensure_ascii=False,
            sort_keys=True,
        )
        if report_payload_key == self._report_payload_key:
            return
        self._report_payload_key = report_payload_key

        self.weekly_report_title.setText(period_title)
        self.report_custom_apply_button.setProperty("active", self._report_custom_mode)
        self.report_custom_apply_button.style().unpolish(self.report_custom_apply_button)
        self.report_custom_apply_button.style().polish(self.report_custom_apply_button)
        self.daily_avg_stress_card.setValue(f"{daily_summary.get('average_stress', 0)}")
        visible_rest_count = int(daily_summary.get("rest_activity_count", 0) or 0)
        if current_date != today:
            visible_rest_count += current_rest_count
        self.daily_rest_count_card.setValue(f"{visible_rest_count} 次")
        self.daily_focus_index_card.setValue(f"{daily_summary.get('average_focus', 0)} / 100")

        daily_title = "每日健康分析报告"
        daily_range = f"{today.isoformat()} 至 {today.isoformat()}"
        weekly_range = f"{period_summary.get('start_date', week_start.isoformat())} 至 {period_summary.get('end_date', week_end.isoformat())}"

        daily_report_md = self._build_health_report_markdown(
            title=daily_title,
            summary=daily_summary,
            range_label=daily_range,
            mode_label="日维度价值观察",
        )
        weekly_report_md = self._build_health_report_markdown(
            title=period_title,
            summary=period_summary,
            range_label=weekly_range,
            mode_label=period_mode,
        )
        daily_report_html = self._build_health_report_html(
            title=daily_title,
            summary=daily_summary,
            range_label=daily_range,
            mode_label="日维度价值观察",
        )
        weekly_report_html = self._build_health_report_html(
            title=period_title,
            summary=period_summary,
            range_label=weekly_range,
            mode_label=period_mode,
        )
        daily_html_changed = daily_report_html != self._latest_daily_report_html
        weekly_html_changed = weekly_report_html != self._latest_weekly_report_html
        self._latest_daily_report_md = daily_report_md
        self._latest_weekly_report_md = weekly_report_md
        self._latest_daily_report_html = daily_report_html
        self._latest_weekly_report_html = weekly_report_html
        if daily_html_changed:
            self.daily_report_view.setHtml(daily_report_html)
            self.daily_report_view.verticalScrollBar().setValue(0)
        if weekly_html_changed:
            self.weekly_report_view.setHtml(weekly_report_html)
            self.weekly_report_view.verticalScrollBar().setValue(0)
        self._write_report_snapshot("daily_latest.md", self._latest_daily_report_md)
        self._write_report_snapshot("daily_latest.html", self._latest_daily_report_html)
        period_file = "custom_latest" if self._report_custom_mode else "weekly_latest"
        self._write_report_snapshot(f"{period_file}.md", self._latest_weekly_report_md)
        self._write_report_snapshot(f"{period_file}.html", self._latest_weekly_report_html)

    def _report_export_bundle(self, report_type: str) -> tuple[str, str, str, QTextBrowser]:
        if report_type == "daily":
            return (
                self._latest_daily_report_md,
                self._latest_daily_report_html,
                "eyemuse_daily_report",
                self.daily_report_view,
            )
        return (
            self._latest_weekly_report_md,
            self._latest_weekly_report_html,
            "eyemuse_period_report",
            self.weekly_report_view,
        )

    def _export_report_markdown(self, report_type: str) -> None:
        content_md, _content_html, default_stem, _browser = self._report_export_bundle(report_type)
        if not content_md:
            self.statusBar().showMessage("当前没有可导出的报告内容", 3200)
            return
        default_path = str(self._report_storage_dir() / f"{default_stem}.md")
        file_path, _ = QFileDialog.getSaveFileName(
            self,
            "导出 Markdown 报告",
            default_path,
            "Markdown Files (*.md)",
        )
        if not file_path:
            return
        Path(file_path).write_text(content_md, encoding="utf-8")
        self.statusBar().showMessage(f"报告已导出到 {file_path}", 3200)

    def _export_report_image(self, report_type: str) -> None:
        _content_md, content_html, default_stem, browser = self._report_export_bundle(report_type)
        if not content_html:
            self.statusBar().showMessage("当前没有可导出的报告图片内容", 3200)
            return
        timestamp = QDateTime.currentDateTime().toString("yyyy-MM-dd_HHmmss_zzz")
        export_dir = self._report_storage_dir()
        default_path = export_dir / f"{default_stem}_{timestamp}.png"
        duplicate_index = 2
        while default_path.exists():
            default_path = export_dir / f"{default_stem}_{timestamp}_{duplicate_index}.png"
            duplicate_index += 1
        file_path, _ = QFileDialog.getSaveFileName(
            self,
            "导出报告图片",
            str(default_path),
            "PNG Files (*.png)",
        )
        if not file_path:
            return
        image = self._render_report_browser_to_image(browser)
        image.save(file_path, "PNG")
        self.statusBar().showMessage(f"报告图片已导出到 {file_path}", 3200)

    def _render_report_browser_to_image(self, browser: QTextBrowser) -> QImage:
        document = browser.document().clone(self)
        document.setTextWidth(max(760, browser.viewport().width() - 24))
        doc_size = document.documentLayout().documentSize().toSize()
        margin = 26
        image = QImage(
            max(1, doc_size.width() + margin * 2),
            max(1, doc_size.height() + margin * 2),
            QImage.Format_ARGB32_Premultiplied,
        )
        image.fill(QColor("#07111f"))
        painter = QPainter(image)
        painter.setRenderHint(QPainter.Antialiasing, True)
        painter.setRenderHint(QPainter.TextAntialiasing, True)
        painter.translate(margin, margin)
        document.drawContents(painter, QRectF(0, 0, doc_size.width(), doc_size.height()))
        painter.end()
        return image

    def _report_insights(self, summary: dict) -> dict[str, object]:
        sample_count = int(summary.get("sample_count", 0) or 0)
        avg_stress = float(summary.get("average_stress", 0) or 0)
        avg_fatigue = float(summary.get("average_fatigue", 0) or 0)
        avg_focus = float(summary.get("average_focus", 0) or 0)
        top_emotion = str(summary.get("top_emotion", "平稳") or "平稳")
        top_signal = str(summary.get("top_signal", "稳定") or "稳定")
        top_behavior = _format_behavior_state(str(summary.get("top_behavior_state", "warming") or "warming"))

        raw_best_state = str(summary.get("best_state", "平稳 / warming") or "平稳 / warming")
        if " / " in raw_best_state:
            state_emotion, state_behavior = raw_best_state.split(" / ", 1)
            best_state_label = f"{state_emotion} / {_format_behavior_state(state_behavior)}"
        else:
            best_state_label = raw_best_state

        best_hour_label = str(summary.get("best_hour_label", "暂无数据") or "暂无数据")
        best_hour_score = float(summary.get("best_hour_score", 0) or 0)
        best_state_score = float(summary.get("best_state_score", 0) or 0)
        trend_summary = str(summary.get("trend_summary", "整体趋势仍在积累中。") or "整体趋势仍在积累中。")
        focus_delta = float(summary.get("focus_delta", 0) or 0)
        stress_delta = float(summary.get("stress_delta", 0) or 0)
        fatigue_delta = float(summary.get("fatigue_delta", 0) or 0)
        highlight_moments = list(summary.get("highlight_moments", []) or [])
        support_message = str(summary.get("support_message", "") or "")
        needs_support = bool(summary.get("needs_support", False))

        if avg_fatigue >= 75 or avg_stress >= 72:
            risk_level = "高"
            recommendation = "优先减轻持续高压任务，安排离屏休息，并降低接下来 24 小时的任务密度。"
        elif avg_focus >= 72 and avg_stress <= 60:
            risk_level = "低"
            recommendation = "当前状态适合持续打磨长期项目，把重要输出压到高质量时段完成。"
        else:
            risk_level = "中"
            recommendation = "建议把任务拆成更清晰的阶段，在状态回落前主动切换到整理、复盘或沟通型任务。"

        if best_hour_label != "暂无数据":
            efficiency_recommendation = f"数据显示你 {best_hour_label} 效率最高，建议把重要任务安排在这个时段。"
        else:
            efficiency_recommendation = "当前样本仍较少，建议继续积累数据后再锁定个人高效时段。"

        long_term_value = (
            f"长期来看，你最有价值的不是单次冲刺，而是把 {best_hour_label if best_hour_label != '暂无数据' else '高质量状态'} "
            "稳定转化为可重复的工作节奏。"
        )
        metric_rows = [
            ("平均压力", f"{summary.get('average_stress', 0)}", "值越低越利于深度工作"),
            ("平均疲劳", f"{summary.get('average_fatigue', 0)}", "值越高越需要恢复与节奏调整"),
            ("平均专注", f"{summary.get('average_focus', 0)}", "值越高越适合安排关键产出"),
            ("键盘活跃", f"{summary.get('average_keyboard_activity', 0)}", "辅助判断输入密度与执行节奏"),
            ("鼠标活跃", f"{summary.get('average_mouse_activity', 0)}", "辅助判断操作负荷与切换频率"),
            ("平均 key-rate", f"{summary.get('average_key_rate_per_min', 0)}", "每分钟输入活跃度"),
        ]
        support_resources = [
            "学校心理中心 / 企业 EAP / 社区心理咨询资源",
            "可信任的家人、朋友或主管，先进行一次低压力沟通",
            "若持续影响睡眠、食欲或日常功能，考虑联系专业心理咨询或精神卫生门诊",
        ] if needs_support else []

        if not highlight_moments:
            highlight_moments = ["本周期暂未识别到低疲劳持续工作时段，建议继续积累数据后再观察。"]

        return {
            "sample_count": sample_count,
            "risk_level": risk_level,
            "recommendation": recommendation,
            "efficiency_recommendation": efficiency_recommendation,
            "best_hour_label": best_hour_label,
            "best_hour_score": best_hour_score,
            "best_state_label": best_state_label,
            "best_state_score": best_state_score,
            "top_emotion": top_emotion,
            "top_signal": top_signal,
            "top_behavior": top_behavior,
            "trend_summary": trend_summary,
            "focus_delta": focus_delta,
            "stress_delta": stress_delta,
            "fatigue_delta": fatigue_delta,
            "highlight_moments": highlight_moments,
            "support_message": support_message,
            "support_resources": support_resources,
            "long_term_value": long_term_value,
            "metric_rows": metric_rows,
            "high_stress_count": int(summary.get("high_stress_count", 0) or 0),
            "high_fatigue_count": int(summary.get("high_fatigue_count", 0) or 0),
            "high_switch_count": int(summary.get("high_switch_count", 0) or 0),
            "keyboard_decline_count": int(summary.get("keyboard_decline_count", 0) or 0),
            "mouse_decline_count": int(summary.get("mouse_decline_count", 0) or 0),
        }

    def _build_health_report_markdown(self, *, title: str, summary: dict, range_label: str, mode_label: str) -> str:
        insights = self._report_insights(summary)
        metric_lines = "\n".join(
            f"| {label} | {value} | {hint} |" for label, value, hint in insights["metric_rows"]
        )
        highlight_lines = "\n".join(f"- {item}" for item in insights["highlight_moments"])
        support_block = ""
        if insights["support_message"]:
            resource_lines = "\n".join(f"- {item}" for item in insights["support_resources"])
            support_block = (
                "## 温和支持提醒\n"
                f"- {insights['support_message']}\n"
                f"{resource_lines}\n\n"
            )

        return (
            f"# {title}\n\n"
            f"> 统计区间：{range_label}  \n"
            f"> 分析模式：{mode_label}\n\n"
            "## 长期价值视角\n"
            f"- {insights['long_term_value']}\n"
            f"- 当前主导情绪为 **{insights['top_emotion']}**，主要信号为 **{insights['top_signal']}**，行为画像为 **{insights['top_behavior']}**。\n"
            f"- 当前风险等级：**{insights['risk_level']}**，共分析 **{insights['sample_count']}** 条样本。\n\n"
            "## 情绪与效率关联分析\n"
            f"- 效率最高时间段：**{insights['best_hour_label']}**，综合效率评分 **{insights['best_hour_score']}**。\n"
            f"- 最佳状态组合：**{insights['best_state_label']}**，综合效率评分 **{insights['best_state_score']}**。\n"
            f"- 建议：{insights['efficiency_recommendation']}\n\n"
            "## 情绪健康周报\n"
            f"- 趋势：{insights['trend_summary']}\n"
            f"- 低疲劳持续工作时段：\n{highlight_lines}\n"
            f"- 总体建议：{insights['recommendation']}\n\n"
            "## 核心指标\n"
            "| 指标 | 数值 | 解读 |\n"
            "| --- | ---: | --- |\n"
            f"{metric_lines}\n\n"
            "## 风险观察\n"
            f"- 高压力样本次数：**{insights['high_stress_count']}**\n"
            f"- 高疲劳样本次数：**{insights['high_fatigue_count']}**\n"
            f"- 高频切换周期次数：**{insights['high_switch_count']}**\n"
            f"- 键盘活跃下降次数：**{insights['keyboard_decline_count']}**\n"
            f"- 鼠标活跃下降次数：**{insights['mouse_decline_count']}**\n\n"
            f"{support_block}"
        )

    def _build_health_report_html(self, *, title: str, summary: dict, range_label: str, mode_label: str) -> str:
        insights = self._report_insights(summary)
        risk_level = str(insights["risk_level"])
        risk_color = {"低": "#25866f", "中": "#a96828", "高": "#b65353"}.get(risk_level, "#a96828")
        metrics_html = "".join(
            "<tr>"
            f"<td width='26%' class='metric-label'>{escape(label)}</td>"
            f"<td width='18%' class='metric-value'>{escape(value)}</td>"
            f"<td width='56%' class='metric-hint'>{escape(hint)}</td>"
            "</tr>"
            for label, value, hint in insights["metric_rows"]
        )
        highlights_html = "".join(
            "<tr>"
            f"<td width='38' valign='top' class='moment-index'>0{index}</td>"
            f"<td valign='top' class='moment-text'>{escape(str(item))}</td>"
            "</tr>"
            for index, item in enumerate(insights["highlight_moments"], start=1)
        )
        resources_html = "".join(
            f"<tr><td width='18' valign='top' class='resource-dot'>•</td><td class='resource-text'>{escape(item)}</td></tr>"
            for item in insights["support_resources"]
        )
        support_html = ""
        if insights["support_message"]:
            support_html = (
                "<tr><td height='14'></td></tr>"
                "<tr><td>"
                "<table width='100%' cellspacing='0' cellpadding='18' class='support-card'>"
                "<tr><td>"
                "<div class='support-tag'>温和支持</div>"
                "<h2>长期异常值得被认真照顾</h2>"
                f"<p>{escape(str(insights['support_message']))}</p>"
                f"<table width='100%' cellspacing='0' cellpadding='3'>{resources_html}</table>"
                "</td></tr></table>"
                "</td></tr>"
            )

        return f"""
        <html>
        <head>
        <style>
            body {{ margin: 0; background-color: #EAF7FF; color: #355E96; font-family: 'Microsoft YaHei UI', 'Segoe UI', sans-serif; }}
            table {{ color: #355E96; }}
            h1 {{ color: #1E4F89; font-size: 28px; font-weight: 700; margin: 8px 0 6px; }}
            h2 {{ color: #28578F; font-size: 18px; font-weight: 700; margin: 4px 0 8px; }}
            p {{ color: #56779F; font-size: 13px; line-height: 1.65; margin: 6px 0; }}
            body {{
                background:
                    radial-gradient(circle at 14% 10%, rgba(56, 189, 248, 0.14), transparent 24%),
                    radial-gradient(circle at 86% 12%, rgba(34, 211, 238, 0.10), transparent 22%),
                    linear-gradient(160deg, #F4FBFF 0%, #E8F6FF 46%, #D7ECFF 100%);
            }}
            .canvas {{ background-color: transparent; }}
            .hero, .section-card, .support-card, .stat-card, .accent-card, .trend-cell {{
                box-shadow: 0 16px 34px rgba(91, 143, 196, 0.18);
            }}
            .hero {{
                background: linear-gradient(145deg, rgba(236, 248, 255, 0.96), rgba(204, 231, 255, 0.92));
                border: 1px solid rgba(255, 255, 255, 0.92);
                border-radius: 22px;
            }}
            .eyebrow {{ color: #3979BC; font-size: 11px; font-weight: 700; letter-spacing: 0.8px; }}
            .range {{ color: #6487B2; font-size: 12px; }}
            .hero-summary {{ color: #355E96; font-size: 14px; line-height: 1.6; }}
            .stat-card {{
                background: linear-gradient(145deg, rgba(230, 245, 255, 0.96), rgba(203, 229, 255, 0.92));
                border: 1px solid rgba(255, 255, 255, 0.90);
                border-radius: 18px;
            }}
            .stat-label {{ color: #6387B4; font-size: 11px; }}
            .stat-value {{ color: #1E4F89; font-size: 20px; font-weight: 700; }}
            .stat-note {{ color: #57799F; font-size: 11px; }}
            .section-card {{
                background: linear-gradient(145deg, rgba(237, 248, 255, 0.96), rgba(211, 234, 255, 0.92));
                border: 1px solid rgba(255, 255, 255, 0.92);
                border-radius: 22px;
            }}
            .section-tag {{ color: #3979BC; font-size: 11px; font-weight: 700; letter-spacing: 0.7px; }}
            .lead {{ color: #355E96; font-size: 14px; line-height: 1.7; }}
            .accent-card {{
                background: linear-gradient(145deg, rgba(222, 242, 255, 0.96), rgba(195, 224, 255, 0.90));
                border-left: 3px solid #5C9DE0;
                border-radius: 18px;
            }}
            .accent-label {{ color: #6387B4; font-size: 11px; }}
            .accent-value {{ color: #1E4F89; font-size: 22px; font-weight: 700; }}
            .accent-copy {{ color: #52749A; font-size: 12px; line-height: 1.55; }}
            .trend-cell {{
                background: linear-gradient(145deg, rgba(233, 246, 255, 0.96), rgba(205, 230, 255, 0.90));
                border: 1px solid rgba(255, 255, 255, 0.90);
                border-radius: 16px;
            }}
            .trend-value {{ color: #3979BC; font-size: 16px; font-weight: 700; }}
            .trend-label {{ color: #6E8DB8; font-size: 10px; }}
            .moment-index {{ color: #3979BC; font-size: 11px; font-weight: 700; padding: 7px 5px; }}
            .moment-text {{ color: #355E96; font-size: 12px; line-height: 1.55; padding: 7px 8px; border-bottom: 1px solid rgba(116, 162, 210, 0.20); }}
            .metric-label {{ color: #5C7EA8; font-size: 12px; padding: 9px 10px; border-bottom: 1px solid rgba(116, 162, 210, 0.20); }}
            .metric-value {{ color: #1E4F89; font-size: 13px; font-weight: 700; padding: 9px 10px; border-bottom: 1px solid rgba(116, 162, 210, 0.20); }}
            .metric-hint {{ color: #6387B4; font-size: 11px; padding: 9px 10px; border-bottom: 1px solid rgba(116, 162, 210, 0.20); }}
            .support-card {{
                background: linear-gradient(145deg, rgba(255, 247, 228, 0.96), rgba(255, 232, 192, 0.92));
                border: 1px solid rgba(227, 180, 104, 0.42);
                border-radius: 22px;
            }}
            .support-tag {{ color: #9B672A; font-size: 11px; font-weight: 700; letter-spacing: 0.5px; }}
            .resource-dot {{ color: #B57A36; font-size: 13px; }}
            .resource-text {{ color: #6E573A; font-size: 12px; line-height: 1.5; }}
        </style>
        </head>
        <body>
            <table width="96%" align="center" cellspacing="0" cellpadding="0" class="canvas">
            <tr><td>
                <table width="100%" cellspacing="0" cellpadding="20" class="hero">
                <tr><td>
                    <div class="eyebrow">{escape(mode_label)}</div>
                    <h1>{escape(title)}</h1>
                    <div class="range">统计区间 · {escape(range_label)}</div>
                    <p class="hero-summary">{escape(str(insights["long_term_value"]))}</p>
                    <table width="100%" cellspacing="8" cellpadding="12">
                    <tr>
                        <td width="34%" valign="top" class="stat-card">
                            <div class="stat-label">效率高峰</div>
                            <div class="stat-value">{escape(str(insights["best_hour_label"]))}</div>
                            <div class="stat-note">综合评分 {escape(str(insights["best_hour_score"]))}</div>
                        </td>
                        <td width="33%" valign="top" class="stat-card">
                            <div class="stat-label">平均专注</div>
                            <div class="stat-value">{escape(str(summary.get("average_focus", 0)))}</div>
                            <div class="stat-note">主导情绪 · {escape(str(insights["top_emotion"]))}</div>
                        </td>
                        <td width="33%" valign="top" class="stat-card">
                            <div class="stat-label">风险等级</div>
                            <div class="stat-value" style="color:{risk_color};">{escape(risk_level)}</div>
                            <div class="stat-note">{escape(str(insights["sample_count"]))} 条有效样本</div>
                        </td>
                    </tr>
                    </table>
                </td></tr>
                </table>
            </td></tr>
            <tr><td height="14"></td></tr>
            <tr><td>
                <table width="100%" cellspacing="0" cellpadding="18" class="section-card">
                <tr><td>
                    <div class="section-tag">LONG-TERM VALUE · 长期价值</div>
                    <h2>把好状态变成可重复的节奏</h2>
                    <p class="lead">{escape(str(insights["recommendation"]))}</p>
                    <p>当前行为画像为 {escape(str(insights["top_behavior"]))}，主要信号为 {escape(str(insights["top_signal"]))}。长期追踪的意义，是让高质量输出不再依赖偶然状态。</p>
                </td></tr>
                </table>
            </td></tr>
            <tr><td height="14"></td></tr>
            <tr><td>
                <table width="100%" cellspacing="0" cellpadding="18" class="section-card">
                <tr><td>
                    <div class="section-tag">EMOTION × EFFICIENCY · 情绪与效率</div>
                    <h2>最值得保护的工作窗口</h2>
                    <table width="100%" cellspacing="8" cellpadding="14">
                    <tr>
                        <td width="50%" valign="top" class="accent-card">
                            <div class="accent-label">高效时间段</div>
                            <div class="accent-value">{escape(str(insights["best_hour_label"]))}</div>
                            <p class="accent-copy">{escape(str(insights["efficiency_recommendation"]))}</p>
                        </td>
                        <td width="50%" valign="top" class="accent-card" style="border-left-color:#f0b45d;">
                            <div class="accent-label">最佳状态组合</div>
                            <div class="accent-value">{escape(str(insights["best_state_label"]))}</div>
                            <p class="accent-copy">综合效率评分 {escape(str(insights["best_state_score"]))}，适合安排深度工作与关键输出。</p>
                        </td>
                    </tr>
                    </table>
                </td></tr>
                </table>
            </td></tr>
            <tr><td height="14"></td></tr>
            <tr><td>
                <table width="100%" cellspacing="0" cellpadding="18" class="section-card">
                <tr>
                    <td width="48%" valign="top">
                        <div class="section-tag">WEEKLY TREND · 周期趋势</div>
                        <h2>状态变化</h2>
                        <p>{escape(str(insights["trend_summary"]))}</p>
                        <table width="100%" cellspacing="6" cellpadding="10">
                        <tr>
                            <td width="33%" class="trend-cell"><div class="trend-value">{escape(str(insights["focus_delta"]))}</div><div class="trend-label">专注变化</div></td>
                            <td width="33%" class="trend-cell"><div class="trend-value">{escape(str(insights["stress_delta"]))}</div><div class="trend-label">压力变化</div></td>
                            <td width="34%" class="trend-cell"><div class="trend-value">{escape(str(insights["fatigue_delta"]))}</div><div class="trend-label">疲劳变化</div></td>
                        </tr>
                        </table>
                    </td>
                    <td width="4%"></td>
                    <td width="48%" valign="top">
                        <div class="section-tag">HIGHLIGHTS · 高光时刻</div>
                        <h2>低疲劳持续工作时段</h2>
                        <table width="100%" cellspacing="0" cellpadding="0">{highlights_html}</table>
                    </td>
                </tr>
                </table>
            </td></tr>
            <tr><td height="14"></td></tr>
            <tr><td>
                <table width="100%" cellspacing="0" cellpadding="18" class="section-card">
                <tr><td>
                    <div class="section-tag">HEALTH SIGNALS · 核心指标</div>
                    <h2>指标明细</h2>
                    <table width="100%" cellspacing="0" cellpadding="0">{metrics_html}</table>
                </td></tr>
                </table>
            </td></tr>
            {support_html}
            </table>
        </body>
        </html>
        """

    def _apply_theme(self) -> None:
        self.setFont(QFont("Segoe UI Variable", 10))
        app = QApplication.instance()
        if app is not None:
            app.setStyle("Fusion")
            palette = QPalette()
            palette.setColor(QPalette.Window, QColor("#07111f"))
            palette.setColor(QPalette.WindowText, QColor("#e6f1ff"))
            palette.setColor(QPalette.Base, QColor("#071521"))
            palette.setColor(QPalette.AlternateBase, QColor("#0b1728"))
            palette.setColor(QPalette.Text, QColor("#e6f1ff"))
            palette.setColor(QPalette.Button, QColor("#10233a"))
            palette.setColor(QPalette.ButtonText, QColor("#eff9ff"))
            palette.setColor(QPalette.Highlight, QColor("#38bdf8"))
            palette.setColor(QPalette.HighlightedText, QColor("#04111f"))
            app.setPalette(palette)

        self.setStyleSheet(
            """
            QMainWindow {
                background:
                    radial-gradient(circle at 18% 10%, rgba(56, 189, 248, 28) 0%, rgba(56, 189, 248, 0) 28%),
                    radial-gradient(circle at 82% 8%, rgba(34, 197, 94, 16) 0%, rgba(34, 197, 94, 0) 24%),
                    linear-gradient(135deg, #050b14 0%, #07111f 35%, #0b1728 100%);
                color: #e6f1ff;
            }
            QMainWindow::separator {
                background: rgba(125, 211, 252, 40);
            }
            #NavBar {
                background: rgba(7, 17, 31, 214);
                border-bottom: 1px solid rgba(125, 211, 252, 42);
            }
            QPushButton#NavButton {
                background: rgba(125, 211, 252, 10);
                border: 1px solid transparent;
                border-radius: 16px;
                color: #b9d6ee;
                padding: 10px 20px;
                min-height: 26px;
                font-weight: 700;
                letter-spacing: 0.2px;
            }
            QPushButton#NavButton:hover {
                background: rgba(56, 189, 248, 20);
                border-color: rgba(125, 211, 252, 42);
                color: #f4fbff;
            }
            QPushButton#NavButton:checked {
                color: #f8fdff;
                background: qlineargradient(x1:0, y1:0, x2:1, y2:1,
                    stop:0 rgba(56, 189, 248, 54), stop:1 rgba(59, 130, 246, 92));
                border: 1px solid rgba(125, 211, 252, 110);
            }
            #WindowControlGroup {
                background: transparent;
                border: none;
            }
            QPushButton#WindowControlButton {
                background: transparent;
                border: none;
                border-radius: 14px;
                padding: 0;
            }
            QPushButton#WindowControlButton:hover {
                background: rgba(148, 163, 184, 26);
            }
            QPushButton#WindowControlButton:pressed {
                background: rgba(148, 163, 184, 42);
            }
            QPushButton#WindowCloseButton:hover {
                background: rgba(239, 68, 68, 30);
            }
            #Page {
                background: transparent;
            }
            #Panel {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:1,
                    stop:0 rgba(10, 20, 37, 232), stop:1 rgba(17, 34, 58, 216));
                border: 1px solid rgba(125, 211, 252, 46);
                border-radius: 26px;
            }
            #SectionTitle {
                color: #f8fdff;
                font-size: 19px;
                font-weight: 700;
            }
            #Title {
                color: #f8fdff;
                font-size: 23px;
                font-weight: 700;
            }
            #Subtitle {
                color: #9fb8d3;
                font-size: 12px;
            }
            #Hint {
                color: #d5e7fb;
                background: rgba(10, 24, 44, 178);
                border: 1px solid rgba(125, 211, 252, 24);
                border-radius: 14px;
                padding: 8px 12px;
            }
            #Badge, #BadgeSecondary, #InlineStatus {
                border-radius: 999px;
                padding: 6px 12px;
                font-weight: 600;
            }
            #Badge {
                color: #04111f;
                background: qlineargradient(x1:0, y1:0, x2:1, y2:1,
                    stop:0 #dff7ff, stop:1 #7dd3fc);
            }
            #BadgeSecondary {
                color: #d0e4f9;
                background: rgba(22, 39, 61, 220);
                border: 1px solid rgba(125, 211, 252, 26);
            }
            QPushButton#CompanionModeButton {
                background: rgba(10, 24, 44, 190);
                border: 1px solid rgba(125, 211, 252, 90);
                border-radius: 999px;
                color: #eff9ff;
                padding: 6px 14px;
                min-height: 12px;
            }
            QPushButton#CompanionModeButton:hover {
                background: rgba(14, 165, 233, 54);
                color: #f8fdff;
            }
            #InlineStatus {
                color: #dbeafe;
                background: rgba(14, 165, 233, 102);
                min-width: 56px;
                qproperty-alignment: AlignCenter;
            }
            #StatCard {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:1,
                    stop:0 rgba(8, 20, 38, 230), stop:1 rgba(16, 35, 58, 216));
                border: 1px solid rgba(125, 211, 252, 54);
                border-radius: 18px;
            }
            #CardTitle {
                color: #90a8c6;
                font-size: 10px;
                letter-spacing: 1px;
                text-transform: uppercase;
            }
            #CardValue {
                color: #f8fdff;
                font-size: 12px;
                font-weight: 600;
            }
            #ConversationView {
                background: rgba(3, 9, 19, 198);
                border: 1px solid rgba(125, 211, 252, 50);
                border-radius: 20px;
                padding: 10px;
            }
            #OverviewPanel {
                background: rgba(3, 9, 19, 198);
                border: 1px solid rgba(125, 211, 252, 50);
                border-radius: 20px;
                padding: 12px;
                color: #e6f1ff;
                font-size: 14px;
            }
            #ReportBrowser {
                background: rgba(4, 12, 24, 210);
                border: 1px solid rgba(125, 211, 252, 38);
                border-radius: 20px;
                padding: 0;
                color: #e6f1ff;
                font-size: 14px;
                selection-background-color: rgba(56, 189, 248, 120);
            }
            QTextBrowser QScrollBar:vertical, QPlainTextEdit QScrollBar:vertical {
                background: transparent;
                width: 11px;
                margin: 6px 2px 6px 2px;
                border: none;
            }
            QTextBrowser QScrollBar::handle:vertical, QPlainTextEdit QScrollBar::handle:vertical {
                background: rgba(56, 189, 248, 86);
                min-height: 52px;
                border-radius: 5px;
            }
            QTextBrowser QScrollBar::handle:vertical:hover, QPlainTextEdit QScrollBar::handle:vertical:hover {
                background: rgba(125, 211, 252, 126);
            }
            QTextBrowser QScrollBar::add-line:vertical,
            QTextBrowser QScrollBar::sub-line:vertical,
            QPlainTextEdit QScrollBar::add-line:vertical,
            QPlainTextEdit QScrollBar::sub-line:vertical {
                height: 0;
                border: none;
                background: transparent;
            }
            QTextBrowser QScrollBar::add-page:vertical,
            QTextBrowser QScrollBar::sub-page:vertical,
            QPlainTextEdit QScrollBar::add-page:vertical,
            QPlainTextEdit QScrollBar::sub-page:vertical {
                background: transparent;
            }
            #CameraPreview {
                background: rgba(2, 8, 18, 194);
                border: 1px dashed rgba(125, 211, 252, 84);
                border-radius: 20px;
                color: #9fb8d3;
                margin-top: 2px;
                margin-bottom: 2px;
            }
            QProgressBar {
                background: rgba(8, 20, 38, 220);
                border: 1px solid rgba(125, 211, 252, 54);
                border-radius: 12px;
                color: #e6f1ff;
                text-align: center;
                min-height: 24px;
            }
            QProgressBar::chunk {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                    stop:0 #67e8f9, stop:1 #3b82f6);
                border-radius: 10px;
            }
            QLineEdit {
                background: rgba(8, 20, 38, 232);
                border: 1px solid rgba(125, 211, 252, 52);
                border-radius: 16px;
                padding: 12px 14px;
                color: #e6f1ff;
            }
            QLineEdit:focus, QDateEdit:focus {
                border: 1px solid rgba(125, 211, 252, 126);
                background: rgba(10, 24, 44, 240);
            }
            QPushButton {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:1,
                    stop:0 #67e8f9, stop:1 #3b82f6);
                border: 1px solid rgba(191, 219, 254, 88);
                border-radius: 16px;
                color: #04111f;
                font-weight: 700;
                padding: 11px 16px;
                min-height: 18px;
            }
            QPushButton#GhostButton {
                background: rgba(10, 24, 44, 160);
                border: 1px solid rgba(125, 211, 252, 48);
                color: #e6f1ff;
            }
            #DashboardSegment {
                background: rgba(10, 24, 44, 196);
                border: 1px solid rgba(125, 211, 252, 48);
                border-radius: 20px;
            }
            QPushButton#DashboardFilterButton {
                background: transparent;
                border: 1px solid transparent;
                color: #bfd3e8;
                padding: 9px 20px;
                min-height: 16px;
                border-radius: 12px;
            }
            QPushButton#DashboardFilterButton:hover {
                background: rgba(14, 165, 233, 34);
                color: #f8fdff;
            }
            QPushButton#DashboardFilterButton:checked {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 rgba(56, 189, 248, 184), stop:1 rgba(14, 165, 233, 184));
                border: 1px solid rgba(125, 211, 252, 140);
                color: #f8fdff;
            }
            #DashboardDateRange {
                background: rgba(10, 24, 44, 196);
                border: 1px solid rgba(125, 211, 252, 48);
                border-radius: 20px;
            }
            #DashboardDateLabel {
                color: #bfd3e8;
                font-size: 12px;
                font-weight: 600;
                padding-right: 4px;
            }
            QDateEdit#DashboardDateEdit {
                background: rgba(4, 12, 24, 210);
                border: 1px solid rgba(125, 211, 252, 60);
                border-radius: 12px;
                color: #e6f1ff;
                padding: 7px 10px;
                min-width: 116px;
            }
            QPushButton#DashboardApplyButton {
                background: rgba(14, 32, 54, 220);
                border: 1px solid rgba(125, 211, 252, 72);
                color: #e6f1ff;
                border-radius: 12px;
                padding: 8px 16px;
                min-height: 16px;
            }
            QPushButton#DashboardApplyButton:hover {
                background: rgba(14, 165, 233, 48);
                color: #f8fdff;
            }
            QPushButton#DashboardApplyButton[active="true"] {
                background: rgba(14, 165, 233, 118);
                border: 1px solid rgba(125, 211, 252, 150);
                color: #f8fdff;
            }
            QPushButton:hover {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 #9ae6ff, stop:1 #38bdf8);
            }
            QPushButton:pressed {
                background: #0ea5e9;
            }
            QCheckBox {
                color: #dbeafe;
                spacing: 8px;
                font-weight: 600;
            }
            QCheckBox::indicator {
                width: 18px;
                height: 18px;
                border-radius: 9px;
                border: 1px solid rgba(125, 211, 252, 72);
                background: rgba(8, 20, 38, 220);
            }
            QCheckBox::indicator:checked {
                background: #22d3ee;
                border-color: #67e8f9;
            }
            QTextBrowser {
                color: #e6f1ff;
                font-size: 14px;
            }
            QMenu {
                background: rgba(7, 17, 31, 245);
                color: #e6f1ff;
                border: 1px solid rgba(125, 211, 252, 40);
                border-radius: 14px;
                padding: 6px;
            }
            QMenu::item {
                padding: 8px 16px;
                border-radius: 10px;
            }
            QMenu::item:selected {
                background: rgba(56, 189, 248, 26);
            }
            QStatusBar {
                background: rgba(7, 17, 31, 214);
                color: #bfd3e8;
                border-top: 1px solid rgba(125, 211, 252, 26);
            }
            QToolTip {
                background: rgba(7, 17, 31, 245);
                color: #e6f1ff;
                border: 1px solid rgba(125, 211, 252, 54);
                padding: 6px 10px;
                border-radius: 10px;
            }
            """
        )

    def _set_mood(self, mood: PetMood, hint: str) -> None:
        if self.mood_badge.text() == mood.value and self.pet_hint.text() == hint:
            self._current_pet_mood = mood
            self._current_pet_hint = hint
            self._refresh_companion_feedback()
            return
        self._current_pet_mood = mood
        self._current_pet_hint = hint
        self.avatar.setMood(mood)
        if self._companion_window is not None:
            self._companion_window.setMood(mood)
        self._update_mode_cards(mood.value)
        self.pet_hint.setText(hint)
        self.statusBar().showMessage(hint, 3000)
        self._refresh_companion_feedback()
        self._schedule_analytics_refresh()

    def _update_mode_cards(self, mood: str) -> None:
        self.mood_badge.setText(mood)
        self.mode_card.setValue(mood)

    def _append_system_message(self, text: str) -> None:
        self._conversation.append(ConversationItem("system", text, self._now()))
        self._refresh_conversation()
        self.event_card.setValue(text)
        self._schedule_analytics_refresh()

    def _inject_message(self, role: str, text: str) -> None:
        if role == "user":
            self.message_input.setText(text)
            self._handle_send()
        else:
            self._append_system_message(text)

    def _submit_user_text(self, text: str, *, from_companion: bool = False) -> None:
        if self._llm_thread is not None:
            self.statusBar().showMessage("上一条回复仍在生成中。", 2500)
            return

        text = text.strip()
        if not text:
            return

        if from_companion and self._companion_window is not None:
            self._companion_window.clear_chat_input()
        else:
            self.message_input.clear()
        self._set_mood(PetMood.listening, "已收到输入，准备生成回应。")
        self._conversation.append(ConversationItem("user", text, self._now()))
        self._refresh_conversation()

        if self._llm_client is not None and getattr(self._llm_client, "configured", False):
            self._start_streaming_reply(text, from_companion=from_companion)
            return

        reply = self._generate_local_reply(text)
        self._set_mood(PetMood.responding, "正在生成本地回应。")
        self._conversation.append(ConversationItem("eyeMuse", reply, self._now()))
        self._refresh_conversation()
        self._set_mood(PetMood.idle, reply)
        if from_companion:
            self._show_companion_message(reply)

    def _handle_send(self) -> None:
        self._submit_user_text(self.message_input.text(), from_companion=False)

    def _start_streaming_reply(self, text: str, *, from_companion: bool = False) -> None:
        if self._llm_client is None:
            return

        self._streaming_user_text = text
        self._streaming_from_companion = from_companion
        self._conversation.append(ConversationItem("eyeMuse", "", self._now()))
        self._streaming_reply_index = len(self._conversation) - 1
        self._refresh_conversation()
        self._set_chat_busy(True)
        self._set_mood(PetMood.responding, "正在流式生成回应。")
        self.event_card.setValue("LLM 流式输出中")
        if from_companion:
            self._show_companion_message("我正在整理回复，请稍等一下。")

        self._llm_thread = QThread(self)
        self._llm_worker = LLMStreamWorker(
            llm_client=self._llm_client,
            user_text=text,
            conversation_items=[
                {"role": item.role, "text": item.text}
                for item in self._conversation[-12:]
            ],
            context_summary=self._current_summary(),
        )
        self._llm_worker.moveToThread(self._llm_thread)
        self._llm_thread.started.connect(self._llm_worker.run)
        self._llm_worker.chunk_received.connect(self._append_stream_chunk)
        self._llm_worker.completed.connect(self._finish_streaming_reply)
        self._llm_worker.failed.connect(self._handle_streaming_error)
        self._llm_thread.finished.connect(self._llm_worker.deleteLater)
        self._llm_thread.finished.connect(self._llm_thread.deleteLater)
        self._llm_thread.start()

    def _append_stream_chunk(self, chunk: str) -> None:
        if self._streaming_reply_index is None or not chunk:
            return
        self._conversation[self._streaming_reply_index].text += chunk
        self._refresh_conversation()
        self.event_card.setValue("LLM 流式输出中")

    def _finish_streaming_reply(self, final_text: str) -> None:
        if self._streaming_reply_index is not None:
            self._conversation[self._streaming_reply_index].text = final_text or self._conversation[self._streaming_reply_index].text
        reply_text = final_text or "回复已完成。"
        self._refresh_conversation()
        self._set_mood(PetMood.idle, reply_text)
        if self._streaming_from_companion:
            self._show_companion_message(reply_text)
        self._teardown_streaming_reply()

    def _handle_streaming_error(self, message: str) -> None:
        partial = ""
        reply_text = "回复暂时中断，请稍后再试。"
        if self._streaming_reply_index is not None:
            partial = self._conversation[self._streaming_reply_index].text.strip()

        if self._streaming_reply_index is not None and not partial:
            fallback = self._generate_local_reply(self._streaming_user_text)
            self._conversation[self._streaming_reply_index].text = fallback
            self._set_mood(PetMood.idle, fallback)
            reply_text = fallback
        elif self._streaming_reply_index is not None:
            self._conversation[self._streaming_reply_index].text += "\n\n[回复中断]"
            self._set_mood(PetMood.alert, "流式回复中断。")
            reply_text = f"{partial}\n\n回复暂时中断。"

        self.event_card.setValue(f"LLM 回退到本地回复：{message}")
        self.camera_note.setText(f"LLM 流式调用异常：{message}")
        self._refresh_conversation()
        if self._streaming_from_companion:
            self._show_companion_message(reply_text)
        self._teardown_streaming_reply()

    def _teardown_streaming_reply(self) -> None:
        if self._llm_thread is not None:
            self._llm_thread.quit()
            self._llm_thread.wait(1500)
        self._llm_thread = None
        self._llm_worker = None
        self._streaming_reply_index = None
        self._streaming_user_text = ""
        self._streaming_from_companion = False
        self._set_chat_busy(False)

    def _set_chat_busy(self, busy: bool) -> None:
        self.message_input.setEnabled(not busy)
        self.send_button.setEnabled(not busy)
        self.clear_button.setEnabled(not busy)
        if self._companion_window is not None:
            self._companion_window.set_chat_busy(busy)

    def _generate_local_reply(self, text: str) -> str:
        lowered = text.lower()
        if any(keyword in lowered for keyword in ("累", "困", "疲劳", "休息", "sleep", "tired")):
            return "我注意到你可能有些疲劳。先休息几分钟，等你缓一缓我再陪你。"
        if any(keyword in lowered for keyword in ("摄像头", "camera", "脸", "面部")):
            return "摄像头链路已经预留好了。当前前端会优先给出本地提示，再逐步接上更稳定的感知。"
        if any(keyword in lowered for keyword in ("你好", "hi", "hello")):
            return "你好，我已经在线。你可以直接输入想法，也可以先打开摄像头看看状态。"
        return f"我收到你的输入：{text}。接下来我会根据状态、摄像头和后续模型接入继续完善回应。"

    def _render_companion_chat_html(self) -> str:
        if not self._conversation:
            return "<p style='color:#94a3b8; text-align:center;'>点一下“对话”，就在这里聊天。</p>"
        html = []
        for item in self._conversation[-6:]:
            if item.role == "user":
                align = "right"
                bubble = "linear-gradient(135deg, rgba(34, 197, 246, 0.28), rgba(37, 99, 235, 0.50))"
                color = "#eff6ff"
            elif item.role == "eyeMuse":
                align = "left"
                bubble = "linear-gradient(135deg, rgba(22, 78, 99, 0.78), rgba(8, 47, 73, 0.92))"
                color = "#dff8ff"
            else:
                align = "center"
                bubble = "linear-gradient(135deg, rgba(30, 41, 59, 0.92), rgba(15, 23, 42, 0.96))"
                color = "#cbd5e1"
            html.append(
                "<div style='margin:6px 0; text-align:%s;'>"
                "<span style='display:inline-block; max-width:92%%; padding:8px 10px; "
                "border-radius:14px; background:%s; border:1px solid rgba(125,211,252,0.18); "
                "box-shadow:0 10px 24px rgba(2,6,23,0.26); color:%s; font-size:12px; line-height:1.5;'>%s</span>"
                "</div>"
                % (align, bubble, color, escape(item.text))
            )
        return "".join(html)

    def _refresh_conversation(self) -> None:
        html = []
        for item in self._conversation[-40:]:
            if item.role == "user":
                bubble = "#1d4ed8"
                title = "你"
            elif item.role == "eyeMuse":
                bubble = "#0f766e"
                title = "EyeMuse"
            else:
                bubble = "#334155"
                title = "系统"
            html.append(
                f'<div style="margin:10px 0; padding:12px 14px; border-radius:16px; background:{bubble}; color:#f8fafc;">'
                f'<div style="font-size:11px; opacity:0.8; margin-bottom:6px;">{title} · {item.timestamp}</div>'
                f'<div style="font-size:14px; line-height:1.6;">{item.text}</div>'
                f'</div>'
            )
        self.conversation_view.setHtml("".join(html) or "<p style='color:#94a3b8;'>暂无消息</p>")
        self.conversation_view.verticalScrollBar().setValue(self.conversation_view.verticalScrollBar().maximum())
        if self._companion_window is not None:
            self._companion_window.set_chat_history_html(self._render_companion_chat_html())
        self._schedule_analytics_refresh()

    def _clear_conversation(self) -> None:
        self._conversation.clear()
        self._refresh_conversation()
        self._append_system_message("会话已清空。")

    def _sustained_companion_signal(self) -> str:
        behavior_state = str(self._behavior_summary.get("behavior_state", "warming"))
        if behavior_state == "fatigued":
            return "fatigue"
        if behavior_state == "anxious":
            return "soothe"

        now = time.monotonic()
        cutoff = now - _COMPANION_SIGNAL_SECONDS
        samples = [sample for sample in self._monitoring_samples if sample.captured_at >= cutoff]
        if len(samples) < 2:
            return "idle"

        sample_span = samples[-1].captured_at - samples[0].captured_at
        if sample_span < _COMPANION_SIGNAL_MIN_SECONDS:
            return "idle"

        sample_count = len(samples)
        fatigue_ratio = sum(sample.fatigue_score >= 70 for sample in samples) / sample_count
        stress_ratio = sum(sample.stress_score >= 70 for sample in samples) / sample_count
        focus_ratio = sum(
            sample.stress_score < 55 and sample.fatigue_score < 55
            for sample in samples
        ) / sample_count

        if fatigue_ratio >= _COMPANION_SIGNAL_RATIO:
            return "fatigue"
        if stress_ratio >= _COMPANION_SIGNAL_RATIO:
            return "soothe"
        if (
            self._local_camera_enabled
            and self._face_count > 0
            and focus_ratio >= _COMPANION_SIGNAL_RATIO
        ):
            return "focus"
        return "idle"

    def _stable_companion_mode(self) -> str:
        now = time.monotonic()
        signal_mode = self._sustained_companion_signal()
        current_mode = self._current_companion_mode

        if signal_mode == current_mode:
            self._companion_candidate_mode = current_mode
            self._companion_candidate_since = now
            return current_mode

        if signal_mode != self._companion_candidate_mode:
            self._companion_candidate_mode = signal_mode
            self._companion_candidate_since = now
            return current_mode

        confirmation_seconds = _COMPANION_FOCUS_CONFIRM_SECONDS
        if signal_mode in {"fatigue", "soothe"}:
            confirmation_seconds = _COMPANION_RISK_CONFIRM_SECONDS

        if current_mode in {"fatigue", "soothe"} and signal_mode not in {"fatigue", "soothe"}:
            confirmation_seconds = _COMPANION_RECOVERY_CONFIRM_SECONDS
        elif current_mode == "rest":
            confirmation_seconds = _COMPANION_BUBBLE_DISPLAY_MS / 1000.0

        if now - self._companion_candidate_since < confirmation_seconds:
            return current_mode

        self._companion_mode_since = now
        self._companion_candidate_mode = signal_mode
        self._companion_candidate_since = now
        return signal_mode

    def _build_companion_feedback(self) -> dict[str, str]:
        if self._rest_timer.isActive():
            mode = "rest"
        else:
            mode = self._stable_companion_mode()

        if mode == "rest":
            return {
                "mode": "rest",
                "bubble": "",
                "marquee": "休息中，安心放松" if self._rest_timer.isActive() else "",
                "show_bubble": False,
            }

        if mode == "focus":
            return {
                "mode": "focus",
                "bubble": "专注模式已开启，我会尽量降低打扰，只保留必要提醒。",
                "marquee": "正在专注中，不打扰",
                "show_bubble": True,
            }

        if mode == "fatigue":
            urgent = self._fatigue_score >= 85
            bubble = "你连续工作有一阵了，眼睛需要休息哦。要不要我带你做 30 秒深呼吸？"
            if urgent:
                bubble = "检测到你已经很疲惫了，我建议立刻离屏休息，也可以切到轻快提神模式。"
            return {
                "mode": "fatigue",
                "bubble": bubble,
                "marquee": "疲劳干预中，记得休息一下" if urgent else "",
                "show_bubble": True,
            }

        if mode == "soothe":
            return {
                "mode": "soothe",
                "bubble": "感觉你有点烦躁，需要我陪你聊聊天吗？我也可以先讲个轻松的小笑话。",
                "marquee": "",
                "show_bubble": True,
            }

        return {
            "mode": "idle",
            "bubble": "",
            "marquee": "",
            "show_bubble": False,
        }

    def _auto_companion_action_text(self, mode: str) -> str:
        if mode == "focus":
            return "专注守护已自动开启，我会尽量降低非必要打扰。"
        if mode == "fatigue":
            if self._fatigue_score >= 85:
                return "已自动切到紧急提神建议：先离屏活动 30 秒，再决定是否继续工作。"
            return "已自动触发疲劳干预，建议先跟着我做 3 轮深呼吸。"
        if mode == "soothe":
            return "已自动切到情绪安抚模式，需要的话我可以继续陪你聊天或讲个轻松笑话。"
        return ""

    def _refresh_companion_feedback(self) -> None:
        if self._companion_window is None:
            return
        payload = self._build_companion_feedback()
        previous_mode = self._current_companion_mode
        self._current_companion_mode = payload["mode"]
        self._companion_window.setMood(self._current_pet_mood)
        self._companion_window.set_camera_enabled(self._local_camera_enabled)
        self._companion_window.set_companion_feedback(
            mode_key=payload["mode"],
            bubble_text=payload["bubble"],
            marquee_text=payload["marquee"],
            show_bubble=bool(payload.get("show_bubble", True)) and payload["mode"] != previous_mode,
            auto_hide_ms=_COMPANION_BUBBLE_DISPLAY_MS if payload["mode"] != previous_mode else 0,
        )
        if payload["mode"] != previous_mode:
            auto_text = self._auto_companion_action_text(payload["mode"])
            if auto_text:
                self._show_companion_message(auto_text)

    def _show_companion_message(self, text: str) -> None:
        self._current_pet_hint = text.strip() or self._current_pet_hint
        if self._companion_window is not None:
            self._companion_window.set_companion_feedback(
                mode_key=self._current_companion_mode,
                bubble_text=self._current_pet_hint,
                marquee_text="正在专注中，不打扰" if self._current_companion_mode == "focus" else "",
                show_bubble=True,
                auto_hide_ms=_COMPANION_BUBBLE_DISPLAY_MS,
            )
        if hasattr(self, "event_card"):
            self.event_card.setValue(self._current_pet_hint)
        self.statusBar().showMessage(self._current_pet_hint, 3200)

    def _start_rest(self, duration_minutes: int) -> None:
        if self._rest_timer.isActive():
            return
        duration_minutes = max(1, min(60, int(duration_minutes)))
        self._rest_duration_seconds = duration_minutes * 60
        self._rest_started_at = time.monotonic()
        self._current_companion_mode = "rest"
        self._companion_mode_since = self._rest_started_at
        self._companion_candidate_mode = "rest"
        self._companion_candidate_since = self._rest_started_at
        self._rest_timer.start()
        if self._companion_window is not None:
            self._companion_window.set_rest_progress(True, self._rest_duration_seconds)
        self._show_companion_message(
            f"休息计时已开始。接下来的 {duration_minutes} 分钟先离开屏幕、活动一下，到时我会提醒你。"
        )

    def _update_rest_countdown(self) -> None:
        if self._rest_duration_seconds <= 0:
            self._rest_timer.stop()
            return
        elapsed_seconds = time.monotonic() - self._rest_started_at
        remaining_seconds = max(0, math.ceil(self._rest_duration_seconds - elapsed_seconds))
        if self._companion_window is not None:
            self._companion_window.set_rest_progress(True, remaining_seconds)
        if remaining_seconds <= 0:
            self._finish_rest()

    def _finish_rest(self) -> None:
        if self._rest_duration_seconds <= 0:
            return
        completed_duration = self._rest_duration_seconds
        self._rest_timer.stop()
        self._rest_duration_seconds = 0
        self._rest_started_at = 0.0
        if self._companion_window is not None:
            self._companion_window.set_rest_progress(False)

        duration_minutes = max(1, round(completed_duration / 60))
        completion_text = (
            f"{duration_minutes} 分钟休息完成，辛苦啦。先眨眨眼、活动肩颈，再决定是否继续当前任务。"
        )
        self._append_system_message(completion_text)
        self._show_companion_message(completion_text)

        snapshot = self._build_realtime_snapshot()
        if self._dashboard_repository is not None and snapshot is not None:
            try:
                self._dashboard_repository.record_rest_activity(
                    snapshot,
                    duration_seconds=completed_duration,
                )
            except Exception as exc:
                self.statusBar().showMessage(f"休息已完成，但统计记录失败：{exc}", 5000)
            else:
                self._dashboard_payload_key = ""
                self._report_payload_key = ""
                self._schedule_analytics_refresh()

    def _handle_companion_chat_submit(self, text: str) -> None:
        self._submit_user_text(text, from_companion=True)
        if self._companion_window is not None:
            self._companion_window.set_chat_busy(self._llm_thread is not None)

    def _handle_companion_toolbar_action(self, action: str) -> None:
        if action in {"home", "dashboard", "report"}:
            self._restore_from_companion_mode()
            self._switch_page(action)
            return
        if action == "camera":
            self._toggle_camera_from_companion()
            return
        if action == "exit":
            self.close()
            return

        if action == "chat":
            if self._companion_window is not None:
                self._companion_window.toggle_chat_panel()

    def _should_show_companion(self) -> bool:
        return hasattr(self, "keep_companion_checkbox") and self.keep_companion_checkbox.isChecked()

    def _handle_companion_presence_toggle(self, state: int) -> None:
        show_companion = Qt.CheckState(state) == Qt.CheckState.Checked
        if show_companion:
            companion_window = self._ensure_companion_window()
            self._place_companion_window()
            companion_window.show()
            companion_window.raise_()
        elif self._companion_window is not None and self._companion_window.isVisible():
            self._companion_window.hide()
        self._update_companion_controls()
        message = "桌宠已显示" if show_companion else "桌宠已关闭"
        self.statusBar().showMessage(message, 2500)

    def _update_companion_controls(self) -> None:
        if not hasattr(self, "minimize_companion_button"):
            return
        show_companion = self._should_show_companion()
        self.minimize_companion_button.setText("最小化并保留桌宠" if show_companion else "普通最小化")

    def _ensure_companion_window(self) -> CompanionPetWindow:
        if self._companion_window is None:
            self._companion_window = CompanionPetWindow()
            self._companion_window.toolbar_action_requested.connect(self._handle_companion_toolbar_action)
            self._companion_window.chat_submitted.connect(self._handle_companion_chat_submit)
            self._companion_window.rest_requested.connect(self._start_rest)
        self._companion_window.setMood(self._current_pet_mood)
        self._companion_window.set_camera_enabled(self._local_camera_enabled)
        self._companion_window.set_chat_busy(self._llm_thread is not None)
        self._companion_window.set_chat_history_html(self._render_companion_chat_html())
        self._refresh_companion_feedback()
        return self._companion_window

    def _place_companion_window(self) -> None:
        companion_window = self._ensure_companion_window()
        if companion_window.isVisible():
            return
        screen = QApplication.primaryScreen()
        if screen is None:
            return
        available_geometry = screen.availableGeometry()
        companion_window.move(
            available_geometry.right() - companion_window.width() - 28,
            available_geometry.bottom() - companion_window.height() - 56,
        )

    def _enter_default_companion_mode(self) -> None:
        if _env_flag("EYEMUSE_AUTOSTART_CAMERA", False):
            self._start_camera()
        if self._should_show_companion():
            companion_window = self._ensure_companion_window()
            self._place_companion_window()
            companion_window.show()
            companion_window.raise_()
        self._update_companion_controls()

    def _enter_companion_mode(self) -> None:
        if not self._should_show_companion():
            if self._companion_window is not None:
                self._companion_window.hide()
            self._update_companion_controls()
            self.showMinimized()
            self.statusBar().showMessage("已普通最小化，桌宠不会保留", 2500)
            return

        companion_window = self._ensure_companion_window()
        self._place_companion_window()
        companion_window.show()
        companion_window.raise_()
        self._update_companion_controls()
        self.hide()

    def _restore_from_companion_mode(self) -> None:
        if self._companion_window is not None and not self._should_show_companion():
            self._companion_window.hide()
        self._update_companion_controls()
        self.showNormal()
        self.raise_()
        self.activateWindow()

    def _toggle_camera_from_companion(self) -> None:
        if self._local_camera_enabled:
            self._stop_camera()
            return
        self._start_camera()

    def _set_camera_toggle_checked(self, checked: bool) -> None:
        self.camera_toggle.blockSignals(True)
        self.camera_toggle.setChecked(checked)
        self.camera_toggle.blockSignals(False)

    def _toggle_camera(self, state: int) -> None:
        if Qt.CheckState(state) == Qt.CheckState.Checked:
            self._start_camera()
        else:
            self._stop_camera()

    def _start_camera(self) -> None:
        if self._camera_worker is not None:
            return

        camera_thread = QThread(self)
        camera_worker = CameraWorker()
        camera_worker.moveToThread(camera_thread)
        camera_thread.started.connect(camera_worker.start)
        camera_worker.frame_ready.connect(self._update_camera_frame)
        camera_worker.status_changed.connect(self._update_camera_status)
        camera_worker.face_count_changed.connect(self._update_face_count)
        camera_worker.analysis_changed.connect(self._update_analysis_metrics)
        camera_worker.open_failed.connect(self._handle_camera_open_failed)
        self.camera_stop_requested.connect(camera_worker.stop)
        camera_worker.finished.connect(camera_thread.quit, Qt.DirectConnection)
        camera_thread.finished.connect(camera_worker.deleteLater)
        camera_thread.finished.connect(self._handle_camera_thread_finished)
        self._camera_worker = camera_worker
        self._camera_thread = camera_thread
        camera_thread.start()

        self._local_camera_enabled = True
        self._set_camera_toggle_checked(True)
        self.camera_card.setValue("开启中")
        self.camera_status.setText("开启中")
        self.camera_note.setText("正在尝试连接本地摄像头，若失败会回退到提示状态。")
        self.stress_card.setValue("校准中")
        self.fatigue_card.setValue("校准中")
        self.analysis_card.setValue("正在初始化分析链路")
        self.heart_rate_card.setValue("Collecting...")
        self.respiration_card.setValue("Collecting...")
        self.hrv_card.setValue("Collecting...")
        self._set_mood(PetMood.listening, "正在观察摄像头状态。")

    def _stop_camera(self) -> None:
        camera_thread = self._camera_thread
        if self._camera_worker is not None and camera_thread is not None and camera_thread.isRunning():
            self.camera_stop_requested.emit()
            if not camera_thread.wait(2500):
                camera_thread.requestInterruption()
                camera_thread.quit()
                camera_thread.wait(500)
        self._camera_worker = None
        self._camera_thread = None
        self._set_camera_toggle_checked(False)
        self._reset_camera_ui()
        self._set_mood(PetMood.offline, "摄像头已关闭，当前处于离线状态。")
        self._schedule_analytics_refresh()

    def _handle_camera_open_failed(self, message: str) -> None:
        self._reset_camera_ui(status=message, inline_status="异常", note=message)
        self._set_mood(PetMood.alert, message)
        self._set_camera_toggle_checked(False)

    def _handle_camera_thread_finished(self) -> None:
        finished_thread = self.sender()
        if finished_thread is not self._camera_thread:
            return
        self._camera_worker = None
        self._camera_thread = None
        if self._local_camera_enabled:
            self._reset_camera_ui(
                status="连接中断",
                inline_status="异常",
                note="摄像头分析线程已停止，可关闭后重新开启。",
            )
            self._set_camera_toggle_checked(False)

    def _reset_camera_ui(
        self,
        *,
        status: str = "关闭",
        inline_status: str = "关闭",
        note: Optional[str] = None,
    ) -> None:
        self._local_camera_enabled = False
        self._face_count = 0
        self._stress_score = 0
        self._fatigue_score = 0
        self._dominant_signal = "none"
        self._calibration_state = "waiting"
        self._heart_rate = None
        self._respiration_rate = None
        self._hrv = None
        self._monitoring_samples.clear()
        self.camera_preview.setPixmap(QPixmap())
        self.camera_preview.setText("摄像头未开启")
        self.camera_card.setValue(status)
        self.camera_status.setText(inline_status)
        self.camera_note.setText(note or "权限提示、失败提示和降级路径都先保留在界面上。")
        self.face_card.setValue("0 个面部")
        self.stress_card.setValue("未开始检测")
        self.fatigue_card.setValue("未开始检测")
        self.analysis_card.setValue("等待开始")
        self.heart_rate_card.setValue("-- bpm")
        self.respiration_card.setValue("-- rpm")
        self.hrv_card.setValue("-- ms")
        self._refresh_companion_feedback()

    def _update_camera_frame(self, image: QImage) -> None:
        if not self.isVisible() or self.isMinimized():
            return
        pixmap = QPixmap.fromImage(image)
        scaled = pixmap.scaled(self.camera_preview.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.camera_preview.setPixmap(scaled)
        self.camera_preview.setText("")

    def _update_camera_status(self, message: str) -> None:
        self.camera_status.setText("异常" if any(flag in message for flag in ("失败", "无法", "异常", "不可用")) else "开启")
        self.camera_card.setValue(message)
        self.camera_note.setText(message)
        self.event_card.setValue(message)
        if any(flag in message for flag in ("失败", "无法", "异常", "不可用")):
            self._set_mood(PetMood.alert, message)
        self._schedule_analytics_refresh()

    def _update_face_count(self, count: int) -> None:
        if count == self._face_count:
            return
        self._face_count = count
        self.face_card.setValue(f"{count} 个面部")
        self._schedule_analytics_refresh()

    def _update_analysis_metrics(self, payload: object) -> None:
        if not isinstance(payload, dict):
            return

        face_count = int(payload.get("face_count", self._face_count))
        stress_score = int(payload.get("stress_score", 0))
        fatigue_score = int(payload.get("fatigue_score", 0))
        dominant_signal = str(payload.get("dominant_signal", "none"))
        calibration_state = str(payload.get("calibration_state", "waiting"))
        calibration_progress = float(payload.get("calibration_progress", 0.0))
        heart_rate_raw = payload.get("heart_rate")
        respiration_rate_raw = payload.get("respiration_rate")
        hrv_raw = payload.get("hrv")
        rppg_progress = float(payload.get("rppg_progress", 0.0))
        heart_rate = float(heart_rate_raw) if isinstance(heart_rate_raw, (int, float)) else None
        respiration_rate = float(respiration_rate_raw) if isinstance(respiration_rate_raw, (int, float)) else None
        hrv = float(hrv_raw) if isinstance(hrv_raw, (int, float)) else None

        self._face_count = face_count
        self._stress_score = stress_score
        self._fatigue_score = fatigue_score
        self._dominant_signal = dominant_signal
        self._calibration_state = calibration_state
        self._heart_rate = heart_rate
        self._respiration_rate = respiration_rate
        self._hrv = hrv
        self._record_monitoring_sample(
            face_count=face_count,
            calibration_state=calibration_state,
            stress_score=stress_score,
            fatigue_score=fatigue_score,
            heart_rate=heart_rate,
            respiration_rate=respiration_rate,
            hrv=hrv,
        )
        self.face_card.setValue(f"{face_count} 个面部")
        self.stress_card.setValue(f"{stress_score} / 100")
        self.fatigue_card.setValue(f"{fatigue_score} / 100")
        self.heart_rate_card.setValue(f"{heart_rate:.0f} bpm" if heart_rate is not None else f"Collecting {int(rppg_progress * 100)}%")
        self.respiration_card.setValue(f"{respiration_rate:.0f} rpm" if respiration_rate is not None else "-- rpm")
        self.hrv_card.setValue(f"{hrv:.0f} ms" if hrv is not None else "-- ms")

        rppg_parts = []
        if heart_rate is not None:
            rppg_parts.append(f"HR {heart_rate:.0f} bpm")
        if respiration_rate is not None:
            rppg_parts.append(f"Resp {respiration_rate:.0f} rpm")
        if hrv is not None:
            rppg_parts.append(f"HRV {hrv:.0f} ms")
        rppg_summary = " | ".join(rppg_parts)
        monitoring_averages = self._monitoring_averages()
        monitoring_mood, monitoring_hint = self._resolve_monitoring_mood(
            face_count=face_count,
            calibration_state=calibration_state,
            averages=monitoring_averages,
        )
        monitoring_mood, monitoring_hint = self._apply_behavior_signal_to_mood(monitoring_mood, monitoring_hint)

        if calibration_state == "calibrating":
            analysis_text = f"校准中 {int(calibration_progress * 100)}%"
            if heart_rate is None and rppg_progress > 0.0:
                analysis_text += f" | rPPG {int(rppg_progress * 100)}%"
            self.analysis_card.setValue(analysis_text)
            self.camera_note.setText(f"正在进行中性面部校准：{analysis_text}")
            self.event_card.setValue(analysis_text)
        elif calibration_state == "ready":
            if dominant_signal != "none":
                analysis_text = f"已就绪 · 主信号 {dominant_signal}"
            else:
                analysis_text = "已就绪"
            if rppg_summary:
                analysis_text = f"{analysis_text} | {rppg_summary}"
            self.analysis_card.setValue(analysis_text)
            if rppg_summary:
                self.camera_note.setText(f"压力 {stress_score}，疲劳 {fatigue_score} | {rppg_summary}")
                self.event_card.setValue(f"压力 {stress_score} / 疲劳 {fatigue_score} | {rppg_summary}")
            else:
                self.camera_note.setText(f"压力 {stress_score}，疲劳 {fatigue_score}。")
                self.event_card.setValue(f"压力 {stress_score} / 疲劳 {fatigue_score}")
        elif calibration_state == "unavailable":
            self.analysis_card.setValue("基础检测模式")
            self.camera_note.setText("已回退到基础面部框选，压力/疲劳分析未启用。")
            self.event_card.setValue("基础面部框选")
        else:
            self.analysis_card.setValue("等待面部")
            if face_count > 0:
                if heart_rate is None and rppg_progress > 0.0:
                    self.camera_note.setText(f"已检测到人脸，等待 MediaPipe 关键点稳定后开始分析。rPPG {int(rppg_progress * 100)}%")
                    self.event_card.setValue(f"rPPG collecting {int(rppg_progress * 100)}%")
                else:
                    self.camera_note.setText("已检测到人脸，等待 MediaPipe 关键点稳定后开始分析。")
                    self.event_card.setValue("已检测到人脸，等待关键点稳定")
            else:
                self.camera_note.setText("等待检测到面部后开始分析。")
                self.event_card.setValue("等待面部")
        self._set_mood(monitoring_mood, monitoring_hint)
        self._schedule_analytics_refresh()

    def _current_summary(self) -> str:
        camera_state = "开启" if self._local_camera_enabled else "关闭"
        summary = f"摄像头 {camera_state}，检测到 {self._face_count} 个面部。"
        if self._heart_rate is not None:
            summary += f" HR {self._heart_rate:.0f} bpm."
        if self._respiration_rate is not None:
            summary += f" Resp {self._respiration_rate:.0f} rpm."
        if self._hrv is not None:
            summary += f" HRV {self._hrv:.0f} ms."
        if bool(self._behavior_summary.get("available")):
            summary += " " + self._behavior_hint_fragment()
        return summary

    @staticmethod
    def _now() -> str:
        return QDateTime.currentDateTime().toString("hh:mm:ss")

    def closeEvent(self, event) -> None:  # noqa: N802
        self._rest_timer.stop()
        if self._llm_thread is not None:
            self._teardown_streaming_reply()
        if self._activity_monitor is not None:
            self._activity_monitor.stop()
        if self._companion_window is not None:
            self._companion_window.close()
        self._stop_camera()
        super().closeEvent(event)

    def _apply_theme(self) -> None:
        apply_modern_theme(self)

    def _build_dashboard_chart_html(self) -> str:
        return build_dashboard_chart_html()

    def _build_dashboard_fallback_html(self, payload: dict) -> str:
        return build_dashboard_fallback_html(payload)

    def _render_companion_chat_html(self) -> str:
        return build_companion_chat_html(self._conversation)

    def _refresh_conversation(self) -> None:
        self.conversation_view.setHtml(build_main_conversation_html(self._conversation))
        self.conversation_view.verticalScrollBar().setValue(self.conversation_view.verticalScrollBar().maximum())
        if self._companion_window is not None:
            self._companion_window.set_chat_history_html(self._render_companion_chat_html())
        self._schedule_analytics_refresh()


def _select_font() -> None:
    if sys.platform.startswith("win"):
        QFontDatabase.addApplicationFont("C:/Windows/Fonts/segoeui.ttf")


def run() -> int:
    app = QApplication.instance() or QApplication(sys.argv)
    app.setQuitOnLastWindowClosed(True)
    _enable_crash_logging()
    _select_font()
    window = EyeMuseWindow()
    window.show()
    return app.exec()
