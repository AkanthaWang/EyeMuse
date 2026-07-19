from __future__ import annotations
from collections import deque
from pathlib import Path
from dataclasses import dataclass
from enum import Enum
from html import escape
import json
import math
import threading
import time
from typing import Optional
import sys

import cv2
from PySide6.QtCore import QDate, QDateTime, QObject, QPoint, QThread, QTimer, Qt, QUrl, Signal, Property, QSize, Slot
from PySide6.QtGui import QColor, QFont, QFontDatabase, QImage, QMovie, QPainter, QPixmap, QPalette
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
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
    QStackedWidget,
    QTextBrowser,
    QVBoxLayout,
    QWidget,
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
_ACTIVITY_PERIOD_SECONDS = 30.0
_ACTIVITY_BASELINE_PERIODS = 10
_ACTIVITY_SWITCH_WINDOW_SECONDS = 2.0


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

    def __init__(self, camera_index: int = 0) -> None:
        super().__init__()
        self._camera_index = camera_index
        self._capture: Optional[cv2.VideoCapture] = None
        self._timer = QTimer(self)
        self._timer.setInterval(33)
        self._timer.timeout.connect(self._read_frame)
        self._analyzer = None
        self._detector = None

    @Slot()
    def start(self) -> None:
        if not self._open_capture():
            return

        if YOLOFaceDetector is not None:
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
        self._timer.stop()
        if self._analyzer is not None and hasattr(self._analyzer, "close"):
            self._analyzer.close()
        self._analyzer = None
        if self._detector is not None and hasattr(self._detector, "close"):
            self._detector.close()
        self._detector = None
        self._cleanup_capture()
        self.status_changed.emit("摄像头已关闭")

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
                "background: rgba(17, 24, 39, 0.92);"
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

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setWindowFlags(Qt.FramelessWindowHint | Qt.Tool | Qt.WindowStaysOnTopHint)
        self.setAttribute(Qt.WA_TranslucentBackground, True)
        self.setObjectName("CompanionPetWindow")
        self.setToolTip("左键拖动，鼠标中键展开下方工具栏")
        self._compact_size = QSize(280, 314)
        self._expanded_size = QSize(280, 338)
        self._chat_expanded_size = QSize(300, 452)
        self.setFixedSize(self._compact_size)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(1)
        self.bubble_label = QLabel("陪伴模式已开启")
        self.bubble_label.setWordWrap(True)
        self.bubble_label.setAlignment(Qt.AlignCenter)
        self.bubble_label.setMinimumHeight(52)
        self.bubble_label.setStyleSheet(
            "background: rgba(255, 255, 255, 0.94);"
            "color: #334155;"
            "border: 1px solid rgba(226, 232, 240, 0.95);"
            "border-radius: 16px;"
            "padding: 10px 12px;"
            "font-size: 12px;"
            "font-weight: 600;"
        )
        self.bubble_label.hide()
        layout.addWidget(self.bubble_label)

        self.marquee_label = QLabel("")
        self.marquee_label.setAlignment(Qt.AlignCenter)
        self.marquee_label.setStyleSheet(
            "color: #0f172a;"
            "background: rgba(14, 165, 233, 0.22);"
            "border: 1px solid rgba(125, 211, 252, 0.38);"
            "border-radius: 12px;"
            "padding: 4px 10px;"
            "font-size: 11px;"
            "font-weight: 700;"
        )
        self.marquee_label.hide()
        layout.addWidget(self.marquee_label)

        self._marquee_source = ""
        self._marquee_index = 0
        self._marquee_timer = QTimer(self)
        self._marquee_timer.setInterval(160)
        self._marquee_timer.timeout.connect(self._advance_marquee)
        self._bubble_timer = QTimer(self)
        self._bubble_timer.setSingleShot(True)
        self._bubble_timer.timeout.connect(self.bubble_label.hide)

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
            "background: rgba(255, 255, 255, 0.96);"
            "border: 1px solid rgba(125, 211, 252, 0.42);"
            "border-radius: 18px;"
        )
        chat_layout = QVBoxLayout(self.chat_frame)
        chat_layout.setContentsMargins(10, 9, 10, 10)
        chat_layout.setSpacing(6)
        self.chat_hint_label = QLabel("和 EyeMuse 说句话")
        self.chat_hint_label.setAlignment(Qt.AlignCenter)
        self.chat_hint_label.setStyleSheet(
            "color: #64748b;"
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
            "color: #334155;"
            "font-size: 12px;"
        )
        self.chat_view.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.chat_view.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.chat_view.setHtml("<p style='color:#94a3b8; text-align:center;'>点一下“对话”，就在这里聊天。</p>")
        self.chat_input = QLineEdit()
        self.chat_input.setPlaceholderText("直接输入文字，和 EyeMuse 对话")
        self.chat_input.setStyleSheet(
            "background: rgba(248,250,252,1.0);"
            "border: 1px solid rgba(148,163,184,0.28);"
            "border-radius: 14px;"
            "padding: 8px 12px;"
            "color: #0f172a;"
            "font-size: 12px;"
        )
        self.chat_input.returnPressed.connect(self._emit_chat_submit)
        self.chat_send_button = QPushButton("说")
        self.chat_send_button.setCursor(Qt.PointingHandCursor)
        self.chat_send_button.setMinimumHeight(30)
        self.chat_send_button.setStyleSheet(
            "QPushButton {"
            "background: rgba(56, 189, 248, 0.95);"
            "border: none;"
            "border-radius: 14px;"
            "color: #082f49;"
            "font-size: 12px;"
            "font-weight: 700;"
            "padding: 6px 14px;"
            "}"
            "QPushButton:hover {"
            "background: rgba(125, 211, 252, 1.0);"
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
            "background: rgba(255, 255, 255, 0.92);"
            "border-radius: 14px;"
            "padding: 1px;"
        )
        toolbar_layout = QHBoxLayout(self.toolbar_frame)
        toolbar_layout.setContentsMargins(8, 3, 8, 3)
        toolbar_layout.setSpacing(4)
        self._toolbar_buttons: dict[str, QPushButton] = {}
        for action_key, label in (
            ("home", "首页"),
            ("chat", "对话"),
            ("camera", "摄像头"),
            ("exit", "退出"),
        ):
            button = QPushButton(label)
            button.setCursor(Qt.PointingHandCursor)
            button.setMinimumHeight(22)
            button.setStyleSheet(
                "QPushButton {"
                "background: rgba(255,255,255,0.0);"
                "border: none;"
                "border-radius: 10px;"
                "color: #94a3b8;"
                "font-size: 11px;"
                "font-weight: 700;"
                "padding: 4px 8px;"
                "}"
                "QPushButton:hover {"
                "background: rgba(148,163,184,0.14);"
                "color: #475569;"
                "}"
            )
            if action_key == "chat":
                button.clicked.connect(self.toggle_chat_panel)
            else:
                button.clicked.connect(lambda _checked=False, key=action_key: self.toolbar_action_requested.emit(key))
            toolbar_layout.addWidget(button)
            self._toolbar_buttons[action_key] = button
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
        self._current_mode = mode_key
        normalized_text = bubble_text.strip()
        if show_bubble and normalized_text:
            self.bubble_label.setText(normalized_text)
            self.bubble_label.show()
            if auto_hide_ms > 0:
                self._bubble_timer.start(auto_hide_ms)
            else:
                self._bubble_timer.stop()
        else:
            self._bubble_timer.stop()
            self.bubble_label.hide()

        if marquee_text.strip():
            padded = f"    {marquee_text.strip()}    "
            self._marquee_source = padded + padded
            self._marquee_index = 0
            self.marquee_label.setText(marquee_text.strip())
            self.marquee_label.show()
            self._marquee_timer.start()
        else:
            self._marquee_timer.stop()
            self._marquee_source = ""
            self.marquee_label.hide()

        bubble_styles = {
            "focus": (
                "background: rgba(219, 234, 254, 0.96);"
                "color: #0f172a;"
                "border: 1px solid rgba(125, 211, 252, 0.95);"
            ),
            "fatigue": (
                "background: rgba(255, 237, 213, 0.96);"
                "color: #7c2d12;"
                "border: 1px solid rgba(251, 191, 36, 0.95);"
            ),
            "soothe": (
                "background: rgba(243, 232, 255, 0.96);"
                "color: #581c87;"
                "border: 1px solid rgba(196, 181, 253, 0.95);"
            ),
        }
        bubble_style = bubble_styles.get(
            mode_key,
            "background: rgba(255, 255, 255, 0.94);"
            "color: #334155;"
            "border: 1px solid rgba(226, 232, 240, 0.95);",
        )
        self.bubble_label.setStyleSheet(
            bubble_style
            + "border-radius: 16px;"
            + "padding: 10px 12px;"
            + "font-size: 12px;"
            + "font-weight: 600;"
        )

    def _advance_marquee(self) -> None:
        if not self._marquee_source:
            return
        window_size = 22
        source = self._marquee_source
        rendered = source[self._marquee_index:self._marquee_index + window_size]
        if len(rendered) < window_size:
            rendered += source[:window_size - len(rendered)]
        self.marquee_label.setText(rendered)
        self._marquee_index = (self._marquee_index + 1) % max(1, len(source) // 2)

    def _toggle_toolbar(self) -> None:
        visible = not self.toolbar_frame.isVisible()
        self.toolbar_frame.setVisible(visible)
        if not visible:
            self.chat_frame.hide()
        self._apply_window_size()

    def _apply_window_size(self) -> None:
        if self.chat_frame.isVisible():
            self.setFixedSize(self._chat_expanded_size)
        elif self.toolbar_frame.isVisible():
            self.setFixedSize(self._expanded_size)
        else:
            self.setFixedSize(self._compact_size)

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
        layout = QVBoxLayout(self)
        layout.setContentsMargins(14, 12, 14, 12)
        layout.setSpacing(3)
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
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("EyeMuse")
        self.resize(1420, 920)
        self.setMinimumSize(1180, 760)

        self._camera_worker: Optional[CameraWorker] = None
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
        self._streaming_reply_index: Optional[int] = None
        self._streaming_user_text: str = ""
        self._current_pet_mood = PetMood.idle
        self._current_pet_hint = "等待用户输入，或开启摄像头观察状态变化。"
        self._current_companion_mode = "idle"
        self._companion_window: Optional[CompanionPetWindow] = None

        self._build_ui()
        self._start_activity_monitor()
        self._apply_theme()
        self._refresh_dashboard_page()
        self._refresh_report_page()
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
        self._refresh_dashboard_page()
        self._refresh_report_page()

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
        frame.setObjectName("NavBar")
        layout = QHBoxLayout(frame)
        layout.setContentsMargins(8, 1, 8, 1)
        layout.setSpacing(0)

        logo_path = Path(__file__).resolve().parents[1] / "assets" / "logo.png"
        self.nav_logo_label = QLabel()
        self.nav_logo_label.setObjectName("NavLogo")
        self.nav_logo_label.setFixedWidth(320)
        self.nav_logo_label.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)
        logo_pixmap = QPixmap(str(logo_path))
        if not logo_pixmap.isNull():
            logo_pixmap = _make_dark_background_transparent(logo_pixmap)
            self.nav_logo_label.setPixmap(logo_pixmap.scaledToHeight(64, Qt.SmoothTransformation))
        else:
            self.nav_logo_label.setText("EyeMuse")

        self.nav_right_spacer = QWidget()
        self.nav_right_spacer.setFixedWidth(320)

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
        layout.addWidget(self.nav_right_spacer, 0, Qt.AlignRight | Qt.AlignVCenter)
        return frame

    def _build_home_page(self) -> QWidget:
        page = QWidget()
        page.setObjectName("Page")
        root_layout = QGridLayout(page)
        root_layout.setContentsMargins(20, 20, 20, 20)
        root_layout.setHorizontalSpacing(18)
        root_layout.setVerticalSpacing(18)

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
        effect.setBlurRadius(30)
        effect.setOffset(0, 10)
        effect.setColor(QColor(0, 0, 0, 90))
        frame.setGraphicsEffect(effect)
        return frame

    def _build_left_panel(self) -> QFrame:
        frame = self._panel_frame()
        layout = QVBoxLayout(frame)
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(16)

        title = QLabel("EyeMuse 桌面宠物")
        title.setObjectName("Title")
        subtitle = QLabel("感知、陪伴、提醒，尽量都留在本地")
        subtitle.setObjectName("Subtitle")

        self.avatar = PetAvatar()
        self.avatar.setObjectName("Avatar")

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
        status_row.addWidget(self.mood_badge)
        status_row.addWidget(self.privacy_badge)
        status_row.addWidget(self.minimize_companion_button)
        status_row.addStretch(1)

        self.pet_hint = QLabel("等待用户输入，或开启摄像头观察状态变化。")
        self.pet_hint.setWordWrap(True)
        self.pet_hint.setObjectName("Hint")

        quick_row = QHBoxLayout()
        quick_row.setSpacing(10)
        self.listen_button = QPushButton("进入聆听")
        self.think_button = QPushButton("思考中")
        self.respond_button = QPushButton("回应一下")
        for button in (self.listen_button, self.think_button, self.respond_button):
            button.setCursor(Qt.PointingHandCursor)
            quick_row.addWidget(button)

        self.listen_button.clicked.connect(lambda: self._set_mood(PetMood.listening, "我在听，你可以继续说。"))
        self.think_button.clicked.connect(lambda: self._set_mood(PetMood.thinking, "我在整理你刚才说的话。"))
        self.respond_button.clicked.connect(lambda: self._set_mood(PetMood.responding, "准备给你一个更自然的回应。"))

        self.local_state_card = StatCard("隐私与保存", "默认仅保留本地会话、摄像头状态与键鼠行为摘要，不主动上传原始数据。")
        self.stress_card = StatCard("压力估计", "未开始检测")
        self.fatigue_card = StatCard("疲劳状态", "未开始检测")
        self.camera_card = StatCard("摄像头", "关闭")
        self.analysis_card = StatCard("分析状态", "等待开始")
        self.event_card = StatCard("最近事件", "等待开始")

        layout.addWidget(title)
        layout.addWidget(subtitle)
        layout.addWidget(self.avatar, 1)
        layout.addLayout(status_row)
        layout.addWidget(self.pet_hint)
        layout.addLayout(quick_row)
        layout.addWidget(self.local_state_card)
        layout.addWidget(self.analysis_card)
        layout.addWidget(self.event_card)
        return frame

    def _build_middle_panel(self) -> QFrame:
        frame = self._panel_frame()
        layout = QVBoxLayout(frame)
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(14)

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
        self.send_button.setCursor(Qt.PointingHandCursor)
        self.send_button.clicked.connect(self._handle_send)
        input_row.addWidget(self.message_input, 1)
        input_row.addWidget(self.send_button)

        action_row = QHBoxLayout()
        self.remind_button = QPushButton("提醒我休息")
        self.energy_button = QPushButton("查看状态")
        self.clear_button = QPushButton("清空对话")
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
        layout.setContentsMargins(18, 18, 18, 18)
        layout.setSpacing(8)

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
        self.face_card.value_label.setStyleSheet("font-size: 12px;")
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
        self.behavior_card.value_label.setStyleSheet("font-size: 12px;")
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
        root_layout.setContentsMargins(20, 20, 20, 20)
        root_layout.setSpacing(16)

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
        chart_layout.setContentsMargins(20, 20, 20, 20)
        chart_layout.setSpacing(10)
        if QWebEngineView is not None:
            self.dashboard_chart_view = QWebEngineView()
            self.dashboard_chart_view.setObjectName("ChartView")
            self.dashboard_chart_view.page().setBackgroundColor(QColor("#020617"))
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
        root_layout.setContentsMargins(20, 20, 20, 20)
        root_layout.setSpacing(18)

        report_grid = QGridLayout()
        report_grid.setHorizontalSpacing(18)
        report_grid.setVerticalSpacing(18)

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
        self.daily_report_view.setObjectName("OverviewPanel")
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
        self.weekly_report_view.setObjectName("OverviewPanel")
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

    def _sync_dashboard_repository(self) -> dict | None:
        if self._dashboard_repository is None:
            return None
        snapshot = self._build_realtime_snapshot()
        if snapshot is not None:
            self._dashboard_repository.record_runtime_snapshot(snapshot)
        return self._dashboard_repository.get_dashboard_payload()

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
                radial-gradient(circle at top left, rgba(34, 197, 94, 0.08), transparent 22%),
                radial-gradient(circle at top right, rgba(56, 189, 248, 0.12), transparent 28%),
                linear-gradient(180deg, rgba(2, 6, 23, 0.98), rgba(15, 23, 42, 0.96));
        }
        .board {
            width: 100%;
            height: 100%;
            padding: 20px;
            box-sizing: border-box;
            display: grid;
            grid-template-columns: 1.5fr 1.1fr 1.1fr;
            grid-template-rows: 1.2fr 1fr 96px;
            row-gap: 28px;
            column-gap: 20px;
        }
        .metric-card, .chart-card {
            background: rgba(15, 23, 42, 0.9);
            border: 1px solid rgba(56, 189, 248, 0.16);
            border-radius: 18px;
            box-shadow: inset 0 0 0 1px rgba(15, 23, 42, 0.36), 0 18px 38px rgba(2, 6, 23, 0.42);
        }
        .metric-row {
            grid-column: 1 / span 3;
            grid-row: 3;
            display: grid;
            grid-template-columns: repeat(6, minmax(0, 1fr));
            gap: 18px;
        }
        .metric-card {
            padding: 14px 18px;
            display: flex;
            flex-direction: column;
            justify-content: center;
        }
        .metric-label {
            color: #94a3b8;
            font-size: 12px;
            margin-bottom: 8px;
        }
        .metric-value {
            color: #f8fafc;
            font-size: 30px;
            font-weight: 700;
            letter-spacing: 0.5px;
        }
        .metric-sub {
            color: #38bdf8;
            font-size: 12px;
            margin-top: 6px;
        }
        .chart-card {
            position: relative;
            padding: 16px;
        }
        .chart-title {
            position: absolute;
            left: 22px;
            top: 18px;
            color: #f8fafc;
            font-size: 15px;
            font-weight: 700;
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
            background: rgba(2, 6, 23, 0.82);
            color: #cbd5e1;
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
            border: 1px solid rgba(248, 113, 113, 0.28);
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
        const palette = ['#38bdf8', '#f59e0b', '#22c55e', '#a78bfa', '#fb7185', '#14b8a6'];
        const textColor = '#cbd5e1';
        const axisColor = 'rgba(148, 163, 184, 0.28)';
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
                tooltip: { trigger: 'axis' },
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
                tooltip: { trigger: 'item' },
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
                        borderColor: '#0f172a',
                        borderWidth: 4,
                        borderRadius: 8,
                    },
                    data: emotionDistribution,
                }],
            }, true);

            window.dashboardCharts.barChart.setOption({
                animation: false,
                color: ['#22c55e'],
                tooltip: { trigger: 'axis', axisPointer: { type: 'shadow' } },
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
                    itemStyle: { borderRadius: [8, 8, 0, 0] },
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
                tooltip: { trigger: 'axis' },
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
                radial-gradient(circle at top left, rgba(34, 197, 94, 0.08), transparent 22%),
                radial-gradient(circle at top right, rgba(56, 189, 248, 0.12), transparent 28%),
                linear-gradient(180deg, rgba(2, 6, 23, 0.98), rgba(15, 23, 42, 0.96));
            color: #e2e8f0;
            font-family: "Segoe UI", "Microsoft YaHei", sans-serif;
        }}
        .board {{
            height: 100%;
            box-sizing: border-box;
            padding: 20px;
            display: grid;
            grid-template-columns: 1.5fr 1.1fr 1.1fr;
            grid-template-rows: 1.2fr 1fr 96px;
            gap: 20px;
        }}
        .card {{
            position: relative;
            background: rgba(15, 23, 42, 0.9);
            border: 1px solid rgba(56, 189, 248, 0.16);
            border-radius: 18px;
            box-shadow: inset 0 0 0 1px rgba(15, 23, 42, 0.36), 0 18px 38px rgba(2, 6, 23, 0.42);
            overflow: hidden;
        }}
        .title {{
            position: absolute;
            left: 22px;
            top: 18px;
            color: #f8fafc;
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
            color: #94a3b8;
            font-size: 12px;
            margin-bottom: 8px;
        }}
        .metric-value {{
            color: #f8fafc;
            font-size: 30px;
            font-weight: 700;
        }}
        .metric-sub {{
            color: #38bdf8;
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
        self._sync_dashboard_repository()
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
        self._sync_dashboard_repository()
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
        report_dir = Path(__file__).resolve().parents[2] / "reports"
        report_dir.mkdir(parents=True, exist_ok=True)
        return report_dir

    def _write_report_snapshot(self, file_name: str, content: str) -> Path:
        path = self._report_storage_dir() / file_name
        path.write_text(content, encoding="utf-8")
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

    def _apply_theme(self) -> None:
        self.setFont(QFont("Segoe UI Variable", 10))
        app = QApplication.instance()
        if app is not None:
            palette = QPalette()
            palette.setColor(QPalette.Window, QColor("#0f172a"))
            palette.setColor(QPalette.WindowText, QColor("#e2e8f0"))
            palette.setColor(QPalette.Base, QColor("#111827"))
            palette.setColor(QPalette.AlternateBase, QColor("#0b1120"))
            palette.setColor(QPalette.Text, QColor("#e2e8f0"))
            palette.setColor(QPalette.Button, QColor("#17213a"))
            palette.setColor(QPalette.ButtonText, QColor("#e2e8f0"))
            palette.setColor(QPalette.Highlight, QColor("#38bdf8"))
            palette.setColor(QPalette.HighlightedText, QColor("#0f172a"))
            app.setPalette(palette)

        self.setStyleSheet(
            """
            QMainWindow {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:1,
                    stop:0 #07111f, stop:0.5 #0f172a, stop:1 #111827);
            }
            #NavBar {
                background: rgba(15, 23, 42, 235);
                border-bottom: 1px solid rgba(148, 163, 184, 70);
            }
            QPushButton#NavButton {
                background: transparent;
                border: none;
                border-radius: 14px;
                color: #cbd5e1;
                padding: 10px 22px;
                min-height: 24px;
                font-weight: 700;
            }
            QPushButton#NavButton:hover {
                background: rgba(56, 189, 248, 20);
                color: #f8fafc;
            }
            QPushButton#NavButton:checked {
                color: #ffffff;
                background: rgba(56, 189, 248, 32);
                border: 1px solid rgba(56, 189, 248, 95);
            }
            #Page {
                background: transparent;
            }
            #Panel {
                background: rgba(15, 23, 42, 200);
                border: 1px solid rgba(148, 163, 184, 60);
                border-radius: 24px;
            }
            #SectionTitle {
                color: #f8fafc;
                font-size: 20px;
                font-weight: 700;
            }
            #Title {
                color: #f8fafc;
                font-size: 24px;
                font-weight: 700;
            }
            #Subtitle {
                color: #94a3b8;
                font-size: 12px;
            }
            #Hint {
                color: #cbd5e1;
                background: rgba(30, 41, 59, 160);
                border-radius: 12px;
                padding: 6px 10px;
            }
            #Badge, #BadgeSecondary, #InlineStatus {
                border-radius: 999px;
                padding: 6px 12px;
                font-weight: 600;
            }
            #Badge {
                color: #082f49;
                background: #7dd3fc;
            }
            #BadgeSecondary {
                color: #cbd5e1;
                background: rgba(51, 65, 85, 210);
            }
            QPushButton#CompanionModeButton {
                background: rgba(15, 23, 42, 150);
                border: 1px solid rgba(56, 189, 248, 85);
                border-radius: 999px;
                color: #e2e8f0;
                padding: 6px 14px;
                min-height: 12px;
            }
            QPushButton#CompanionModeButton:hover {
                background: rgba(14, 165, 233, 46);
                color: #f8fafc;
            }
            #InlineStatus {
                color: #dbeafe;
                background: rgba(14, 165, 233, 90);
                min-width: 56px;
                qproperty-alignment: AlignCenter;
            }
            #StatCard {
                background: rgba(15, 23, 42, 200);
                border: 1px solid rgba(56, 189, 248, 80);
                border-radius: 16px;
            }
            #CardTitle {
                color: #94a3b8;
                font-size: 10px;
                letter-spacing: 1px;
                text-transform: uppercase;
            }
            #CardValue {
                color: #f8fafc;
                font-size: 12px;
                font-weight: 600;
            }
            #ConversationView {
                background: rgba(2, 6, 23, 190);
                border: 1px solid rgba(148, 163, 184, 80);
                border-radius: 18px;
                padding: 10px;
            }
            #OverviewPanel {
                background: rgba(2, 6, 23, 190);
                border: 1px solid rgba(148, 163, 184, 80);
                border-radius: 18px;
                padding: 12px;
                color: #e2e8f0;
                font-size: 14px;
            }
            #CameraPreview {
                background: rgba(2, 6, 23, 190);
                border: 1px dashed rgba(148, 163, 184, 100);
                border-radius: 18px;
                color: #94a3b8;
                margin-top: 2px;
                margin-bottom: 2px;
            }
            QProgressBar {
                background: rgba(15, 23, 42, 220);
                border: 1px solid rgba(148, 163, 184, 80);
                border-radius: 12px;
                color: #e2e8f0;
                text-align: center;
                min-height: 24px;
            }
            QProgressBar::chunk {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                    stop:0 #22d3ee, stop:1 #3b82f6);
                border-radius: 10px;
            }
            QLineEdit {
                background: rgba(15, 23, 42, 230);
                border: 1px solid rgba(148, 163, 184, 80);
                border-radius: 16px;
                padding: 12px 14px;
                color: #e2e8f0;
            }
            QPushButton {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 #38bdf8, stop:1 #0ea5e9);
                border: none;
                border-radius: 16px;
                color: #0f172a;
                font-weight: 700;
                padding: 11px 16px;
                min-height: 18px;
            }
            QPushButton#GhostButton {
                background: rgba(15, 23, 42, 120);
                border: 1px solid rgba(226, 232, 240, 80);
                color: #e2e8f0;
            }
            #DashboardSegment {
                background: rgba(15, 23, 42, 185);
                border: 1px solid rgba(56, 189, 248, 78);
                border-radius: 18px;
            }
            QPushButton#DashboardFilterButton {
                background: transparent;
                border: 1px solid transparent;
                color: #cbd5e1;
                padding: 9px 20px;
                min-height: 16px;
                border-radius: 12px;
            }
            QPushButton#DashboardFilterButton:hover {
                background: rgba(14, 165, 233, 34);
                color: #f8fafc;
            }
            QPushButton#DashboardFilterButton:checked {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 rgba(56, 189, 248, 180), stop:1 rgba(14, 165, 233, 180));
                border: 1px solid rgba(125, 211, 252, 140);
                color: #f8fafc;
            }
            #DashboardDateRange {
                background: rgba(15, 23, 42, 185);
                border: 1px solid rgba(148, 163, 184, 78);
                border-radius: 18px;
            }
            #DashboardDateLabel {
                color: #cbd5e1;
                font-size: 12px;
                font-weight: 600;
                padding-right: 4px;
            }
            QDateEdit#DashboardDateEdit {
                background: rgba(2, 6, 23, 180);
                border: 1px solid rgba(56, 189, 248, 70);
                border-radius: 12px;
                color: #e2e8f0;
                padding: 7px 10px;
                min-width: 116px;
            }
            QPushButton#DashboardApplyButton {
                background: rgba(30, 41, 59, 210);
                border: 1px solid rgba(148, 163, 184, 90);
                color: #e2e8f0;
                border-radius: 12px;
                padding: 8px 16px;
                min-height: 16px;
            }
            QPushButton#DashboardApplyButton:hover {
                background: rgba(14, 165, 233, 48);
                color: #f8fafc;
            }
            QPushButton#DashboardApplyButton[active="true"] {
                background: rgba(14, 165, 233, 118);
                border: 1px solid rgba(125, 211, 252, 150);
                color: #f8fafc;
            }
            QPushButton:hover {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 #7dd3fc, stop:1 #38bdf8);
            }
            QPushButton:pressed {
                background: #0284c7;
            }
            QCheckBox {
                color: #e2e8f0;
                spacing: 8px;
                font-weight: 600;
            }
            QCheckBox::indicator {
                width: 18px;
                height: 18px;
                border-radius: 9px;
                border: 1px solid rgba(148, 163, 184, 160);
                background: rgba(15, 23, 42, 220);
            }
            QCheckBox::indicator:checked {
                background: #22c55e;
                border-color: #22c55e;
            }
            QTextBrowser {
                color: #e2e8f0;
                font-size: 14px;
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
        self._refresh_dashboard_page()
        self._refresh_report_page()

    def _update_mode_cards(self, mood: str) -> None:
        self.mood_badge.setText(mood)
        self.mode_card.setValue(mood)

    def _append_system_message(self, text: str) -> None:
        self._conversation.append(ConversationItem("system", text, self._now()))
        self._refresh_conversation()
        self.event_card.setValue(text)
        self._refresh_dashboard_page()
        self._refresh_report_page()

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
            self._start_streaming_reply(text)
            return

        reply = self._generate_local_reply(text)
        self._set_mood(PetMood.responding, "正在生成本地回应。")
        self._conversation.append(ConversationItem("eyeMuse", reply, self._now()))
        self._refresh_conversation()
        self._set_mood(PetMood.idle, reply)

    def _handle_send(self) -> None:
        self._submit_user_text(self.message_input.text(), from_companion=False)

    def _start_streaming_reply(self, text: str) -> None:
        if self._llm_client is None:
            return

        self._streaming_user_text = text
        self._conversation.append(ConversationItem("eyeMuse", "", self._now()))
        self._streaming_reply_index = len(self._conversation) - 1
        self._refresh_conversation()
        self._set_chat_busy(True)
        self._set_mood(PetMood.responding, "正在流式生成回应。")
        self.event_card.setValue("LLM 流式输出中")

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
        self._refresh_conversation()
        self._set_mood(PetMood.idle, final_text or "回复已完成。")
        self._teardown_streaming_reply()

    def _handle_streaming_error(self, message: str) -> None:
        partial = ""
        if self._streaming_reply_index is not None:
            partial = self._conversation[self._streaming_reply_index].text.strip()

        if self._streaming_reply_index is not None and not partial:
            fallback = self._generate_local_reply(self._streaming_user_text)
            self._conversation[self._streaming_reply_index].text = fallback
            self._set_mood(PetMood.idle, fallback)
        elif self._streaming_reply_index is not None:
            self._conversation[self._streaming_reply_index].text += "\n\n[回复中断]"
            self._set_mood(PetMood.alert, "流式回复中断。")

        self.event_card.setValue(f"LLM 回退到本地回复：{message}")
        self.camera_note.setText(f"LLM 流式调用异常：{message}")
        self._refresh_conversation()
        self._teardown_streaming_reply()

    def _teardown_streaming_reply(self) -> None:
        if self._llm_thread is not None:
            self._llm_thread.quit()
            self._llm_thread.wait(1500)
        self._llm_thread = None
        self._llm_worker = None
        self._streaming_reply_index = None
        self._streaming_user_text = ""
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
                bubble = "#dbeafe"
                color = "#0f172a"
            elif item.role == "eyeMuse":
                align = "left"
                bubble = "#ecfeff"
                color = "#164e63"
            else:
                align = "center"
                bubble = "#f8fafc"
                color = "#64748b"
            html.append(
                "<div style='margin:6px 0; text-align:%s;'>"
                "<span style='display:inline-block; max-width:92%%; padding:8px 10px; "
                "border-radius:14px; background:%s; color:%s; font-size:12px; line-height:1.5;'>%s</span>"
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
        self._refresh_dashboard_page()
        self._refresh_report_page()

    def _clear_conversation(self) -> None:
        self._conversation.clear()
        self._refresh_conversation()
        self._append_system_message("会话已清空。")

    def _build_companion_feedback(self) -> dict[str, str]:
        hint = self._current_pet_hint.strip() or "陪伴模式已开启。"
        behavior_state = str(self._behavior_summary.get("behavior_state", "warming"))
        emotion = self._emotion_tendency()

        if self._local_camera_enabled and self._face_count > 0 and emotion in {"专注", "平稳"} and self._stress_score < 55 and self._fatigue_score < 55:
            return {
                "mode": "focus",
                "bubble": "专注模式已开启，我会尽量降低打扰，只保留必要提醒。",
                "marquee": "正在专注中，不打扰",
                "show_bubble": True,
            }

        if self._fatigue_score >= 72 or behavior_state == "fatigued":
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

        if self._stress_score >= 72 or behavior_state == "anxious" or emotion == "焦虑":
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
            auto_hide_ms=4200 if payload["mode"] != previous_mode else 0,
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
                auto_hide_ms=3800,
            )
        if hasattr(self, "event_card"):
            self.event_card.setValue(self._current_pet_hint)
        self.statusBar().showMessage(self._current_pet_hint, 3200)

    def _handle_companion_chat_submit(self, text: str) -> None:
        self._submit_user_text(text, from_companion=True)
        if self._companion_window is not None:
            self._companion_window.set_chat_busy(self._llm_thread is not None)

    def _handle_companion_toolbar_action(self, action: str) -> None:
        if action == "home":
            self._restore_from_companion_mode()
            self._switch_page("home")
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

    def _ensure_companion_window(self) -> CompanionPetWindow:
        if self._companion_window is None:
            self._companion_window = CompanionPetWindow()
            self._companion_window.toolbar_action_requested.connect(self._handle_companion_toolbar_action)
            self._companion_window.chat_submitted.connect(self._handle_companion_chat_submit)
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
        self._start_camera()
        self._enter_companion_mode()

    def _enter_companion_mode(self) -> None:
        companion_window = self._ensure_companion_window()
        self._place_companion_window()
        companion_window.show()
        companion_window.raise_()
        self.hide()

    def _restore_from_companion_mode(self) -> None:
        if self._companion_window is not None:
            self._companion_window.hide()
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

        self._camera_worker = CameraWorker()
        self._camera_worker.frame_ready.connect(self._update_camera_frame)
        self._camera_worker.status_changed.connect(self._update_camera_status)
        self._camera_worker.face_count_changed.connect(self._update_face_count)
        self._camera_worker.analysis_changed.connect(self._update_analysis_metrics)
        self._camera_worker.open_failed.connect(self._handle_camera_open_failed)
        self._camera_worker.start()

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
        if self._camera_worker is not None:
            self._camera_worker.stop()
        self._camera_worker = None
        self._set_camera_toggle_checked(False)
        self._reset_camera_ui()
        self._set_mood(PetMood.offline, "摄像头已关闭，当前处于离线状态。")
        self._refresh_dashboard_page()
        self._refresh_report_page()

    def _handle_camera_open_failed(self, message: str) -> None:
        self._camera_worker = None
        self._reset_camera_ui(status=message, inline_status="异常", note=message)
        self._set_mood(PetMood.alert, message)
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
        self._refresh_dashboard_page()
        self._refresh_report_page()

    def _update_face_count(self, count: int) -> None:
        self._face_count = count
        self.face_card.setValue(f"{count} 个面部")
        self._refresh_dashboard_page()
        self._refresh_report_page()

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
        self._refresh_companion_feedback()
        self._set_mood(monitoring_mood, monitoring_hint)
        self._refresh_dashboard_page()
        self._refresh_report_page()

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
        if self._llm_thread is not None:
            self._teardown_streaming_reply()
        if self._activity_monitor is not None:
            self._activity_monitor.stop()
        if self._companion_window is not None:
            self._companion_window.close()
        self._stop_camera()
        super().closeEvent(event)


def _select_font() -> None:
    if sys.platform.startswith("win"):
        QFontDatabase.addApplicationFont("C:/Windows/Fonts/segoeui.ttf")


def run() -> int:
    app = QApplication.instance() or QApplication(sys.argv)
    _select_font()
    window = EyeMuseWindow()
    window.show()
    return app.exec()
