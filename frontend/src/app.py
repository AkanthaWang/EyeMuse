from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Optional
import sys

import cv2
from PySide6.QtCore import QDateTime, QObject, QThread, QTimer, Qt, Signal, Property, QSize, Slot
from PySide6.QtGui import QColor, QFont, QFontDatabase, QImage, QPainter, QPixmap, QPalette
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QFrame,
    QGraphicsDropShadowEffect,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QPushButton,
    QSizePolicy,
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


class PetMood(str, Enum):
    idle = "idle"
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


class PetAvatar(QWidget):
    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._mood = PetMood.idle
        self._breath = 0.0
        self._timer = QTimer(self)
        self._timer.setInterval(32)
        self._timer.timeout.connect(self._tick)
        self._timer.start()
        self.setMinimumSize(320, 320)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

    def setMood(self, mood: PetMood) -> None:
        if self._mood == mood:
            return
        self._mood = mood
        self.update()

    def mood(self) -> PetMood:
        return self._mood

    mood = Property(str, mood, setMood)

    def _tick(self) -> None:
        self._breath += 0.045
        if self._breath > 6.28318:
            self._breath = 0.0
        self.update()

    def paintEvent(self, event) -> None:  # noqa: N802
        del event
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        painter.fillRect(self.rect(), QColor("#111827"))

        width = self.width()
        height = self.height()
        breath_offset = int(8 * (1 + (self._breath % 3.14159) / 3.14159))

        body_w = int(width * 0.56)
        body_h = int(height * 0.56)
        body_x = (width - body_w) // 2
        body_y = (height - body_h) // 2 + breath_offset // 2

        mood_colors = {
            PetMood.idle: (QColor("#7dd3fc"), QColor("#0f172a")),
            PetMood.listening: (QColor("#34d399"), QColor("#052e16")),
            PetMood.thinking: (QColor("#fbbf24"), QColor("#3b2f0a")),
            PetMood.responding: (QColor("#60a5fa"), QColor("#1e3a8a")),
            PetMood.alert: (QColor("#fb7185"), QColor("#4c0519")),
            PetMood.offline: (QColor("#94a3b8"), QColor("#1e293b")),
        }
        accent, shadow = mood_colors[self._mood]

        painter.setPen(Qt.NoPen)
        painter.setBrush(QColor(15, 23, 42, 180))
        painter.drawRoundedRect(body_x + 12, body_y + 16, body_w, body_h, 36, 36)

        painter.setBrush(accent)
        painter.drawRoundedRect(body_x, body_y, body_w, body_h, 36, 36)

        painter.setBrush(QColor(255, 255, 255, 220))
        eye_y = body_y + int(body_h * 0.38)
        eye_left_x = body_x + int(body_w * 0.31)
        eye_right_x = body_x + int(body_w * 0.61)
        eye_w = int(body_w * 0.09)
        eye_h = int(body_h * 0.10)
        if self._mood == PetMood.alert:
            eye_h = max(4, eye_h // 2)
        painter.drawEllipse(eye_left_x, eye_y, eye_w, eye_h)
        painter.drawEllipse(eye_right_x, eye_y, eye_w, eye_h)

        if self._mood in {PetMood.thinking, PetMood.responding}:
            mouth_w = int(body_w * 0.18)
            mouth_h = 5
            mouth_x = body_x + (body_w - mouth_w) // 2
            mouth_y = body_y + int(body_h * 0.66)
            painter.setBrush(shadow)
            painter.drawRoundedRect(mouth_x, mouth_y, mouth_w, mouth_h, 3, 3)
        elif self._mood == PetMood.listening:
            mouth_w = int(body_w * 0.15)
            mouth_h = int(body_h * 0.04)
            mouth_x = body_x + (body_w - mouth_w) // 2
            mouth_y = body_y + int(body_h * 0.66)
            painter.setBrush(QColor(255, 255, 255, 220))
            painter.drawRoundedRect(mouth_x, mouth_y, mouth_w, mouth_h, 5, 5)
        elif self._mood == PetMood.alert:
            painter.setBrush(QColor("#fff1f2"))
            painter.drawEllipse(body_x + int(body_w * 0.45), body_y + int(body_h * 0.64), int(body_w * 0.08), int(body_h * 0.08))

        ear_w = int(body_w * 0.18)
        ear_h = int(body_h * 0.15)
        painter.setBrush(accent.lighter(120))
        painter.drawRoundedRect(body_x + int(body_w * 0.08), body_y - ear_h // 2, ear_w, ear_h, 18, 18)
        painter.drawRoundedRect(body_x + int(body_w * 0.74), body_y - ear_h // 2, ear_w, ear_h, 18, 18)


class StatCard(QFrame):
    def __init__(self, title: str, value: str, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setObjectName("StatCard")
        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 14, 16, 14)
        layout.setSpacing(4)
        self.title_label = QLabel(title)
        self.title_label.setObjectName("CardTitle")
        self.value_label = QLabel(value)
        self.value_label.setObjectName("CardValue")
        self.value_label.setWordWrap(True)
        layout.addWidget(self.title_label)
        layout.addWidget(self.value_label)

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
        self._llm_client = self._create_llm_client()
        self._streaming_reply_index: Optional[int] = None
        self._streaming_user_text: str = ""

        self._build_ui()
        self._apply_theme()
        self._append_system_message("EyeMuse 前端原型已就绪，输入文本或打开摄像头开始交互。")

    @staticmethod
    def _create_llm_client():
        if LLMClient is None:
            return None
        try:
            return LLMClient()
        except Exception:
            return None

    def _build_ui(self) -> None:
        central = QWidget(self)
        self.setCentralWidget(central)

        root_layout = QGridLayout(central)
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

        self.statusBar().showMessage("本地优先，摄像头默认关闭")

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
        status_row.addWidget(self.mood_badge)
        status_row.addWidget(self.privacy_badge)
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

        self.local_state_card = StatCard("隐私与保存", "默认仅保留本地会话与状态，不主动上传摄像头原始数据。")
        self.stress_card = StatCard("压力估计", "未开始检测")
        self.fatigue_card = StatCard("疲劳状态", "未开始检测")
        self.camera_card = StatCard("摄像头", "关闭")

        layout.addWidget(title)
        layout.addWidget(subtitle)
        layout.addWidget(self.avatar, 1)
        layout.addLayout(status_row)
        layout.addWidget(self.pet_hint)
        layout.addLayout(quick_row)
        layout.addWidget(self.local_state_card)
        layout.addWidget(self.stress_card)
        layout.addWidget(self.fatigue_card)
        layout.addWidget(self.camera_card)
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
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(14)

        title = QLabel("感知面板")
        title.setObjectName("Title")
        subtitle = QLabel("摄像头默认关闭，开启后会显示帧画面与检测结果")
        subtitle.setObjectName("Subtitle")

        self.camera_preview = QLabel("摄像头未开启")
        self.camera_preview.setObjectName("CameraPreview")
        self.camera_preview.setAlignment(Qt.AlignCenter)
        self.camera_preview.setMinimumSize(QSize(360, 250))
        self.camera_preview.setScaledContents(False)

        camera_controls = QHBoxLayout()
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

        self.face_card = StatCard("面部检测", "0 个面部")
        self.analysis_card = StatCard("分析状态", "等待开始")
        self.mode_card = StatCard("当前模式", "idle")
        self.event_card = StatCard("最近事件", "等待开始")

        layout.addWidget(title)
        layout.addWidget(subtitle)
        layout.addWidget(self.camera_preview, 1)
        layout.addLayout(camera_controls)
        layout.addWidget(self.camera_note)
        layout.addWidget(self.face_card)
        layout.addWidget(self.analysis_card)
        layout.addWidget(self.mode_card)
        layout.addWidget(self.event_card)
        return frame

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
            #Panel {
                background: rgba(15, 23, 42, 200);
                border: 1px solid rgba(148, 163, 184, 60);
                border-radius: 24px;
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
                border-radius: 14px;
                padding: 10px 12px;
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
            #InlineStatus {
                color: #dbeafe;
                background: rgba(14, 165, 233, 90);
            }
            #StatCard {
                background: rgba(15, 23, 42, 200);
                border: 1px solid rgba(56, 189, 248, 80);
                border-radius: 18px;
            }
            #CardTitle {
                color: #94a3b8;
                font-size: 11px;
                letter-spacing: 1px;
                text-transform: uppercase;
            }
            #CardValue {
                color: #f8fafc;
                font-size: 14px;
                font-weight: 600;
            }
            #ConversationView {
                background: rgba(2, 6, 23, 190);
                border: 1px solid rgba(148, 163, 184, 80);
                border-radius: 18px;
                padding: 10px;
            }
            #CameraPreview {
                background: rgba(2, 6, 23, 190);
                border: 1px dashed rgba(148, 163, 184, 100);
                border-radius: 18px;
                color: #94a3b8;
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
        self.avatar.setMood(mood)
        self._update_mode_cards(mood.value)
        self.pet_hint.setText(hint)
        self.statusBar().showMessage(hint, 3000)

    def _update_mode_cards(self, mood: str) -> None:
        self.mood_badge.setText(mood)
        self.mode_card.setValue(mood)

    def _append_system_message(self, text: str) -> None:
        self._conversation.append(ConversationItem("system", text, self._now()))
        self._refresh_conversation()
        self.event_card.setValue(text)

    def _inject_message(self, role: str, text: str) -> None:
        if role == "user":
            self.message_input.setText(text)
            self._handle_send()
        else:
            self._append_system_message(text)

    def _handle_send(self) -> None:
        if self._llm_thread is not None:
            self.statusBar().showMessage("上一条回复仍在生成中。", 2500)
            return

        text = self.message_input.text().strip()
        if not text:
            return

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

    def _generate_local_reply(self, text: str) -> str:
        lowered = text.lower()
        if any(keyword in lowered for keyword in ("累", "困", "疲劳", "休息", "sleep", "tired")):
            return "我注意到你可能有些疲劳。先休息几分钟，等你缓一缓我再陪你。"
        if any(keyword in lowered for keyword in ("摄像头", "camera", "脸", "面部")):
            return "摄像头链路已经预留好了。当前前端会优先给出本地提示，再逐步接上更稳定的感知。"
        if any(keyword in lowered for keyword in ("你好", "hi", "hello")):
            return "你好，我已经在线。你可以直接输入想法，也可以先打开摄像头看看状态。"
        return f"我收到你的输入：{text}。接下来我会根据状态、摄像头和后续模型接入继续完善回应。"

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

    def _clear_conversation(self) -> None:
        self._conversation.clear()
        self._refresh_conversation()
        self._append_system_message("会话已清空。")

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
        self.camera_card.setValue("开启中")
        self.camera_status.setText("开启中")
        self.camera_note.setText("正在尝试连接本地摄像头，若失败会回退到提示状态。")
        self.stress_card.setValue("校准中")
        self.fatigue_card.setValue("校准中")
        self.analysis_card.setValue("正在初始化分析链路")
        self._set_mood(PetMood.listening, "正在观察摄像头状态。")

    def _stop_camera(self) -> None:
        if self._camera_worker is not None:
            self._camera_worker.stop()
        self._camera_worker = None
        self._reset_camera_ui()
        self._set_mood(PetMood.idle, "摄像头已关闭。")

    def _handle_camera_open_failed(self, message: str) -> None:
        self._camera_worker = None
        self._reset_camera_ui(status=message, inline_status="异常", note=message)
        self._set_mood(PetMood.alert, message)

        self.camera_toggle.blockSignals(True)
        self.camera_toggle.setChecked(False)
        self.camera_toggle.blockSignals(False)

    def _reset_camera_ui(
        self,
        *,
        status: str = "关闭",
        inline_status: str = "关闭",
        note: Optional[str] = None,
    ) -> None:
        self._local_camera_enabled = False
        self._face_count = 0
        self.camera_preview.setPixmap(QPixmap())
        self.camera_preview.setText("摄像头未开启")
        self.camera_card.setValue(status)
        self.camera_status.setText(inline_status)
        self.camera_note.setText(note or "权限提示、失败提示和降级路径都先保留在界面上。")
        self.face_card.setValue("0 个面部")
        self.stress_card.setValue("未开始检测")
        self.fatigue_card.setValue("未开始检测")
        self.analysis_card.setValue("等待开始")

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

    def _update_face_count(self, count: int) -> None:
        self._face_count = count
        self.face_card.setValue(f"{count} 个面部")

    def _update_analysis_metrics(self, payload: object) -> None:
        if not isinstance(payload, dict):
            return

        face_count = int(payload.get("face_count", self._face_count))
        stress_score = int(payload.get("stress_score", 0))
        fatigue_score = int(payload.get("fatigue_score", 0))
        dominant_signal = str(payload.get("dominant_signal", "none"))
        calibration_state = str(payload.get("calibration_state", "waiting"))
        calibration_progress = float(payload.get("calibration_progress", 0.0))

        self._face_count = face_count
        self.face_card.setValue(f"{face_count} 个面部")
        self.stress_card.setValue(f"{stress_score} / 100")
        self.fatigue_card.setValue(f"{fatigue_score} / 100")

        if calibration_state == "calibrating":
            analysis_text = f"校准中 {int(calibration_progress * 100)}%"
            self.analysis_card.setValue(analysis_text)
            self.camera_note.setText(f"正在进行中性面部校准：{analysis_text}")
            self.event_card.setValue(analysis_text)
            self._set_mood(PetMood.listening, "正在校准压力分析基线。")
        elif calibration_state == "ready":
            if dominant_signal != "none":
                analysis_text = f"已就绪 · 主信号 {dominant_signal}"
            else:
                analysis_text = "已就绪"
            self.analysis_card.setValue(analysis_text)
            self.camera_note.setText(f"压力 {stress_score}，疲劳 {fatigue_score}。")
            self.event_card.setValue(f"压力 {stress_score} / 疲劳 {fatigue_score}")
            if stress_score >= 80 or fatigue_score >= 80:
                self._set_mood(PetMood.alert, f"压力 {stress_score}，疲劳 {fatigue_score}，建议休息。")
            elif stress_score >= 60 or fatigue_score >= 60:
                self._set_mood(PetMood.thinking, f"压力 {stress_score}，疲劳 {fatigue_score}。")
        elif calibration_state == "unavailable":
            self.analysis_card.setValue("基础检测模式")
            self.camera_note.setText("已回退到基础面部框选，压力/疲劳分析未启用。")
            self.event_card.setValue("基础面部框选")
        else:
            self.analysis_card.setValue("等待面部")
            if face_count > 0:
                self.camera_note.setText("已检测到人脸，等待 MediaPipe 关键点稳定后开始分析。")
                self.event_card.setValue("已检测到人脸，等待关键点稳定")
            else:
                self.camera_note.setText("等待检测到面部后开始分析。")
                self.event_card.setValue("等待面部")

    def _current_summary(self) -> str:
        camera_state = "开启" if self._local_camera_enabled else "关闭"
        return f"摄像头 {camera_state}，检测到 {self._face_count} 个面部。"

    @staticmethod
    def _now() -> str:
        return QDateTime.currentDateTime().toString("hh:mm:ss")

    def closeEvent(self, event) -> None:  # noqa: N802
        if self._llm_thread is not None:
            self._teardown_streaming_reply()
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
