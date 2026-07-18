from __future__ import annotations
from pathlib import Path
from dataclasses import dataclass
from enum import Enum
import json
from typing import Optional
import sys

import cv2
from PySide6.QtCore import QDate, QDateTime, QObject, QThread, QTimer, Qt, QUrl, Signal, Property, QSize, Slot
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

class PetAvatar(QLabel):
    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._mood = PetMood.idle
        self._movie: Optional[QMovie] = None
        self._max_movie_size = QSize(320, 320)
        self.setMinimumSize(240, 255)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.setAlignment(Qt.AlignCenter)

        self._asset_dir = Path(__file__).resolve().parents[1] / "assets"
        self._mood_assets = {
            PetMood.idle: "idle.gif",
            PetMood.listening: "listening.gif",
            PetMood.thinking: "thinking.gif",
            PetMood.responding: "responding.gif",
            PetMood.alert: "alert.gif",
            PetMood.offline: "offline.gif",
        }

        self.setStyleSheet(
            "background: rgba(17, 24, 39, 0.92);"
            "border-radius: 24px;"
        )
        self.setMood(PetMood.idle)

    def _target_render_size(self) -> QSize:
        available_size = QSize(
            max(1, int(self.width() * 0.86)),
            max(1, int(self.height() * 0.86)),
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

    def setMood(self, mood: PetMood) -> None:
        if self._mood == mood and self._movie is not None:
            return

        self._mood = mood
        file_name = self._mood_assets.get(mood, "idle.gif")
        asset_path = self._asset_dir / file_name

        if not asset_path.exists():
            fallback_path = self._asset_dir / "idle.gif"
            if fallback_path.exists():
                asset_path = fallback_path
            else:
                if self._movie is not None:
                    self._movie.stop()
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

        new_movie = QMovie(str(asset_path))
        new_movie.setCacheMode(QMovie.CacheAll)
        self._movie = new_movie
        self._movie.frameChanged.connect(self._handle_movie_frame_changed)
        self._movie.start()
        self._movie.jumpToFrame(0)
        self._render_current_frame()
        self.setText("")

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        self._render_current_frame()

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
        self._stress_score = 0
        self._fatigue_score = 0
        self._dominant_signal = "none"
        self._calibration_state = "waiting"
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
        self._latest_custom_report_md = ""
        self._streaming_reply_index: Optional[int] = None
        self._streaming_user_text: str = ""

        self._build_ui()
        self._apply_theme()
        self._refresh_dashboard_page()
        self._refresh_report_page()
        self._append_system_message("EyeMuse 前端原型已就绪，输入文本或打开摄像头开始交互。")

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

        self.statusBar().showMessage("本地优先，摄像头默认关闭")

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
            self.nav_logo_label.setPixmap(
                logo_pixmap.scaledToHeight(64, Qt.SmoothTransformation)
            )
        else:
            self.nav_logo_label.setText("EyeMuse")

        self.nav_right_spacer = QWidget()
        self.nav_right_spacer.setFixedWidth(320)

        self.home_nav_button = QPushButton("主页面")
        self.dashboard_nav_button = QPushButton("可视化分析")
        self.report_nav_button = QPushButton("健康报告")

        for button in (
            self.dashboard_nav_button,
            self.home_nav_button,
            self.report_nav_button,
        ):
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
        status_row.addWidget(self.mood_badge)
        status_row.addWidget(self.privacy_badge)
        status_row.addStretch(1)

        self.pet_hint = QLabel("等待用户输入，或开启摄像头观察状态变化。")
        self.pet_hint.setWordWrap(True)
        self.pet_hint.setObjectName("Hint")
        self.pet_hint.hide()

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
            page_name = {
                "home": "主页面",
                "dashboard": "可视化分析大屏",
                "report": "健康报告页面",
            }.get(page, "主页面")
            self.statusBar().showMessage(f"已切换到{page_name}", 2500)

    def _focus_score(self) -> int:
        score = 100
        score -= int(self._stress_score * 0.4)
        score -= int(self._fatigue_score * 0.45)
        if not self._local_camera_enabled:
            score -= 10
        if self._face_count == 0 and self._local_camera_enabled:
            score -= 15
        return max(0, min(100, score))

    def _emotion_tendency(self) -> str:
        if self._fatigue_score >= 80:
            return "疲惫"
        if self._stress_score >= 80:
            return "焦虑"
        if self._stress_score >= 55 or self._fatigue_score >= 55:
            return "专注波动"
        if self._face_count > 0 and self._local_camera_enabled:
            return "专注"
        return "平稳"

    def _build_realtime_snapshot(self):
        if RealtimeSnapshot is None:
            return None
        event_text = "等待开始"
        if hasattr(self, "event_card"):
            event_text = self.event_card.value_label.text()
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
            grid-template-columns: repeat(4, 1fr);
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
                color: palette,
                tooltip: { trigger: 'item' },
                legend: {
                    bottom: 6,
                    textStyle: { color: textColor },
                },
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
                setStatus(error.message || '图表加载失败', true);
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
            fallback_html = (
                "<h3 style='color:#f8fafc;'>图表降级模式</h3>"
                "<p style='color:#cbd5e1;'>当前运行环境未启用 Qt WebEngine，分析页已回退为历史数据概要。</p>"
                f"<p style='color:#cbd5e1;'>历史样本：<b>{len(payload.get('line_categories', []))}</b></p>"
                f"<p style='color:#cbd5e1;'>平均专注：<b>{payload.get('averages', {}).get('avg_focus', 0)}</b></p>"
                f"<p style='color:#cbd5e1;'>平均压力：<b>{payload.get('averages', {}).get('avg_stress', 0)}</b></p>"
            )
            self.dashboard_chart_fallback.setHtml(fallback_html)

    def _refresh_dashboard_page(self) -> None:
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
        for period, button in self._dashboard_filter_buttons.items():
            button.setChecked(period == self._dashboard_period)
        self.dashboard_custom_apply_button.setProperty("active", self._dashboard_period == "custom")
        self.dashboard_custom_apply_button.style().unpolish(self.dashboard_custom_apply_button)
        self.dashboard_custom_apply_button.style().polish(self.dashboard_custom_apply_button)

        payload_key = json.dumps(payload, ensure_ascii=False, sort_keys=True)
        if getattr(self, "_dashboard_payload_key", "") != payload_key:
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
                period_summary = self._dashboard_repository.get_report_summary(
                    start_date=custom_start,
                    end_date=custom_end,
                )
                period_title = "自定义时间健康分析报告"
            else:
                period_summary = self._dashboard_repository.get_report_summary(
                    start_date=week_start,
                    end_date=week_end,
                )
                period_title = "每周健康分析报告"
        else:
            daily_summary = {}
            period_summary = {}
            period_title = "每周健康分析报告"

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
        top_emotion = summary.get("top_emotion", "平稳")
        top_signal = summary.get("top_signal", "稳定")
        high_stress_count = summary.get("high_stress_count", 0)
        high_fatigue_count = summary.get("high_fatigue_count", 0)
        peak_stress = summary.get("peak_stress", 0)
        peak_fatigue = summary.get("peak_fatigue", 0)
        lowest_focus = summary.get("lowest_focus", 0)
        events = summary.get("events", [])

        if avg_fatigue >= 75 or high_fatigue_count >= 3:
            risk_level = "高"
            suggestion = "优先降低连续用眼时长，安排 10 分钟离屏休息，并在下一阶段减少高强度任务。"
        elif avg_stress >= 70 or high_stress_count >= 3:
            risk_level = "中高"
            suggestion = "优先做任务拆分与节奏降噪，避免多任务并发，加入呼吸放松或白噪音干预。"
        elif avg_focus >= 72:
            risk_level = "低"
            suggestion = "保持当前节奏，把高价值任务集中放在状态最稳的时间段，避免被低优先级事务打断。"
        else:
            risk_level = "中"
            suggestion = "建议采用 45-10 或 50-10 的学习工作节奏，控制压力峰值，穿插短休息。"

        key_points = [
            f"**主导情绪**：{top_emotion}，当前主要行为信号为 **{top_signal}**。",
            f"**核心风险等级**：{risk_level}，期间共记录 **{sample_count}** 条有效样本。",
            f"**优先建议**：{suggestion}",
        ]
        event_lines = "\n".join(f"- {item}" for item in events) if events else "- 本时段暂无显著事件波动记录。"

        return (
            f"# {title}\n\n"
            f"> 统计区间：{range_label}  \n"
            f"> 分析模式：{mode_label}\n\n"
            "## 划重点\n"
            f"- {key_points[0]}\n"
            f"- {key_points[1]}\n"
            f"- {key_points[2]}\n\n"
            "## 核心指标表\n\n"
            "| 指标 | 数值 | 解读 |\n"
            "| --- | ---: | --- |\n"
            f"| 平均压力 | {avg_stress} | 压力值越高，越需要减少外界干扰 |\n"
            f"| 平均疲劳 | {avg_fatigue} | 疲劳值越高，越需要休息与节奏调整 |\n"
            f"| 平均专注 | {avg_focus} | 专注值越高，越适合安排核心任务 |\n"
            f"| 压力峰值 | {peak_stress} | 观察是否存在明显高压时段 |\n"
            f"| 疲劳峰值 | {peak_fatigue} | 观察是否出现连续高疲劳趋势 |\n"
            f"| 最低专注 | {lowest_focus} | 用于判断注意力最低谷 |\n\n"
            "## 风险观察\n"
            f"- 高压力样本次数：**{high_stress_count}**\n"
            f"- 高疲劳样本次数：**{high_fatigue_count}**\n"
            f"- 建议重点关注：**{top_emotion} / {top_signal}** 组合出现的时间段\n\n"
            "## 近期事件记录\n"
            f"{event_lines}\n\n"
            "## 重要分析建议\n"
            f"1. **优先级一**：{suggestion}\n"
            "2. **优先级二**：把高价值任务安排在专注度更高的时间段，避免在高疲劳段继续硬撑。\n"
            "3. **优先级三**：若连续多次出现高压力或高疲劳，建议启用更明确的休息提醒与轻任务切换策略。\n"
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
            #HeroPanel {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                    stop:0 rgba(13, 71, 161, 220), stop:1 rgba(14, 116, 144, 180));
                border: 1px solid rgba(125, 211, 252, 80);
                border-radius: 24px;
            }
            #HeroTitle {
                color: #eff6ff;
                font-size: 26px;
                font-weight: 700;
            }
            #HeroSubtitle {
                color: #dbeafe;
                font-size: 13px;
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
        self.avatar.setMood(mood)
        self._update_mode_cards(mood.value)
        self.pet_hint.setText(hint)
        self.statusBar().showMessage(hint, 3000)
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
        self._refresh_dashboard_page()
        self._refresh_report_page()

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
        self._refresh_dashboard_page()
        self._refresh_report_page()

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
        self._stress_score = 0
        self._fatigue_score = 0
        self._dominant_signal = "none"
        self._calibration_state = "waiting"
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

        self._face_count = face_count
        self._stress_score = stress_score
        self._fatigue_score = fatigue_score
        self._dominant_signal = dominant_signal
        self._calibration_state = calibration_state
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
        self._refresh_dashboard_page()
        self._refresh_report_page()

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
