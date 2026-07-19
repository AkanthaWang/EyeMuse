from __future__ import annotations

from PySide6.QtGui import QColor, QFont, QPalette
from PySide6.QtWidgets import QApplication, QMainWindow


def apply_modern_theme(window: QMainWindow) -> None:
    app = QApplication.instance()
    if app is not None:
        app.setStyle("Fusion")
        palette = QPalette()
        palette.setColor(QPalette.Window, QColor("#D4EBFF"))
        palette.setColor(QPalette.WindowText, QColor("#31598E"))
        palette.setColor(QPalette.Base, QColor("#DFF2FF"))
        palette.setColor(QPalette.AlternateBase, QColor("#D0E8FF"))
        palette.setColor(QPalette.Text, QColor("#3A679F"))
        palette.setColor(QPalette.Button, QColor("#D0E8FF"))
        palette.setColor(QPalette.ButtonText, QColor("#2E5A95"))
        palette.setColor(QPalette.Highlight, QColor("#7EA9F4"))
        palette.setColor(QPalette.HighlightedText, QColor("#24436E"))
        app.setPalette(palette)

    window.setFont(QFont("Microsoft YaHei UI", 10))
    window.setStyleSheet(_build_stylesheet())


def _build_stylesheet() -> str:
    return """
    QMainWindow {
        background:
            radial-gradient(circle at 14% 12%, rgba(166, 239, 255, 0.62), transparent 26%),
            radial-gradient(circle at 86% 10%, rgba(173, 187, 255, 0.42), transparent 22%),
            linear-gradient(180deg, #EDF9FF 0%, #D9EEFF 46%, #C7E3FB 100%);
        color: #3A679F;
    }
    #NavBar {
        background: rgba(239, 248, 255, 0.34);
        border-bottom: 1px solid rgba(255, 255, 255, 0.72);
    }
    #NavLogo {
        color: #2E5A95;
        font-size: 18px;
        font-weight: 600;
    }
    #WindowControlGroup {
        background: transparent;
    }
    QPushButton#WindowControlButton,
    QPushButton#WindowCloseButton {
        background: rgba(255, 255, 255, 0.30);
        border: 1px solid rgba(255, 255, 255, 0.72);
        border-radius: 10px;
        color: #5A7DB1;
        min-width: 30px;
        min-height: 30px;
        padding: 0;
    }
    QPushButton#WindowControlButton:hover {
        background: rgba(255, 255, 255, 0.48);
    }
    QPushButton#WindowCloseButton:hover {
        background: rgba(255, 234, 214, 0.82);
        border-color: rgba(255, 255, 255, 0.84);
    }
    QPushButton#WindowControlButton:pressed,
    QPushButton#WindowCloseButton:pressed {
        background: rgba(255, 255, 255, 0.62);
    }
    QPushButton {
        background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
            stop:0 rgba(118, 185, 255, 0.98), stop:1 rgba(73, 135, 239, 0.98));
        border: 1px solid rgba(255, 255, 255, 0.82);
        border-radius: 14px;
        color: #164A83;
        padding: 8px 14px;
        min-height: 30px;
        font-weight: 600;
    }
    QPushButton:hover {
        background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
            stop:0 rgba(139, 201, 255, 0.98), stop:1 rgba(92, 152, 247, 0.98));
    }
    QPushButton:pressed {
        background: rgba(79, 141, 239, 0.98);
    }
    QPushButton:disabled {
        color: rgba(58, 103, 159, 0.46);
        background: rgba(220, 237, 255, 0.42);
        border-color: rgba(255, 255, 255, 0.56);
    }
    QPushButton#NavButton,
    QPushButton#DashboardFilterButton {
        background: rgba(228, 242, 255, 0.44);
        border: 1px solid rgba(255, 255, 255, 0.72);
        color: #597CAE;
        min-height: 28px;
        padding: 6px 12px;
    }
    QPushButton#NavButton:hover,
    QPushButton#DashboardFilterButton:hover {
        color: #2E5C98;
        border-color: rgba(255, 255, 255, 0.84);
        background: rgba(208, 230, 255, 0.64);
    }
    QPushButton#NavButton:checked,
    QPushButton#DashboardFilterButton:checked {
        color: #1C528E;
        border-color: rgba(255, 255, 255, 0.88);
        background: rgba(169, 207, 255, 0.78);
    }
    QPushButton#DashboardApplyButton,
    QPushButton#PrimaryActionButton {
        background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
            stop:0 rgba(128, 190, 255, 0.98), stop:1 rgba(79, 136, 241, 0.98));
        border-color: rgba(255, 255, 255, 0.84);
    }
    QPushButton#SecondaryActionButton,
    QPushButton#CompanionModeButton,
    QPushButton#GhostActionButton,
    QPushButton#GhostButton {
        background: rgba(229, 243, 255, 0.46);
        border: 1px solid rgba(255, 255, 255, 0.80);
        color: #4470A7;
    }
    QPushButton#SecondaryActionButton:hover,
    QPushButton#CompanionModeButton:hover,
    QPushButton#GhostActionButton:hover,
    QPushButton#GhostButton:hover {
        background: rgba(211, 232, 255, 0.66);
    }
    #Page {
        background: transparent;
    }
    #Panel {
        background: qlineargradient(x1:0, y1:0, x2:1, y2:1,
            stop:0 rgba(233, 246, 255, 0.62), stop:1 rgba(198, 224, 255, 0.50));
        border: 1px solid rgba(255, 255, 255, 0.84);
        border-radius: 16px;
    }
    #SectionTitle,
    #Title {
        color: #2F5C98;
        font-weight: 600;
    }
    #Title {
        font-size: 18px;
    }
    #SectionTitle {
        font-size: 18px;
    }
    #Subtitle {
        color: #6E8DB8;
        font-size: 12px;
    }
    #Hint,
    #OverviewPanel,
    #ReportBrowser,
    #ConversationView,
    #CameraPreview {
        background: qlineargradient(x1:0, y1:0, x2:1, y2:1,
            stop:0 rgba(220, 239, 255, 0.60), stop:1 rgba(193, 220, 255, 0.50));
        border: 1px solid rgba(255, 255, 255, 0.84);
        border-radius: 14px;
        color: #3B679E;
    }
    #Hint {
        padding: 10px 12px;
        color: rgba(59, 103, 158, 0.88);
    }
    #Badge,
    #BadgeSecondary,
    #InlineStatus {
        border-radius: 999px;
        padding: 5px 12px;
        font-weight: 600;
    }
    #Badge {
        color: #1E5590;
        background: rgba(212, 235, 255, 0.82);
        border: 1px solid rgba(255, 255, 255, 0.84);
    }
    #BadgeSecondary {
        color: #5278AB;
        background: rgba(236, 247, 255, 0.48);
        border: 1px solid rgba(255, 255, 255, 0.82);
    }
    #InlineStatus {
        color: #3E6AA1;
        background: rgba(223, 239, 255, 0.60);
        border: 1px solid rgba(255, 255, 255, 0.82);
    }
    #StatCard {
        background: qlineargradient(x1:0, y1:0, x2:1, y2:1,
            stop:0 rgba(226, 242, 255, 0.64), stop:1 rgba(193, 220, 255, 0.52));
        border: 1px solid rgba(255, 255, 255, 0.84);
        border-radius: 14px;
    }
    #CardTitle {
        color: #6F8DB8;
        font-size: 10px;
        letter-spacing: 0.9px;
        text-transform: uppercase;
    }
    #CardValue {
        color: #355E96;
        font-size: 14px;
        font-weight: 600;
    }
    #ConversationView {
        padding: 12px;
    }
    #OverviewPanel,
    #ReportBrowser {
        padding: 12px;
        font-size: 14px;
    }
    #CameraPreview {
        color: #6387B4;
    }
    #DashboardSegment,
    #DashboardDateRange {
        background: rgba(232, 244, 255, 0.46);
        border: 1px solid rgba(255, 255, 255, 0.82);
        border-radius: 14px;
    }
    #DashboardDateEdit {
        background: rgba(216, 235, 255, 0.58);
        color: #355E96;
        border: 1px solid rgba(255, 255, 255, 0.82);
        border-radius: 12px;
        padding: 6px 10px;
        selection-background-color: rgba(124, 149, 232, 0.24);
    }
    #DashboardDateLabel {
        color: #6E8DB8;
    }
    QTextBrowser, QPlainTextEdit {
        background: transparent;
        color: #355E96;
    }
    QTextBrowser QScrollBar:vertical, QPlainTextEdit QScrollBar:vertical {
        background: transparent;
        width: 10px;
        margin: 6px 2px;
        border: none;
    }
    QTextBrowser QScrollBar::handle:vertical, QPlainTextEdit QScrollBar::handle:vertical {
        background: rgba(120, 171, 241, 0.56);
        min-height: 44px;
        border-radius: 5px;
    }
    QLineEdit {
        background: rgba(216, 235, 255, 0.58);
        border: 1px solid rgba(255, 255, 255, 0.82);
        border-radius: 14px;
        padding: 9px 12px;
        color: #355E96;
        selection-background-color: rgba(124, 149, 232, 0.22);
    }
    QLineEdit:focus, QDateEdit:focus {
        border-color: rgba(255, 255, 255, 0.92);
    }
    QDateEdit {
        background: rgba(216, 235, 255, 0.58);
        border: 1px solid rgba(255, 255, 255, 0.82);
        border-radius: 12px;
        padding: 5px 10px;
        color: #355E96;
    }
    QCheckBox {
        color: #456FA7;
        spacing: 8px;
    }
    QCheckBox::indicator {
        width: 16px;
        height: 16px;
        border-radius: 5px;
        border: 1px solid rgba(255, 255, 255, 0.80);
        background: rgba(223, 239, 255, 0.64);
    }
    QCheckBox::indicator:checked {
        background: rgba(124, 149, 232, 0.84);
        border-color: rgba(255, 255, 255, 0.88);
    }
    QProgressBar {
        background: rgba(218, 236, 255, 0.58);
        border: 1px solid rgba(255, 255, 255, 0.82);
        border-radius: 12px;
        color: #355E96;
        text-align: center;
        min-height: 24px;
    }
    QProgressBar::chunk {
        background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
            stop:0 rgba(255, 240, 194, 0.96), stop:1 rgba(106, 187, 255, 0.96));
        border-radius: 10px;
    }
    """
