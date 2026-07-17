from __future__ import annotations

from .common import FaceAnalysisResult, FaceDetectionResult, FaceRegion
from .mediapipe_analyzer import MediaPipeFaceAnalyzer
from .yolo_detector import YOLOFaceDetector


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
    detector_error = None
    analyzer_error = None

    try:
        detector = YOLOFaceDetector(min_detection_confidence=min_detection_confidence)
    except Exception as exc:
        detector_error = exc

    try:
        analyzer = MediaPipeFaceAnalyzer(
            min_face_detection_confidence=min_detection_confidence,
            min_face_presence_confidence=min_detection_confidence,
            min_tracking_confidence=min_detection_confidence,
        )
    except Exception as exc:
        analyzer_error = exc

    if detector is None and analyzer is None:
        raise RuntimeError(
            f"Unable to initialize face detection stack. YOLO: {detector_error}; MediaPipe: {analyzer_error}"
        )

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

            detection_result = detector.detect(frame) if detector is not None else None
            annotated_frame = detector.annotate(frame, detection_result) if detection_result is not None else frame

            if analyzer is not None:
                analysis_result = analyzer.detect(
                    frame,
                    detection_result.regions if detection_result is not None else None,
                )
                if analysis_result.face_detected or detection_result is None:
                    annotated_frame = analyzer.annotate(frame, analysis_result)

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
