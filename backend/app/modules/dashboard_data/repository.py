from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import date, datetime, timedelta
import json
import math
from pathlib import Path
import random
import sqlite3
from typing import Any


@dataclass(frozen=True)
class RealtimeSnapshot:
    recorded_at: str
    mood: str
    emotion: str
    stress_score: int
    fatigue_score: int
    focus_score: int
    face_count: int
    dominant_signal: str
    event_text: str
    camera_enabled: bool
    source: str = "runtime"


class DashboardRepository:
    def __init__(self, db_path: Path | None = None) -> None:
        root = Path(__file__).resolve().parents[4]
        data_dir = root / "data"
        data_dir.mkdir(parents=True, exist_ok=True)
        self._db_path = db_path or data_dir / "eyemuse.db"
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self._db_path)
        connection.row_factory = sqlite3.Row
        return connection

    def _initialize(self) -> None:
        with self._connect() as connection:
            cursor = connection.cursor()
            cursor.executescript(
                """
                CREATE TABLE IF NOT EXISTS runtime_state (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS metric_snapshots (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    recorded_at TEXT NOT NULL,
                    mood TEXT NOT NULL,
                    emotion TEXT NOT NULL,
                    stress_score INTEGER NOT NULL,
                    fatigue_score INTEGER NOT NULL,
                    focus_score INTEGER NOT NULL,
                    face_count INTEGER NOT NULL,
                    dominant_signal TEXT NOT NULL,
                    event_text TEXT NOT NULL,
                    camera_enabled INTEGER NOT NULL,
                    source TEXT NOT NULL
                );

                CREATE INDEX IF NOT EXISTS idx_metric_snapshots_recorded_at
                ON metric_snapshots(recorded_at);
                """
            )
            demo_count = cursor.execute(
                "SELECT COUNT(*) AS count FROM metric_snapshots WHERE source = 'demo'"
            ).fetchone()["count"]
            if demo_count < 240:
                cursor.execute("DELETE FROM metric_snapshots WHERE source = 'demo'")
                self._seed_demo_data(connection)

    def _seed_demo_data(self, connection: sqlite3.Connection) -> None:
        rng = random.Random(42)
        now = datetime.now().replace(second=0, microsecond=0)
        start = (now - timedelta(days=45)).replace(hour=0, minute=0)
        rows: list[tuple[Any, ...]] = []

        sample_count = 45 * 8
        for index in range(sample_count):
            current = start + timedelta(hours=3 * index)
            day_phase = index / 8.0
            intra_phase = (index % 8) / 2.0
            stress_score = int(48 + 13 * math.sin(day_phase / 1.7) + 10 * math.sin(intra_phase) + rng.randint(-6, 6))
            fatigue_score = int(44 + 16 * math.cos(day_phase / 2.8) + 12 * math.cos(intra_phase / 1.5) + rng.randint(-5, 5))
            stress_score = max(18, min(95, stress_score))
            fatigue_score = max(15, min(92, fatigue_score))
            focus_score = max(12, min(96, 100 - int(stress_score * 0.42 + fatigue_score * 0.38)))
            face_count = 1 if rng.random() > 0.1 else 0

            if fatigue_score >= 76:
                emotion = "疲惫"
                mood = "alert"
                dominant_signal = "eye_squint"
                event_text = "Demo：检测到疲劳上升"
            elif stress_score >= 72:
                emotion = "焦虑"
                mood = "thinking"
                dominant_signal = "brow_furrow"
                event_text = "Demo：压力偏高，建议放松"
            elif focus_score >= 72:
                emotion = "专注"
                mood = "responding"
                dominant_signal = "none"
                event_text = "Demo：专注度稳定"
            else:
                emotion = "平稳"
                mood = "idle"
                dominant_signal = "none"
                event_text = "Demo：状态平稳"

            rows.append(
                (
                    current.isoformat(timespec="minutes"),
                    mood,
                    emotion,
                    stress_score,
                    fatigue_score,
                    focus_score,
                    face_count,
                    dominant_signal,
                    event_text,
                    1,
                    "demo",
                )
            )

        connection.executemany(
            """
            INSERT INTO metric_snapshots (
                recorded_at, mood, emotion, stress_score, fatigue_score, focus_score,
                face_count, dominant_signal, event_text, camera_enabled, source
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            rows,
        )
        connection.commit()

    @staticmethod
    def _period_window(
        period: str,
        start_date: date | None = None,
        end_date: date | None = None,
    ) -> tuple[datetime, datetime, str, str]:
        today_start = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
        if period == "custom" and start_date is not None and end_date is not None:
            range_start = datetime.combine(start_date, datetime.min.time())
            range_end = datetime.combine(end_date + timedelta(days=1), datetime.min.time())
            days = max(1, (range_end - range_start).days)
            if days <= 2:
                return range_start, range_end, "%m-%d %H:%M", "hour"
            return range_start, range_end, "%m-%d", "day"
        if period == "week":
            return today_start - timedelta(days=7), today_start, "%m-%d", "day"
        if period == "month":
            return today_start - timedelta(days=30), today_start, "%m-%d", "day"
        return today_start - timedelta(days=1), today_start, "%H:%M", "hour"

    def record_runtime_snapshot(self, snapshot: RealtimeSnapshot) -> None:
        encoded = json.dumps(asdict(snapshot), ensure_ascii=False)
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO runtime_state (key, value, updated_at)
                VALUES ('current_snapshot', ?, ?)
                ON CONFLICT(key) DO UPDATE SET
                    value = excluded.value,
                    updated_at = excluded.updated_at
                """,
                (encoded, snapshot.recorded_at),
            )

            latest_runtime = connection.execute(
                """
                SELECT recorded_at
                FROM metric_snapshots
                WHERE source = 'runtime'
                ORDER BY recorded_at DESC
                LIMIT 1
                """
            ).fetchone()

            should_insert = True
            if latest_runtime is not None:
                latest_dt = datetime.fromisoformat(latest_runtime["recorded_at"])
                current_dt = datetime.fromisoformat(snapshot.recorded_at)
                should_insert = current_dt - latest_dt >= timedelta(seconds=45)

            if should_insert:
                connection.execute(
                    """
                    INSERT INTO metric_snapshots (
                        recorded_at, mood, emotion, stress_score, fatigue_score, focus_score,
                        face_count, dominant_signal, event_text, camera_enabled, source
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        snapshot.recorded_at,
                        snapshot.mood,
                        snapshot.emotion,
                        snapshot.stress_score,
                        snapshot.fatigue_score,
                        snapshot.focus_score,
                        snapshot.face_count,
                        snapshot.dominant_signal,
                        snapshot.event_text,
                        int(snapshot.camera_enabled),
                        snapshot.source,
                    ),
                )
            connection.commit()

    def get_dashboard_payload(
        self,
        period: str = "day",
        *,
        start_date: date | None = None,
        end_date: date | None = None,
    ) -> dict[str, Any]:
        start_at, end_at, label_format, bucket_mode = self._period_window(period, start_date, end_date)
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT recorded_at, emotion, stress_score, fatigue_score, focus_score,
                       face_count, dominant_signal, event_text, camera_enabled
                FROM metric_snapshots
                WHERE recorded_at >= ? AND recorded_at < ?
                ORDER BY recorded_at ASC
                """
                ,
                (start_at.isoformat(timespec="minutes"), end_at.isoformat(timespec="minutes")),
            ).fetchall()

            emotion_rows = connection.execute(
                """
                SELECT emotion, COUNT(*) AS count
                FROM metric_snapshots
                WHERE recorded_at >= ? AND recorded_at < ?
                GROUP BY emotion
                ORDER BY count DESC
                """
                ,
                (start_at.isoformat(timespec="minutes"), end_at.isoformat(timespec="minutes")),
            ).fetchall()

            signal_rows = connection.execute(
                """
                SELECT CASE
                    WHEN dominant_signal = 'none' THEN '稳定'
                    ELSE dominant_signal
                END AS label,
                COUNT(*) AS count
                FROM metric_snapshots
                WHERE recorded_at >= ? AND recorded_at < ?
                GROUP BY label
                ORDER BY count DESC
                """
                ,
                (start_at.isoformat(timespec="minutes"), end_at.isoformat(timespec="minutes")),
            ).fetchall()

            current_state = connection.execute(
                "SELECT value FROM runtime_state WHERE key = 'current_snapshot'"
            ).fetchone()

        grouped: dict[str, dict[str, Any]] = {}
        for row in rows:
            recorded_at = datetime.fromisoformat(row["recorded_at"])
            if bucket_mode == "hour":
                bucket = recorded_at.replace(minute=0, second=0, microsecond=0)
            else:
                bucket = recorded_at.replace(hour=0, minute=0, second=0, microsecond=0)
            key = bucket.isoformat()
            stats = grouped.setdefault(
                key,
                {
                    "label": bucket.strftime(label_format),
                    "stress": [],
                    "fatigue": [],
                    "focus": [],
                },
            )
            stats["stress"].append(row["stress_score"])
            stats["fatigue"].append(row["fatigue_score"])
            stats["focus"].append(row["focus_score"])

        categories = [grouped[key]["label"] for key in grouped]
        stress_series = [round(sum(grouped[key]["stress"]) / len(grouped[key]["stress"]), 1) for key in grouped]
        fatigue_series = [round(sum(grouped[key]["fatigue"]) / len(grouped[key]["fatigue"]), 1) for key in grouped]
        focus_series = [round(sum(grouped[key]["focus"]) / len(grouped[key]["focus"]), 1) for key in grouped]

        averages = {
            "avg_stress": round(sum(stress_series) / max(1, len(stress_series)), 1),
            "avg_fatigue": round(sum(fatigue_series) / max(1, len(fatigue_series)), 1),
            "avg_focus": round(sum(focus_series) / max(1, len(focus_series)), 1),
        }
        period_label = {
            "day": "前一天",
            "week": "前一周",
            "month": "前一个月",
            "custom": (
                f"{start_date.strftime('%Y-%m-%d')} 至 {end_date.strftime('%Y-%m-%d')}"
                if start_date is not None and end_date is not None
                else "自定义日期"
            ),
        }.get(period, "前一天")
        return {
            "line_categories": categories,
            "line_series": {
                "压力": stress_series,
                "疲劳": fatigue_series,
                "专注": focus_series,
            },
            "emotion_distribution": [
                {"name": row["emotion"], "value": row["count"]}
                for row in emotion_rows
            ],
            "signal_distribution": [
                {"name": row["label"], "value": row["count"]}
                for row in signal_rows
            ],
            "current_snapshot": json.loads(current_state["value"]) if current_state else None,
            "averages": averages,
            "period": period,
            "period_label": period_label,
            "sample_count": len(rows),
            "range_start": start_at.isoformat(timespec="minutes"),
            "range_end": end_at.isoformat(timespec="minutes"),
            "latest_events": [
                row["event_text"]
                for row in rows[-5:]
                if row["event_text"]
            ],
        }

    def get_report_payload(self) -> dict[str, Any]:
        with self._connect() as connection:
            today_rows = connection.execute(
                """
                SELECT emotion, stress_score, fatigue_score, focus_score
                FROM metric_snapshots
                WHERE recorded_at >= ?
                ORDER BY recorded_at ASC
                """,
                ((datetime.now() - timedelta(days=1)).isoformat(timespec="minutes"),),
            ).fetchall()

            week_rows = connection.execute(
                """
                SELECT emotion, stress_score, fatigue_score, focus_score
                FROM metric_snapshots
                WHERE recorded_at >= ?
                ORDER BY recorded_at ASC
                """,
                ((datetime.now() - timedelta(days=7)).isoformat(timespec="minutes"),),
            ).fetchall()

        def _avg(rows: list[sqlite3.Row], field: str) -> float:
            if not rows:
                return 0.0
            return round(sum(row[field] for row in rows) / len(rows), 1)

        emotion_counter: dict[str, int] = {}
        for row in week_rows:
            emotion_counter[row["emotion"]] = emotion_counter.get(row["emotion"], 0) + 1
        top_emotion = max(emotion_counter, key=emotion_counter.get) if emotion_counter else "平稳"

        return {
            "today_average_stress": _avg(list(today_rows), "stress_score"),
            "today_average_fatigue": _avg(list(today_rows), "fatigue_score"),
            "today_average_focus": _avg(list(today_rows), "focus_score"),
            "week_average_stress": _avg(list(week_rows), "stress_score"),
            "week_average_fatigue": _avg(list(week_rows), "fatigue_score"),
            "week_average_focus": _avg(list(week_rows), "focus_score"),
            "top_emotion": top_emotion,
            "week_samples": len(week_rows),
        }

    def get_report_summary(
        self,
        *,
        start_date: date,
        end_date: date,
    ) -> dict[str, Any]:
        range_start = datetime.combine(start_date, datetime.min.time())
        range_end = datetime.combine(end_date + timedelta(days=1), datetime.min.time())
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT emotion, stress_score, fatigue_score, focus_score, dominant_signal, event_text
                FROM metric_snapshots
                WHERE recorded_at >= ? AND recorded_at < ?
                ORDER BY recorded_at ASC
                """,
                (
                    range_start.isoformat(timespec="minutes"),
                    range_end.isoformat(timespec="minutes"),
                ),
            ).fetchall()

        def _avg(field: str) -> float:
            if not rows:
                return 0.0
            return round(sum(row[field] for row in rows) / len(rows), 1)

        emotion_counter: dict[str, int] = {}
        signal_counter: dict[str, int] = {}
        for row in rows:
            emotion_counter[row["emotion"]] = emotion_counter.get(row["emotion"], 0) + 1
            signal_key = "稳定" if row["dominant_signal"] == "none" else row["dominant_signal"]
            signal_counter[signal_key] = signal_counter.get(signal_key, 0) + 1

        top_emotion = max(emotion_counter, key=emotion_counter.get) if emotion_counter else "平稳"
        top_signal = max(signal_counter, key=signal_counter.get) if signal_counter else "稳定"
        high_stress_count = sum(1 for row in rows if row["stress_score"] >= 75)
        high_fatigue_count = sum(1 for row in rows if row["fatigue_score"] >= 75)
        rest_activity_count = sum(
            1
            for row in rows
            if any(keyword in (row["event_text"] or "") for keyword in ("休息", "放松", "离屏", "呼吸", "白噪音"))
        )
        events = [row["event_text"] for row in rows[-5:] if row["event_text"]]

        return {
            "start_date": start_date.isoformat(),
            "end_date": end_date.isoformat(),
            "sample_count": len(rows),
            "average_stress": _avg("stress_score"),
            "average_fatigue": _avg("fatigue_score"),
            "average_focus": _avg("focus_score"),
            "peak_stress": max((row["stress_score"] for row in rows), default=0),
            "peak_fatigue": max((row["fatigue_score"] for row in rows), default=0),
            "lowest_focus": min((row["focus_score"] for row in rows), default=0),
            "top_emotion": top_emotion,
            "top_signal": top_signal,
            "high_stress_count": high_stress_count,
            "high_fatigue_count": high_fatigue_count,
            "rest_activity_count": rest_activity_count,
            "events": events,
        }
