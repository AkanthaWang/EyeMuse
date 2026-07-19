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
    key_rate_per_min: float = 0.0
    keyboard_active_seconds: int = 0
    keyboard_activity: float = 0.0
    keyboard_declined: bool = False
    mouse_distance: float = 0.0
    mouse_active_seconds: int = 0
    mouse_activity: float = 0.0
    mouse_declined: bool = False
    modality_switches: int = 0
    behavior_state: str = "warming"
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
                    key_rate_per_min REAL NOT NULL DEFAULT 0,
                    keyboard_active_seconds INTEGER NOT NULL DEFAULT 0,
                    keyboard_activity REAL NOT NULL DEFAULT 0,
                    keyboard_declined INTEGER NOT NULL DEFAULT 0,
                    mouse_distance REAL NOT NULL DEFAULT 0,
                    mouse_active_seconds INTEGER NOT NULL DEFAULT 0,
                    mouse_activity REAL NOT NULL DEFAULT 0,
                    mouse_declined INTEGER NOT NULL DEFAULT 0,
                    modality_switches INTEGER NOT NULL DEFAULT 0,
                    behavior_state TEXT NOT NULL DEFAULT 'warming',
                    source TEXT NOT NULL
                );

                CREATE INDEX IF NOT EXISTS idx_metric_snapshots_recorded_at
                ON metric_snapshots(recorded_at);
                """
            )
            self._ensure_metric_snapshot_columns(cursor)
            demo_stats = cursor.execute(
                """
                SELECT COUNT(*) AS count, MAX(recorded_at) AS latest_at
                FROM metric_snapshots
                WHERE source = 'demo'
                """
            ).fetchone()
            expected_demo_count = 46 * 8
            expected_latest_at = (
                datetime.now() - timedelta(days=1)
            ).replace(hour=21, minute=0, second=0, microsecond=0)
            latest_demo_at = (
                datetime.fromisoformat(demo_stats["latest_at"])
                if demo_stats["latest_at"]
                else None
            )
            demo_data_is_stale = (
                demo_stats["count"] < expected_demo_count
                or latest_demo_at is None
                or latest_demo_at < expected_latest_at
            )
            if demo_data_is_stale:
                cursor.execute("DELETE FROM metric_snapshots WHERE source = 'demo'")
                self._seed_demo_data(connection)
            self._ensure_demo_rest_events(cursor)

    @staticmethod
    def _ensure_metric_snapshot_columns(cursor: sqlite3.Cursor) -> None:
        existing_columns = {
            row["name"]
            for row in cursor.execute("PRAGMA table_info(metric_snapshots)").fetchall()
        }
        required_columns = {
            "key_rate_per_min": "REAL NOT NULL DEFAULT 0",
            "keyboard_active_seconds": "INTEGER NOT NULL DEFAULT 0",
            "keyboard_activity": "REAL NOT NULL DEFAULT 0",
            "keyboard_declined": "INTEGER NOT NULL DEFAULT 0",
            "mouse_distance": "REAL NOT NULL DEFAULT 0",
            "mouse_active_seconds": "INTEGER NOT NULL DEFAULT 0",
            "mouse_activity": "REAL NOT NULL DEFAULT 0",
            "mouse_declined": "INTEGER NOT NULL DEFAULT 0",
            "modality_switches": "INTEGER NOT NULL DEFAULT 0",
            "behavior_state": "TEXT NOT NULL DEFAULT 'warming'",
        }
        for column_name, definition in required_columns.items():
            if column_name not in existing_columns:
                cursor.execute(f"ALTER TABLE metric_snapshots ADD COLUMN {column_name} {definition}")

    @staticmethod
    def _ensure_demo_rest_events(cursor: sqlite3.Cursor) -> None:
        cursor.execute(
            """
            UPDATE metric_snapshots
            SET event_text = CASE strftime('%H', recorded_at)
                WHEN '12' THEN 'Demo：完成 10 分钟离屏休息，状态恢复稳定'
                WHEN '18' THEN 'Demo：完成 3 轮呼吸放松，压力有所缓解'
                ELSE event_text
            END
            WHERE source = 'demo'
              AND strftime('%H', recorded_at) IN ('12', '18')
              AND event_text NOT LIKE '%离屏休息%'
              AND event_text NOT LIKE '%呼吸放松%'
            """
        )

    def _seed_demo_data(self, connection: sqlite3.Connection) -> None:
        rng = random.Random(42)
        now = datetime.now().replace(second=0, microsecond=0)
        start = (now - timedelta(days=46)).replace(hour=0, minute=0)
        rows: list[tuple[Any, ...]] = []

        sample_count = 46 * 8
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
            key_rate_per_min = round(max(18.0, min(220.0, focus_score * 2.1 + rng.randint(-22, 22))), 1)
            keyboard_active_seconds = max(5, min(30, int(key_rate_per_min / 8 + rng.randint(-2, 2))))
            keyboard_activity = round(
                0.7 * min(key_rate_per_min / 200.0, 1.0) + 0.3 * (keyboard_active_seconds / 30.0),
                3,
            )
            mouse_distance = round(max(280.0, min(7600.0, 5200 - fatigue_score * 28 + rng.randint(-480, 480))), 1)
            mouse_active_seconds = max(4, min(30, int(mouse_distance / 260 + rng.randint(-2, 2))))
            mouse_activity = round(
                0.5 * min(mouse_distance / 6000.0, 1.0) + 0.5 * (mouse_active_seconds / 30.0),
                3,
            )
            modality_switches = max(1, int((stress_score + focus_score) / 12) + rng.randint(-2, 2))

            if fatigue_score >= 76:
                emotion = "疲惫"
                mood = "alert"
                dominant_signal = "eye_squint"
                event_text = "Demo：检测到疲劳上升"
                keyboard_declined = 1
                mouse_declined = 1
                behavior_state = "fatigued"
            elif stress_score >= 72:
                emotion = "焦虑"
                mood = "thinking"
                dominant_signal = "brow_furrow"
                event_text = "Demo：压力偏高，建议放松"
                keyboard_declined = 0
                mouse_declined = 0
                modality_switches = max(modality_switches, 12)
                behavior_state = "anxious"
            elif focus_score >= 72:
                emotion = "专注"
                mood = "responding"
                dominant_signal = "none"
                event_text = "Demo：专注度稳定"
                keyboard_declined = 0
                mouse_declined = 0
                behavior_state = "steady"
            else:
                emotion = "平稳"
                mood = "idle"
                dominant_signal = "none"
                event_text = "Demo：状态平稳"
                keyboard_declined = 0
                mouse_declined = 0
                behavior_state = "warming"

            if current.hour == 12:
                event_text = "Demo：完成 10 分钟离屏休息，状态恢复稳定"
            elif current.hour == 18:
                event_text = "Demo：完成 3 轮呼吸放松，压力有所缓解"

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
                    key_rate_per_min,
                    keyboard_active_seconds,
                    keyboard_activity,
                    keyboard_declined,
                    mouse_distance,
                    mouse_active_seconds,
                    mouse_activity,
                    mouse_declined,
                    modality_switches,
                    behavior_state,
                    "demo",
                )
            )

        connection.executemany(
            """
            INSERT INTO metric_snapshots (
                recorded_at, mood, emotion, stress_score, fatigue_score, focus_score,
                face_count, dominant_signal, event_text, camera_enabled,
                key_rate_per_min, keyboard_active_seconds, keyboard_activity, keyboard_declined,
                mouse_distance, mouse_active_seconds, mouse_activity, mouse_declined,
                modality_switches, behavior_state, source
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
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
                        face_count, dominant_signal, event_text, camera_enabled,
                        key_rate_per_min, keyboard_active_seconds, keyboard_activity, keyboard_declined,
                        mouse_distance, mouse_active_seconds, mouse_activity, mouse_declined,
                        modality_switches, behavior_state, source
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
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
                        snapshot.key_rate_per_min,
                        snapshot.keyboard_active_seconds,
                        snapshot.keyboard_activity,
                        int(snapshot.keyboard_declined),
                        snapshot.mouse_distance,
                        snapshot.mouse_active_seconds,
                        snapshot.mouse_activity,
                        int(snapshot.mouse_declined),
                        snapshot.modality_switches,
                        snapshot.behavior_state,
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
                       face_count, dominant_signal, event_text, camera_enabled,
                       keyboard_activity, mouse_activity, modality_switches, behavior_state
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
                    "keyboard_activity": [],
                    "mouse_activity": [],
                    "switches": [],
                },
            )
            stats["stress"].append(row["stress_score"])
            stats["fatigue"].append(row["fatigue_score"])
            stats["focus"].append(row["focus_score"])
            stats["keyboard_activity"].append(float(row["keyboard_activity"]))
            stats["mouse_activity"].append(float(row["mouse_activity"]))
            stats["switches"].append(int(row["modality_switches"]))

        categories = [grouped[key]["label"] for key in grouped]
        stress_series = [round(sum(grouped[key]["stress"]) / len(grouped[key]["stress"]), 1) for key in grouped]
        fatigue_series = [round(sum(grouped[key]["fatigue"]) / len(grouped[key]["fatigue"]), 1) for key in grouped]
        focus_series = [round(sum(grouped[key]["focus"]) / len(grouped[key]["focus"]), 1) for key in grouped]
        keyboard_series = [round(sum(grouped[key]["keyboard_activity"]) / len(grouped[key]["keyboard_activity"]), 3) for key in grouped]
        mouse_series = [round(sum(grouped[key]["mouse_activity"]) / len(grouped[key]["mouse_activity"]), 3) for key in grouped]
        switch_series = [round(sum(grouped[key]["switches"]) / len(grouped[key]["switches"]), 1) for key in grouped]

        averages = {
            "avg_stress": round(sum(stress_series) / max(1, len(stress_series)), 1),
            "avg_fatigue": round(sum(fatigue_series) / max(1, len(fatigue_series)), 1),
            "avg_focus": round(sum(focus_series) / max(1, len(focus_series)), 1),
            "avg_keyboard_activity": round(sum(keyboard_series) / max(1, len(keyboard_series)), 3),
            "avg_mouse_activity": round(sum(mouse_series) / max(1, len(mouse_series)), 3),
            "avg_switches": round(sum(switch_series) / max(1, len(switch_series)), 1),
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
                "键盘活跃": keyboard_series,
                "鼠标活跃": mouse_series,
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
                SELECT recorded_at, emotion, stress_score, fatigue_score, focus_score, dominant_signal, event_text,
                       key_rate_per_min, keyboard_activity, keyboard_declined,
                       mouse_distance, mouse_activity, mouse_declined,
                       modality_switches, behavior_state
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
        behavior_counter: dict[str, int] = {}
        for row in rows:
            emotion_counter[row["emotion"]] = emotion_counter.get(row["emotion"], 0) + 1
            signal_key = "稳定" if row["dominant_signal"] == "none" else row["dominant_signal"]
            signal_counter[signal_key] = signal_counter.get(signal_key, 0) + 1
            behavior_counter[row["behavior_state"]] = behavior_counter.get(row["behavior_state"], 0) + 1

        top_emotion = max(emotion_counter, key=emotion_counter.get) if emotion_counter else "平稳"
        top_signal = max(signal_counter, key=signal_counter.get) if signal_counter else "稳定"
        top_behavior_state = max(behavior_counter, key=behavior_counter.get) if behavior_counter else "warming"
        high_stress_count = sum(1 for row in rows if row["stress_score"] >= 75)
        high_fatigue_count = sum(1 for row in rows if row["fatigue_score"] >= 75)
        keyboard_decline_count = sum(1 for row in rows if row["keyboard_declined"])
        mouse_decline_count = sum(1 for row in rows if row["mouse_declined"])
        high_switch_count = sum(1 for row in rows if row["modality_switches"] >= 10)
        rest_activity_count = sum(
            1
            for row in rows
            if any(keyword in (row["event_text"] or "") for keyword in ("休息", "放松", "离屏", "呼吸", "白噪音"))
        )
        events = [row["event_text"] for row in rows[-5:] if row["event_text"]]

        def _avg_list(values: list[float]) -> float:
            if not values:
                return 0.0
            return round(sum(values) / len(values), 1)

        def _efficiency_score(row: sqlite3.Row) -> float:
            return round(
                max(
                    0.0,
                    min(
                        100.0,
                        row["focus_score"] * 0.58
                        + row["keyboard_activity"] * 16.0
                        + row["mouse_activity"] * 12.0
                        - row["stress_score"] * 0.18
                        - row["fatigue_score"] * 0.14,
                    ),
                ),
                1,
            )

        hourly_scores: dict[int, list[float]] = {}
        state_scores: dict[str, list[float]] = {}
        low_fatigue_work_samples: list[tuple[datetime, float, float]] = []
        focus_series: list[float] = []
        stress_series: list[float] = []
        fatigue_series: list[float] = []

        for row in rows:
            score = _efficiency_score(row)
            try:
                recorded_at = datetime.fromisoformat(str(row["recorded_at"]))
                hourly_scores.setdefault(recorded_at.hour, []).append(score)
                is_low_fatigue = row["fatigue_score"] <= 45
                is_working = (
                    row["focus_score"] >= 50
                    and (
                        row["keyboard_activity"] >= 0.08
                        or row["mouse_activity"] >= 0.08
                        or row["behavior_state"] == "focused"
                    )
                )
                if is_low_fatigue and is_working:
                    low_fatigue_work_samples.append(
                        (recorded_at, float(row["fatigue_score"]), float(row["focus_score"]))
                    )
            except ValueError:
                pass

            state_label = f"{row['emotion']} / {row['behavior_state']}"
            state_scores.setdefault(state_label, []).append(score)

            focus_series.append(float(row["focus_score"]))
            stress_series.append(float(row["stress_score"]))
            fatigue_series.append(float(row["fatigue_score"]))

        best_hour = None
        best_hour_label = "暂无数据"
        best_hour_score = 0.0
        if hourly_scores:
            best_hour, values = max(hourly_scores.items(), key=lambda item: _avg_list(item[1]))
            best_hour_label = f"{best_hour:02d}:00-{(best_hour + 2) % 24:02d}:00"
            best_hour_score = _avg_list(values)

        best_state = "平稳 / warming"
        best_state_score = 0.0
        if state_scores:
            best_state, values = max(state_scores.items(), key=lambda item: _avg_list(item[1]))
            best_state_score = _avg_list(values)

        low_fatigue_periods: list[list[tuple[datetime, float, float]]] = []
        for sample in low_fatigue_work_samples:
            if not low_fatigue_periods or sample[0] - low_fatigue_periods[-1][-1][0] > timedelta(minutes=5):
                low_fatigue_periods.append([sample])
            else:
                low_fatigue_periods[-1].append(sample)

        period_stats: list[tuple[float, float, str]] = []
        for period_rows in low_fatigue_periods:
            if len(period_rows) < 2:
                continue
            period_start = period_rows[0][0]
            period_end = period_rows[-1][0]
            duration_minutes = max(1.0, (period_end - period_start).total_seconds() / 60.0 + 1.0)
            average_period_fatigue = _avg_list([sample[1] for sample in period_rows])
            average_period_focus = _avg_list([sample[2] for sample in period_rows])
            if period_start.date() == period_end.date():
                time_range = (
                    f"{period_start.strftime('%m-%d %H:%M')}-"
                    f"{period_end.strftime('%H:%M')}"
                )
            else:
                time_range = (
                    f"{period_start.strftime('%m-%d %H:%M')}-"
                    f"{period_end.strftime('%m-%d %H:%M')}"
                )
            period_text = (
                f"{time_range} · 低疲劳持续工作 {round(duration_minutes)} 分钟，"
                f"平均疲劳度 {average_period_fatigue}，平均专注度 {average_period_focus}"
            )
            period_stats.append((duration_minutes, average_period_fatigue, period_text))

        highlight_moments = [
            text
            for _duration, _fatigue, text in sorted(
                period_stats,
                key=lambda item: (-item[0], item[1]),
            )[:3]
        ]

        split_index = len(rows) // 2
        early_focus = _avg_list(focus_series[:split_index])
        late_focus = _avg_list(focus_series[split_index:])
        early_stress = _avg_list(stress_series[:split_index])
        late_stress = _avg_list(stress_series[split_index:])
        early_fatigue = _avg_list(fatigue_series[:split_index])
        late_fatigue = _avg_list(fatigue_series[split_index:])
        focus_delta = round(late_focus - early_focus, 1)
        stress_delta = round(late_stress - early_stress, 1)
        fatigue_delta = round(late_fatigue - early_fatigue, 1)

        if focus_delta >= 5 and stress_delta <= 0:
            trend_summary = "本周期后半段专注度明显提升，说明节奏逐步进入稳定区。"
        elif focus_delta <= -5 or fatigue_delta >= 5:
            trend_summary = "本周期后半段出现效率回落，建议检查任务密度和恢复节奏。"
        else:
            trend_summary = "整体波动可控，效率与情绪状态保持在相对稳定区间。"

        anomaly_ratio = 0.0 if not rows else round(max(high_stress_count, high_fatigue_count) / len(rows), 3)
        needs_support = len(rows) >= 12 and _avg("fatigue_score") >= 72 and anomaly_ratio >= 0.3
        support_message = (
            "如果这种高疲劳或高压力状态已经持续数周，建议和可信任的人聊聊，或联系学校/企业 EAP、心理咨询与精神卫生资源。"
            if needs_support
            else ""
        )

        return {
            "start_date": start_date.isoformat(),
            "end_date": end_date.isoformat(),
            "sample_count": len(rows),
            "average_stress": _avg("stress_score"),
            "average_fatigue": _avg("fatigue_score"),
            "average_focus": _avg("focus_score"),
            "average_keyboard_activity": _avg("keyboard_activity"),
            "average_mouse_activity": _avg("mouse_activity"),
            "average_key_rate_per_min": _avg("key_rate_per_min"),
            "average_mouse_distance": _avg("mouse_distance"),
            "peak_stress": max((row["stress_score"] for row in rows), default=0),
            "peak_fatigue": max((row["fatigue_score"] for row in rows), default=0),
            "lowest_focus": min((row["focus_score"] for row in rows), default=0),
            "top_emotion": top_emotion,
            "top_signal": top_signal,
            "top_behavior_state": top_behavior_state,
            "high_stress_count": high_stress_count,
            "high_fatigue_count": high_fatigue_count,
            "keyboard_decline_count": keyboard_decline_count,
            "mouse_decline_count": mouse_decline_count,
            "high_switch_count": high_switch_count,
            "rest_activity_count": rest_activity_count,
            "events": events,
            "best_hour": best_hour,
            "best_hour_label": best_hour_label,
            "best_hour_score": best_hour_score,
            "best_state": best_state,
            "best_state_score": best_state_score,
            "trend_summary": trend_summary,
            "focus_delta": focus_delta,
            "stress_delta": stress_delta,
            "fatigue_delta": fatigue_delta,
            "highlight_moments": highlight_moments,
            "needs_support": needs_support,
            "support_message": support_message,
        }
