"""实时数据提供者：模拟工业实时数据 + 旋转门压缩 + 注入检索上下文。"""
import json
import random
import sqlite3
from datetime import datetime, timedelta
from typing import Dict, List, Optional

from loguru import logger

_LIVE_SCHEMA = """
CREATE TABLE IF NOT EXISTS live_data (
    entity_id    TEXT NOT NULL,
    timestamp    TEXT NOT NULL,
    value        REAL NOT NULL,
    unit         TEXT,
    status       TEXT,
    trend_points TEXT
);
"""

# 实体类型 -> 模拟数据范围配置
_SIM_RANGES: Dict[str, dict] = {
    "Sensor": {
        "TI": {"unit": "℃", "min": 20, "max": 200, "normal_low": 80, "normal_high": 90},
        "FI": {"unit": "m³/h", "min": 0, "max": 50, "normal_low": 18, "normal_high": 22},
        "PI": {"unit": "MPa", "min": 0, "max": 5, "normal_low": 0.5, "normal_high": 0.8},
        "SI": {"unit": "rpm", "min": 0, "max": 300, "normal_low": 110, "normal_high": 130},
        "ZT": {"unit": "%", "min": 0, "max": 100, "normal_low": 40, "normal_high": 70},
        "LI": {"unit": "%", "min": 0, "max": 100, "normal_low": 30, "normal_high": 70},
        "AI": {"unit": "%", "min": 0, "max": 100, "normal_low": 20, "normal_high": 60},
        "default": {"unit": "", "min": 0, "max": 100, "normal_low": 30, "normal_high": 70},
    },
    "ControlLoop": {
        "TIC": {"unit": "%", "min": 0, "max": 100, "normal_low": 40, "normal_high": 70},
        "FIC": {"unit": "%", "min": 0, "max": 100, "normal_low": 30, "normal_high": 60},
        "PIC": {"unit": "%", "min": 0, "max": 100, "normal_low": 30, "normal_high": 60},
        "LIC": {"unit": "%", "min": 0, "max": 100, "normal_low": 30, "normal_high": 60},
        "default": {"unit": "%", "min": 0, "max": 100, "normal_low": 30, "normal_high": 70},
    },
    "Device": {
        "default": {"unit": "运行状态", "min": 0, "max": 1, "normal_low": 1, "normal_high": 1},
    },
}

# 支持实时数据的实体类型
_LIVE_TYPES = {"Device", "Sensor", "ControlLoop"}


def _get_range(entity_id: str, entity_type: str) -> dict:
    """根据实体类型和 ID 前缀推断模拟数据范围。"""
    type_cfg = _SIM_RANGES.get(entity_type, {})
    for prefix in sorted(type_cfg.keys(), key=len, reverse=True):
        if prefix != "default" and entity_id.startswith(prefix):
            return type_cfg[prefix]
    return type_cfg.get("default", {"unit": "", "min": 0, "max": 100, "normal_low": 30, "normal_high": 70})


def _swinging_door_compress(
    points: List[tuple], tolerance: float = 0.02
) -> List[tuple]:
    """旋转门压缩算法。

    points: [(timestamp_str, value), ...]
    tolerance: 容差比例（相对于数据范围）
    返回压缩后的关键点列表。
    """
    if len(points) <= 2:
        return points

    values = [p[1] for p in points]
    val_range = max(values) - min(values)
    if val_range == 0:
        return [points[0], points[-1]]

    door = val_range * tolerance
    kept = [points[0]]
    anchor_idx = 0

    for i in range(1, len(points)):
        # 计算当前点相对于锚点的变化率
        anchor_val = points[anchor_idx][1]
        cur_val = points[i][1]
        delta = cur_val - anchor_val

        # 如果超出容差门，保存当前点的前一个点作为关键点
        if abs(delta) > door:
            if i - 1 > anchor_idx:
                kept.append(points[i - 1])
            kept.append(points[i])
            anchor_idx = i

    # 确保最后一个点被保留
    if kept[-1] != points[-1]:
        kept.append(points[-1])

    return kept


def _determine_status(value: float, cfg: dict) -> str:
    """根据当前值和正常范围判断状态。"""
    nl, nh = cfg["normal_low"], cfg["normal_high"]
    val_range = cfg["max"] - cfg["min"]
    if val_range == 0:
        return "normal"
    margin = val_range * 0.1
    if value < nl - margin or value > nh + margin:
        return "alarm"
    if value > nh - margin or value < nl + margin:
        if value > nh:
            return "rising"
        if value < nl:
            return "falling"
    return "normal"


class LiveDataProvider:
    """实时数据提供者：管理 SQLite live_data 表，支持模拟生成、旋转门压缩、查询。"""

    def __init__(self, db_path: str):
        self.conn = sqlite3.connect(db_path, check_same_thread=False, timeout=30)
        self.conn.execute("PRAGMA journal_mode=WAL")
        self.conn.execute("PRAGMA busy_timeout=30000")
        self.conn.executescript(_LIVE_SCHEMA)
        self.conn.commit()

    def reset(self):
        self.conn.execute("DELETE FROM live_data")
        self.conn.commit()

    def generate_sim_data(
        self,
        nodes: List[dict],
        num_points: int = 20,
        interval_minutes: float = 1.5,
        tolerance: float = 0.02,
    ) -> int:
        """为 Device/Sensor/ControlLoop 类型节点生成模拟实时数据。

        返回生成的实体数量。
        """
        self.reset()
        now = datetime.now()
        count = 0
        for node in nodes:
            ntype = node.get("type", "")
            if ntype not in _LIVE_TYPES:
                continue
            nid = node["id"]
            cfg = _get_range(nid, ntype)

            # 生成原始采样点
            base_val = random.uniform(cfg["normal_low"], cfg["normal_high"])
            # 随机选择趋势方向
            trend_dir = random.choice([-1, 0, 1])
            trend_magnitude = (cfg["max"] - cfg["min"]) * random.uniform(0.01, 0.05)

            raw_points = []
            for i in range(num_points):
                ts = now - timedelta(minutes=(num_points - 1 - i) * interval_minutes)
                noise = random.gauss(0, (cfg["max"] - cfg["min"]) * 0.005)
                val = base_val + trend_dir * trend_magnitude * i + noise
                val = max(cfg["min"], min(cfg["max"], val))
                raw_points.append((ts.strftime("%H:%M:%S"), round(val, 2)))

            # 旋转门压缩
            compressed = _swinging_door_compress(raw_points, tolerance)

            # 当前值 = 最后一个点
            cur_val = raw_points[-1][1]
            status = _determine_status(cur_val, cfg)

            trend_json = json.dumps(
                [{"t": p[0], "v": p[1]} for p in compressed], ensure_ascii=False
            )

            self.conn.execute(
                "INSERT INTO live_data (entity_id, timestamp, value, unit, status, trend_points) VALUES (?,?,?,?,?,?)",
                (
                    nid,
                    now.strftime("%Y-%m-%dT%H:%M:%S"),
                    cur_val,
                    cfg["unit"],
                    status,
                    trend_json,
                ),
            )
            count += 1

        self.conn.commit()
        logger.info("生成 %d 个实体的模拟实时数据", count)
        return count

    def get(self, entity_id: str) -> Optional[dict]:
        """查询单个实体的最新实时数据。"""
        row = self.conn.execute(
            "SELECT entity_id, timestamp, value, unit, status, trend_points FROM live_data WHERE entity_id=? ORDER BY timestamp DESC LIMIT 1",
            (entity_id,),
        ).fetchone()
        if not row:
            return None
        return {
            "entity_id": row[0],
            "timestamp": row[1],
            "value": row[2],
            "unit": row[3],
            "status": row[4],
            "trend_points": json.loads(row[5] or "[]"),
        }

    def get_batch(self, entity_ids: List[str]) -> Dict[str, dict]:
        """批量查询多个实体的实时数据。"""
        result = {}
        for eid in entity_ids:
            data = self.get(eid)
            if data:
                result[eid] = data
        return result

    def all_live(self) -> List[dict]:
        """返回所有实时数据记录。"""
        rows = self.conn.execute(
            "SELECT entity_id, timestamp, value, unit, status, trend_points FROM live_data ORDER BY entity_id"
        ).fetchall()
        return [
            {
                "entity_id": r[0],
                "timestamp": r[1],
                "value": r[2],
                "unit": r[3],
                "status": r[4],
                "trend_points": json.loads(r[5] or "[]"),
            }
            for r in rows
        ]

    def format_for_prompt(self, entity_ids: List[str]) -> str:
        """将命中的实体实时数据格式化为 prompt 文本。"""
        batch = self.get_batch(entity_ids)
        if not batch:
            return ""
        lines = []
        ts = ""
        for eid, data in batch.items():
            ts = data["timestamp"]
            trend = data["trend_points"]
            if len(trend) >= 2:
                trend_str = "→".join(f"{p['v']}" for p in trend)
            else:
                trend_str = f"{data['value']}"
            status_map = {"normal": "正常", "rising": "上升", "falling": "下降", "alarm": "报警"}
            status_cn = status_map.get(data["status"], data["status"])
            lines.append(
                f"- {eid}: 当前 {data['value']}{data['unit']}, 状态: {status_cn}, "
                f"趋势: {trend_str}"
            )
        return f"【实时数据 | 采集时间: {ts}】\n" + "\n".join(lines)
