"""SQLite 图存储：节点/边/文档块/事实块（超边）。"""
import json
import sqlite3
from typing import Dict, List, Optional, Tuple

import numpy as np
from loguru import logger

from .models import Edge, Node
from .exceptions import StoreError

_SCHEMA = """
CREATE TABLE IF NOT EXISTS nodes (
    id TEXT PRIMARY KEY,
    type TEXT NOT NULL,
    label TEXT,
    properties TEXT
);
CREATE TABLE IF NOT EXISTS edges (
    src TEXT NOT NULL,
    rel TEXT NOT NULL,
    dst TEXT NOT NULL,
    source_doc TEXT,
    PRIMARY KEY (src, rel, dst)
);
CREATE TABLE IF NOT EXISTS chunks (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    doc TEXT NOT NULL,
    text TEXT NOT NULL,
    embedding BLOB
);
CREATE TABLE IF NOT EXISTS facts (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    subject TEXT NOT NULL,
    text TEXT NOT NULL,
    source_doc TEXT,
    embedding BLOB
);
CREATE INDEX IF NOT EXISTS idx_edges_src ON edges(src);
CREATE INDEX IF NOT EXISTS idx_edges_dst ON edges(dst);
"""


def _to_blob(vec: np.ndarray) -> bytes:
    return np.asarray(vec, dtype=np.float32).tobytes()


def _from_blob(blob: bytes) -> np.ndarray:
    return np.frombuffer(blob, dtype=np.float32)


class GraphStore:
    def __init__(self, db_path: str):
        try:
            self.conn = sqlite3.connect(db_path, check_same_thread=False, timeout=30)
            self.conn.execute("PRAGMA journal_mode=WAL")
            self.conn.execute("PRAGMA busy_timeout=30000")
            self.conn.executescript(_SCHEMA)
            self.conn.commit()
        except sqlite3.Error as exc:
            raise StoreError(str(exc), f"数据库打开失败: {db_path}")

    def reset(self):
        try:
            for table in ("nodes", "edges", "chunks", "facts", "live_data"):
                try:
                    self.conn.execute(f"DELETE FROM {table}")
                except sqlite3.OperationalError:
                    pass
            self.conn.commit()
        except sqlite3.Error as exc:
            raise StoreError(str(exc), "数据库重置失败，可能被其他进程锁定")

    # ---------- 写入 ----------
    def add_node(self, node_id: str, node_type: str, label: str = "", properties: Optional[dict] = None):
        self.conn.execute(
            "INSERT OR REPLACE INTO nodes (id, type, label, properties) VALUES (?,?,?,?)",
            (node_id, node_type, label or node_id, json.dumps(properties or {}, ensure_ascii=False)),
        )

    def add_edge(self, src: str, rel: str, dst: str, source_doc: str = ""):
        self.conn.execute(
            "INSERT OR REPLACE INTO edges (src, rel, dst, source_doc) VALUES (?,?,?,?)",
            (src, rel, dst, source_doc),
        )

    def add_chunk(self, doc: str, text: str, embedding: np.ndarray):
        self.conn.execute(
            "INSERT INTO chunks (doc, text, embedding) VALUES (?,?,?)",
            (doc, text, _to_blob(embedding)),
        )

    def add_fact(self, subject: str, text: str, source_doc: str, embedding: np.ndarray):
        self.conn.execute(
            "INSERT INTO facts (subject, text, source_doc, embedding) VALUES (?,?,?,?)",
            (subject, text, source_doc, _to_blob(embedding)),
        )

    def commit(self):
        self.conn.commit()

    # ---------- 检索 ----------
    def _load_embeddings(self, table: str) -> Tuple[List[tuple], np.ndarray]:
        rows = self.conn.execute(f"SELECT * FROM {table}").fetchall()
        if not rows:
            return [], np.zeros((0, 1), dtype=np.float32)
        vecs = np.stack([_from_blob(r[-1]) for r in rows])
        return rows, vecs

    def all_chunks(self) -> Tuple[List[tuple], np.ndarray]:
        """rows: (id, doc, text, embedding)"""
        return self._load_embeddings("chunks")

    def all_facts(self) -> Tuple[List[tuple], np.ndarray]:
        """rows: (id, subject, text, source_doc, embedding)"""
        return self._load_embeddings("facts")

    def get_node(self, node_id: str) -> Optional[dict]:
        row = self.conn.execute(
            "SELECT id, type, label, properties FROM nodes WHERE id=?", (node_id,)
        ).fetchone()
        if not row:
            return None
        node = Node(id=row[0], type=row[1], label=row[2], properties=json.loads(row[3] or "{}"))
        return node.model_dump()

    def neighbors(self, node_ids: List[str], hops: int = 1) -> List[Tuple[str, str, str]]:
        """返回指定节点集合 N 跳内的所有边 (src, rel, dst)。"""
        frontier = set(node_ids)
        seen_edges: set = set()
        for _ in range(hops):
            if not frontier:
                break
            placeholders = ",".join("?" * len(frontier))
            params = list(frontier) * 2
            rows = self.conn.execute(
                f"SELECT src, rel, dst FROM edges WHERE src IN ({placeholders}) OR dst IN ({placeholders})",
                params,
            ).fetchall()
            next_frontier = set()
            for src, rel, dst in rows:
                if (src, rel, dst) not in seen_edges:
                    seen_edges.add((src, rel, dst))
                    next_frontier.update((src, dst))
            frontier = next_frontier - frontier
        return sorted(seen_edges)

    def stats(self) -> Dict[str, int]:
        result = {}
        for table in ("nodes", "edges", "chunks", "facts"):
            result[table] = self.conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
        return result

    def all_edges(self) -> List[Tuple[str, str, str]]:
        return self.conn.execute("SELECT src, rel, dst FROM edges").fetchall()

    def all_nodes(self) -> List[dict]:
        rows = self.conn.execute("SELECT id, type, label, properties FROM nodes").fetchall()
        return [
            Node(id=r[0], type=r[1], label=r[2], properties=json.loads(r[3] or "{}")).model_dump()
            for r in rows
        ]
