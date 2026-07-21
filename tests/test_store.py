"""GraphStore SQLite 存储测试。"""
import os
import tempfile

import numpy as np
import pytest

from src.store import GraphStore


@pytest.fixture
def store():
    """临时数据库的 GraphStore。"""
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        db_path = f.name
    s = GraphStore(db_path)
    yield s
    s.conn.close()
    os.unlink(db_path)
    for ext in ("-wal", "-shm"):
        p = db_path + ext
        if os.path.exists(p):
            os.unlink(p)


class TestGraphStoreCRUD:
    def test_add_and_get_node(self, store):
        store.add_node("R-101", "Device", "反应釜", {"range": "0~150℃"})
        store.commit()
        node = store.get_node("R-101")
        assert node is not None
        assert node["id"] == "R-101"
        assert node["type"] == "Device"
        assert node["label"] == "反应釜"
        assert node["properties"]["range"] == "0~150℃"

    def test_get_nonexistent_node(self, store):
        assert store.get_node("nonexistent") is None

    def test_add_edge(self, store):
        store.add_node("A", "Device", "A")
        store.add_node("B", "Device", "B")
        store.add_edge("A", "causes", "B", "doc1")
        store.commit()
        edges = store.all_edges()
        assert ("A", "causes", "B") in edges

    def test_add_chunk(self, store):
        vec = np.array([0.1, 0.2, 0.3], dtype=np.float32)
        store.add_chunk("doc1", "text content", vec)
        store.commit()
        rows, matrix = store.all_chunks()
        assert len(rows) == 1
        assert rows[0][1] == "doc1"
        assert rows[0][2] == "text content"
        assert matrix.shape == (1, 3)

    def test_add_fact(self, store):
        vec = np.array([0.4, 0.5], dtype=np.float32)
        store.add_fact("R-101", "fact text", "doc1", vec)
        store.commit()
        rows, matrix = store.all_facts()
        assert len(rows) == 1
        assert rows[0][1] == "R-101"
        assert rows[0][2] == "fact text"
        assert matrix.shape == (1, 2)


class TestGraphStoreNeighbors:
    def test_one_hop(self, store):
        store.add_edge("A", "causes", "B")
        store.add_edge("B", "causes", "C")
        store.commit()
        edges = store.neighbors(["A"], hops=1)
        assert ("A", "causes", "B") in edges

    def test_two_hops(self, store):
        store.add_edge("A", "causes", "B")
        store.add_edge("B", "causes", "C")
        store.commit()
        edges = store.neighbors(["A"], hops=2)
        assert ("A", "causes", "B") in edges
        assert ("B", "causes", "C") in edges

    def test_no_edges(self, store):
        edges = store.neighbors(["X"], hops=1)
        assert edges == []


class TestGraphStoreStats:
    def test_empty_stats(self, store):
        stats = store.stats()
        assert stats["nodes"] == 0
        assert stats["edges"] == 0
        assert stats["chunks"] == 0
        assert stats["facts"] == 0

    def test_populated_stats(self, store):
        store.add_node("A", "Device", "A")
        store.add_edge("A", "causes", "B")
        store.add_chunk("doc", "text", np.zeros(3, dtype=np.float32))
        store.commit()
        stats = store.stats()
        assert stats["nodes"] == 1
        assert stats["edges"] == 1
        assert stats["chunks"] == 1


class TestGraphStoreReset:
    def test_reset_clears_all(self, store):
        store.add_node("A", "Device", "A")
        store.add_edge("A", "causes", "B")
        store.add_chunk("doc", "text", np.zeros(3, dtype=np.float32))
        store.commit()
        store.reset()
        stats = store.stats()
        assert stats["nodes"] == 0
        assert stats["edges"] == 0
