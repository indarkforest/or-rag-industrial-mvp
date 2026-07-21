"""Pydantic 数据模型测试。"""
import pytest
from pydantic import ValidationError

from src.models import (
    AppConfig,
    DataConfig,
    EmbeddingConfig,
    EvalRow,
    EvalScore,
    ModelConfig,
    Node,
    QAResult,
    RetrievalConfig,
)


class TestModelConfig:
    def test_defaults(self):
        cfg = ModelConfig()
        assert cfg.api_key == ""
        assert cfg.temperature == 0.1

    def test_custom_values(self):
        cfg = ModelConfig(base_url="http://localhost:8000", api_key="sk-test", name="gpt-4")
        assert cfg.base_url == "http://localhost:8000"
        assert cfg.name == "gpt-4"


class TestEmbeddingConfig:
    def test_valid_provider(self):
        cfg = EmbeddingConfig(provider="hashing")
        assert cfg.provider == "hashing"

    def test_invalid_provider(self):
        with pytest.raises(ValidationError):
            EmbeddingConfig(provider="invalid")


class TestDataConfig:
    def test_valid_chunk_size(self):
        cfg = DataConfig(chunk_size=800)
        assert cfg.chunk_size == 800

    def test_too_small_chunk_size(self):
        with pytest.raises(ValidationError):
            DataConfig(chunk_size=50)


class TestRetrievalConfig:
    def test_defaults(self):
        cfg = RetrievalConfig()
        assert cfg.top_k_chunks == 4
        assert cfg.expand_hops == 1

    def test_invalid_hops(self):
        with pytest.raises(ValidationError):
            RetrievalConfig(expand_hops=10)

    def test_invalid_k(self):
        with pytest.raises(ValidationError):
            RetrievalConfig(top_k_chunks=0)


class TestAppConfig:
    def test_full_config(self):
        raw = {
            "model": {"base_url": "http://x", "api_key": "k", "name": "m"},
            "embedding": {"provider": "hashing"},
            "data": {"chunk_size": 500},
            "retrieval": {"top_k_chunks": 3, "expand_hops": 2},
        }
        cfg = AppConfig(**raw)
        assert cfg.model.name == "m"
        assert cfg.embedding.provider == "hashing"
        assert cfg.data.chunk_size == 500
        assert cfg.retrieval.expand_hops == 2

    def test_to_dict(self):
        cfg = AppConfig()
        d = cfg.to_dict()
        assert isinstance(d, dict)
        assert "model" in d
        assert "embedding" in d

    def test_missing_section_uses_defaults(self):
        cfg = AppConfig(model={"api_key": "k"})
        assert cfg.embedding.provider == "api"
        assert cfg.data.chunk_size == 800


class TestNode:
    def test_basic(self):
        node = Node(id="R-101", type="Device", label="反应釜")
        assert node.id == "R-101"
        assert node.label == "反应釜"
        assert node.properties == {}

    def test_with_properties(self):
        node = Node(id="TI-101", type="Sensor", properties={"range": "0~150℃"})
        assert node.properties["range"] == "0~150℃"


class TestQAResult:
    def test_basic(self):
        r = QAResult(retriever="naive-rag", question="测试问题", answer="测试回答")
        assert r.retriever == "naive-rag"
        assert r.contexts == []

    def test_model_dump(self):
        r = QAResult(retriever="og-rag", question="Q", contexts=["c1"], answer="A")
        d = r.model_dump()
        assert d["retriever"] == "og-rag"
        assert d["contexts"] == ["c1"]


class TestEvalModels:
    def test_eval_score_defaults(self):
        s = EvalScore()
        assert s.correctness == 0.0
        assert s.comment == ""

    def test_eval_row(self):
        row = EvalRow(id="q1", question="Q", retriever="naive-rag", answer="A")
        assert row.id == "q1"
        assert row.type == ""
