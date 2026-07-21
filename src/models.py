"""Pydantic 数据模型：统一类型定义与配置校验。"""
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, field_validator


# ── 配置模型 ──

class ModelConfig(BaseModel):
    base_url: str = "https://api.minimaxi.com/v1"
    api_key: str = ""
    name: str = "MiniMax-M3"
    temperature: float = 0.1


class EmbeddingConfig(BaseModel):
    provider: str = "api"
    base_url: str = "https://api.minimaxi.com/v1"
    api_key: str = ""
    name: str = "embo-01"

    @field_validator("provider")
    @classmethod
    def valid_provider(cls, v: str) -> str:
        if v not in ("api", "hashing"):
            raise ValueError(f"provider 必须是 'api' 或 'hashing'，当前: {v}")
        return v


class DataConfig(BaseModel):
    docs_dir: str = "data/docs"
    ontology_path: str = "data/ontology/industrial_ontology.jsonld"
    questions_path: str = "data/questions.yaml"
    db_path: str = "data/kg.sqlite"
    chunk_size: int = 800

    @field_validator("chunk_size")
    @classmethod
    def positive_chunk(cls, v: int) -> int:
        if v < 100:
            raise ValueError(f"chunk_size 不能小于 100，当前: {v}")
        return v


class RetrievalConfig(BaseModel):
    top_k_chunks: int = 4
    top_k_facts: int = 6
    expand_hops: int = 1

    @field_validator("top_k_chunks", "top_k_facts")
    @classmethod
    def positive_k(cls, v: int) -> int:
        if v < 1:
            raise ValueError(f"top_k 不能小于 1，当前: {v}")
        return v

    @field_validator("expand_hops")
    @classmethod
    def valid_hops(cls, v: int) -> int:
        if v < 0 or v > 5:
            raise ValueError(f"expand_hops 应在 0-5 之间，当前: {v}")
        return v


class AppConfig(BaseModel):
    """顶层应用配置，对应 config.yaml。"""
    model: ModelConfig = Field(default_factory=ModelConfig)
    embedding: EmbeddingConfig = Field(default_factory=EmbeddingConfig)
    data: DataConfig = Field(default_factory=DataConfig)
    retrieval: RetrievalConfig = Field(default_factory=RetrievalConfig)

    def to_dict(self) -> dict:
        """转换为普通 dict，兼容现有代码的 dict 访问。"""
        return self.model_dump()


# ── 图谱数据模型 ──

class Node(BaseModel):
    id: str
    type: str
    label: str = ""
    properties: Dict[str, Any] = Field(default_factory=dict)


class Edge(BaseModel):
    src: str
    rel: str
    dst: str
    source_doc: str = ""


# ── 问答结果模型 ──

class QAResult(BaseModel):
    retriever: str
    question: str
    contexts: List[str] = Field(default_factory=list)
    answer: str = ""


# ── 评估模型 ──

class EvalScore(BaseModel):
    correctness: float = 0.0
    completeness: float = 0.0
    faithfulness: float = 0.0
    comment: str = ""


class EvalRow(BaseModel):
    id: str
    type: str = ""
    question: str
    retriever: str
    answer: str
    correctness: float = 0.0
    completeness: float = 0.0
    faithfulness: float = 0.0
    comment: str = ""


class EvalSummary(BaseModel):
    correctness: float = 0.0
    completeness: float = 0.0
    faithfulness: float = 0.0


class EvalResult(BaseModel):
    rows: List[EvalRow] = Field(default_factory=list)
    summary: Dict[str, EvalSummary] = Field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "rows": [r.model_dump() for r in self.rows],
            "summary": {k: v.model_dump() for k, v in self.summary.items()},
        }
