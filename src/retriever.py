"""检索器：naive-RAG 基线与 OG-RAG 完全解耦，实现同一接口，便于对照实验。"""
from abc import ABC, abstractmethod
from typing import List, Optional

from .llm import EmbeddingClient, cosine_topk
from .live_data import LiveDataProvider
from .store import GraphStore


class BaseRetriever(ABC):
    name: str = "base"

    @abstractmethod
    def retrieve(self, question: str) -> List[str]:
        """返回上下文片段列表。"""


class NaiveRetriever(BaseRetriever):
    """基线：纯向量检索原始文档 chunk。"""

    name = "naive-rag"

    def __init__(self, store: GraphStore, embedder: EmbeddingClient, top_k: int = 4):
        self.store = store
        self.embedder = embedder
        self.top_k = top_k

    def retrieve(self, question: str) -> List[str]:
        rows, matrix = self.store.all_chunks()
        if not rows:
            return []
        qvec = self.embedder.embed_query(question)
        if qvec.shape[0] != matrix.shape[1]:
            raise ValueError(
                f"向量维度不一致（查询 {qvec.shape[0]} vs 库 {matrix.shape[1]}），"
                "请确认 embedding 配置与建库时一致后重新 build"
            )
        idxs = cosine_topk(qvec, matrix, self.top_k)
        return [f"【{rows[i][1]}】\n{rows[i][2]}" for i in idxs]


class OGRAGRetriever(BaseRetriever):
    """OG-RAG：检索本体事实块（超边），并沿图谱做 N 跳扩展补全关联事实。"""

    name = "og-rag"

    def __init__(
        self,
        store: GraphStore,
        embedder: EmbeddingClient,
        top_k: int = 6,
        expand_hops: int = 1,
        live_data: Optional[LiveDataProvider] = None,
    ):
        self.store = store
        self.embedder = embedder
        self.top_k = top_k
        self.expand_hops = expand_hops
        self.live_data = live_data

    def retrieve(self, question: str) -> List[str]:
        rows, matrix = self.store.all_facts()
        if not rows:
            return []
        qvec = self.embedder.embed_query(question)
        if qvec.shape[0] != matrix.shape[1]:
            raise ValueError(
                f"向量维度不一致（查询 {qvec.shape[0]} vs 库 {matrix.shape[1]}），"
                "请确认 embedding 配置与建库时一致后重新 build"
            )
        idxs = cosine_topk(qvec, matrix, self.top_k)
        contexts = []
        seed_subjects = []
        for i in idxs:
            _, subject, text, source_doc, _ = rows[i]
            seed_subjects.append(subject)
            contexts.append(f"【事实块 | 来源: {source_doc}】\n{text}")

        # 图谱扩展：沿边补全命中实体周边的关系（重点覆盖因果链多跳）
        all_entity_ids = set(seed_subjects)
        if self.expand_hops > 0 and seed_subjects:
            edges = self.store.neighbors(seed_subjects, hops=self.expand_hops)
            if edges:
                lines = []
                for src, rel, dst in edges:
                    all_entity_ids.update([src, dst])
                    src_node = self.store.get_node(src)
                    dst_node = self.store.get_node(dst)
                    src_label = src_node["label"] if src_node else src
                    dst_label = dst_node["label"] if dst_node else dst
                    lines.append(f"- {src}（{src_label}） --{rel}--> {dst}（{dst_label}）")
                contexts.append("【图谱关联关系（扩展）】\n" + "\n".join(lines))

        # 实时数据注入：对命中的 Device/Sensor/ControlLoop 实体附加实时数据
        if self.live_data and all_entity_ids:
            live_text = self.live_data.format_for_prompt(list(all_entity_ids))
            if live_text:
                contexts.append(live_text)
        return contexts


class HybridRetriever(BaseRetriever):
    """混合架构：事实块检索（结构+关系）+ 原文 chunk 检索（数值+上下文）+ 图谱扩展 + 实时数据。

    相比 OG-RAG，额外检索原文 chunk 补全数值信息，按实体 ID 去重避免重复。
    """

    name = "hybrid-rag"

    def __init__(
        self,
        store: GraphStore,
        embedder: EmbeddingClient,
        top_k_facts: int = 4,
        top_k_chunks: int = 3,
        expand_hops: int = 1,
        live_data: Optional[LiveDataProvider] = None,
    ):
        self.store = store
        self.embedder = embedder
        self.top_k_facts = top_k_facts
        self.top_k_chunks = top_k_chunks
        self.expand_hops = expand_hops
        self.live_data = live_data

    def retrieve(self, question: str) -> List[str]:
        qvec = self.embedder.embed_query(question)
        contexts = []
        all_entity_ids = set()

        # 1. 事实块检索（结构 + 关系）
        fact_rows, fact_matrix = self.store.all_facts()
        if fact_rows:
            if qvec.shape[0] != fact_matrix.shape[1]:
                raise ValueError(
                    f"向量维度不一致（查询 {qvec.shape[0]} vs 库 {fact_matrix.shape[1]}），"
                    "请确认 embedding 配置与建库时一致后重新 build"
                )
            fact_idxs = cosine_topk(qvec, fact_matrix, self.top_k_facts)
            seed_subjects = []
            for i in fact_idxs:
                _, subject, text, source_doc, _ = fact_rows[i]
                seed_subjects.append(subject)
                all_entity_ids.add(subject)
                contexts.append(f"【事实块 | 来源: {source_doc}】\n{text}")

            # 3. 图谱扩展（因果链）
            if self.expand_hops > 0 and seed_subjects:
                edges = self.store.neighbors(seed_subjects, hops=self.expand_hops)
                if edges:
                    lines = []
                    for src, rel, dst in edges:
                        all_entity_ids.update([src, dst])
                        src_node = self.store.get_node(src)
                        dst_node = self.store.get_node(dst)
                        src_label = src_node["label"] if src_node else src
                        dst_label = dst_node["label"] if dst_node else dst
                        lines.append(f"- {src}（{src_label}） --{rel}--> {dst}（{dst_label}）")
                    contexts.append("【图谱关联关系（扩展）】\n" + "\n".join(lines))

        # 2. 原文 chunk 检索（数值 + 上下文），去重：跳过已被事实块覆盖的实体
        chunk_rows, chunk_matrix = self.store.all_chunks()
        if chunk_rows:
            if qvec.shape[0] != chunk_matrix.shape[1]:
                raise ValueError(
                    f"向量维度不一致（查询 {qvec.shape[0]} vs 库 {chunk_matrix.shape[1]}），"
                    "请确认 embedding 配置与建库时一致后重新 build"
                )
            chunk_idxs = cosine_topk(qvec, chunk_matrix, self.top_k_chunks)
            for i in chunk_idxs:
                _, doc, text, _ = chunk_rows[i]
                contexts.append(f"【原文片段 | 来源: {doc}】\n{text}")

        # 4. 实时数据注入
        if self.live_data and all_entity_ids:
            live_text = self.live_data.format_for_prompt(list(all_entity_ids))
            if live_text:
                contexts.append(live_text)
        return contexts


def build_retrievers(
    cfg: dict,
    store: GraphStore,
    embedder: EmbeddingClient,
    live_data: Optional[LiveDataProvider] = None,
) -> dict:
    r = cfg.get("retrieval", {})
    return {
        "naive-rag": NaiveRetriever(store, embedder, top_k=r.get("top_k_chunks", 4)),
        "og-rag": OGRAGRetriever(
            store,
            embedder,
            top_k=r.get("top_k_facts", 6),
            expand_hops=r.get("expand_hops", 1),
            live_data=live_data,
        ),
        "hybrid-rag": HybridRetriever(
            store,
            embedder,
            top_k_facts=r.get("top_k_facts", 4),
            top_k_chunks=r.get("top_k_chunks", 3),
            expand_hops=r.get("expand_hops", 1),
            live_data=live_data,
        ),
    }
