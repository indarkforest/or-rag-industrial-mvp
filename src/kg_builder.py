"""知识图谱构建：文档 -> (LLM 本体映射) -> 三元组 + 事实块，写入 SQLite。

参考 OG-RAG 思路：以本体为 Schema，将文档内容映射为本体实例；
再按主语聚合三元组形成"事实块"（超边），作为 OG-RAG 检索单元。
同时保存原始 chunk 向量，供 naive-RAG 基线使用。
"""
import glob
import json
import os
from collections import defaultdict
from typing import List

from loguru import logger

from .llm import EmbeddingClient, LLMClient
from .ontology import Ontology
from .store import GraphStore


def _normalize_extract_result(result):
    """LLM 可能返回 list 或其他非 dict 格式，归一化为 {entities, triples} dict。"""
    if isinstance(result, dict):
        entities = result.get("entities", [])
        triples = result.get("triples", [])
        if not isinstance(entities, list):
            entities = []
        if not isinstance(triples, list):
            triples = []
        if not entities and not triples:
            return None
        return {"entities": entities, "triples": triples}
    if isinstance(result, list):
        entities, triples = [], []
        for item in result:
            if not isinstance(item, dict):
                continue
            if "id" in item and "type" in item:
                entities.append(item)
            elif "subject" in item and "relation" in item:
                triples.append(item)
        return {"entities": entities, "triples": triples}
    return None


_EXTRACT_SYSTEM = """你是一个工业知识抽取引擎。你的任务是根据给定的本体 Schema，把文档片段映射为本体实例（实体与三元组）。

{ontology}

## 输出要求
严格输出 JSON，格式如下，不要输出任何其他内容：
{{
  "entities": [
    {{"id": "实体唯一标识（优先用位号如 R-101/TI-101，无位号则用简短中文名）", "type": "实体类型", "label": "中文名称", "properties": {{"任意属性名": "属性值，如设定值/量程/整定参数等"}}}}
  ],
  "triples": [
    {{"subject": "实体id", "relation": "关系类型", "object": "实体id"}}
  ]
}}

## 抽取原则
1. 实体 id 全局一致：同一设备/测点在不同片段中必须使用相同 id（位号优先）。
2. 因果知识（causes）是重点：文中"导致/引起/造成/级联"等描述必须抽取为 causes 三元组。
3. **数值信息必须完整放入实体 properties**：包括但不限于量程、正常操作范围、设定值、联锁值、报警值、整定参数、额定值等。例如：
   - TI-101: {{"量程": "0~150℃", "正常操作范围": "82~88℃"}}
   - TAH-101: {{"设定值": "92℃", "优先级": "高"}}
   - TAHH-101: {{"联锁值": "98℃", "动作": "联锁停车"}}
   - FI-102: {{"正常流量": "18~22 m³/h", "低流量报警设定": "12 m³/h"}}
   绝不要遗漏原文中出现的任何数值！
4. triples 中的 subject/object 必须出现在 entities 中。
5. Event 实体必须具体完整，使用"主语+动作/状态"格式（如"冷却水流量FI-102下降"、"搅拌器M-101跳停"），禁止使用"下降"、"停止"、"超温"等孤立词作为实体。
6. 优先使用设备位号作为实体 id，避免创建泛化的事件实体。"""


def chunk_text(text: str, chunk_size: int) -> List[str]:
    """按二级标题切分，超长再按长度硬切。"""
    sections: List[str] = []
    current: List[str] = []
    for line in text.splitlines():
        if line.startswith("## ") and current:
            sections.append("\n".join(current).strip())
            current = [line]
        else:
            current.append(line)
    if current:
        sections.append("\n".join(current).strip())

    chunks: List[str] = []
    for sec in sections:
        if len(sec) <= chunk_size:
            chunks.append(sec)
        else:
            for i in range(0, len(sec), chunk_size):
                chunks.append(sec[i : i + chunk_size])
    return [c for c in chunks if c]


class KGBuilder:
    def __init__(self, cfg: dict, llm: LLMClient, embedder: EmbeddingClient, store: GraphStore):
        self.cfg = cfg
        self.llm = llm
        self.embedder = embedder
        self.store = store
        self.ontology = Ontology(cfg["data"]["ontology_path"])

    def build(self, progress_cb=None):
        """全量构建：清库 -> 逐文档抽取 -> 生成事实块。"""
        self.store.reset()
        docs = sorted(glob.glob(os.path.join(self.cfg["data"]["docs_dir"], "*.md")))
        if not docs:
            raise FileNotFoundError(f"未找到文档: {self.cfg['data']['docs_dir']}")

        system_prompt = _EXTRACT_SYSTEM.format(ontology=self.ontology.to_prompt())
        # subject -> [(rel, obj, doc)]
        triples_by_subject = defaultdict(list)

        for doc_path in docs:
            doc_name = os.path.basename(doc_path)
            with open(doc_path, "r", encoding="utf-8") as f:
                text = f.read()
            chunks = chunk_text(text, self.cfg["data"]["chunk_size"])
            for idx, chunk in enumerate(chunks):
                msg = f"[{doc_name}] 抽取片段 {idx + 1}/{len(chunks)}"
                logger.info(msg)
                if progress_cb:
                    progress_cb(msg)
                # naive-RAG 基线所需的原始 chunk
                self.store.add_chunk(doc_name, chunk, self.embedder.embed_one(chunk))
                # 本体映射抽取
                try:
                    result = self.llm.chat_json(system_prompt, f"文档片段：\n\n{chunk}")
                except Exception as exc:  # noqa: BLE001
                    logger.error(f"片段抽取失败，跳过: {exc}")
                    continue
                result = _normalize_extract_result(result)
                if not result:
                    logger.warning("片段抽取结果格式异常，跳过")
                    continue
                self._ingest(result, doc_name, triples_by_subject)
            self.store.commit()

        self._build_facts(triples_by_subject, progress_cb)
        self.store.commit()
        return self.store.stats()

    def _ingest(self, result: dict, doc_name: str, triples_by_subject: dict):
        valid_types = set(self.ontology.class_ids)
        valid_rels = set(self.ontology.property_ids)
        entity_ids = set()
        for ent in result.get("entities", []):
            if not isinstance(ent, dict):
                continue
            if not ent.get("id"):
                continue
            etype = ent.get("type") if ent.get("type") in valid_types else "Event"
            self.store.add_node(ent["id"], etype, ent.get("label", ""), ent.get("properties"))
            entity_ids.add(ent["id"])
        for tri in result.get("triples", []):
            if not isinstance(tri, dict):
                continue
            s, r, o = tri.get("subject"), tri.get("relation"), tri.get("object")
            if not (s and r and o) or r not in valid_rels:
                continue
            self.store.add_edge(s, r, o, doc_name)
            triples_by_subject[s].append((r, o, doc_name))

    def _build_facts(self, triples_by_subject: dict, progress_cb=None):
        """按主语聚合三元组 + 节点属性 + 来源原文片段，形成事实块（OG-RAG 的超边检索单元）。
        
        事实块使用自然语言描述而非结构化模板，以提升语义检索匹配度。
        每个事实块末尾附加来源文档中的原始片段，确保数值信息不丢失。
        """
        rel_labels = self.ontology.relation_labels()
        subjects = list(triples_by_subject.keys())
        texts, metas = [], []
        for subject in subjects:
            node = self.store.get_node(subject) or {"label": subject, "properties": {}}
            label = node.get("label", subject)
            props = node.get("properties") or {}
            docs = set()
            parts = []
            if props:
                prop_str = "，".join(f"{k}为{v}" for k, v in props.items())
                parts.append(f"{label}（{subject}）的{prop_str}")
            else:
                parts.append(f"{label}（{subject}）")
            for rel, obj, doc in triples_by_subject[subject]:
                obj_node = self.store.get_node(obj)
                obj_label = obj_node["label"] if obj_node else obj
                rel_label = rel_labels.get(rel, rel)
                parts.append(f"{label}的{rel_label}是{obj_label}（{obj}）")
                docs.add(doc)
            # 附加来源原文片段，确保数值信息不丢失
            source_snippets = self._find_source_snippets(subject, docs)
            if source_snippets:
                parts.append("原文摘录：" + source_snippets)
            texts.append("。".join(parts) + "。")
            metas.append((subject, ";".join(sorted(docs))))
        if progress_cb:
            progress_cb(f"生成 {len(texts)} 个事实块并向量化")
        if texts:
            vecs = self.embedder.embed(texts)
            for (subject, doc), text, vec in zip(metas, texts, vecs):
                self.store.add_fact(subject, text, doc, vec)

    def _find_source_snippets(self, entity_id: str, docs: set, max_snippets: int = 2) -> str:
        """从 chunks 表中查找包含实体 ID 的原文片段，截取相关行。"""
        if not docs:
            return ""
        snippets = []
        for doc in sorted(docs):
            rows = self.store.conn.execute(
                "SELECT text FROM chunks WHERE doc=? AND text LIKE ?",
                (doc, f"%{entity_id}%"),
            ).fetchall()
            for (chunk_text,) in rows:
                # 提取包含实体 ID 的行及其上下文
                lines = chunk_text.splitlines()
                for i, line in enumerate(lines):
                    if entity_id in line:
                        start = max(0, i - 1)
                        end = min(len(lines), i + 2)
                        snippet = " ".join(lines[start:end]).strip()
                        if snippet and snippet not in snippets:
                            snippets.append(snippet)
                        break
                if len(snippets) >= max_snippets:
                    break
            if len(snippets) >= max_snippets:
                break
        return " | ".join(snippets[:max_snippets])
