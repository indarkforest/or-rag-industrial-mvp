"""LLM 与 Embedding 客户端。LLM 基于 OpenAI 兼容接口，Embedding 使用 MiniMax 原生 API。"""
import json
import re
from typing import List

import numpy as np
import requests
from loguru import logger
from openai import OpenAI
from sklearn.feature_extraction.text import HashingVectorizer

from .exceptions import EmbeddingError, LLMError


class LLMClient:
    def __init__(self, cfg: dict):
        m = cfg["model"]
        self.client = OpenAI(base_url=m["base_url"], api_key=m["api_key"])
        self.model = m["name"]
        self.temperature = m.get("temperature", 0.1)

    def chat(self, system: str, user: str) -> str:
        try:
            resp = self.client.chat.completions.create(
                model=self.model,
                temperature=self.temperature,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
            )
            return resp.choices[0].message.content or ""
        except Exception as exc:
            raise LLMError(str(exc), f"LLM 调用失败（{self.model}）: {exc}")

    def chat_json(self, system: str, user: str):
        """要求 LLM 返回 JSON，并做容错解析（剥离 markdown 代码块等）。

        返回值可能是 dict 或 list，调用方需自行判断类型。
        """
        text = self.chat(system, user)
        return parse_json_loose(text)


def parse_json_loose(text: str):
    """从 LLM 输出中尽力解析 JSON。"""
    text = text.strip()
    # 剥离 ```json ... ```
    fence = re.search(r"```(?:json)?\s*(.*?)```", text, re.S)
    if fence:
        text = fence.group(1).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        # 截取第一个 { 到最后一个 } 之间
        start, end = text.find("{"), text.rfind("}")
        if start >= 0 and end > start:
            try:
                return json.loads(text[start : end + 1])
            except json.JSONDecodeError:
                pass
        start, end = text.find("["), text.rfind("]")
        if start >= 0 and end > start:
            try:
                return json.loads(text[start : end + 1])
            except json.JSONDecodeError:
                pass
        raise ValueError(f"无法从 LLM 输出中解析 JSON: {text[:200]}")


class EmbeddingClient:
    """provider=api 时调用 MiniMax 原生 /embeddings 接口；失败或 provider=hashing 时使用本地 HashingVectorizer 兜底。"""

    def __init__(self, cfg: dict):
        e = cfg.get("embedding", {}) or {}
        self.provider = e.get("provider", "hashing")
        self.model = e.get("name", "embo-01")
        self._api_failed = False
        self._hasher = HashingVectorizer(
            n_features=1024, analyzer="char_wb", ngram_range=(1, 3), norm="l2"
        )
        if self.provider == "api":
            self.base_url = e.get("base_url", "https://api.minimaxi.com/v1").rstrip("/")
            self.api_key = e.get("api_key", "")
            self._embed_url = f"{self.base_url}/embeddings"

    def _api_embed(self, texts: List[str], embed_type: str = "db") -> np.ndarray:
        """调用 MiniMax 原生 embedding API（非 OpenAI 兼容格式）。"""
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        payload = {"model": self.model, "texts": texts, "type": embed_type}
        resp = requests.post(self._embed_url, headers=headers, json=payload, timeout=60)
        resp.raise_for_status()
        data = resp.json()
        vectors = data.get("vectors") or data.get("data", [])
        if not vectors:
            raise ValueError("API 返回空向量")
        if isinstance(vectors[0], dict):
            vecs = np.array([v["embedding"] for v in vectors], dtype=np.float32)
        else:
            vecs = np.array(vectors, dtype=np.float32)
        norms = np.linalg.norm(vecs, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return vecs / norms

    def embed(self, texts: List[str]) -> np.ndarray:
        if self.provider == "api" and not self._api_failed:
            try:
                return self._api_embed(texts, embed_type="db")
            except Exception as exc:  # noqa: BLE001
                logger.warning(f"Embedding API 调用失败，降级为本地 hashing 向量: {exc}")
                self._api_failed = True
        return self._hasher.transform(texts).toarray().astype(np.float32)

    def embed_one(self, text: str) -> np.ndarray:
        return self.embed([text])[0]

    def embed_query(self, text: str) -> np.ndarray:
        """查询向量化，使用 type=query 以匹配 MiniMax 的检索优化。"""
        if self.provider == "api" and not self._api_failed:
            try:
                return self._api_embed([text], embed_type="query")[0]
            except Exception as exc:  # noqa: BLE001
                logger.warning(f"Embedding query API 失败，降级: {exc}")
                self._api_failed = True
        return self._hasher.transform([text]).toarray().astype(np.float32)[0]


def cosine_topk(query_vec: np.ndarray, matrix: np.ndarray, k: int) -> List[int]:
    """返回相似度最高的 k 个行索引（matrix 每行已归一化）。"""
    if matrix.shape[0] == 0:
        return []
    scores = matrix @ query_vec
    order = np.argsort(-scores)
    return order[: min(k, len(order))].tolist()
