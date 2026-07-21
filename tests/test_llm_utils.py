"""LLM 工具函数测试（不涉及 API 调用）。"""
import numpy as np
import pytest

from src.llm import cosine_topk, parse_json_loose


class TestParseJsonLoose:
    def test_plain_json(self):
        result = parse_json_loose('{"a": 1, "b": 2}')
        assert result == {"a": 1, "b": 2}

    def test_json_in_code_fence(self):
        result = parse_json_loose('```json\n{"a": 1}\n```')
        assert result == {"a": 1}

    def test_json_in_plain_fence(self):
        result = parse_json_loose('```\n{"a": 1}\n```')
        assert result == {"a": 1}

    def test_json_with_surrounding_text(self):
        result = parse_json_loose('Here is the result:\n{"a": 1, "b": 2}\nDone.')
        assert result == {"a": 1, "b": 2}

    def test_json_list(self):
        result = parse_json_loose('[1, 2, 3]')
        assert result == [1, 2, 3]

    def test_json_list_with_surrounding_text(self):
        result = parse_json_loose('Result:\n[{"id": "A"}, {"id": "B"}]\nEnd.')
        assert len(result) == 2
        assert result[0]["id"] == "A"

    def test_invalid_json_raises(self):
        with pytest.raises(ValueError):
            parse_json_loose("not json at all")

    def test_nested_json(self):
        result = parse_json_loose('{"entities": [{"id": "R-101"}], "triples": []}')
        assert "entities" in result
        assert len(result["entities"]) == 1


class TestCosineTopK:
    def test_empty_matrix(self):
        qvec = np.array([1.0, 0.0])
        matrix = np.zeros((0, 2), dtype=np.float32)
        assert cosine_topk(qvec, matrix, 3) == []

    def test_basic_topk(self):
        qvec = np.array([1.0, 0.0], dtype=np.float32)
        matrix = np.array([
            [1.0, 0.0],   # most similar
            [0.0, 1.0],   # least similar
            [0.7, 0.7],   # medium
        ], dtype=np.float32)
        # normalize rows
        matrix = matrix / np.linalg.norm(matrix, axis=1, keepdims=True)
        qvec = qvec / np.linalg.norm(qvec)
        idxs = cosine_topk(qvec, matrix, 2)
        assert idxs[0] == 0  # most similar first
        assert len(idxs) == 2

    def test_k_larger_than_matrix(self):
        qvec = np.array([1.0, 0.0], dtype=np.float32)
        matrix = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
        matrix = matrix / np.linalg.norm(matrix, axis=1, keepdims=True)
        qvec = qvec / np.linalg.norm(qvec)
        idxs = cosine_topk(qvec, matrix, 10)
        assert len(idxs) == 2
