"""KGBuilder 纯函数测试（不涉及 LLM 调用）。"""
import pytest

from src.kg_builder import _normalize_extract_result, chunk_text


class TestNormalizeExtractResult:
    def test_dict_with_entities_and_triples(self):
        result = {
            "entities": [{"id": "R-101", "type": "Device"}],
            "triples": [{"subject": "R-101", "relation": "causes", "object": "TI-101"}],
        }
        normalized = _normalize_extract_result(result)
        assert normalized is not None
        assert len(normalized["entities"]) == 1
        assert len(normalized["triples"]) == 1

    def test_dict_empty(self):
        result = {"entities": [], "triples": []}
        assert _normalize_extract_result(result) is None

    def test_dict_with_non_list_fields(self):
        result = {"entities": "not a list", "triples": 123}
        normalized = _normalize_extract_result(result)
        # entities/triples 归一化为空 list 后，两者都空则返回 None
        assert normalized is None

    def test_list_format(self):
        result = [
            {"id": "R-101", "type": "Device", "label": "反应釜"},
            {"subject": "R-101", "relation": "causes", "object": "TI-101"},
        ]
        normalized = _normalize_extract_result(result)
        assert normalized is not None
        assert len(normalized["entities"]) == 1
        assert len(normalized["triples"]) == 1

    def test_list_with_invalid_items(self):
        result = ["not a dict", {"id": "A", "type": "Device"}, 123]
        normalized = _normalize_extract_result(result)
        assert len(normalized["entities"]) == 1

    def test_none_input(self):
        assert _normalize_extract_result(None) is None

    def test_string_input(self):
        assert _normalize_extract_result("not a dict") is None


class TestChunkText:
    def test_single_section(self):
        text = "Some content without headers"
        chunks = chunk_text(text, 800)
        assert len(chunks) == 1
        assert chunks[0] == "Some content without headers"

    def test_multiple_sections(self):
        text = "## Section 1\nContent 1\n## Section 2\nContent 2"
        chunks = chunk_text(text, 800)
        assert len(chunks) == 2

    def test_long_section_split(self):
        text = "A" * 1500
        chunks = chunk_text(text, 500)
        assert len(chunks) == 3
        assert all(len(c) <= 500 for c in chunks)

    def test_empty_text(self):
        chunks = chunk_text("", 800)
        assert chunks == [""] or chunks == []

    def test_section_with_newlines(self):
        text = "## Header\nLine 1\nLine 2\nLine 3"
        chunks = chunk_text(text, 800)
        assert len(chunks) == 1
        assert "Line 1" in chunks[0]
        assert "Line 3" in chunks[0]
