"""评估模块纯函数测试（不涉及 LLM 调用）。"""
import pytest

from src.evaluate import format_report


class TestFormatReport:
    def test_basic_report(self):
        result = {
            "rows": [
                {
                    "id": "q1",
                    "type": "causal",
                    "question": "为什么温度升高？",
                    "retriever": "naive-rag",
                    "answer": "因为冷却不足",
                    "correctness": 4,
                    "completeness": 3,
                    "faithfulness": 5,
                    "comment": "基本正确",
                },
            ],
            "summary": {
                "naive-rag": {"correctness": 4.0, "completeness": 3.0, "faithfulness": 5.0},
                "og-rag": {"correctness": 5.0, "completeness": 4.0, "faithfulness": 5.0},
            },
        }
        report = format_report(result)
        assert "对照评估报告" in report
        assert "naive-rag" in report
        assert "og-rag" in report
        assert "为什么温度升高？" in report
        assert "因为冷却不足" in report

    def test_empty_report(self):
        result = {"rows": [], "summary": {}}
        report = format_report(result)
        assert "对照评估报告" in report
        assert "汇总" in report

    def test_multiple_rows(self):
        result = {
            "rows": [
                {
                    "id": "q1",
                    "type": "",
                    "question": "Q1",
                    "retriever": "naive-rag",
                    "answer": "A1",
                    "correctness": 3,
                    "completeness": 3,
                    "faithfulness": 4,
                    "comment": "c1",
                },
                {
                    "id": "q1",
                    "type": "",
                    "question": "Q1",
                    "retriever": "og-rag",
                    "answer": "A2",
                    "correctness": 5,
                    "completeness": 4,
                    "faithfulness": 5,
                    "comment": "c2",
                },
            ],
            "summary": {
                "naive-rag": {"correctness": 3.0, "completeness": 3.0, "faithfulness": 4.0},
                "og-rag": {"correctness": 5.0, "completeness": 4.0, "faithfulness": 5.0},
            },
        }
        report = format_report(result)
        assert "naive-rag" in report
        assert "og-rag" in report
        assert report.count("### q1") == 2
