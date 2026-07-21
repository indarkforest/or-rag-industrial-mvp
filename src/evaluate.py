"""对照评估：同一问题集分别用 naive-RAG 与 OG-RAG 回答，LLM-as-judge 打分。"""
from typing import Dict, List

import yaml
from loguru import logger

from .agent import QAAgent
from .llm import LLMClient
from .models import EvalResult, EvalRow, EvalScore

_JUDGE_SYSTEM = """你是严格的评估员。根据参考答案要点，对候选回答打分。

评分维度（各 1~5 分，5 为最好）：
- correctness：回答与参考答案要点的一致性、准确性
- completeness：要点覆盖是否完整（因果链是否完整）
- faithfulness：是否存在编造/与参考答案矛盾的内容

严格输出 JSON，不要输出其他内容：
{"correctness": 分数, "completeness": 分数, "faithfulness": 分数, "comment": "一句话点评"}"""


def load_questions(path: str) -> List[dict]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)["questions"]


def judge(llm: LLMClient, question: str, reference: str, answer: str) -> EvalScore:
    user = f"## 问题\n{question}\n\n## 参考答案要点\n{reference}\n\n## 候选回答\n{answer}"
    try:
        result = llm.chat_json(_JUDGE_SYSTEM, user)
        if not isinstance(result, dict):
            logger.warning(f"评分 LLM 返回非 dict: {type(result).__name__}")
            return EvalScore(comment="评分格式异常")
        return EvalScore(
            correctness=float(result.get("correctness", 0)),
            completeness=float(result.get("completeness", 0)),
            faithfulness=float(result.get("faithfulness", 0)),
            comment=result.get("comment", ""),
        )
    except Exception as exc:  # noqa: BLE001
        logger.error(f"评分失败: {exc}")
        return EvalScore(comment=f"评分失败: {exc}")


def run_eval(cfg: dict, llm: LLMClient, agents: Dict[str, QAAgent], progress_cb=None) -> dict:
    """返回 {"rows": [...], "summary": {retriever: 平均分}}。"""
    questions = load_questions(cfg["data"]["questions_path"])
    rows: List[EvalRow] = []
    for q in questions:
        for name, agent in agents.items():
            msg = f"[{q['id']}] {name} 回答中..."
            logger.info(msg)
            if progress_cb:
                progress_cb(msg)
            result = agent.answer(q["question"])
            scores = judge(llm, q["question"], q["reference"], result["answer"])
            rows.append(EvalRow(
                id=q["id"],
                type=q.get("type", ""),
                question=q["question"],
                retriever=name,
                answer=result["answer"],
                correctness=scores.correctness,
                completeness=scores.completeness,
                faithfulness=scores.faithfulness,
                comment=scores.comment,
            ))

    summary = {}
    for name in agents:
        subset = [r for r in rows if r.retriever == name]
        n = max(len(subset), 1)
        summary[name] = {
            "correctness": round(sum(r.correctness for r in subset) / n, 2),
            "completeness": round(sum(r.completeness for r in subset) / n, 2),
            "faithfulness": round(sum(r.faithfulness for r in subset) / n, 2),
        }
    return {"rows": [r.model_dump() for r in rows], "summary": summary}


def format_report(result: dict) -> str:
    lines = ["# OG-RAG vs naive-RAG 对照评估报告", "", "## 汇总（平均分，满分 5）", ""]
    lines.append("| 检索方式 | 准确性 | 完整性 | 忠实度 |")
    lines.append("|---|---|---|---|")
    for name, s in result["summary"].items():
        lines.append(f"| {name} | {s['correctness']} | {s['completeness']} | {s['faithfulness']} |")
    lines.append("")
    lines.append("## 明细")
    for row in result["rows"]:
        lines.append("")
        lines.append(f"### {row['id']} [{row['retriever']}] （{row['type']}）")
        lines.append(f"**问题**：{row['question']}")
        lines.append(
            f"**评分**：correctness={row['correctness']} completeness={row['completeness']} "
            f"faithfulness={row['faithfulness']} —— {row['comment']}"
        )
        lines.append(f"**回答**：\n{row['answer']}")
    return "\n".join(lines)
