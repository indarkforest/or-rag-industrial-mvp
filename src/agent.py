"""问答 Agent：检索器提供上下文，LLM 基于上下文回答。"""
from typing import Dict, List

from .llm import LLMClient
from .retriever import BaseRetriever

_ANSWER_SYSTEM = """你是工业智能助手，服务于化工装置的运行与工艺人员。
请仅依据提供的上下文回答问题：
1. 回答要具体，引用位号与数值（如设定值、范围）。
2. 涉及因果分析时，按因果链逐步说明（A -> B -> C）。
3. 上下文中没有的信息，明确说明"依据现有资料无法确定"，不要编造。
4. 如果上下文中包含【实时数据】，请结合实时数据进行分析，注意数据带有时间戳，需判断时效性。
5. 实时数据仅供参考，结论应优先基于文档知识，再结合实时数据做趋势分析和预警判断。
6. 回答使用中文，简洁分点。"""


class QAAgent:
    def __init__(self, llm: LLMClient, retriever: BaseRetriever):
        self.llm = llm
        self.retriever = retriever

    def answer(self, question: str) -> Dict:
        contexts: List[str] = self.retriever.retrieve(question)
        context_text = "\n\n".join(contexts) if contexts else "（未检索到任何上下文）"
        user = f"## 上下文\n{context_text}\n\n## 问题\n{question}"
        answer = self.llm.chat(_ANSWER_SYSTEM, user)
        return {
            "retriever": self.retriever.name,
            "question": question,
            "contexts": contexts,
            "answer": answer,
        }
