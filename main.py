"""CLI 入口：
python main.py build            # 构建知识图谱与索引
python main.py query -q "问题" [-r naive-rag|og-rag]
python main.py eval             # 对照评估，输出 eval_report.md
"""
import argparse
import logging

from src.agent import QAAgent
from src.config import load_config, project_path
from src.evaluate import format_report, run_eval
from src.kg_builder import KGBuilder
from src.llm import EmbeddingClient, LLMClient
from src.retriever import build_retrievers
from src.store import GraphStore

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


def main():
    parser = argparse.ArgumentParser(description="OG-RAG 工业场景效果验证 MVP")
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("build", help="构建知识图谱与索引")

    q = sub.add_parser("query", help="单次问答")
    q.add_argument("-q", "--question", required=True)
    q.add_argument("-r", "--retriever", default="hybrid-rag", choices=["naive-rag", "og-rag", "hybrid-rag"])

    sub.add_parser("eval", help="对照评估")

    args = parser.parse_args()
    cfg = load_config()
    llm = LLMClient(cfg)
    embedder = EmbeddingClient(cfg)
    store = GraphStore(cfg["data"]["db_path"])

    if args.command == "build":
        stats = KGBuilder(cfg, llm, embedder, store).build()
        print(f"构建完成: {stats}")

    elif args.command == "query":
        retrievers = build_retrievers(cfg, store, embedder)
        agent = QAAgent(llm, retrievers[args.retriever])
        result = agent.answer(args.question)
        print(f"\n=== 检索方式: {result['retriever']} ===")
        print("\n--- 检索到的上下文 ---")
        for ctx in result["contexts"]:
            print(ctx)
            print("-" * 40)
        print("\n--- 回答 ---")
        print(result["answer"])

    elif args.command == "eval":
        retrievers = build_retrievers(cfg, store, embedder)
        agents = {name: QAAgent(llm, r) for name, r in retrievers.items()}
        result = run_eval(cfg, llm, agents)
        report = format_report(result)
        report_path = project_path("eval_report.md")
        with open(report_path, "w", encoding="utf-8") as f:
            f.write(report)
        print("\n=== 汇总 ===")
        for name, s in result["summary"].items():
            print(f"{name}: {s}")
        print(f"\n完整报告已写入: {report_path}")


if __name__ == "__main__":
    main()
