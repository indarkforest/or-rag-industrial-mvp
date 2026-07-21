"""Streamlit Web 界面：知识库构建 / 问答对比 / 对照评估 / 图谱浏览。
启动：streamlit run app.py
"""
import glob
import os

import pandas as pd
import streamlit as st

from src.agent import QAAgent
from src.config import load_config
from src.evaluate import format_report, run_eval
from src.kg_builder import KGBuilder
from src.live_data import LiveDataProvider
from src.llm import EmbeddingClient, LLMClient
from src.retriever import build_retrievers
from src.store import GraphStore

st.set_page_config(page_title="OG-RAG 工业验证 MVP", page_icon="🏭", layout="wide")


def _build_cfg(api_key: str, model_name: str, base_url: str, embed_model: str,
               top_k_chunks: int, top_k_facts: int, expand_hops: int) -> dict:
    """根据页面配置构建 cfg dict。"""
    cfg = load_config()
    if api_key:
        cfg["model"]["api_key"] = api_key
        cfg["embedding"]["api_key"] = api_key
    if model_name:
        cfg["model"]["name"] = model_name
    if base_url:
        cfg["model"]["base_url"] = base_url
        cfg["embedding"]["base_url"] = base_url
    if embed_model:
        cfg["embedding"]["name"] = embed_model
    cfg["retrieval"]["top_k_chunks"] = top_k_chunks
    cfg["retrieval"]["top_k_facts"] = top_k_facts
    cfg["retrieval"]["expand_hops"] = expand_hops
    return cfg


def _get_ctx():
    """根据 session_state 中的配置创建所有客户端实例。API Key 为空时返回 None。"""
    api_key = st.session_state.get("api_key", "")
    if not api_key:
        # 尝试从环境变量读取
        import os
        api_key = os.environ.get("MINIMAX_API_KEY", "")
        if api_key:
            st.session_state["api_key"] = api_key
    if not api_key:
        return None, None, None, None, None
    cfg = _build_cfg(
        api_key,
        st.session_state.get("model_name", ""),
        st.session_state.get("base_url", ""),
        st.session_state.get("embed_model", ""),
        st.session_state.get("top_k_chunks", 4),
        st.session_state.get("top_k_facts", 6),
        st.session_state.get("expand_hops", 1),
    )
    try:
        llm = LLMClient(cfg)
        embedder = EmbeddingClient(cfg)
        store = GraphStore(cfg["data"]["db_path"])
        live_data = LiveDataProvider(cfg["data"]["db_path"])
        return cfg, llm, embedder, store, live_data
    except Exception:
        return None, None, None, None, None


# ── 侧边栏配置 ──
with st.sidebar:
    st.header("⚙️ 配置")
    st.caption("API Key 和模型参数在此配置，无需手动编辑 config.yaml")

    # 从 config.yaml 读取默认值
    _default_cfg = load_config()
    _default_key = _default_cfg["model"].get("api_key", "")
    _default_model = _default_cfg["model"].get("name", "MiniMax-M3")
    _default_base = _default_cfg["model"].get("base_url", "https://api.minimaxi.com/v1")
    _default_embed = _default_cfg.get("embedding", {}).get("name", "embo-01")
    _r = _default_cfg.get("retrieval", {})

    api_key = st.text_input("API Key", value=_default_key, type="password",
                            help="MiniMax API Key，优先级高于 config.yaml")
    model_name = st.text_input("LLM 模型", value=_default_model,
                               help="如 MiniMax-M3、MiniMax-M1 等")
    base_url = st.text_input("API Base URL", value=_default_base,
                             help="OpenAI 兼容接口地址")
    embed_model = st.text_input("Embedding 模型", value=_default_embed,
                                help="如 embo-01")

    st.divider()
    st.subheader("检索参数")
    top_k_chunks = st.slider("naive-RAG top_k", 1, 10, _r.get("top_k_chunks", 4),
                             help="原文 chunk 检索数量")
    top_k_facts = st.slider("OG-RAG top_k", 1, 10, _r.get("top_k_facts", 6),
                            help="事实块检索数量")
    expand_hops = st.slider("图谱扩展跳数", 0, 3, _r.get("expand_hops", 1),
                            help="0=不扩展，1=1跳邻居，2=2跳因果链")

    # 保存到 session_state
    st.session_state["api_key"] = api_key
    st.session_state["model_name"] = model_name
    st.session_state["base_url"] = base_url
    st.session_state["embed_model"] = embed_model
    st.session_state["top_k_chunks"] = top_k_chunks
    st.session_state["top_k_facts"] = top_k_facts
    st.session_state["expand_hops"] = expand_hops

    if st.button("应用配置", type="primary"):
        st.rerun()

    st.divider()
    st.caption("💡 修改配置后点击「应用配置」生效")


cfg, llm, embedder, store, live_data = _get_ctx()

st.title("工业场景效果验证 MVP")
st.caption("混合场景检索（hybrid-rag） vs  本体驱动检索（OG-RAG） vs 纯向量检索（naive-RAG）")

if cfg is None:
    st.warning("⚠️ 请在左侧侧边栏配置 API Key 后点击「应用配置」")
    st.stop()


def _extract_final_answer(text: str) -> str:
    """从 LLM 输出中提取最终回答，剥离 <think>...</think> 思考过程。
    """
    import re
    # 去除 <think>...</think> 块（含可能的属性变体）
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
    # 清理多余空行
    text = re.sub(r"\n{3,}", "\n\n", text).strip()
    return text


_COMPARE_SYSTEM = """你是工业AI回答质量评估专家。请对三种检索方式（naive-rag、og-rag、hybrid-rag）针对同一问题的回答进行对比分析。

输出格式要求（Markdown）：
1. 先给出每个回答的简要评价（2-3句话）
2. 然后用一个表格对比以下维度（满分5分）：
   - 因果链完整性
   - 数值准确性
   - 机理深度
   - 实时数据利用
   - 可操作性
   - 忠实度
3. 最后给出排名和总结

请客观评价，基于回答内容本身的质量，不要因为回答长短而偏向。"""


def _generate_comparison(llm_client, question: str, results: dict) -> str:
    """调用 LLM 生成三路回答的对比分析。"""
    parts = [f"## 问题\n{question}\n"]
    for name in ["naive-rag", "og-rag", "hybrid-rag"]:
        if name in results:
            answer = results[name]["answer"]
            parts.append(f"## {name} 的回答\n{answer}\n")
    user = "\n".join(parts)
    return llm_client.chat(_COMPARE_SYSTEM, user)

stats = store.stats()
c1, c2, c3, c4 = st.columns(4)
c1.metric("实体节点", stats["nodes"])
c2.metric("关系边", stats["edges"])
c3.metric("文档块", stats["chunks"])
c4.metric("事实块（超边）", stats["facts"])

tab_build, tab_qa, tab_eval, tab_live, tab_graph = st.tabs(["📥 构建知识库", "💬 问答对比", "📊 对照评估", "📡 实时数据", "🕸 图谱浏览"])

with tab_build:
    # ── 文档管理 ──
    st.subheader("📄 文档管理")
    docs_dir = cfg["data"]["docs_dir"]
    os.makedirs(docs_dir, exist_ok=True)

    # 上传文档（动态 key：上传后计数器+1，rerun 后是全新 uploader，状态自动清空）
    upload_key = f"doc_uploader_{st.session_state.get('upload_counter', 0)}"
    uploaded = st.file_uploader("上传文档（仅限 .md）", type=["md"], accept_multiple_files=True,
                                key=upload_key)
    if uploaded:
        for f in uploaded:
            if not f.name.endswith(".md"):
                st.error(f"不支持的格式: {f.name}，仅允许 .md 文件")
                continue
            path = os.path.join(docs_dir, f.name)
            with open(path, "wb") as out:
                out.write(f.getbuffer())
            st.success(f"已上传: {f.name}")
        # 递增计数器，下次渲染时 uploader key 变化 → 全新组件
        st.session_state["upload_counter"] = st.session_state.get("upload_counter", 0) + 1
        st.rerun()

    # 列出已有文档
    existing = sorted(glob.glob(os.path.join(docs_dir, "*.md")))
    if existing:
        st.markdown(f"**当前文档（{len(existing)} 个）**")
        for doc_path in existing:
            doc_name = os.path.basename(doc_path)
            col_name, col_size, col_del = st.columns([5, 2, 1])
            with col_name:
                st.text(doc_name)
            with col_size:
                size_kb = os.path.getsize(doc_path) / 1024
                st.text(f"{size_kb:.1f} KB")
            with col_del:
                if st.button("删除", key=f"del_{doc_name}"):
                    os.remove(doc_path)
                    st.success(f"已删除: {doc_name}")
                    st.rerun()
    else:
        st.info("暂无文档，请上传 .md 文件")

    st.divider()

    # ── 知识库构建 ──
    st.subheader("🔨 构建知识库")
    st.markdown("按本体 Schema 抽取实体/三元组并向量化。**将清空并重建现有库。**")
    if st.button("开始构建", type="primary"):
        if not existing:
            st.error("没有文档可构建，请先上传。")
        else:
            progress_box = st.empty()
            with st.spinner("构建中（每个片段调用一次 LLM，请耐心等待）..."):
                builder = KGBuilder(cfg, llm, embedder, store)
                result = builder.build(progress_cb=lambda m: progress_box.info(m))
            st.success(f"构建完成: {result}")
            st.rerun()

with tab_qa:
    question = st.text_input("输入问题", placeholder="例：如果仪表风压力低于 0.4 MPa，会对 R-101 的温度产生什么影响？")
    mode = st.radio("检索方式", ["三路对比（全部）", "hybrid-rag", "og-rag", "naive-rag"], horizontal=True)
    col_btn, col_chk = st.columns([1, 3])
    with col_btn:
        ask_btn = st.button("提问", type="primary")
        enable_compare = st.checkbox("对比分析", value=True)
    if ask_btn and question:
        if stats["chunks"] == 0:
            st.error("知识库为空，请先在「构建知识库」页构建。")
        else:
            retrievers = build_retrievers(cfg, store, embedder, live_data=live_data)
            names = ["naive-rag", "og-rag", "hybrid-rag"] if mode.startswith("三路") else [mode]
            qa_results = {}
            cols = st.columns(len(names))
            for col, name in zip(cols, names):
                with col:
                    st.subheader(f"🔎 {name}")
                    with st.spinner(f"{name} 回答中..."):
                        result = QAAgent(llm, retrievers[name]).answer(question)
                    qa_results[name] = result
                    answer = result["answer"]
                    final_answer = _extract_final_answer(answer)
                    st.markdown(final_answer)
                    with st.expander(f"检索到的上下文（{len(result['contexts'])} 段）", expanded=False):
                        for ctx in result["contexts"]:
                            st.text(ctx)
                            st.divider()
                    with st.expander("查看完整模型输出（含思考过程）", expanded=False):
                        st.text(answer)
            st.session_state["qa_results"] = qa_results
            st.session_state["qa_question"] = question

            # 勾选了对比分析且有多路结果时，自动执行对比
            if enable_compare and len(qa_results) > 1:
                st.divider()
                with st.spinner("LLM 正在生成对比分析..."):
                    comparison = _generate_comparison(llm, question, qa_results)
                st.subheader("三路回答对比分析")
                st.markdown(comparison)

with tab_eval:
    st.markdown("对 `data/questions.yaml` 中的问题集，分别用三种检索方式回答并由 LLM 评分。")
    if st.button("运行对照评估", type="primary"):
        if stats["chunks"] == 0:
            st.error("知识库为空，请先构建。")
        else:
            retrievers = build_retrievers(cfg, store, embedder, live_data=live_data)
            agents = {name: QAAgent(llm, r) for name, r in retrievers.items()}
            progress_box = st.empty()
            with st.spinner("评估中，每题需调用 LLM 多次，耗时较长..."):
                result = run_eval(cfg, llm, agents, progress_cb=lambda m: progress_box.info(m))
            st.session_state["eval_result"] = result
    if "eval_result" in st.session_state:
        result = st.session_state["eval_result"]
        st.subheader("汇总（平均分，满分 5）")
        st.dataframe(pd.DataFrame(result["summary"]).T, width='stretch')
        st.subheader("明细")
        detail = pd.DataFrame(result["rows"])[
            ["id", "type", "retriever", "correctness", "completeness", "faithfulness", "comment"]
        ]
        st.dataframe(detail, width='stretch')
        st.download_button(
            "下载完整报告 (Markdown)",
            format_report(result),
            file_name="eval_report.md",
            mime="text/markdown",
        )
        for row in result["rows"]:
            with st.expander(f"{row['id']} [{row['retriever']}] {row['question'][:40]}..."):
                st.markdown(row["answer"])

with tab_live:
    st.markdown("为知识图谱中的 **Device / Sensor / ControlLoop** 类型实体生成模拟实时数据，OG-RAG 检索时自动注入。")
    col_gen, col_info = st.columns([1, 2])
    with col_gen:
        if st.button("生成模拟实时数据", type="primary"):
            if stats["nodes"] == 0:
                st.error("知识库为空，请先构建知识库。")
            else:
                nodes = store.all_nodes()
                count = live_data.generate_sim_data(nodes)
                st.success(f"已为 {count} 个实体生成模拟实时数据（含旋转门压缩趋势）")
                st.rerun()
    with col_info:
        live_rows = live_data.all_live()
        st.metric("已接入实时数据实体数", len(live_rows))

    if live_rows:
        st.subheader("实时数据总览")
        display_rows = []
        for r in live_rows:
            trend = r["trend_points"]
            if len(trend) >= 2:
                trend_str = "→".join(f"{p['v']}" for p in trend)
            else:
                trend_str = f"{r['value']}"
            status_map = {"normal": "正常", "rising": "上升", "falling": "下降", "alarm": "报警"}
            display_rows.append({
                "实体ID": r["entity_id"],
                "当前值": f"{r['value']} {r['unit']}",
                "状态": status_map.get(r["status"], r["status"]),
                "趋势": trend_str,
                "压缩点数": len(trend),
                "采集时间": r["timestamp"],
            })
        st.dataframe(pd.DataFrame(display_rows), width='stretch', height=400)

        st.subheader("按实体查看详情")
        entity_ids = [r["entity_id"] for r in live_rows]
        selected = st.selectbox("选择实体", entity_ids)
        if selected:
            detail = live_data.get(selected)
            if detail:
                node = store.get_node(selected)
                label = node["label"] if node else selected
                st.markdown(f"**{label}（{selected}）**")
                c1, c2, c3 = st.columns(3)
                c1.metric("当前值", f"{detail['value']} {detail['unit']}")
                c2.metric("状态", detail["status"])
                c3.metric("趋势点数", len(detail["trend_points"]))
                st.caption(f"采集时间: {detail['timestamp']}")
                if detail["trend_points"]:
                    chart_data = pd.DataFrame(
                        [{"时间": p["t"], "值": p["v"]} for p in detail["trend_points"]]
                    )
                    st.line_chart(chart_data.set_index("时间"), width='stretch')
    else:
        st.info("暂无实时数据，请先点击「生成模拟实时数据」。")

with tab_graph:
    if stats["nodes"] == 0:
        st.info("图谱为空，请先构建知识库。")
    else:
        nodes = store.all_nodes()
        edges = store.all_edges()
        col_a, col_b = st.columns(2)
        with col_a:
            st.subheader(f"实体（{len(nodes)}）")
            st.dataframe(
                pd.DataFrame(
                    [{"id": n["id"], "类型": n["type"], "名称": n["label"], "属性": str(n["properties"])} for n in nodes]
                ),
                width='stretch',
                height=420,
            )
        with col_b:
            st.subheader(f"关系（{len(edges)}）")
            st.dataframe(
                pd.DataFrame(edges, columns=["主体", "关系", "客体"]),
                width='stretch',
                height=420,
            )
        st.subheader("因果链视图（causes 关系）")
        causal = [e for e in edges if e[1] == "causes"]
        if causal:
            lines = ["digraph G { rankdir=LR; node [shape=box, fontname=\"Microsoft YaHei\"];"]
            for src, _, dst in causal:
                lines.append(f'"{src}" -> "{dst}";')
            lines.append("}")
            st.graphviz_chart("\n".join(lines))
        else:
            st.info("暂无 causes 关系。")
