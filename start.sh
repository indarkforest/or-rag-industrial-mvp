#!/bin/bash
set -e

echo "============================================"
echo "  工业场景效果验证 MVP - 一键启动"
echo "============================================"
echo

# 检查 Python
if ! command -v python3 &> /dev/null; then
    echo "[错误] 未检测到 python3，请先安装 Python 3.10+"
    exit 1
fi

# 创建虚拟环境
if [ ! -d "venv" ]; then
    echo "[1/3] 创建虚拟环境..."
    python3 -m venv venv
else
    echo "[1/3] 虚拟环境已存在"
fi

# 激活虚拟环境
source venv/bin/activate

# 安装依赖
echo "[2/3] 检查依赖..."
pip install -r requirements.txt -q

# 启动
echo "[3/3] 启动应用..."
echo
echo "============================================"
echo "  应用启动后请在浏览器中访问:"
echo "  http://localhost:8501"
echo
echo "  首次使用请在左侧侧边栏配置 API Key"
echo "  按 Ctrl+C 可停止应用"
echo "============================================"
echo

streamlit run app.py --server.port 8501
