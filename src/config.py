"""配置加载：config.yaml + 环境变量（环境变量优先）。"""
import os

import yaml

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def project_path(*parts: str) -> str:
    return os.path.join(_PROJECT_ROOT, *parts)


def load_config(path: str = None) -> dict:
    cfg_path = path or project_path("config.yaml")
    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    # 环境变量覆盖 API Key
    env_key = os.environ.get("MINIMAX_API_KEY", "")
    if env_key:
        cfg["model"]["api_key"] = env_key
        if not cfg.get("embedding", {}).get("api_key"):
            cfg["embedding"]["api_key"] = env_key
    if cfg.get("embedding") and not cfg["embedding"].get("api_key"):
        cfg["embedding"]["api_key"] = cfg["model"].get("api_key", "")

    # data 路径转为绝对路径
    for key in ("docs_dir", "ontology_path", "questions_path", "db_path"):
        cfg["data"][key] = project_path(cfg["data"][key])
    return cfg
