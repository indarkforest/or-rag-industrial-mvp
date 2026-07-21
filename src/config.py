"""配置加载：config.yaml + 环境变量（环境变量优先）+ Pydantic 校验。"""
import os

import yaml
from loguru import logger

from .exceptions import ConfigError
from .models import AppConfig

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def project_path(*parts: str) -> str:
    return os.path.join(_PROJECT_ROOT, *parts)


def load_config(path: str = None) -> dict:
    cfg_path = path or project_path("config.yaml")
    try:
        with open(cfg_path, "r", encoding="utf-8") as f:
            raw = yaml.safe_load(f)
    except FileNotFoundError:
        raise ConfigError(f"配置文件不存在: {cfg_path}", "配置文件 config.yaml 不存在，请检查项目完整性")
    except yaml.YAMLError as exc:
        raise ConfigError(f"YAML 解析失败: {exc}", "config.yaml 格式错误，请检查 YAML 语法")

    # 环境变量覆盖 API Key
    env_key = os.environ.get("MINIMAX_API_KEY", "")
    if env_key:
        raw.setdefault("model", {})["api_key"] = env_key
        if not raw.get("embedding", {}).get("api_key"):
            raw.setdefault("embedding", {})["api_key"] = env_key
    if raw.get("embedding") and not raw["embedding"].get("api_key"):
        raw["embedding"]["api_key"] = raw.get("model", {}).get("api_key", "")

    # Pydantic 校验
    try:
        cfg = AppConfig(**raw)
        logger.info(f"配置校验通过: model={cfg.model.name}, embedding={cfg.embedding.provider}")
    except Exception as exc:
        raise ConfigError(f"配置校验失败: {exc}", f"config.yaml 配置项有误: {exc}")

    # 转为 dict，data 路径转为绝对路径
    result = cfg.to_dict()
    for key in ("docs_dir", "ontology_path", "questions_path", "db_path"):
        result["data"][key] = project_path(result["data"][key])
    return result
