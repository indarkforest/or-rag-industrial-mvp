"""统一日志配置：loguru 结构化日志 + 控制台 + 文件输出。"""
import sys
from pathlib import Path

from loguru import logger

# 项目根目录（src/ 的上一级）
_PROJECT_ROOT = Path(__file__).resolve().parent.parent


def setup_logger(log_dir: str = None, level: str = "INFO"):
    """初始化日志配置。每次调用都重新配置 handler。

    - 控制台：彩色输出，INFO 及以上
    - 文件：按天轮转，保留 7 天，DEBUG 及以上
    """
    if log_dir is None:
        log_dir = str(_PROJECT_ROOT / "logs")

    logger.remove()

    logger.add(
        sys.stderr,
        level=level,
        format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | "
               "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
        colorize=True,
    )

    Path(log_dir).mkdir(parents=True, exist_ok=True)
    logger.add(
        f"{log_dir}/app_{{time:YYYY-MM-DD}}.log",
        level="DEBUG",
        format="{time:YYYY-MM-DD HH:mm:ss.SSS} | {level: <8} | "
               "{name}:{function}:{line} | {message}",
        rotation="00:00",
        retention="7 days",
        encoding="utf-8",
    )


def get_logger():
    """获取 logger 实例（如未初始化则先初始化）。"""
    return logger
