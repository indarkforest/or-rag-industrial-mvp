"""分层异常体系：按模块分类，提供用户友好提示。"""
from loguru import logger


class AppError(Exception):
    """应用基础异常，所有自定义异常的父类。"""

    def __init__(self, message: str, user_message: str = None):
        super().__init__(message)
        self.user_message = user_message or message
        logger.error(f"{self.__class__.__name__}: {message}")


class ConfigError(AppError):
    """配置相关错误（缺失、格式错误、校验失败）。"""

    def __init__(self, message: str, user_message: str = None):
        user_message = user_message or f"配置错误: {message}"
        super().__init__(message, user_message)


class LLMError(AppError):
    """LLM 调用相关错误（API 异常、响应解析失败）。"""

    def __init__(self, message: str, user_message: str = None):
        user_message = user_message or f"LLM 调用失败: {message}"
        super().__init__(message, user_message)


class EmbeddingError(AppError):
    """Embedding 调用相关错误。"""

    def __init__(self, message: str, user_message: str = None):
        user_message = user_message or f"向量化失败: {message}"
        super().__init__(message, user_message)


class KGBuildError(AppError):
    """知识图谱构建相关错误。"""

    def __init__(self, message: str, user_message: str = None):
        user_message = user_message or f"知识图谱构建失败: {message}"
        super().__init__(message, user_message)


class RetrievalError(AppError):
    """检索相关错误。"""

    def __init__(self, message: str, user_message: str = None):
        user_message = user_message or f"检索失败: {message}"
        super().__init__(message, user_message)


class StoreError(AppError):
    """存储相关错误（SQLite 异常）。"""

    def __init__(self, message: str, user_message: str = None):
        user_message = user_message or f"数据存储错误: {message}"
        super().__init__(message, user_message)
