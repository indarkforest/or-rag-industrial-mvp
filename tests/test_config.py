"""配置加载与校验测试。"""
import os
import tempfile

import pytest
import yaml

from src.config import load_config, project_path
from src.exceptions import ConfigError


class TestLoadConfig:
    def test_valid_config(self):
        cfg = load_config()
        assert "model" in cfg
        assert "embedding" in cfg
        assert "data" in cfg
        assert "retrieval" in cfg

    def test_config_paths_are_absolute(self):
        cfg = load_config()
        for key in ("docs_dir", "ontology_path", "questions_path", "db_path"):
            path = cfg["data"][key]
            assert os.path.isabs(path), f"{key} should be absolute: {path}"

    def test_env_var_overrides_api_key(self, monkeypatch):
        monkeypatch.setenv("MINIMAX_API_KEY", "env-test-key")
        cfg = load_config()
        assert cfg["model"]["api_key"] == "env-test-key"

    def test_missing_config_file(self):
        with pytest.raises(ConfigError, match="config"):
            load_config("/nonexistent/path/config.yaml")

    def test_invalid_yaml(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write("model: [invalid\n  yaml: {")
            path = f.name
        try:
            with pytest.raises(ConfigError):
                load_config(path)
        finally:
            os.unlink(path)

    def test_invalid_config_values(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump({
                "model": {"api_key": "k", "name": "m"},
                "embedding": {"provider": "invalid_provider"},
                "data": {"chunk_size": 10},
            }, f)
            path = f.name
        try:
            with pytest.raises(ConfigError, match="validation"):
                load_config(path)
        finally:
            os.unlink(path)


class TestProjectPath:
    def test_returns_absolute(self):
        p = project_path("config.yaml")
        assert os.path.isabs(p)

    def test_joins_parts(self):
        p = project_path("data", "docs", "file.md")
        assert "data" in p
        assert "docs" in p
        assert "file.md" in p
