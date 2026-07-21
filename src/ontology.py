"""本体加载：解析 JSON-LD，生成供 LLM 抽取使用的 Schema 描述。"""
import json
from typing import Dict, List


class Ontology:
    def __init__(self, path: str):
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        self.classes: List[dict] = []
        self.properties: List[dict] = []
        for item in data.get("@graph", []):
            raw_id = item.get("@id", "")
            if not raw_id:
                continue
            entry = {
                "id": raw_id.split(":")[-1],
                "label": item.get("rdfs:label", ""),
                "comment": item.get("rdfs:comment", ""),
            }
            if item.get("@type") == "rdfs:Class":
                self.classes.append(entry)
            elif item.get("@type") == "rdfs:Property":
                entry["domain"] = item.get("onto:domain", "").split(":")[-1]
                entry["range"] = item.get("onto:range", "").split(":")[-1]
                self.properties.append(entry)

    @property
    def class_ids(self) -> List[str]:
        return [c["id"] for c in self.classes]

    @property
    def property_ids(self) -> List[str]:
        return [p["id"] for p in self.properties]

    def to_prompt(self) -> str:
        """生成本体 Schema 的自然语言描述，注入抽取 prompt。"""
        lines = ["## 实体类型（type 必须从中选择）"]
        for c in self.classes:
            lines.append(f"- {c['id']}（{c['label']}）：{c['comment']}")
        lines.append("")
        lines.append("## 关系类型（relation 必须从中选择）")
        for p in self.properties:
            lines.append(
                f"- {p['id']}（{p['label']}，{p.get('domain','?')} -> {p.get('range','?')}）：{p['comment']}"
            )
        return "\n".join(lines)

    def relation_labels(self) -> Dict[str, str]:
        return {p["id"]: p["label"] for p in self.properties}
