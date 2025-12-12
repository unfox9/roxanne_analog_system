from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Optional


@dataclass
class ActionSchema:
    """
    負責：
    - 讀 action_mapping.txt -> channels(list[dict])
    - 定義 action_dim（用 max_index+1，比 len(channels) 穩）
    - 做最基本的 group 分類（先沿用你 JointMapper 的規則）
    """
    channels: List[dict]
    action_dim: int
    groups: Dict[str, List[dict]]

    @staticmethod
    def _infer_group(name: str) -> str:
        # 先照你原本 JointMapper 的分類邏輯 :contentReference[oaicite:3]{index=3}
        if "Left leg" in name:
            return "left_leg"
        if "Right leg" in name:
            return "right_leg"
        if ("Spine" in name) or ("Chest" in name):
            return "spine"
        if "Tail_" in name:
            return "tail"
        return "extra"

    @classmethod
    def from_mapping_file(
        cls,
        path: str,
        expected_dim: Optional[int] = None,
        strict_contiguous: bool = True,
    ) -> "ActionSchema":
        channels: List[dict] = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                parts = [p.strip() for p in line.split(",")]
                if len(parts) < 3:
                    continue
                idx = int(parts[0])
                name = parts[1]
                axis = parts[2].upper()
                if axis not in ("X", "Y", "Z"):
                    # 先保守跳過，避免怪資料把 schema 弄壞
                    continue
                channels.append({"index": idx, "name": name, "axis": axis})

        channels.sort(key=lambda c: c["index"])
        if not channels:
            raise ValueError(f"No channels parsed from mapping file: {path}")

        # action_dim 用 max_index+1，比 len(channels) 更穩（避免中間缺 index 時爆炸）
        max_idx = max(c["index"] for c in channels)
        action_dim = int(max_idx) + 1

        # 驗證 index
        indices = [c["index"] for c in channels]
        if len(set(indices)) != len(indices):
            raise ValueError("Duplicate indices found in action mapping file.")

        if strict_contiguous:
            # 期望 0..max_idx 全都存在
            expected = set(range(action_dim))
            missing = sorted(list(expected - set(indices)))
            if missing:
                raise ValueError(f"Action mapping indices not contiguous, missing: {missing[:20]} ...")

        if expected_dim is not None and action_dim != int(expected_dim):
            raise ValueError(f"mapping action_dim({action_dim}) != expected_dim({expected_dim})")

        # 分組
        groups: Dict[str, List[dict]] = {
            "left_leg": [],
            "right_leg": [],
            "spine": [],
            "tail": [],
            "extra": [],
        }
        for c in channels:
            g = cls._infer_group(c["name"])
            groups[g].append(c)

        return cls(channels=channels, action_dim=action_dim, groups=groups)
