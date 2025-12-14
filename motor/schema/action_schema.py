from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Pattern, Sequence, Tuple
import re
from collections import defaultdict


def _sep(word: str) -> str:
    return rf"(?:^|[_\s]){word}(?:$|[_\s])"


@dataclass(frozen=True)
class GroupRule:
    part: str
    pattern: str
    midline: bool = False
    axis: Optional[str] = None

    def compile(self) -> Pattern[str]:
        return re.compile(self.pattern, re.IGNORECASE)


DEFAULT_RULES: List[GroupRule] = [
    GroupRule("eye", r"(eye|eyeball|iris|pupil|eyelid|blink|gaze)"),
    GroupRule("mouth", r"(jaw|mouth|lip|tongue|teeth|smile|frown|cheek)", midline=True),
    GroupRule("ear", _sep("ear")),
    GroupRule("finger", r"(finger|thumb)"),
    GroupRule("toe", _sep("toes")),
    GroupRule("wrist", _sep("wrist")),
    GroupRule("elbow", _sep("elbow")),
    GroupRule("arm", rf"{_sep('arm')}|shoulder|scapul|clavic"),
    GroupRule("ankle", rf"{_sep('ankle')}|hock"),
    GroupRule("knee", _sep("knee")),
    GroupRule("leg", rf"{_sep('leg')}|hip|thigh"),
    GroupRule("tail", r"^tail_|(?:^|[_\s])tail(?:$|[_\s])", midline=True),
    GroupRule(
        "spine", rf"{_sep('spine')}|{_sep('chest')}|waist|upper\s+chest", midline=True
    ),
    GroupRule("neck", _sep("neck"), midline=True),
    GroupRule("head", _sep("head"), midline=True),
]


class ActionSchema:
    def __init__(
        self,
        channels: list[Dict],
        rules: Sequence[GroupRule] = DEFAULT_RULES,
        strict_contiguous: bool = True,
    ):
        self.rules: List[Tuple[GroupRule, Pattern[str]]] = [
            (r, r.compile()) for r in rules
        ]

        channels = sorted(channels, key=lambda c: int(c["index"]))
        self.channels: List[Dict] = channels
        self.action_dim: int = (
            (max(int(c["index"]) for c in channels) + 1) if channels else 0
        )

        if strict_contiguous:
            expected = list(range(self.action_dim))
            got = [int(c["index"]) for c in channels]
            if got != expected:
                raise ValueError(
                    f"Non-contiguous / non-0-based mapping indices.\n"
                    f"expected: {expected[:10]}...{expected[-10:]}\n"
                    f"got     : {got[:10]}...{got[-10:]}"
                )

        self.groups: Dict[str, List[Dict]] = defaultdict(list)
        for c in self.channels:
            name = str(c["name"])
            axis = str(c["axis"]).upper()

            side = self._detect_side(name)
            part = self._match_part(name=name, axis=axis)
            group = self._make_group(
                part=part, side=side, midline=self._is_midline(part)
            )

            c["side"] = side
            c["part"] = part
            c["group"] = group
            c["tags"] = self._make_tags(part=part, side=side, axis=axis)

            self.groups[group].append(c)

        if "extra" not in self.groups:
            self.groups["extra"] = []

    @classmethod
    def from_mapping_file(
        cls,
        path: str,
        rules: Sequence[GroupRule] = DEFAULT_RULES,
        strict_contiguous: bool = True,
    ) -> "ActionSchema":
        channels: List[Dict] = []
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
                channels.append({"index": idx, "name": name, "axis": axis})
        return cls(channels=channels, rules=rules, strict_contiguous=strict_contiguous)

    def _detect_side(self, name: str) -> Optional[str]:
        if re.search(r"\bleft\b", name, flags=re.IGNORECASE) or re.search(
            r"(?:^|_)L(?:$|_)", name
        ):
            return "l"
        if re.search(r"\bright\b", name, flags=re.IGNORECASE) or re.search(
            r"(?:^|_)R(?:$|_)", name
        ):
            return "r"
        return None

    def _match_part(self, name: str, axis: str) -> str:
        for rule, cre in self.rules:
            if rule.axis is not None and rule.axis.upper() != axis:
                continue
            if cre.search(name):
                return rule.part
        return "extra"

    def _is_midline(self, part: str) -> bool:
        return part in {"spine", "tail", "neck", "head", "mouth"}

    def _make_group(self, part: str, side: Optional[str], midline: bool) -> str:
        if part == "extra":
            return "extra"
        if midline or side is None:
            return part
        return f"{part}_{side}"

    def _make_tags(self, part: str, side: Optional[str], axis: str) -> List[str]:
        tags = [part, f"axis_{axis.lower()}"]
        if side in ("l", "r"):
            tags.append(f"side_{side}")
        return tags

    def get(self, group: str) -> List[Dict]:
        return self.groups.get(group, [])

    def summary(self) -> str:
        keys = sorted(self.groups.keys())
        lines = [f"action_dim={self.action_dim}"]
        for k in keys:
            lines.append(f"{k:10s}: {len(self.groups[k])}")
        return "\n".join(lines)
