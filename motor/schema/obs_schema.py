from __future__ import annotations
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Literal
import numpy as np


ObsLevel = Literal["reflex", "motor", "policy"]


@dataclass(frozen=True)
class ObsItem:
    name: str
    dim: int
    level: ObsLevel
    required: bool = True
    clip: Optional[float] = None
    normalize: Optional[str] = None
    desc: str = ""


class ObsSchema:
    def __init__(self, items: List[ObsItem]):
        self.items = items
        self._by_level: Dict[ObsLevel, List[ObsItem]] = {
            "reflex": [],
            "motor": [],
            "policy": [],
        }
        for it in items:
            self._by_level[it.level].append(it)

    def items_by_level(self, level: ObsLevel) -> List[ObsItem]:
        return list(self._by_level[level])

    def dim(self, level: Optional[ObsLevel] = None) -> int:
        if level is None:
            return sum(it.dim for it in self.items)
        return sum(it.dim for it in self._by_level[level])

    def summary(self) -> str:
        lines = ["ObsSchema:"]
        for lvl in ["reflex", "motor", "policy"]:
            d = self.dim(lvl)
            lines.append(f"  {lvl:7s}: dim={d}")
            for it in self._by_level[lvl]:
                req = "REQ" if it.required else "opt"
                lines.append(f"    - {it.name:24s} ({it.dim}) [{req}]")
        return "\n".join(lines)

    def validate(self, obs_dict: Dict[str, np.ndarray]):
        for it in self.items:
            if it.required and it.name not in obs_dict:
                raise KeyError(f"Missing required obs: {it.name}")
            if it.name in obs_dict:
                x = np.asarray(obs_dict[it.name])
                if x.size != it.dim:
                    raise ValueError(
                        f"Obs '{it.name}' dim mismatch: "
                        f"expected {it.dim}, got {x.size}"
                    )

    def build_vector(
        self,
        obs_dict: Dict[str, np.ndarray],
        level: Optional[ObsLevel] = None,
    ) -> np.ndarray:
        items = self.items if level is None else self._by_level[level]
        vec = []
        for it in items:
            if it.name not in obs_dict:
                if it.required:
                    raise KeyError(f"Missing required obs: {it.name}")
                else:
                    vec.append(np.zeros((it.dim,), dtype=np.float32))
                    continue

            x = np.asarray(obs_dict[it.name], dtype=np.float32).reshape(-1)

            if it.clip is not None:
                x = np.clip(x, -it.clip, it.clip)

            vec.append(x)

        if not vec:
            return np.zeros((0,), dtype=np.float32)
        return np.concatenate(vec, axis=0)
