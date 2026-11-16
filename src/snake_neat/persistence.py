"""Saving / loading a genome in JSON format (standard library)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .config import GameConfig
from .genome import Genome

FORMAT = "snake-neat/1"


def save_genome(path: str | Path, genome: Genome, game_cfg: GameConfig, meta: dict[str, Any] | None = None) -> None:
    data = {"format": FORMAT, "game": game_cfg.to_dict(), "genome": genome.to_dict(), "meta": meta or {}}
    Path(path).write_text(json.dumps(data, indent=1))


def load_genome(path: str | Path) -> tuple[Genome, GameConfig, dict[str, Any]]:
    data = json.loads(Path(path).read_text())
    if data.get("format") != FORMAT:
        raise ValueError(f"unknown format: {data.get('format')!r}")
    return Genome.from_dict(data["genome"]), GameConfig.from_dict(data["game"]), data.get("meta", {})
