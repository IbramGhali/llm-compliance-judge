"""Settings from environment variables (and an optional local .env file).

Every setting is documented in .env.example. Defaults keep the project offline.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _load_dotenv(path: Path) -> None:
    """Minimal .env reader: KEY=VALUE lines, existing environment wins."""
    if not path.is_file():
        return
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


def _env(name: str, default: str) -> str:
    return os.environ.get(name, default) or default


@dataclass(frozen=True)
class Settings:
    provider: str = "fake"
    extraction_model: str = "claude-haiku-5-5"
    assessment_model: str = "claude-opus-5-5"
    effort: str = "medium"
    confidence_threshold: float = 0.6
    fake_quote_drift: float = 0.03
    gate_tolerance: float = 0.02
    data_dir: Path = field(default_factory=lambda: PROJECT_ROOT / "data")
    db_path: Path = field(default_factory=lambda: PROJECT_ROOT / ".judge" / "judge.db")
    rulebook_path: Path = field(default_factory=lambda: PROJECT_ROOT / "config" / "rulebook.yaml")
    pricing_path: Path = field(default_factory=lambda: PROJECT_ROOT / "config" / "pricing.yaml")
    prompts_dir: Path = field(default_factory=lambda: PROJECT_ROOT / "prompts")
    golden_path: Path = field(default_factory=lambda: PROJECT_ROOT / "golden" / "labels.jsonl")
    baseline_path: Path = field(default_factory=lambda: PROJECT_ROOT / "eval" / "baseline.json")

    @classmethod
    def from_env(cls) -> Settings:
        _load_dotenv(PROJECT_ROOT / ".env")
        defaults = cls()
        provider = _env("JUDGE_LLM_PROVIDER", defaults.provider).lower()
        return cls(
            provider=provider,
            extraction_model=_env("JUDGE_EXTRACTION_MODEL", defaults.extraction_model),
            assessment_model=_env("JUDGE_ASSESSMENT_MODEL", defaults.assessment_model),
            effort=_env("JUDGE_EFFORT", defaults.effort),
            confidence_threshold=float(
                _env("JUDGE_CONFIDENCE_THRESHOLD", str(defaults.confidence_threshold))
            ),
            fake_quote_drift=float(_env("JUDGE_FAKE_QUOTE_DRIFT", str(defaults.fake_quote_drift))),
            gate_tolerance=float(_env("JUDGE_GATE_TOLERANCE", str(defaults.gate_tolerance))),
            data_dir=Path(_env("JUDGE_DATA_DIR", str(defaults.data_dir))),
            db_path=Path(_env("JUDGE_DB_PATH", str(defaults.db_path))),
            rulebook_path=Path(_env("JUDGE_RULEBOOK", str(defaults.rulebook_path))),
        )
