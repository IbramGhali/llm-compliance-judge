"""Wiring: build the pipeline and its dependencies from settings."""

from __future__ import annotations

from dataclasses import dataclass

from compliance_judge.config import Settings
from compliance_judge.llm import make_client
from compliance_judge.pipeline.runner import Pipeline
from compliance_judge.prompts import load_prompts
from compliance_judge.rulebook import Rulebook, load_rulebook
from compliance_judge.store import Store


@dataclass
class App:
    settings: Settings
    rulebook: Rulebook
    store: Store
    pipeline: Pipeline


def build_app(settings: Settings | None = None) -> App:
    settings = settings or Settings.from_env()
    rulebook = load_rulebook(settings.rulebook_path)
    store = Store(settings.db_path)
    pipeline = Pipeline(
        settings, make_client(settings), rulebook, load_prompts(settings.prompts_dir), store
    )
    return App(settings, rulebook, store, pipeline)
