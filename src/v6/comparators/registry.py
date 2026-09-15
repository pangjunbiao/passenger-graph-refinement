"""Closed registry for Step-11 established baselines only."""

from __future__ import annotations

from typing import Any, Mapping

from src.v6.comparators.base import BaselineAdapter
from src.v6.comparators.bertopic import BERTopicAdapter
from src.v6.comparators.btm import BTMAdapter
from src.v6.comparators.combinedtm import CombinedTMAdapter
from src.v6.comparators.gsdmm import GSDMMAdapter
from src.v6.comparators.lda import LDAAdapter
from src.v6.comparators.nmf import NMFAdapter


REGISTRY: dict[str, type[BaselineAdapter]] = {
    adapter.method_id: adapter
    for adapter in (
        LDAAdapter,
        NMFAdapter,
        GSDMMAdapter,
        BTMAdapter,
        BERTopicAdapter,
        CombinedTMAdapter,
    )
}


def make_adapter(method_id: str, config: Mapping[str, Any]) -> BaselineAdapter:
    try:
        adapter_type = REGISTRY[str(method_id)]
    except KeyError as exc:
        raise KeyError(f"Unknown or non-Step-11 baseline: {method_id}") from exc
    return adapter_type(config)
