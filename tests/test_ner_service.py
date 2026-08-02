from typing import Any

import numpy as np
import pytest

from rag_ner.config import Settings
from rag_ner.runtime import InferenceRuntime
from rag_ner.services.ner_service import (
    ModelCacheError,
    ModelOutputError,
    NerService,
    _normalize_entities,
)


def test_service_loads_cached_model_warms_up_and_normalizes_entities(tmp_path) -> None:
    calls: dict[str, Any] = {}
    model_dir = tmp_path / "raner"
    model_dir.mkdir()
    for filename in ("configuration.json", "pytorch_model.bin", "vocab.txt"):
        (model_dir / filename).touch()

    def download(*args: Any, **kwargs: Any) -> str:
        calls["download"] = (args, kwargs)
        return str(model_dir)

    def pipeline_factory(**kwargs: Any):
        calls["pipeline"] = kwargs

        def predict(text: str, **predict_kwargs: Any):
            calls.setdefault("predictions", []).append((text, predict_kwargs))
            if text == "模型加载测试":
                return {"output": []}
            return {
                "output": [
                    {"type": "PER", "start": 0, "end": 3, "span": text[:3]},
                    {
                        "type": "LOC",
                        "start": np.int64(4),
                        "end": np.int64(6),
                        "span": "北京",
                    },
                ]
            }

        return predict

    service = NerService(
        Settings(model_cache_dir="/tmp/models"),
        pipeline_factory=pipeline_factory,
        snapshot_downloader=download,
        runtime_resolver=lambda _: InferenceRuntime("cpu"),
    )

    assert service.recognize("孙燕姿在北京") == [
        {"type": "PER", "start": 0, "end": 3, "span": "孙燕姿"},
        {"type": "LOC", "start": 4, "end": 6, "span": "北京"},
    ]
    assert service.ready
    assert service.device == "cpu"
    assert calls["pipeline"] == {"model": str(model_dir), "device": "cpu"}
    assert calls["download"][1]["local_files_only"] is True
    assert calls["predictions"] == [
        ("模型加载测试", {"return_prob": False}),
        ("孙燕姿在北京", {"return_prob": False}),
    ]


@pytest.mark.parametrize(
    "result",
    [
        None,
        {},
        {"output": "not-a-list"},
        {"output": [{"type": "UNKNOWN", "start": 0, "end": 1, "span": "北"}]},
        {"output": [{"type": "LOC", "start": True, "end": 1, "span": "北"}]},
        {"output": [{"type": "LOC", "start": 0, "end": 2, "span": "北京"}]},
    ],
)
def test_invalid_model_output_is_not_silently_treated_as_no_entities(result) -> None:
    with pytest.raises(ModelOutputError):
        _normalize_entities(result, "北")


def test_missing_local_model_has_an_actionable_error() -> None:
    def missing_download(*_: Any, **__: Any) -> str:
        raise ValueError("missing")

    service = NerService(
        Settings(),
        pipeline_factory=lambda **_: None,
        snapshot_downloader=missing_download,
        runtime_resolver=lambda _: InferenceRuntime("cpu"),
    )

    with pytest.raises(ModelCacheError, match="scripts/download_model.py"):
        service.load()
