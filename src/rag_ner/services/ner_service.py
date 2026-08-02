"""Thread-safe lifecycle management for the RaNER ModelScope pipeline."""

import threading
from collections.abc import Callable, Mapping
from numbers import Integral
from pathlib import Path
from typing import Any

from rag_ner.config import Settings
from rag_ner.runtime import InferenceRuntime, resolve_inference_runtime

_ENTITY_TYPES = frozenset({"PER", "ORG", "LOC", "GPE"})
_REQUIRED_MODEL_FILES = ("configuration.json", "pytorch_model.bin", "vocab.txt")
_WARMUP_TEXT = "模型加载测试"


class ModelOutputError(RuntimeError):
    """Raised when ModelScope returns data outside the RaNER output contract."""


class ModelCacheError(RuntimeError):
    """Raised when the configured local ModelScope cache is unavailable."""


class NerService:
    def __init__(
        self,
        app_settings: Settings,
        pipeline_factory: Callable[..., Any] | None = None,
        snapshot_downloader: Callable[..., str] | None = None,
        runtime_resolver: Callable[[Settings], InferenceRuntime] = resolve_inference_runtime,
    ) -> None:
        self.settings = app_settings
        self._pipeline_factory = pipeline_factory
        self._snapshot_downloader = snapshot_downloader
        self._runtime_resolver = runtime_resolver
        self._pipeline: Any | None = None
        self._runtime: InferenceRuntime | None = None
        self._load_lock = threading.Lock()
        self._predict_lock = threading.Lock()

    @property
    def ready(self) -> bool:
        return self._pipeline is not None

    @property
    def device(self) -> str | None:
        return self._runtime.device if self._runtime else None

    def load(self) -> None:
        if self.ready:
            return
        with self._load_lock:
            if self.ready:
                return
            runtime = self._runtime_resolver(self.settings)
            downloader = self._snapshot_downloader
            factory = self._pipeline_factory
            if downloader is None or factory is None:
                try:
                    from modelscope.hub.snapshot_download import snapshot_download
                    from modelscope.pipelines import pipeline
                    from modelscope.utils.constant import Tasks
                except ModuleNotFoundError as exc:
                    raise RuntimeError(
                        "NER dependencies are missing; run uv sync --extra ml --locked"
                    ) from exc
                downloader = downloader or snapshot_download
                if factory is None:

                    def default_pipeline_factory(**kwargs: Any) -> Any:
                        return pipeline(Tasks.named_entity_recognition, **kwargs)

                    factory = default_pipeline_factory

            try:
                model_dir = downloader(
                    self.settings.model_id,
                    cache_dir=self.settings.model_cache_dir,
                    revision=self.settings.model_revision,
                    local_files_only=True,
                )
            except (OSError, ValueError) as exc:
                raise ModelCacheError(
                    "RaNER model is not available in the local cache; run "
                    "uv run --extra ml --locked python scripts/download_model.py"
                ) from exc
            _validate_model_dir(model_dir)

            candidate = factory(model=model_dir, device=runtime.device)
            # ModelScope prepares the device lazily on its first call. Warm up
            # before reporting ready so CUDA/OOM and output-contract failures
            # happen during application startup instead of the first request.
            warmup_result = candidate(_WARMUP_TEXT, return_prob=False)
            _normalize_entities(warmup_result, _WARMUP_TEXT)

            self._runtime = runtime
            self._pipeline = candidate

    def recognize(self, text: str) -> list[dict[str, Any]]:
        self.load()
        with self._predict_lock:
            assert self._pipeline is not None
            result = self._pipeline(text, return_prob=False)
        return _normalize_entities(result, text)


def _validate_model_dir(model_dir: str) -> None:
    path = Path(model_dir)
    missing = [name for name in _REQUIRED_MODEL_FILES if not (path / name).is_file()]
    if missing:
        missing_files = ", ".join(missing)
        raise ModelCacheError(
            f"RaNER model cache is incomplete at {path}: missing {missing_files}; "
            "run uv run --extra ml --locked python scripts/download_model.py"
        )


def _normalize_entities(result: Any, text: str) -> list[dict[str, Any]]:
    if not isinstance(result, Mapping):
        raise ModelOutputError("RaNER returned a non-object result")
    entities = result.get("output")
    if not isinstance(entities, list):
        raise ModelOutputError("RaNER result is missing the output entity list")

    normalized: list[dict[str, Any]] = []
    for entity in entities:
        if not isinstance(entity, Mapping):
            raise ModelOutputError("RaNER returned a non-object entity")
        entity_type = entity.get("type")
        start = entity.get("start")
        end = entity.get("end")
        span = entity.get("span")
        if not isinstance(entity_type, str) or entity_type not in _ENTITY_TYPES:
            raise ModelOutputError(f"RaNER returned an unknown entity type: {entity_type!r}")
        if not isinstance(span, str) or not span:
            raise ModelOutputError("RaNER returned an invalid entity span")
        if (
            isinstance(start, bool)
            or isinstance(end, bool)
            or not isinstance(start, Integral)
            or not isinstance(end, Integral)
        ):
            raise ModelOutputError("RaNER returned non-integer entity offsets")
        start_index = int(start)
        end_index = int(end)
        if start_index < 0 or end_index <= start_index or end_index > len(text):
            raise ModelOutputError("RaNER returned out-of-range entity offsets")
        if text[start_index:end_index] != span:
            raise ModelOutputError("RaNER entity span does not match its offsets")
        normalized.append(
            {"type": entity_type, "start": start_index, "end": end_index, "span": span}
        )
    return normalized
