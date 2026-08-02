import pytest

from rag_ner.config import Settings
from rag_ner.runtime import RuntimeConfigurationError, resolve_inference_runtime


def test_cpu_requires_no_torch() -> None:
    assert resolve_inference_runtime(Settings(device="cpu")).device == "cpu"


def test_invalid_device_is_rejected() -> None:
    with pytest.raises(RuntimeConfigurationError, match="cpu or cuda"):
        resolve_inference_runtime(Settings(device="mps"))


def test_cuda_device_index_must_be_numeric(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("rag_ner.runtime.sys.platform", "linux")
    with pytest.raises(RuntimeConfigurationError, match="numeric"):
        resolve_inference_runtime(Settings(device="cuda:one"))
