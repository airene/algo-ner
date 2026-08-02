"""Validate the two supported inference runtimes before model loading."""

import sys
from dataclasses import dataclass

from rag_ner.config import Settings


@dataclass(frozen=True)
class InferenceRuntime:
    device: str


class RuntimeConfigurationError(RuntimeError):
    """Raised when the requested accelerator cannot be used safely."""


def resolve_inference_runtime(app_settings: Settings) -> InferenceRuntime:
    requested = app_settings.device.strip().lower()
    if requested == "cpu":
        return InferenceRuntime(device="cpu")

    if not requested.startswith("cuda:"):
        raise RuntimeConfigurationError("DEVICE must be cpu or cuda:<non-negative device index>")
    if sys.platform == "darwin":
        raise RuntimeConfigurationError("macOS deployment supports DEVICE=cpu only")
    if sys.platform != "linux":
        raise RuntimeConfigurationError("GPU deployment supports Debian/Linux x86_64 only")

    try:
        device_index = int(requested.removeprefix("cuda:"))
    except ValueError as exc:
        raise RuntimeConfigurationError("DEVICE must use a numeric CUDA device index") from exc
    if device_index < 0:
        raise RuntimeConfigurationError("CUDA device index must not be negative")

    try:
        import torch
    except ModuleNotFoundError as exc:
        raise RuntimeConfigurationError(
            "PyTorch is missing; run uv sync --extra ml --locked"
        ) from exc

    if not torch.cuda.is_available():
        raise RuntimeConfigurationError(
            "CUDA is unavailable; refusing to silently fall back to CPU"
        )
    if device_index >= torch.cuda.device_count():
        raise RuntimeConfigurationError(
            f"CUDA device {device_index} does not exist; found "
            f"{torch.cuda.device_count()} device(s)"
        )
    return InferenceRuntime(device=f"cuda:{device_index}")
