"""Download the configured RaNER model into the service cache."""

from modelscope.hub.snapshot_download import snapshot_download

from rag_ner.config import settings


def main() -> None:
    model_dir = snapshot_download(
        settings.model_id,
        cache_dir=settings.model_cache_dir,
        revision=settings.model_revision,
    )
    print(model_dir)


if __name__ == "__main__":
    main()
