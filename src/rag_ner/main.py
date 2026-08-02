"""FastAPI application for local RaNER serving."""

import asyncio
import logging
import secrets
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Literal

import uvicorn
from fastapi import Depends, FastAPI, Header, HTTPException, status
from pydantic import BaseModel, ConfigDict, Field, field_validator

from rag_ner import __version__
from rag_ner.config import Settings, settings
from rag_ner.services.ner_service import NerService


class NerRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    text: str = Field(min_length=1)

    @field_validator("text")
    @classmethod
    def reject_blank_text(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("Text must contain a non-whitespace character")
        return value


class Entity(BaseModel):
    type: Literal["PER", "ORG", "LOC", "GPE"]
    start: int = Field(ge=0)
    end: int = Field(gt=0)
    span: str = Field(min_length=1)


class NerResponse(BaseModel):
    entities: list[Entity]


def create_app(app_settings: Settings = settings, service: NerService | None = None) -> FastAPI:
    logging.basicConfig(level=logging.INFO)
    ner_service = service or NerService(app_settings)
    request_gate = asyncio.Semaphore(app_settings.max_concurrent_requests)

    @asynccontextmanager
    async def lifespan(_: FastAPI) -> AsyncIterator[None]:
        await asyncio.to_thread(ner_service.load)
        yield

    app = FastAPI(title="rag-ner", version=__version__, lifespan=lifespan)
    app.state.ner_service = ner_service

    def authorize(authorization: str | None = Header(default=None)) -> None:
        if not app_settings.api_key:
            return
        scheme, _, credentials = (authorization or "").partition(" ")
        key_matches = secrets.compare_digest(
            credentials.strip().encode("utf-8"), app_settings.api_key.encode("utf-8")
        )
        if scheme.lower() != "bearer" or not key_matches:
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid API key")

    @app.get("/healthz")
    def healthz() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/readyz")
    def readyz() -> dict[str, str]:
        if not ner_service.ready:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="NER model is not ready",
            )
        return {"status": "ready", "device": ner_service.device or "unknown"}

    @app.post("/v1/ner", response_model=NerResponse, dependencies=[Depends(authorize)])
    async def recognize(payload: NerRequest) -> NerResponse:
        if len(payload.text) > app_settings.max_input_characters:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
                detail=f"Text exceeds {app_settings.max_input_characters} character limit",
            )
        if request_gate.locked():
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Server is busy; retry later",
                headers={"Retry-After": "1"},
            )
        async with request_gate:
            entities = await asyncio.to_thread(ner_service.recognize, payload.text)
        return NerResponse(entities=[Entity.model_validate(entity) for entity in entities])

    return app


app = create_app()


def run_server(app_settings: Settings = settings, application: FastAPI | None = None) -> None:
    """Start the single-process server using the bind address from Settings."""
    uvicorn.run(
        application if application is not None else app,
        host=app_settings.app_host,
        port=app_settings.app_port,
    )


if __name__ == "__main__":
    run_server()
