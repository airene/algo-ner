import asyncio

import httpx

from rag_ner.config import Settings
from rag_ner.main import create_app


class FakeNerService:
    ready = True
    device = "cpu"

    def load(self) -> None:
        return None

    def recognize(self, text: str):
        return [{"type": "PER", "start": 0, "end": len(text), "span": text}]


def request(app, method: str, path: str, **kwargs):
    async def run():
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            return await client.request(method, path, **kwargs)

    return asyncio.run(run())


def test_ner_response_and_readiness() -> None:
    app = create_app(Settings(), FakeNerService())

    assert request(app, "GET", "/readyz").json() == {"status": "ready", "device": "cpu"}
    response = request(app, "POST", "/v1/ner", json={"text": "孙燕姿"})
    assert response.status_code == 200
    assert response.json() == {
        "entities": [{"type": "PER", "start": 0, "end": 3, "span": "孙燕姿"}]
    }


def test_api_key_and_input_limit_are_enforced() -> None:
    app = create_app(Settings(api_key="secret", max_input_characters=3), FakeNerService())

    assert request(app, "POST", "/v1/ner", json={"text": "abc"}).status_code == 401
    assert (
        request(
            app,
            "POST",
            "/v1/ner",
            headers={"Authorization": "Bearer secret"},
            json={"text": "abcd"},
        ).status_code
        == 422
    )


def test_large_json_envelope_does_not_override_the_text_character_contract() -> None:
    app = create_app(Settings(max_input_characters=3), FakeNerService())

    response = request(
        app,
        "POST",
        "/v1/ner",
        content=" " * 5_000 + '{"text":"abc"}',
        headers={"Content-Type": "application/json"},
    )

    assert response.status_code == 200


def test_blank_text_and_unknown_fields_are_rejected() -> None:
    app = create_app(Settings(), FakeNerService())

    assert request(app, "POST", "/v1/ner", json={"text": " \n "}).status_code == 422
    assert request(app, "POST", "/v1/ner", json={"text": "北京", "typo": True}).status_code == 422
