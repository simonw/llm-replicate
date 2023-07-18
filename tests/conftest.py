import pytest
import responses
from responses import matchers


@pytest.fixture
def user_path(tmpdir):
    dir = tmpdir / "llm.datasette.io"
    dir.mkdir()
    return dir


@pytest.fixture(autouse=True)
def env_setup(monkeypatch, user_path):
    monkeypatch.setenv("LLM_USER_PATH", str(user_path))
    monkeypatch.setenv(
        "REPLICATE_API_TOKEN", "fba5fb4826c5f9d7caf3fd7f49ff066c21065ddf"
    )


@pytest.fixture
def mocked_responses():
    with responses.RequestsMock() as rsps:
        rsps.get(
            "https://api.replicate.com/v1/models/simonw/hello-world/versions/v1",
            json={
                "id": "v1",
                "created_at": "2020-01-01T00:00:00",
                "cog_version": "1.0",
                "openapi_schema": {},
            },
        )
        rsps.post(
            "https://api.replicate.com/v1/predictions",
            match=[
                matchers.json_params_matcher(
                    {"version": "v1", "input": {"prompt": "world"}}
                )
            ],
            json={
                "id": "p1",
                "version": "v1",
                "urls": {
                    "get": "https://api.replicate.com/v1/predictions/p1",
                    "cancel": "https://api.replicate.com/v1/predictions/p1/cancel",
                },
                "created_at": "2022-04-26T20:00:40.658234Z",
                "completed_at": "2022-04-26T20:02:27.648305Z",
                "source": "api",
                "status": "processing",
                "input": {"text": "world"},
                "output": None,
                "error": None,
                "logs": "",
            },
        )
        rsps.get(
            "https://api.replicate.com/v1/predictions/p1",
            json={
                "id": "p1",
                "version": "v1",
                "urls": {
                    "get": "https://api.replicate.com/v1/predictions/p1",
                    "cancel": "https://api.replicate.com/v1/predictions/p1/cancel",
                },
                "created_at": "2022-04-26T20:00:40.658234Z",
                "completed_at": "2022-04-26T20:02:27.648305Z",
                "source": "api",
                "status": "succeeded",
                "input": {"text": "world"},
                "output": ["hello world"],
                "error": None,
                "logs": "",
            },
        )
        yield rsps
