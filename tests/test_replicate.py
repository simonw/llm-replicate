from click.testing import CliRunner
import json
from llm.cli import cli
import pathlib
import pytest
import sqlite_utils
from unittest.mock import Mock, patch

flan_t5 = json.loads(
    (pathlib.Path(__file__).parent / "replicate-flan-t5-xl.json").read_text()
)


@patch("replicate.Client")
def test_replicate_prompt(mock_class, user_path):
    runner = CliRunner()

    mock_client = Mock()
    mock_run_output = ["hello", " world"]
    mock_client.run.return_value = mock_run_output
    mock_class.return_value = mock_client

    (user_path / "replicate").mkdir()
    (user_path / "replicate" / "fetch-models.json").write_text(
        json.dumps([flan_t5]),
        "utf-8",
    )
    result = runner.invoke(cli, ["-m", "replicate-flan-t5-xl", "say hi"])
    assert result.exit_code == 0, result.output
    assert result.output == "hello world\n"
    call = mock_client.run.call_args_list[0]
    assert call.args == (
        "replicate/flan-t5-xl:7a216605843d87f5426a10d2cc6940485a232336ed04d655ef86b91e020e9210",
    )
    assert call.kwargs == {"input": {"prompt": "say hi"}}


@pytest.mark.parametrize("chat", (True, False))
def test_add_model(user_path, requests_mock, chat):
    requests_mock.get(
        "https://api.replicate.com/v1/models/a16z-infra/llama13b-v2-chat",
        json={"latest_version": {"id": "llama2-id-123"}},
    )
    runner = CliRunner()
    result = runner.invoke(
        cli,
        [
            "replicate",
            "add",
            "a16z-infra/llama13b-v2-chat",
            "--alias",
            "llama2",
        ]
        + (["--chat"] if chat else []),
    )
    assert result.exit_code == 0, result.output
    # Should be in the models.json
    models = json.loads((user_path / "replicate" / "models.json").read_text("utf-8"))
    expected = {
        "model": "a16z-infra/llama13b-v2-chat",
        "model_id": "a16z-infra-llama13b-v2-chat",
        "version": "llama2-id-123",
        "aliases": ["llama2"],
    }
    if chat:
        expected["chat"] = True
    assert models == [expected]


@patch("replicate.Client")
def test_chat_model_prompt(mock_class, user_path, requests_mock):
    # First register that model, re-using existing test
    test_add_model(user_path, requests_mock, chat=True)

    # Set up mock
    mock_client = Mock()
    mock_run_output = ["hello", " world"]
    mock_client.run.return_value = mock_run_output
    mock_class.return_value = mock_client

    # Run the prompt
    runner = CliRunner()
    result = runner.invoke(cli, ["-m", "llama2", "say hi"])
    assert result.exit_code == 0, result.output
    assert result.output == "hello world\n"
    call = mock_client.run.call_args_list[0]
    assert call.args == ("a16z-infra/llama13b-v2-chat:llama2-id-123",)
    assert call.kwargs == {"input": {"prompt": "User: say hi\nAssistant:"}}

    # Add a continued prompt
    result2 = runner.invoke(cli, ["-c", "and again"])
    call = mock_client.run.call_args_list[1]
    assert call.args == ("a16z-infra/llama13b-v2-chat:llama2-id-123",)
    assert call.kwargs == {
        "input": {
            "prompt": "User: say hi\nAssistant: hello world\nUser: and again\nAssistant:"
        }
    }

    # Check it was correctly logged
    db = sqlite_utils.Database(str(user_path / "logs.db"))
    # Should be one conversation
    conversations = list(db["conversations"].rows)
    assert len(conversations) == 1
    conversation = conversations[0]
    # With two responses
    assert db["responses"].count == 2
    responses = list(
        db.query(
            """
            select model, prompt, prompt_json, response from responses
            where conversation_id = ? order by id
            """,
            (conversation["id"],),
        )
    )
    assert responses == [
        {
            "model": "replicate-a16z-infra-llama13b-v2-chat",
            "prompt": "say hi",
            "prompt_json": '{"lines": ["User: say hi\\n", "Assistant:"]}',
            "response": "hello world",
        },
        {
            "model": "replicate-a16z-infra-llama13b-v2-chat",
            "prompt": "and again",
            "prompt_json": '{"lines": ["User: say hi\\n", "Assistant: hello world\\n", "User: and again\\n", "Assistant:"]}',
            "response": "hello world",
        },
    ]


@pytest.mark.parametrize("skip_existing", (False, True))
def test_fetch_predictions(user_path, requests_mock, skip_existing):
    db = sqlite_utils.Database(str(user_path / "logs.db"))
    # Register a model with a version_id to test _model_guess
    (user_path / "replicate").mkdir()
    (user_path / "replicate" / "models.json").write_text(
        json.dumps(
            [
                {
                    "model": "a16z-infra/llama13b-v2-chat",
                    "model_id": "a16z-infra-llama13b-v2-chat",
                    "version": "version-1",
                }
            ]
        ),
        "utf-8",
    )
    assert not db["replicate_predictions"].exists()
    if skip_existing:
        db["replicate_predictions"].insert(
            {
                "id": "prediction-2",
                "blah": "blah",
                "_model_guess": None,
                "version": "version-2",
                "completed_at": "2021-08-31T18:00:00.000000Z",
            },
            pk="id",
        )

    requests_mock.get(
        "https://api.replicate.com/v1/predictions",
        json={
            "next": "https://api.replicate.com/v1/predictions?cursor=2",
            "results": [
                {
                    "id": "prediction-1",
                    "urls": {"get": "https://api.replicate.com/v1/predictions/1"},
                    "version": "version-1",
                }
            ],
        },
    )
    requests_mock.get(
        "https://api.replicate.com/v1/predictions?cursor=2",
        json={
            "next": None,
            "results": [
                {
                    "id": "prediction-2",
                    "urls": {"get": "https://api.replicate.com/v1/predictions/2"},
                    "version": "version-2",
                }
            ],
        },
    )
    pred1 = requests_mock.get(
        "https://api.replicate.com/v1/predictions/1",
        json={
            "id": "prediction-1",
            "blah": "blah",
            "version": "version-1",
            "completed_at": "2021-08-31T18:00:00.000000Z",
        },
    )
    pred2 = requests_mock.get(
        "https://api.replicate.com/v1/predictions/2",
        json={
            "id": "prediction-2",
            "blah": "blah",
            "version": "version-2",
            "completed_at": "2021-08-31T18:00:00.000000Z",
        },
    )
    runner = CliRunner()
    result = runner.invoke(
        cli, ["replicate", "fetch-predictions"], catch_exceptions=False
    )
    assert result.exit_code == 0, result.output
    assert db["replicate_predictions"].count == 2
    assert db["replicate_predictions"].get("prediction-1") == {
        "id": "prediction-1",
        "blah": "blah",
        "_model_guess": "a16z-infra/llama13b-v2-chat",
        "version": "version-1",
        "completed_at": "2021-08-31T18:00:00.000000Z",
    }
    assert db["replicate_predictions"].get("prediction-2") == {
        "id": "prediction-2",
        "blah": "blah",
        "_model_guess": None,
        "version": "version-2",
        "completed_at": "2021-08-31T18:00:00.000000Z",
    }

    if skip_existing:
        # Only first prediction URL should have been fetched
        assert pred1.called
        assert not pred2.called
    else:
        # Both should have been fetched
        assert pred1.called
        assert pred2.called
