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
