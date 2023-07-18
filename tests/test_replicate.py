from click.testing import CliRunner
import json
from llm.cli import cli
import pathlib
from unittest.mock import Mock, patch

flan_t5 = json.loads(
    (pathlib.Path(__file__).parent / "replicate-flan-t5-xl.json").read_text()
)


@patch("llm_replicate.vendored_replicate.Client")
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


def test_add_model(user_path, requests_mock):
    requests_mock.get(
        "https://api.replicate.com/v1/models/a16z-infra/llama13b-v2-chat",
        json={"latest_version": {"id": "llama2-id-123"}},
    )
    runner = CliRunner()
    result = runner.invoke(
        cli, ["replicate", "add", "a16z-infra/llama13b-v2-chat", "--alias", "llama2"]
    )
    assert result.exit_code == 0, result.output
    # Should be in the models.json
    models = json.loads((user_path / "replicate" / "models.json").read_text("utf-8"))
    assert models == [
        {
            "model": "a16z-infra/llama13b-v2-chat",
            "model_id": "a16z-infra-llama13b-v2-chat",
            "version": "llama2-id-123",
            "aliases": ["llama2"],
        }
    ]
