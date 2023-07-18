import click
import json
import llm
import replicate
import requests
import yaml


@llm.hookimpl
def register_commands(cli):
    @cli.group(name="replicate")
    def replicate():
        "Commands for working with models hosted on Replicate"

    @replicate.command(name="fetch-models")
    @click.option("--key", "-k", help="Replicate API key")
    def fetch_models(key):
        """
        Fetch details of models in the language-models collection

        https://replicate.com/collections/language-models
        """
        token = llm.get_key(key, "replicate", env_var="REPLICATE_API_TOKEN")
        if not token:
            raise click.ClickException(
                "Pass --key, store a 'replicate' key or set the REPLICATE_API_TOKEN environment variable."
            )
        r = requests.get(
            "https://api.replicate.com/v1/collections/language-models",
            headers={"Authorization": "Token {}".format(token)},
        )
        if r.status_code != 200:
            raise click.ClickException("Error fetching models: {}".format(r.text))
        models = r.json()["models"]
        json_path = config_dir() / "fetch-models.json"
        with open(json_path, "w") as f:
            json.dump(models, f, indent=2)


@llm.hookimpl
def register_models(register):
    # First do cached JSON collection
    json_path = config_dir() / "fetch-models.json"
    if json_path.exists():
        models = json.loads(json_path.read_text())
        for details in models:
            register(ReplicateModel(details))

    # TODO: Models the user registered themselves
    pass


class ReplicateModel(llm.Model):
    model_id = "replicate"
    needs_key = "replicate"
    key_env_var = "REPLICATE_API_TOKEN"

    def __init__(self, details):
        self.model_id = "replicate-{}".format(details["name"])
        self.details = details

    def execute(self, prompt, stream, response, conversation):
        version_id = self.details["latest_version"]["id"]
        client = replicate.Client(api_token=self.get_key())
        output = client.run(
            "{owner}/{name}:{version_id}".format(
                owner=self.details["owner"],
                name=self.details["name"],
                version_id=version_id,
            ),
            input={"prompt": prompt.prompt},
        )
        yield from output


def config_dir():
    dir_path = llm.user_dir() / "replicate"
    if not dir_path.exists():
        dir_path.mkdir()
    return dir_path
