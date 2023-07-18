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
        response = requests.get(
            "https://api.replicate.com/v1/collections/language-models",
            headers={"Authorization": "Token {}".format(token)},
        )
        if response.status_code != 200:
            raise click.ClickException(
                "Error fetching models: {}".format(response.text)
            )
        models = response.json()["models"]
        json_path = config_dir() / "fetch-models.json"
        with open(json_path, "w") as fp:
            json.dump(models, fp, indent=2)

    @replicate.command(name="add")
    @click.argument("model_id")
    @click.option("aliases", "--alias", multiple=True, help="Aliases for this model")
    @click.option("--version", help="Model version (defaults to latest)")
    @click.option("--key", "-k", help="Replicate API key")
    def add_model(model_id, aliases, version, key):
        """
        Register additional Replicate models with LLM

        Example usage:

            llm replicate add joehoover/falcon-40b-instruct
        """
        if not version:
            # Fetch latest version from Replicate API
            token = llm.get_key(key, "replicate", env_var="REPLICATE_API_TOKEN")
            if not token:
                raise click.ClickException(
                    "Pass --key, store a 'replicate' key or set the REPLICATE_API_TOKEN environment variable."
                )
            response = requests.get(
                "https://api.replicate.com/v1/models/{}".format(model_id),
                headers={"Authorization": "Token {}".format(token)},
            )
            if response.status_code != 200:
                raise click.ClickException(
                    "Error fetching model details: {}".format(response.text)
                )
            model_details = response.json()
            version = model_details["latest_version"]["id"]
        # Add to models.json
        models_path = config_dir() / "models.json"
        if models_path.exists():
            models = json.loads(models_path.read_text())
        else:
            models = []
        models.append(
            {
                "model": model_id,
                "model_id": model_id.replace("/", "-"),
                "version": version,
                "aliases": aliases,
            }
        )
        models_path.write_text(json.dumps(models, indent=2))

    @replicate.command(name="edit-models")
    def edit_models():
        """
        Edit registered models using the default $EDITOR
        """
        models_path = config_dir() / "models.json"
        if not models_path.exists():
            models_path.write_text("[]")
        click.edit(filename=str(models_path))


@llm.hookimpl
def register_models(register):
    # First do cached JSON collection
    fetch_models_path = config_dir() / "fetch-models.json"
    if fetch_models_path.exists():
        models = json.loads(fetch_models_path.read_text())
        for details in models:
            register(
                ReplicateModel(
                    owner=details["owner"],
                    name=details["name"],
                    version_id=details["latest_version"]["id"],
                )
            )

    models_path = config_dir() / "models.json"
    if models_path.exists():
        more_models = json.loads(models_path.read_text())
        for info in more_models:
            register(
                ReplicateModel(
                    owner=info["model"].split("/")[0],
                    name=info["model"].split("/")[1],
                    version_id=info["version"],
                )
            )


class ReplicateModel(llm.Model):
    model_id = "replicate"
    needs_key = "replicate"
    key_env_var = "REPLICATE_API_TOKEN"

    def __init__(self, owner, name, version_id):
        model_id = "replicate-{}-{}".format(owner, name)
        if model_id.startswith("replicate-replicate-"):
            model_id = model_id[len("replicate-") :]
        self.model_id = model_id
        self.version_id = version_id
        self.name = name
        self.owner = owner

    def execute(self, prompt, stream, response, conversation):
        client = replicate.Client(api_token=self.get_key())
        output = client.run(
            "{owner}/{name}:{version_id}".format(
                owner=self.owner,
                name=self.name,
                version_id=self.version_id,
            ),
            input={"prompt": prompt.prompt},
        )
        yield from output


def config_dir():
    dir_path = llm.user_dir() / "replicate"
    if not dir_path.exists():
        dir_path.mkdir()
    return dir_path
