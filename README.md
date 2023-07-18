# llm-replicate

[![PyPI](https://img.shields.io/pypi/v/llm-replicate.svg)](https://pypi.org/project/llm-replicate/)
[![Changelog](https://img.shields.io/github/v/release/simonw/llm-replicate?include_prereleases&label=changelog)](https://github.com/simonw/llm-replicate/releases)
[![Tests](https://github.com/simonw/llm-replicate/workflows/Test/badge.svg)](https://github.com/simonw/llm-replicate/actions?query=workflow%3ATest)
[![License](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](https://github.com/simonw/llm-replicate/blob/main/LICENSE)

[LLM](https://llm.datasette.io/) plugin for models hosted on Replicate

## Installation

Install this plugin in the same environment as LLM.
```bash
llm install llm-replicate
```
## Configuration

You will need an API key from Replicate. You can [obtain one here](https://replicate.com/account/api-tokens).

You can set that as an environment variable called `REPLICATE_API_TOKEN`, or add it to the `llm` set of saved keys using:

```bash
llm keys set replicate
```
```
Enter key: <paste key here>
```
To fetch and save details of [the default collection](https://replicate.com/collections/language-models) of language models hosted on Replicate, run this:
```bash
llm replicate fetch-models
```

## Usage

Run `llm models list` to see the list of models:

```bash
llm models list
```
Then run a prompt through a specific model like this:
```bash
llm -m replicate-joehoover-falcon-40b-instruct "Ten great names for a pet pelican"
```

## Registering extra models

To register additional models that are not included in the default [Language models collection](https://replicate.com/collections/language-models), find their ID on Replicate and use the `llm replicate add` command.

For example, to add the [joehoover/falcon-40b-instruct](https://replicate.com/joehoover/falcon-40b-instruct) model, run this:

```bash
llm replicate add joehoover/falcon-40b-instruct --alias falcon
```
This adds the model with the alias `falcon` - you can have 0 or more aliases for a model.

Now you can run it like this:
```bash
llm -m replicate-joehoover-falcon-40b-instruct "Three reasons to get a pet falcon"
```
Or using the alias like this:
```bash
llm -m falcon "Three reasons to get a pet falcon"
```
You can edit the list of models you have registered using the default `$EDITOR` like this:
```bash
llm replicate edit-models
```

## Development

To set up this plugin locally, first checkout the code. Then create a new virtual environment:

    cd llm-palm
    python3 -m venv venv
    source venv/bin/activate

Now install the dependencies and test dependencies:

    pip install -e '.[test]'

To run the tests:

    pytest
