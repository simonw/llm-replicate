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

To fetch and save details of the default set of Replicate models, run this:
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

## Development

To set up this plugin locally, first checkout the code. Then create a new virtual environment:

    cd llm-palm
    python3 -m venv venv
    source venv/bin/activate

Now install the dependencies and test dependencies:

    pip install -e '.[test]'

To run the tests:

    pytest
