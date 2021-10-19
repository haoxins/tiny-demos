#!/usr/bin/env zsh

poetry install
poetry run black .
poetry build
poetry run main
