#!/usr/bin/env zsh

pip install \
  --force-reinstall \
  ./demo/dist/hi-0.1.2-py3-none-any.whl

pip freeze | grep hi > requirements.txt
