#!/usr/bin/env zsh

pip install \
  --force-reinstall \
  ./demo/dist/hi-0.3.6-py3-none-any.whl

# pip freeze | grep hi > requirements.txt

python3 -m hi \
  --ns dev \
  --name hello \
  --version 12306
