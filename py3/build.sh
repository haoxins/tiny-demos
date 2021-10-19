#!/usr/bin/env zsh

# python3 -m grpc_tools.protoc \
#   -I./grpcdemo/protos \
#   --python_out=./grpcdemo/protos \
#   --grpc_python_out=./grpcdemo/protos \
#   ./grpcdemo/protos/test.proto

poetry install
poetry run black .
poetry build
