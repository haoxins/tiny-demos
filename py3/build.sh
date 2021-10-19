#!/usr/bin/env zsh

python3 -m grpc_tools.protoc \
  -I./grpcdemo/protos \
  --python_out=./grpcdemo \
  --grpc_python_out=./grpcdemo \
  ./grpcdemo/protos/test.proto
