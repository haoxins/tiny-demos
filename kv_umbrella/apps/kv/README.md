
# KV

```bash
mix new kv --module KV

mix compile

mix test
# run particular test
mix test test/kv_test.exs:5
# fmt
mix format
# env
MIX_ENV=prod mix compile
# test `distributed`
iex --sname bar -S mix
elixir --sname foo -S mix test --only distributed
```
