# TryPhx

```bash
# update hex
mix local.hex

mix archive.install \
  https://github.com/phoenixframework/archives/raw/master/1.4-dev/phx_new.ez

mix phx.new PROJECT_NAME \
  --umbrella \
  --no-webpack \
  --no-html

mix ecto.create

mix phx.server
```
