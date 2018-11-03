# Since configuration is shared in umbrella projects, this file
# should only configure the :try_phx application itself
# and only for organization purposes. All other config goes to
# the umbrella root.
use Mix.Config

# Configure your database
config :try_phx, TryPhx.Repo,
  username: "phx",
  password: "phx",
  database: "try_phx_test",
  hostname: "localhost",
  port: "16002",
  pool: Ecto.Adapters.SQL.Sandbox
