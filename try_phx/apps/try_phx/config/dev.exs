# Since configuration is shared in umbrella projects, this file
# should only configure the :try_phx application itself
# and only for organization purposes. All other config goes to
# the umbrella root.
use Mix.Config

# Configure your database
config :try_phx, TryPhx.Repo,
  username: "root",
  password: "root",
  database: "try_phx_dev",
  hostname: "localhost",
  pool_size: 10
