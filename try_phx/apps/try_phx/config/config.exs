# Since configuration is shared in umbrella projects, this file
# should only configure the :try_phx application itself
# and only for organization purposes. All other config goes to
# the umbrella root.
use Mix.Config

config :try_phx, ecto_repos: [TryPhx.Repo]

import_config "#{Mix.env}.exs"
