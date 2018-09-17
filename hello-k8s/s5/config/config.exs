# This file is responsible for configuring your application
# and its dependencies with the aid of the Mix.Config module.
#
# This configuration file is loaded before any dependency and
# is restricted to this project.
use Mix.Config

# General application configuration
config :s5,
  ecto_repos: [S5.Repo]

# Configures the endpoint
config :s5, S5Web.Endpoint,
  url: [host: "localhost"],
  secret_key_base: "fK3Vsa/yEOH4YqMcvfIi8JHEp5Q6L7xd4/gCk8kjuXKH22HRDANo0z9FxZZJKYOp",
  render_errors: [view: S5Web.ErrorView, accepts: ~w(html json)],
  pubsub: [name: S5.PubSub,
           adapter: Phoenix.PubSub.PG2]

# Configures Elixir's Logger
config :logger, :console,
  format: "$time $metadata[$level] $message\n",
  metadata: [:user_id]

# Import environment specific config. This must remain at the bottom
# of this file so it overrides the configuration defined above.
import_config "#{Mix.env}.exs"
