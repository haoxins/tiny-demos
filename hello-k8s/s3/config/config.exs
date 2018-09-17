# This file is responsible for configuring your application
# and its dependencies with the aid of the Mix.Config module.
#
# This configuration file is loaded before any dependency and
# is restricted to this project.
use Mix.Config

# General application configuration
config :s3,
  ecto_repos: [S3.Repo]

# Configures the endpoint
config :s3, S3Web.Endpoint,
  url: [host: "localhost"],
  secret_key_base: "Ok905FYcMzZiukAwOWGOCYNgDwsWOqBP5rmoacTnMtDIdP8qy9qs3mfgWfGo+oD9",
  render_errors: [view: S3Web.ErrorView, accepts: ~w(html json)],
  pubsub: [name: S3.PubSub,
           adapter: Phoenix.PubSub.PG2]

# Configures Elixir's Logger
config :logger, :console,
  format: "$time $metadata[$level] $message\n",
  metadata: [:user_id]

# Import environment specific config. This must remain at the bottom
# of this file so it overrides the configuration defined above.
import_config "#{Mix.env}.exs"
