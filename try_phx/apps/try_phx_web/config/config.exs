use Mix.Config

# General application configuration
config :try_phx_web,
  ecto_repos: [TryPhx.Repo],
  generators: [context_app: :try_phx]

# Configures the endpoint
config :try_phx_web, TryPhxWeb.Endpoint,
  url: [host: "localhost"],
  secret_key_base: "2uX9ZBLkPjBHoKAha0Sm6ksPTRLgZlDfXr5ztFtLuqr7boKozOCPJPt9GkMeH4Qm",
  render_errors: [view: TryPhxWeb.ErrorView, accepts: ~w(json)],
  pubsub: [name: TryPhxWeb.PubSub,
           adapter: Phoenix.PubSub.PG2]

# Import environment specific config. This must remain at the bottom
# of this file so it overrides the configuration defined above.
import_config "#{Mix.env}.exs"
