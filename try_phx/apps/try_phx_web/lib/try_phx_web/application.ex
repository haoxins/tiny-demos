defmodule TryPhxWeb.Application do
  @moduledoc false

  use Application

  def start(_type, _args) do
    children = [
      TryPhxWeb.Endpoint,
    ]

    opts = [strategy: :one_for_one, name: TryPhxWeb.Supervisor]
    Supervisor.start_link(children, opts)
  end

  def config_change(changed, _new, removed) do
    TryPhxWeb.Endpoint.config_change(changed, removed)
    :ok
  end
end
