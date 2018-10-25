defmodule TryPhx.Application do
  @moduledoc false

  use Application

  def start(_type, _args) do
    Supervisor.start_link([
      TryPhx.Repo,
    ], strategy: :one_for_one, name: TryPhx.Supervisor)
  end
end
