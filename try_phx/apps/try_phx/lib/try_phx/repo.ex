defmodule TryPhx.Repo do
  use Ecto.Repo,
    otp_app: :try_phx,
    adapter: Ecto.Adapters.Postgres
end
