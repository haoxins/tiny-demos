defmodule TryPhxWeb.Router do
  use TryPhxWeb, :router

  pipeline :api do
    plug :accepts, ["json"]
  end

  scope "/api", TryPhxWeb do
    pipe_through :api
  end
end
