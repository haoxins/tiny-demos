defmodule S2Web.PageController do
  use S2Web, :controller

  def index(conn, _params) do
    render conn, "index.html"
  end
end
