defmodule S6Web.PageController do
  use S6Web, :controller

  def index(conn, _params) do
    render conn, "index.html"
  end
end
