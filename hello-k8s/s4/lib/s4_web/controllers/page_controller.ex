defmodule S4Web.PageController do
  use S4Web, :controller

  def index(conn, _params) do
    render conn, "index.html"
  end
end
