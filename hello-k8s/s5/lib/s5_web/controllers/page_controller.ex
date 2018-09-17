defmodule S5Web.PageController do
  use S5Web, :controller

  def index(conn, _params) do
    render conn, "index.html"
  end
end
