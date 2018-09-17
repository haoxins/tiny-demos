defmodule S1Web.PageController do
  use S1Web, :controller

  def index(conn, _params) do
    render conn, "index.html"
  end
end
