defmodule S3Web.PageController do
  use S3Web, :controller

  def index(conn, _params) do
    render conn, "index.html"
  end
end
