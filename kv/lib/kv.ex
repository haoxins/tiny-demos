defmodule KV do
  @moduledoc """
  Documentation for KV.
  """

  @doc """
  """
  use Application
  def start(_type, _args) do
    KV.Supervisor.start_link(name: KV.Supervisor)
  end
end
