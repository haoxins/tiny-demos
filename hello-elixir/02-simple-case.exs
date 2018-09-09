case {1, 2, 3} do
  {1, x, 3} when x > 0 ->
    IO.puts("Will match")

  _ ->
    IO.puts("...")
end

f = fn
  x, y when x > 0 -> x + y
  x, y -> x * y
end

IO.puts(f.(1, 3))
IO.puts(f.(-1, 3))
