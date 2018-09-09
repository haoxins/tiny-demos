x = 1
1 = x

{a, b, c} = {:hello, "world", 18}

IO.puts(a)
IO.puts(b)
IO.puts(c)

{:ok, result} = {:ok, "bingo"}

# { :ok, result } = { :error, "bingo" } # error - (MatchError) no match of right hand side value: {:error, "bingo"}

IO.puts("--- --- ---")

[head | tail] = [1, 2, 3]
IO.puts(head)
IO.puts(tail)

IO.puts("--- --- ---")

list = [1, 2, 3]
IO.puts([0 | list])
