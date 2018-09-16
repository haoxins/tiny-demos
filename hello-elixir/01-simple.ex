name = "Elixir"
IO.puts("hello #{name}")
IO.puts("hello" <> ", #{name}")

add = fn a, b -> a + b end

IO.puts(add.(1, 3))

IO.puts("--- 2 ---")

# true
IO.puts(1 == 1.0)
# false
IO.puts(1 === 1.0)
# false
IO.puts('false' == "false")
# double-quotes -> binary string, single-quotes -> char list
to_string('123')
to_charlist("123")

l1 = [1, 2] ++ [3, 4]

# IO.puts(l1)

l2 = [false, 1, true, 2, false] -- [false, true]

# IO.puts(l2)
