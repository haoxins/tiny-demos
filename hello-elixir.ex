# fmt: mix format *.ex
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

_ = [1, 2] ++ [3, 4]

# IO.puts(l1)

_ = [false, 1, true, 2, false] -- [false, true]

# IO.puts(l2)

# case
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

# pattern matching
x = 1
1 = x

{a, b, c} = {:hello, "world", 18}

IO.puts(a)
IO.puts(b)
IO.puts(c)

{:ok, _} = {:ok, "bingo"}

# { :ok, result } = { :error, "bingo" } # error - (MatchError) no match of right hand side value: {:error, "bingo"}

IO.puts("--- --- ---")

[head | tail] = [1, 2, 3]
IO.puts(head)
IO.puts(tail)

IO.puts("--- --- ---")

list = [1, 2, 3]
IO.puts([0 | list])

# function capture

f = &(&1 + 1)
IO.puts(f.(1))

f2 = &List.flatten(&1, &2)
f2.([1, [[2], 3]], [4, 5])

# comprehension
for n <- 1..9, do: _ = n * n

### List: ordering
list = [{:a, 1}, {:b, 2}]
_ = list ++ [c: 3]

IO.puts(list[:a])

# if 1
if false, do: :this, else: :that

# equal to
if(false, do: :this, else: :that)

# equal to
if(false, [{:do, :this}, {:else, :that}])

# match
[a: a, b: b] = [a: 1, b: 3]
IO.puts("match a: #{a}, b: #{b}")

### Map: no ordering
m = %{:a => 1, 2 => :b}
IO.puts("map: #{m[:a]} #{m[2]} #{m[:c]}")

# pattern match: will match as long as possible
%{} = m
%{:a => c} = m
IO.puts(c)
m2 = %{m | :a => "new a"}
IO.puts(m2[:a])

users = [
  hx: %{name: "haoxin", age: 22, languages: ["Javascript", "Kotlin", "Golang"]},
  ms: %{name: "mishao", age: 18, languages: ["a", "b", "c"]}
]

IO.puts(users[:hx].age)
_ = put_in(users[:hx].age, 24)

# process
pid = spawn(fn -> 1 + 2 end)
pid2 = spawn(fn -> 1 + 2 end)
pid3 = spawn(fn -> 1 + 2 end)

IO.puts("#{inspect(self())}")
IO.puts("#{inspect(pid)}")
IO.puts("#{inspect(pid2)}")
IO.puts("#{inspect(pid3)}")

IO.puts(Process.alive?(self()))
IO.puts(Process.alive?(pid))
IO.puts(Process.alive?(pid2))
IO.puts(Process.alive?(pid3))

# send, receive
send(self(), {:hello, "hx"})

receive do
  {:hello, msg} -> IO.puts(msg)
end

receive do
  {:nothing, msg} -> msg
after
  1_100 -> IO.puts("nothing, after 1s")
end

defmodule User do
  defstruct name: "hx", age: 18
end
