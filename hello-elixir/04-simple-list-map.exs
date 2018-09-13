### List: ordering
list = [{:a, 1}, {:b, 2}]
list ++ [c: 3]

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
users = put_in users[:hx].age, 24
