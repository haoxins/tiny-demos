import LeanSnippet

def main : IO Unit := do
  let s := "Lean4"
  IO.println s!"Hello, {s}!"
