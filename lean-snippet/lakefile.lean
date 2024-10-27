import Lake
open Lake DSL

package "lean-snippet" where
  version := v!"0.1.0"

lean_lib «LeanSnippet» where

@[default_target]
lean_exe "lean-snippet" where
  root := `Main
