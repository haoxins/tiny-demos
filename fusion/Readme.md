
```zsh
cd arrow-ballista
cargo build --release
RUST_LOG=info ./target/release/ballista-scheduler
RUST_LOG=info ./target/release/ballista-executor \
  --concurrent-tasks 4 \
  -p 50051

cargo run --release --bin sql
```
