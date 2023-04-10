
```zsh
docker run \
  -p 8000:8000 \
  -p 8001:8001 \
  ghcr.io/arroyosystems/arroyo-single:multi-arch

docker run \
  -p 9093:9093 \
  -e KRAFT_CONTAINER_HOST_NAME=kafka \
  -e KRAFT_CREATE_TOPICS=test,demo \
  -e KRAFT_PARTITIONS_PER_TOPIC=3 \
  moeenz/docker-kafka-kraft
```

```zsh
cd arrow-ballista
cargo build --release
RUST_LOG=info ./target/release/ballista-scheduler
RUST_LOG=info ./target/release/ballista-executor \
  --concurrent-tasks 4 \
  -p 50051

cargo run --release --bin sql
```
