
```
docker volume create postgres-volume

docker run --name arroyo-postgres \
  --env POSTGRES_USER=arroyo \
  --env POSTGRES_PASSWORD=arroyo \
  --env POSTGRES_DB=arroyo \
  --volume postgres-volume:/var/lib/postgresql/data \
  --publish 5432:5432 \
  --detach postgres
```
