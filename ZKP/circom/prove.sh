BASENAME=1

node ./generate_witness.js \
  ./${BASENAME}_js/${BASENAME}.wasm \
  ./input.json \
  ./witness.wtns

snarkjs groth16 prove \
  ${BASENAME}_0001.zkey \
  ./witness.wtns \
  ./proof.json \
  ./public.json
