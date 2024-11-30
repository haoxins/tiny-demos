BASENAME=1

snarkjs groth16 verify \
  ./${BASENAME}_verification_key.json \
  ./public.json \
  ./proof.json
