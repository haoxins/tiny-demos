BASENAME=1

snarkjs powersoftau new bn128 12 pot12_0000.ptau -v
snarkjs powersoftau contribute pot12_0000.ptau pot12_0001.ptau --name="Foo" -v
snarkjs powersoftau prepare phase2 pot12_0001.ptau pot12_final.ptau -v

snarkjs groth16 setup \
  ${BASENAME}.r1cs \
  pot12_final.ptau \
  ${BASENAME}_0000.zkey

snarkjs zkey contribute \
  ${BASENAME}_0000.zkey \
  ${BASENAME}_0001.zkey \
  --name="Foo" -v

snarkjs zkey export verificationkey \
  ${BASENAME}_0001.zkey \
  ${BASENAME}_verification_key.json
