BASENAME=1

echo '{"a": "3","b": "11"}' > input.json

circom ./${BASENAME}.circom --r1cs --wasm
