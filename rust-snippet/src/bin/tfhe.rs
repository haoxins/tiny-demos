use tfhe::prelude::*;
use tfhe::{ConfigBuilder, FheUint8, FheUint32, generate_keys, set_server_key};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let config = ConfigBuilder::default().build();

    let (client_key, server_keys) = generate_keys(config);

    let clear_a = 1344u32;
    let clear_b = 5u32;
    let clear_c = 7u8;

    // FheUint32: Encrypted equivalent to u32
    let mut encrypted_a = FheUint32::try_encrypt(clear_a, &client_key)?;
    let encrypted_b = FheUint32::try_encrypt(clear_b, &client_key)?;

    // FheUint8: Encrypted equivalent to u8
    let encrypted_c = FheUint8::try_encrypt(clear_c, &client_key)?;

    // On the server side:
    set_server_key(server_keys);

    // Clear equivalent computations: 1344 * 5 = 6720
    let encrypted_res_mul = &encrypted_a * &encrypted_b;

    // Clear equivalent computations: 6720 >> 5 = 210
    encrypted_a = &encrypted_res_mul >> &encrypted_b;

    // Clear equivalent computations: let casted_a = a as u8;
    let casted_a: FheUint8 = encrypted_a.cast_into();

    // Clear equivalent computations: min(210, 7) = 7
    let encrypted_res_min = &casted_a.min(&encrypted_c);

    // Operation between clear and encrypted data:
    // Clear equivalent computations: 7 & 1 = 1
    let encrypted_res = encrypted_res_min & 1_u8;

    // Decrypting on the client side:
    let clear_res: u8 = encrypted_res.decrypt(&client_key);
    assert_eq!(clear_res, 1_u8);

    Ok(())
}
