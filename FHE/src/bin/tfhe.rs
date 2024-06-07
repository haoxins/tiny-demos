use tfhe::prelude::*;
use tfhe::{generate_keys, set_server_key, ConfigBuilder, FheUint32, FheUint8};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let config = ConfigBuilder::default().build();

    let (client_key, server_keys) = generate_keys(config);

    let clear_a = 1344u32;
    let clear_b = 5u32;
    let clear_c = 7u8;

    let mut encrypted_a = FheUint32::try_encrypt(clear_a, &client_key)?;
    let encrypted_b = FheUint32::try_encrypt(clear_b, &client_key)?;

    let encrypted_c = FheUint8::try_encrypt(clear_c, &client_key)?;

    set_server_key(server_keys);

    let encrypted_res_mul = &encrypted_a * &encrypted_b;

    encrypted_a = &encrypted_res_mul >> &encrypted_b;

    let casted_a: FheUint8 = encrypted_a.cast_into();

    let encrypted_res_min = &casted_a.min(&encrypted_c);

    let encrypted_res = encrypted_res_min & 1_u8;

    let clear_res: u8 = encrypted_res.decrypt(&client_key);
    assert_eq!(clear_res, 1_u8);

    Ok(())
}
