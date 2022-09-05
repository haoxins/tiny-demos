use bincode::serialize as to_bincode;
use serde_cbor::to_vec as to_cbor;
use serde_derive::Serialize;
use serde_json::to_string as to_json;

#[derive(Serialize)]
struct City {
    name: String,
    population: usize,
    latitude: f64,
    longitude: f64,
}

fn main() {
    let calabar = City {
        name: String::from("Calabar"),
        population: 470_000,
        latitude: 4.95,
        longitude: 8.33,
    };

    let as_json = to_json(&calabar).unwrap();
    let as_cbor = to_cbor(&calabar).unwrap();
    let as_bincode = to_bincode(&calabar).unwrap();

    println!("JSON:");
    println!("{}", &as_json);
    println!("CBOR:");
    println!("{:?}", &as_cbor);
    println!("Bincode:");
    println!("{:?}", &as_bincode);
    println!("JSON as utf8:");
    println!("{}", String::from_utf8_lossy(as_json.as_bytes()));
    println!("CBOR as utf8:");
    println!("{:?}", String::from_utf8_lossy(&as_cbor));
    println!("Bincode as utf8:");
    println!("{:?}", String::from_utf8_lossy(&as_bincode));
}
