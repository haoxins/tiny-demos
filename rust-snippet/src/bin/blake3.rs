use blake3;
use sha2::{Digest, Sha256};

use std::time::SystemTime;

fn main() {
    let s = b"123456789012345678901234567890";

    // println!("blake3 {}", blake3::hash(s));

    // let mut h = Sha256::new();
    // h.update(s);
    // println!("sha256 {:?}", h.finalize());

    // println!("md5 {:?}", md5::compute(s));

    let start = SystemTime::now();
    for _ in 0..10_000_000 {
        let mut h = Sha256::new();
        h.update(s);
        h.finalize();
    }
    println!("sha2 (256) times {:?}", start.elapsed().unwrap());

    let start = SystemTime::now();
    for _ in 0..10_000_000 {
        md5::compute(s);
    }
    println!("md5 times {:?}", start.elapsed().unwrap());

    let start = SystemTime::now();
    for _ in 0..10_000_000 {
        blake3::hash(s);
    }
    println!("blake3 times {:?}", start.elapsed().unwrap());
}
