extern crate rand;
use rand::RngCore;
use std::fmt;
use std::fmt::Display;
#[derive(Debug)]
struct MacAddress([u8; 6]);
impl Display for MacAddress {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "{:02x}:{:02x}:{:02x}:{:02x}:{:02x}:{:02x}",
            self.0[0], self.0[1], self.0[2], self.0[3], self.0[4], self.0[5]
        )
    }
}
impl MacAddress {
    fn new() -> Self {
        let mut octets: [u8; 6] = [0; 6];
        rand::thread_rng().fill_bytes(&mut octets);
        octets[0] |= 0b_0000_0011;
    }
    fn is_local(&self) -> bool {
        self.0[0] & 0b_0000_0010 == 0b_0000_0010
    fn is_unicast(&self) -> bool {
        self.0[0] & 0b_0000_0001 == 0b_0000_0001
    let mac = MacAddress::new();
