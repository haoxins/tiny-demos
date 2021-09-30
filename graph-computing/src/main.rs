use hmac::{Hmac, Mac, NewMac};
use sha2::Sha256;
// Note that this protocol is not perfect:
// it allows replays.
fn send_message(key: &[u8], message: &[u8]) -> Vec<u8> {
    let mut mac = Hmac::<Sha256>::new(key.into());
    mac.update(message);
    mac.finalize().into_bytes().to_vec()
}

fn receive_message(key: &[u8], message: &[u8], authentication_tag: &[u8]) -> bool {
    let mut mac = Hmac::<Sha256>::new(key.into());
    mac.update(message);
    mac.verify(&authentication_tag).is_ok()
}

fn main() {
    struct Point {
        x: i32,
        y: i32,
    }
    let p = Point { x: 1, y: 2 };
    let r: &Point = &p;
    let rr: &&Point = &r;
    let rrr: &&&Point = &rr;
    assert_eq!(rrr.x, 1);
    assert_eq!(rrr.y, 2);
    let key: [u8; 64] = [1; 64];
    let msg: [u8; 64] = [9; 64];
    let tag = send_message(&key, &msg);
    let ok = receive_message(&key, &msg, &tag);

    let x = 1;
    let y = 1;
    let rx = &x;
    let ry = &y;
    let rrx = &rx;
    let rry = &ry;
    assert!(rrx <= rry);
    assert!(rrx == rry);
    assert!(rrx >= rry);
    assert_eq!(ok, true)
}
