// @TODO:
//   如果两个算符 A 与 B 是可对易的, 那么, A 的所有本征子空间在 B 的作用下都是整体不变的.
use qomo::{Bra3, Ket3};

fn main() {
    let k = Ket3::new(1.0, 0.0, 0.0);
    let b = Bra3::new(1.0, 0.0, 0.0);
    let k2 = Ket3::new(3.0, 2.0, 1.0);
    println!("got x: {:?}", k * b * k2);

    let k = Ket3::new(0.0, 1.0, 0.0);
    let b = Bra3::new(0.0, 1.0, 0.0);
    let k2 = Ket3::new(3.0, 2.0, 1.0);

    println!("got y: {:?}", k * b * k2);

    let k = Ket3::new(0.0, 0.0, 1.0);
    let b = Bra3::new(0.0, 0.0, 1.0);
    let k2 = Ket3::new(3.0, 2.0, 1.0);

    println!("got z: {:?}", k * b * k2);
}
