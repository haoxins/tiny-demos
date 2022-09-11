use byteorder::{LittleEndian, ReadBytesExt, WriteBytesExt};
use std::io::Cursor;

fn write_numbers_to_file() -> (u32, i8, f64) {
    let mut w = vec![];

    let one: u32 = 1;
    let two: i8 = 2;
    let three: f64 = 3.0;

    w.write_u32::<LittleEndian>(one).unwrap();
    println!("w: {:?}", &w);

    w.write_i8(two).unwrap();
    println!("w: {:?}", &w);

    w.write_f64::<LittleEndian>(three).unwrap();
    println!("w: {:?}", &w);

    (one, two, three)
}

fn read_numbers_from_file() -> (u32, i8, f64) {
    let mut r = Cursor::new(vec![1, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 8, 64]);
    let one = r.read_u32::<LittleEndian>().unwrap();
    let two = r.read_i8().unwrap();
    let three = r.read_f64::<LittleEndian>().unwrap();

    (one, two, three)
}

fn main() {
    let (one, two, three) = write_numbers_to_file();
    println!("one: {}, two: {}, three: {}", one, two, three);

    let (one_, two_, three_) = read_numbers_from_file();
    println!("one_: {}, two_: {}, three_: {}", one_, two_, three_);

    assert_eq!(one, one_);
    assert_eq!(two, two_);
    assert_eq!(three, three_);
}
