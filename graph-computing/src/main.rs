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
}
