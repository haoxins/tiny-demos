fn main() {
    let x = 1;
    let y = 1;
    let rx = &x;
    let ry = &y;
    let rrx = &rx;
    let rry = &ry;
    assert!(rrx <= rry);
    assert!(rrx == rry);
    assert!(rrx >= rry);
}
