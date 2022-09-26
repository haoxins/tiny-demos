fn gcd(mut n: u64, mut m: u64) -> u64 {
    assert!(n != 0 && m != 0);
    while m != 0 {
        if m < n {
            let t = m;
            m = n;
            n = t;
        }
        m = m % n;
    }
    n
}

fn main() {
    println!("{}", gcd(14, 15));
}

#[test]
fn test_gcd() {
    assert_eq!(gcd(7, 13), 1);
    assert_eq!(gcd(2 * 5, 3 * 5), 5);
}
