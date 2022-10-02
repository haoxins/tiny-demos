use std::env;
use std::str::FromStr;

        numbers.push(u64::from_str(&arg).expect("error parsing argument"));
    for m in &numbers[1..] {
        d = gcd(d, *m);
    }
