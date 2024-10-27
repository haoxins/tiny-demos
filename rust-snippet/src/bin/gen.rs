// `gen` introduced from edition 2024
#![feature(gen_blocks)]

fn main() {
    let xs = vec![1, 2, 2, 3, 3, 3, 4, 4, 4, 4];
    let ys: Vec<_> = rl_encode(xs).collect();
    println!("{:?}", ys);
}

fn rl_encode<I: IntoIterator<Item = u8>>(xs: I) -> impl Iterator<Item = u8> {
    gen {
        let mut xs = xs.into_iter();
        let (Some(mut cur), mut n) = (xs.next(), 0) else {
            return;
        };
        for x in xs {
            if x == cur && n < u8::MAX {
                n += 1;
            } else {
                yield n;
                yield cur;
                (cur, n) = (x, 0);
            }
        }
        yield n;
        yield cur;
    }
    .into_iter()
}
