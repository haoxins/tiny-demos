use dyn_stack::{GlobalPodBuffer, PodStack};
use num_complex::ComplexFloat;
use std::time::Duration;
use tfhe_fft::c64;
use tfhe_fft::ordered::{Method, Plan};

fn main() {
    const N: usize = 4;
    let plan = Plan::new(4, Method::Measure(Duration::from_millis(10)));
    let mut scratch_memory = GlobalPodBuffer::new(plan.fft_scratch().unwrap());
    let mut stack = PodStack::new(&mut scratch_memory);

    let data = [
        c64::new(1.0, 0.0),
        c64::new(2.0, 0.0),
        c64::new(3.0, 0.0),
        c64::new(4.0, 0.0),
    ];

    let mut transformed_fwd = data;
    plan.fwd(&mut transformed_fwd, stack);

    let mut transformed_inv = transformed_fwd;
    plan.inv(&mut transformed_inv, stack);

    for (actual, expected) in transformed_inv.iter().map(|z| z / N as f64).zip(data) {
        assert!((expected - actual).abs() < 1e-9);
    }
}
