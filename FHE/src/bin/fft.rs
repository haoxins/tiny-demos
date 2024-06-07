use concrete_fft::c64;
use concrete_fft::ordered::{Method, Plan};
use dyn_stack::{GlobalPodBuffer, PodStack, ReborrowMut};
use num_complex::ComplexFloat;
use std::time::Duration;

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
    plan.fwd(&mut transformed_fwd, stack.rb_mut());

    let mut transformed_inv = transformed_fwd;
    plan.inv(&mut transformed_inv, stack.rb_mut());

    for (actual, expected) in transformed_inv.iter().map(|z| z / N as f64).zip(data) {
        assert!((expected - actual).abs() < 1e-9);
    }
}
