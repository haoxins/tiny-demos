use num::Complex;

fn square_add_loop(c: Complex<f64>) {
    let mut z = Complex::new(0.0, 0.0);
    loop {
        z = z * z + c;
    }
}
