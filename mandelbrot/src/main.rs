
fn main() {
    let mandelbrot = calculate_mandelbrot(1000, 2.0, 1.0, -1.0, 1.0, 100, 24);
    render_mandelbrot(mandelbrot);
}
