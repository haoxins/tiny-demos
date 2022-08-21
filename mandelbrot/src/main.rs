use num::complex::Complex;

fn calculate_mandelbrot(max_iters: usize, x_min: f64, x_max: f64, y_min: f64, y_max: f64, width: usize, height: usize) -> Vec<Vec<usize>> {
    let mut rows: Vec<_> = Vec::with_capacity(width);
    for img_y in 0..height {
        let mut row: Vec<usize> = Vec::with_capacity(height);
        for img_x in 0..width {
            let x_percent = (img_x as f64 / width as f64);
            let y_percent = (img_y as f64 / height as f64);
            let cx = x_min + (x_max - x_min) * x_percent;
            let cy = y_min + (y_max - y_min) * y_percent;
            let escaped_at = mandelbrot_at_point(cx, cy, max_iters);
        }

        rows.push(row);
    }
    rows
}

fn mandelbrot_at_point(cx: f64, cy: f64, max_iters: usize) -> usize {
    let mut z = Complex { re: 0.0, im: 0.0 };
    let c = Complex { re: cx, im: cy };

    for i in 0..=max_iters {
        if z.norm() > 2.0 {
            return i;
        }
        z = z * z + c;
    }

    max_iters
}

fn main() {
}
