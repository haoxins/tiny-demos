use nalgebra::{
    base::{Matrix2, Matrix3},
    Vector2,
};
use num_complex::Complex64;

fn main() {
    let t1 = Matrix3::new(1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0);

    let t2 = Matrix3::new(9.0, 8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0);

    let vals = t1.eigenvalues().unwrap();
    println!("eigenvalues: {:?}", vals);
    let cvals = t1.complex_eigenvalues();
    println!("complex eigenvalues: {:?}", cvals);

    // Det
    let det1 = (t1 * t2).determinant();
    let det2 = t1.determinant() * t2.determinant();
    println!("Det");
    println!("det1: {det1}");
    println!("det2: {det2}");

    // Tr
    let tr1 = (t1 * t2).trace();
    let tr2 = (t2 * t1).trace();
    println!("Tr");
    println!("tr1: {tr1}");
    println!("tr2: {tr2}");

    // 厄米共轭 (伴随)
    let x = Vector2::new(Complex64::new(1.0, 2.0), Complex64::new(3.0, 4.0));
    let y = Vector2::new(Complex64::new(5.0, 6.0), Complex64::new(7.0, 8.0));
    let m = Matrix2::new(
        Complex64::new(1.0, 1.0),
        Complex64::new(2.0, 2.0),
        Complex64::new(3.0, 3.0),
        Complex64::new(4.0, 4.0),
    );

    let r1 = x.dot(&(m * &y));
    let r2 = (m.adjoint().transpose() * x).dot(&y);

    println!("r1: {r1}");
    println!("r2: {r2}");
}
