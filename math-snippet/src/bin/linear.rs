use nalgebra::base::Matrix3;

// use num::Float;

fn main() {
    let t1 = Matrix3::new(1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0);

    let t2 = Matrix3::new(9.0, 8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0);

    // Det
    let det1 = (t1 * t2).determinant();
    let det2 = t1.determinant() * t2.determinant();
    println!("det1: {}", det1);
    println!("det2: {}", det2);

    // Tr
    let tr1 = (t1 * t2).trace();
    let tr2 = (t2 * t1).trace();
    println!("tr1: {}", tr1);
    println!("tr2: {}", tr2);
}
