// https://en.wikipedia.org/wiki/Dual_space
use nalgebra::base::{Matrix3, Vector3};

use num::Float;

fn main() {
    let e1 = Vector3::new(1.0, 0.0, 0.0);
    let e2 = Vector3::new(0.0, 2.0, 0.0);
    let e3 = Vector3::new(0.0, 0.0, 3.0);

    print!("e1 = {}", e1);
    print!("e2 = {}", e2);
    print!("e3 = {}", e3);

    let stp = e1.dot(&e2.cross(&e3));
    println!("[e1 e2 e3] = e1 * (e2 x e3) = {}", stp);
    let stp = e1.cross(&e2).dot(&e3);
    println!("[e1 e2 e3] = e1 x e2 * e3 = {}", stp);

    let g_ij = Matrix3::new(
        e1.dot(&e1),
        e1.dot(&e2),
        e1.dot(&e3),
        e2.dot(&e1),
        e2.dot(&e2),
        e2.dot(&e3),
        e3.dot(&e1),
        e3.dot(&e2),
        e3.dot(&e3),
    );

    let g_ij_det = g_ij.determinant();

    print!("g_ij = {}", g_ij);
    print!("g_ij det = {}\n", g_ij_det);

    let e_1 = e2.cross(&e3) * g_ij_det.sqrt();
    let e_2 = e3.cross(&e1) * g_ij_det.sqrt();
    let e_3 = e1.cross(&e2) * g_ij_det.sqrt();

    print!("e_1 = {}", e_1);
    print!("e_2 = {}", e_2);
    print!("e_3 = {}", e_3);

    let g_ji = Matrix3::new(
        e_1.dot(&e_1),
        e_1.dot(&e_2),
        e_1.dot(&e_3),
        e_2.dot(&e_1),
        e_2.dot(&e_2),
        e_2.dot(&e_3),
        e_3.dot(&e_1),
        e_3.dot(&e_2),
        e_3.dot(&e_3),
    );

    let g_ji_det = g_ji.determinant();

    print!("g_ji = {}", g_ji);
    print!("g_ji det = {}\n", g_ji_det);

    let e_1 = g_ij * e1;
    let e_2 = g_ij * e2;
    let e_3 = g_ij * e3;

    print!("e_1 = {}", e_1);
    print!("e_2 = {}", e_2);
    print!("e_3 = {}", e_3);

    let stp = e_1.dot(&e_2.cross(&e_3));
    println!("[e_1 e_2 e_3] = e_1 * (e_2 x e_3) = {}", stp);
    let stp = e_1.cross(&e_2).dot(&e_3);
    println!("[e_1 e_2 e_3] = e_1 x e_2 * e_3 = {}", stp);
}
