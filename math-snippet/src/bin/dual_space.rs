// https://en.wikipedia.org/wiki/Dual_space
use nalgebra::base::{Matrix3, Vector3};

use num::Float;

fn main() {
    let e1 = Vector3::new(1.0, 0.0, 0.0);
    let e2 = Vector3::new(0.0, 2.0, 0.0);
    let e3 = Vector3::new(0.0, 0.0, 3.0);

    println!("e1 = {} e2 = {} e3 = {}", e1, e2, e3);

    let stp1 = e1.dot(&e2.cross(&e3));
    println!("[e1 e2 e3] = e1 * (e2 x e3) = {}", stp1);
    let stp1 = e1.cross(&e2).dot(&e3);
    println!(
        "[e1 e2 e3] = e1 x e2 * e3 = {} (should be equivalent)",
        stp1
    );

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

    println!("g_ij = {} g_ij det = {}", g_ij, g_ij_det);

    // Solution: 1
    let det_s1: f64 = g_ij_det.sqrt();

    let e_1 = e2.cross(&e3) / det_s1;
    let e_2 = e3.cross(&e1) / det_s1;
    let e_3 = e1.cross(&e2) / det_s1;

    println!("The results from solution 1");
    println!("e_1 = {} e_2 = {} e_3 = {}", e_1, e_2, e_3);

    let stp_1 = e_1.dot(&e_2.cross(&e_3));
    println!("[e_1 e_2 e_3] = e_1 * (e_2 x e_3) = {}", stp_1);
    println!(
        "[e1 e2 e3] * [e_1 e_2 e_3] = {} (should be 1)",
        stp1 * stp_1
    );

    // Solution: 2

    let g_ji = g_ij.try_inverse().unwrap();

    let g_ji_det = g_ji.determinant();

    println!("The results from solution 2");
    println!("g_ji = {} g_ji det = {}", g_ji, g_ji_det);
    println!(
        "g_ij det * g_ji det = {} (should be 1)",
        g_ij_det * g_ji_det
    );

    let e_1 = g_ji * e1;
    let e_2 = g_ji * e2;
    let e_3 = g_ji * e3;

    println!("e_1 = {} e_2 = {} e_3 = {}", e_1, e_2, e_3);

    let stp_1 = e_1.dot(&e_2.cross(&e_3));
    println!("[e_1 e_2 e_3] = e_1 * (e_2 x e_3) = {}", stp_1);
    println!(
        "[e1 e2 e3] * [e_1 e_2 e_3] = {} (should be 1)",
        stp1 * stp_1
    );

    // Solution: 3
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

    println!("The results from solution 3");
    println!("g_ji = {} g_ji det = {}", g_ji, g_ji_det);
    println!(
        "g_ij det * g_ji det = {} (should be 1)",
        g_ij_det * g_ji_det
    );

    let e_1 = g_ji * e1;
    let e_2 = g_ji * e2;
    let e_3 = g_ji * e3;

    println!("e_1 = {} e_2 = {} e_3 = {}", e_1, e_2, e_3);

    let stp_1 = e_1.dot(&e_2.cross(&e_3));
    println!("[e_1 e_2 e_3] = e_1 * (e_2 x e_3) = {}", stp_1);
    println!(
        "[e1 e2 e3] * [e_1 e_2 e_3] = {} (should be 1)",
        stp1 * stp_1
    );
}
