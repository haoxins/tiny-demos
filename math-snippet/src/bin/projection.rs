use nalgebra::base::{RowVector3, Vector3};

fn main() {
    let a = Vector3::new(1.0, 0.0, 0.0);
    let a_t = RowVector3::new(1.0, 0.0, 0.0);

    let v = Vector3::new(3.0, 2.0, 1.0);

    let r1 = a_t.kronecker(&a) * &v;
    println!("r1: {:?}", r1);
    let r2 = a_t * a.dot(&v);
    println!("r2: {:?}", r2);

    let a = Vector3::new(0.0, 1.0, 0.0);
    let a_t = RowVector3::new(0.0, 1.0, 0.0);

    let v = Vector3::new(3.0, 2.0, 1.0);

    let r1 = a_t.kronecker(&a) * &v;
    println!("r1: {:?}", r1);
    let r2 = a_t * a.dot(&v);
    println!("r2: {:?}", r2);
}
