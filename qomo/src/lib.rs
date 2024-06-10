use std::ops::Mul;

use nalgebra::{
    base::{Matrix, RowVector3, Vector3},
    ArrayStorage, Const, Scalar,
};
use num_traits::{MulAdd, Num, NumAssignOps};

pub struct Ket3<T> {
    pub v: Vector3<T>,
}

impl<T> Ket3<T> {
    pub fn new(x: T, y: T, z: T) -> Self {
        Self {
            v: Vector3::new(x, y, z),
        }
    }
}

impl<T: Scalar + Num + NumAssignOps + MulAdd> Mul<Bra3<T>> for Ket3<T> {
    type Output = Matrix<T, Const<3>, Const<3>, ArrayStorage<T, 3, 3>>;

    fn mul(self, rhs: Bra3<T>) -> Self::Output {
        self.v.kronecker(&rhs.v)
    }
}

pub struct Bra3<T> {
    pub v: RowVector3<T>,
}

impl<T> Bra3<T> {
    pub fn new(x: T, y: T, z: T) -> Self {
        Self {
            v: RowVector3::new(x, y, z),
        }
    }
}

impl<T: Scalar + Num + NumAssignOps + MulAdd> Mul<Ket3<T>> for Bra3<T> {
    type Output = T;

    fn mul(self, rhs: Ket3<T>) -> Self::Output {
        self.v.dot(&rhs.v.transpose())
    }
}

// pub type Ket1<T> = Ket<T, 1>;
// pub type Ket2<T> = Ket<T, 2>;
// pub type Ket3<T> = Ket<T, 3>;
// pub type Ket4<T> = Ket<T, 4>;
// pub type Ket5<T> = Ket<T, 5>;
// pub type Ket6<T> = Ket<T, 6>;

// pub type Bra1<T> = Bra<T, 1>;
// pub type Bra2<T> = Bra<T, 2>;
// pub type Bra3<T> = Bra<T, 3>;
// pub type Bra4<T> = Bra<T, 4>;
// pub type Bra5<T> = Bra<T, 5>;
// pub type Bra6<T> = Bra<T, 6>;

// pub struct Ket<T, const R: usize> {
// }

// pub struct Bra<T, const R: usize> {
// }

#[cfg(test)]
mod tests {
    use nalgebra::base::SquareMatrix;

    use super::*;

    #[test]
    fn bra_ket() {
        let k = Ket3::new(1.0, 2.0, 3.0);
        let b = Bra3::new(3.0, 2.0, 1.0);

        assert_eq!(b * k, 10.0);

        let k = Ket3::new(1.0, 0.0, 0.0);
        let b = Bra3::new(0.0, 2.0, 0.0);
        assert_eq!(
            k * b,
            SquareMatrix::<f64, Const<3>, ArrayStorage<f64, 3, 3>>::new(
                0.0, 2.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
            )
        );
    }
}
