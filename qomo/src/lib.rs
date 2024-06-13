use std::fmt::{self, Debug};
use std::ops::Mul;
use std::usize;

use nalgebra::{base::Matrix, ArrayStorage, Const, Scalar};
use num_traits::{MulAdd, Num, NumAssignOps};

pub type Ket2<T> = Ket<T, 2>;
pub type Ket3<T> = Ket<T, 3>;
pub type Ket4<T> = Ket<T, 4>;

pub type Bra2<T> = Bra<T, 2>;
pub type Bra3<T> = Bra<T, 3>;
pub type Bra4<T> = Bra<T, 4>;

#[derive(Clone, Copy)]
pub struct Ket<T, const R: usize> {
    pub v: Matrix<T, Const<R>, Const<1>, ArrayStorage<T, R, 1>>,
}

macro_rules! ket_impl(
    ($($R: expr, [$($num: ident),*] $(;)*)*) => {$(
        impl<T> Ket<T, $R> {
            // #[inline]
            pub const fn new($($num: T),*) -> Self {
                Self {
                    v: Matrix::<T, Const<$R>, Const<1>, ArrayStorage<T, $R, 1>>::new($($num),*)
                }
            }
        }

        impl<T: Scalar> Debug for Ket<T, $R> {
            fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
                write!(f, "{:?}", self.v.data.as_slice())
            }
        }

        impl<T: Scalar + Num + NumAssignOps + MulAdd> Mul<Bra<T, $R>> for Ket<T, $R> {
            type Output = Matrix<T, Const<$R>, Const<$R>, ArrayStorage<T, $R, $R>>;

            fn mul(self, rhs: Bra<T, $R>) -> Self::Output {
                self.v.kronecker(&rhs.v)
            }
        }

        impl<T: Scalar + Num + NumAssignOps + MulAdd> Mul<Ket<T, $R>>
            for Matrix<T, Const<$R>, Const<$R>, ArrayStorage<T, $R, $R>> {
            type Output = Ket<T, $R>;

            fn mul(self, rhs: Ket<T, $R>) -> Self::Output {
                Ket { v: self * rhs.v }
            }
        }

        // // https://stackoverflow.com/questions/63119000/why-am-i-required-to-cover-t-in-impl-foreigntraitlocaltype-for-t-e0210
        impl<T: Scalar + Num + NumAssignOps + MulAdd> Mul<Ket<T, $R>> for (T,) {
            type Output = Ket<T, $R>;

            fn mul(self, rhs: Ket<T, $R>) -> Self::Output {
                Ket { v: rhs.v * self.0 }
            }
        }

        impl<T: Scalar + Num + NumAssignOps + MulAdd> Mul<T> for Ket<T, $R> {
            type Output = Ket<T, $R>;

            fn mul(self, rhs: T) -> Self::Output {
                Ket { v: self.v * rhs }
            }
        }
    )*}
);

ket_impl!(
    2, [a, b];
    3, [a, b, c];
    4, [a, b, c, d];
);

#[derive(Clone, Copy)]
pub struct Bra<T, const R: usize> {
    pub v: Matrix<T, Const<1>, Const<R>, ArrayStorage<T, 1, R>>,
}

macro_rules! bra_impl(
    ($($R: expr, [$($num: ident),*] $(;)*)*) => {$(
        impl<T> Bra<T, $R> {
            pub fn new($($num: T),*) -> Self {
                Self {
                    v: Matrix::<T, Const<1>, Const<$R>, ArrayStorage<T, 1, $R>>::new($($num),*),
                }
            }
        }

        impl<T: Scalar> Debug for Bra<T, $R> {
            fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
                write!(f, "{:?}", self.v.data.as_slice())
            }
        }

        impl<T: Scalar + Num + NumAssignOps + MulAdd> Mul<Ket<T, $R>> for Bra<T, $R> {
            type Output = T;

            fn mul(self, rhs: Ket<T, $R>) -> Self::Output {
                self.v.dot(&rhs.v.transpose())
            }
        }
    )*}
);

bra_impl!(
    2, [a, b];
    3, [a, b, c];
    4, [a, b, c, d];
);

#[cfg(test)]
mod tests {
    use nalgebra::base::{SquareMatrix, Vector3};

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

        let k = Ket3::new(1.0, 0.0, 0.0);
        let b = Bra3::new(0.0, 2.0, 0.0);
        let k2 = Ket3::new(0.0, 3.0, 0.0);
        let r = k * b * k2;
        assert_eq!(r.v, Vector3::new(6.0, 0.0, 0.0));

        let r = (2.0,) * k;
        assert_eq!(r.v, Vector3::new(2.0, 0.0, 0.0));

        let r = k * 2.0;
        assert_eq!(r.v, Vector3::new(2.0, 0.0, 0.0));
    }
}
