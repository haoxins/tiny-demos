mod very_generic {

    #[derive(Clone, Copy, Debug)]
    struct Complex<T> {
        re: T,
        im: T,
    }

    use std::ops::Add;

    impl<L, R> Add<Complex<R>> for Complex<L>
    where
        L: Add<R>,
    {
        type Output = Complex<L::Output>;
        fn add(self, rhs: Complex<R>) -> Self::Output {
            Complex {
                re: self.re + rhs.re,
                im: self.im + rhs.im,
            }
        }
    }
}

mod impl_compound {

    #[derive(Clone, Copy, Debug)]
    struct Complex<T> {
        re: T,
        im: T,
    }

    use std::ops::AddAssign;

    impl<T> AddAssign for Complex<T>
    where
        T: AddAssign<T>,
    {
        fn add_assign(&mut self, rhs: Complex<T>) {
            self.re += rhs.re;
            self.im += rhs.im;
        }
    }
}

mod derive_partialeq {
    #[derive(Clone, Copy, Debug, PartialEq)]
    struct Complex<T> {
        re: T,
        im: T,
    }
}

mod derive_everything {
    #[derive(Clone, Copy, Debug, Eq, PartialEq)]
    struct Complex<T> {
        /// Real portion of the complex number
        re: T,

        /// Imaginary portion of the complex number
        im: T,
    }
}

mod formatting {
    #[test]
    fn complex() {
        #[derive(Copy, Clone, Debug)]
        struct Complex {
            re: f64,
            im: f64,
        }

        let third = Complex {
            re: -0.5,
            im: f64::sqrt(0.75),
        };
        println!("{:?}", third);

        use std::fmt;

        impl fmt::Display for Complex {
            fn fmt(&self, dest: &mut fmt::Formatter) -> fmt::Result {
                let im_sign = if self.im < 0.0 { '-' } else { '+' };
                write!(dest, "{} {} {}i", self.re, im_sign, f64::abs(self.im))
            }
        }

        let one_twenty = Complex {
            re: -0.5,
            im: 0.866,
        };
        assert_eq!(format!("{}", one_twenty), "-0.5 + 0.866i");

        let two_forty = Complex {
            re: -0.5,
            im: -0.866,
        };
        assert_eq!(format!("{}", two_forty), "-0.5 - 0.866i");
    }

    #[test]
    fn complex_fancy() {
        #[derive(Copy, Clone, Debug)]
        struct Complex {
            re: f64,
            im: f64,
        }

        use std::fmt;

        impl fmt::Display for Complex {
            fn fmt(&self, dest: &mut fmt::Formatter) -> fmt::Result {
                let (re, im) = (self.re, self.im);
                if dest.alternate() {
                    let abs = f64::sqrt(re * re + im * im);
                    let angle = f64::atan2(im, re) / std::f64::consts::PI * 180.0;
                    write!(dest, "{} ∠ {}°", abs, angle)
                } else {
                    let im_sign = if im < 0.0 { '-' } else { '+' };
                    write!(dest, "{} {} {}i", re, im_sign, f64::abs(im))
                }
            }
        }

        let ninety = Complex { re: 0.0, im: 2.0 };
        assert_eq!(format!("{}", ninety), "0 + 2i");
        assert_eq!(format!("{:#}", ninety), "2 ∠ 90°");
    }
}
