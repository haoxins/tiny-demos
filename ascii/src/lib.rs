mod my_ascii {
    #[derive(Debug, Eq, PartialEq)]
    pub struct Ascii(
        Vec<u8>,
    );

    impl Ascii {
        pub fn from_bytes(bytes: Vec<u8>) -> Result<Ascii, NotAsciiError> {
            if bytes.iter().any(|&byte| !byte.is_ascii()) {
                return Err(NotAsciiError(bytes));
            }
            Ok(Ascii(bytes))
        }
    }

    #[derive(Debug, Eq, PartialEq)]
    pub struct NotAsciiError(pub Vec<u8>);

    impl From<Ascii> for String {
        fn from(ascii: Ascii) -> String {
            unsafe { String::from_utf8_unchecked(ascii.0) }
        }
    }

    impl Ascii {
        /// Construct an `Ascii` value from `bytes`, without checking
        /// whether `bytes` actually contains well-formed ASCII.
        ///
        /// This constructor is infallible, and returns an `Ascii` directly,
        /// rather than a `Result<Ascii, NotAsciiError>` as the `from_bytes`
        /// constructor does.
        ///
        /// # Safety
        ///
        /// The caller must ensure that `bytes` contains only ASCII
        /// characters: bytes no greater than 0x7f. Otherwise, the effect is
        /// undefined.
        pub unsafe fn from_bytes_unchecked(bytes: Vec<u8>) -> Ascii {
            Ascii(bytes)
        }
    }
}
#[test]
fn good_ascii() {
    use my_ascii::Ascii;

    let bytes: Vec<u8> = b"ASCII and ye shall receive".to_vec();

    let ascii: Ascii = Ascii::from_bytes(bytes).unwrap();

    let string = String::from(ascii);

    assert_eq!(string, "ASCII and ye shall receive");
}

#[test]
fn bad_ascii() {
    use my_ascii::Ascii;

    let bytes = vec![0xf7, 0xbf, 0xbf, 0xbf];

    let ascii = unsafe { Ascii::from_bytes_unchecked(bytes) };

    let bogus: String = ascii.into();

    assert_eq!(bogus.chars().next().unwrap() as u32, 0x1fffff);
}
