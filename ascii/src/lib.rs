mod my_ascii {
    #[derive(Debug, Eq, PartialEq)]
    pub struct Ascii(Vec<u8>);

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
        pub unsafe fn from_bytes_unchecked(bytes: Vec<u8>) -> Ascii {
            Ascii(bytes)
        }
    }
}
