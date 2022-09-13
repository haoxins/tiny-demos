use std::error;
use std::fmt;
use std::fs::File;
use std::io;
use std::net;
use std::net::Ipv6Addr;

#[derive(Debug)]
enum CustomError {
    IO(io::Error),
    Parsing(net::AddrParseError),
}

impl fmt::Display for CustomError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{:?}", self)
    }
}

impl error::Error for CustomError {}

fn main() -> Result<(), CustomError> {
    let _f = File::open("not-exists.txt").map_err(CustomError::IO)?;

    let _localhost = "::1".parse::<Ipv6Addr>().map_err(CustomError::Parsing)?;

    Ok(())
}
