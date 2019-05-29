extern crate tokio;

use tokio::io;
use tokio::net::TcpStream;
use tokio::prelude::*;

pub fn main() -> Result<(), Box<std::error::Error>> {
    let addr = "127.0.0.1:8765".parse()?;

    let client = TcpStream::connect(&addr)
        .and_then(|stream| {
            io::write_all(stream, "hello world\n").then(|result| {
                Ok(())
            })
        })
        .map_err(|err| {
        });

    tokio::run(client);

    Ok(())
}
