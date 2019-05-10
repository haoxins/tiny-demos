extern crate tokio;

use tokio::io;
use tokio::net::TcpStream;
use tokio::prelude::*;

pub fn main() -> Result<(), Box<std::error::Error>> {
    let addr = "127.0.0.1:8765".parse()?;

    let client = TcpStream::connect(&addr)
        .and_then(|stream| {
            println!("created stream");
            io::write_all(stream, "hello world\n").then(|result| {
                println!("wrote to stream; success={:?}", result.is_ok());
                Ok(())
            })
        })
        .map_err(|err| {
            println!("connection error = {:?}", err);
        });

    tokio::run(client);
    println!("Stream has been created and written to.");

    Ok(())
}
