// extern crate futures;
extern crate tokio;

use tokio::io;
use tokio::net::TcpListener;
use tokio::prelude::*;

fn main() {
    let addr = "127.0.0.1:8765".parse().unwrap();
    let listener = TcpListener::bind(&addr).unwrap();

    let server = listener
        .incoming()
        .for_each(|socket| {
            let (reader, writer) = socket.split();
            let amount = io::copy(reader, writer);

            let msg = amount.then(|result| {
                match result {
                    Ok((amount, _, _)) => println!("wrote {} bytes", amount),
                    Err(e) => println!("error: {}", e),
                }

                Ok(())
            });

            // spawn the task that handles the client connection socket on to the
            // tokio runtime. This means each client connection will be handled
            // concurrently
            tokio::spawn(msg);
            Ok(())
        })
        .map_err(|err| {
            // Handle error by printing to STDOUT.
            println!("accept error = {:?}", err);
        });

    println!("server running on localhost:8765");

    tokio::run(server);
}
