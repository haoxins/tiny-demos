#![warn(rust_2018_idioms)]
#![allow(elided_lifetimes_in_paths)]

use std::error::Error;
use std::io;

fn http_get_main(url: &str) -> Result<(), Box<dyn Error>> {
    let mut resp = reqwest::blocking::get(url)?;
    if !resp.status().is_success() {
        Err(format!("{}", resp.status()))?;
    }

    let stdout = io::stdout();
    io::copy(&mut resp, &mut stdout.lock())?;

    Ok(())
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() != 2 {
        eprintln!("usage: http-get URL");
        return;
    }

    let url = &args[1];
    println!("The request URL is {}", url);

    if let Err(err) = http_get_main(url) {
        eprintln!("error: {}", err);
    }
}
