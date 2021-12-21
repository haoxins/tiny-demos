use std::error::Error;
use std::io;

async fn http_get(url: &str) -> Result<(), Box<dyn Error>> {
    let mut resp = reqwest::get(url).await?;
    if !resp.status().is_success() {
        Err(format!("{}", resp.status()))?;
    }

    let stdout = io::stdout();
    io::copy(&mut resp, &mut stdout.lock())?;

    Ok(())
}

#[tokio::main]
async fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() != 2 {
        eprintln!("usage: http-get URL");
        return;
    }

    let url = &args[1];
    println!("The request URL is {}", url);

    if let Err(err) = http_get(url) {
        eprintln!("error: {}", err);
    }
}
