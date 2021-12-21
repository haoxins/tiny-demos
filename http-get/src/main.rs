use std::error::Error;

async fn http_get(url: &str) -> Result<(), Box<dyn Error>> {
    let resp = reqwest::get(url).await?;
    if !resp.status().is_success() {
        Err(format!("{}", resp.status()))?;
    }

    let text = resp.text().await?;
    println!("body: {}", text);

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

    if let Err(err) = http_get(url).await {
        eprintln!("error: {}", err);
    }
}
