use async_std::io::prelude::*;
use async_std::net;
use std::io::Result;

async fn request(host: &str, port: u16, path: &str) -> Result<String> {
    let mut socket = net::TcpStream::connect((host, port)).await?;

    let request = format!("GET {} HTTP/1.1\r\nHost: {}\r\n\r\n", path, host);
    socket.write_all(request.as_bytes()).await?;
    socket.shutdown(net::Shutdown::Write)?;

    let mut response = String::new();
    socket.read_to_string(&mut response).await?;

    Ok(response)
}

async fn requests(reqs: Vec<(String, u16, String)>) -> Vec<Result<String>> {
    use async_std::task;

    let mut handles = vec![];
    for (host, port, path) in reqs {
        handles.push(task::spawn_local(async move {
            request(&host, port, &path).await
        }));
    }

    let mut results = vec![];
    for handle in handles {
        results.push(handle.await);
    }

    results
}

#[tokio::main]
async fn main() {
    let reqs = vec![
        ("baidu.com".to_string(), 80, "/".to_string()),
        ("douban.com".to_string(), 80, "/".to_string()),
        ("zhihu.com".to_string(), 80, "/".to_string()),
    ];

    let results = requests(reqs).await;
    for result in results {
        match result {
            Ok(response) => println!("{}", response),
            Err(err) => eprintln!("error: {}", err),
        }
    }
}
