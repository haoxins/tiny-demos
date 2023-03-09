use axum::{http::StatusCode, response::IntoResponse, routing::post, Json, Router};
use serde::{Deserialize, Serialize};

use std::net::SocketAddr;

#[derive(Deserialize, Debug)]
struct GcdParams {
    n: u64,
    m: u64,
}

#[tokio::main]
async fn main() {
    let app = Router::new().route("/gcd", post(handle_gcd));

    let addr = SocketAddr::from(([127, 0, 0, 1], 3000));
    axum::Server::bind(&addr)
        .serve(app.into_make_service())
        .await
        .unwrap();
}

fn handle_gcd(Json(input): Json<Numbers>) -> Result<impl IntoResponse, StatusCode> {
    println!("The input is {:?}", input);

    let result = gcd(input);

    Ok(Json(result))
}

#[derive(Debug, Deserialize)]
struct Numbers {
    m: u64,
    n: u64,
}

#[derive(Debug, Serialize)]
struct Results {
    n: u64,
}

fn gcd(mut nums: Numbers) -> u64 {
    assert!(nums.n != 0 && nums.m != 0);
    while nums.n != 0 {
        if nums.n < nums.m {
            let t = nums.n;
            nums.n = nums.m;
            nums.m = t;
        }
        nums.n = nums.n % nums.m;
    }
    nums.m
}

#[test]
fn test_gcd() {
    assert_eq!(gcd(Numbers { m: 7, n: 13 }), 1);
    assert_eq!(gcd(Numbers { m: 2 * 5, n: 3 * 5 }), 5);
}
