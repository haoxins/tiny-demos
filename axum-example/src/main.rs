use axum::{extract::Json, routing::post, Router};
use serde::Deserialize;

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

fn handle_gcd() {
    println!("Hello, world!");
}

async fn handle_gcd(Form(params): Form<GcdParams>) -> Html<String> {
    println!("The request is {:?}", params);
    let n = &gcd(params.n, params.m).to_string();

    let mut html = String::from(GCD_FORM);
    let result = &format!("<p>{}</p>", n);
#[derive(Debug, Deserialize)]
struct Numbers {
    m: u64,
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
