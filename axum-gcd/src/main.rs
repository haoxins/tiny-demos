use axum::{
    extract::Form,
    response::Html,
    routing::{get, post},
    Router,
};
use serde::Deserialize;

use std::net::SocketAddr;

#[derive(Deserialize, Debug)]
struct GcdParams {
    n: u64,
    m: u64,
}

#[tokio::main]
async fn main() {
    let app = Router::new()
        .route("/", get(index))
        .route("/gcd", post(handle_gcd));

    let addr = SocketAddr::from(([127, 0, 0, 1], 3000));
    axum::Server::bind(&addr)
        .serve(app.into_make_service())
        .await
        .expect("server error");
}

async fn index() -> Html<&'static str> {
    Html(GCD_FORM)
}

async fn handle_gcd(Form(params): Form<GcdParams>) -> Html<String> {
    println!("The request is {:?}", params);
    let n = &gcd(params.n, params.m).to_string();

    let mut html = String::from(GCD_FORM);
    let result = &format!("<p>{}</p>", n);
    html.push_str(result);

    Html(html)
}

fn gcd(mut n: u64, mut m: u64) -> u64 {
    assert!(n != 0 && m != 0);
    while m != 0 {
        if m < n {
            let t = m;
            m = n;
            n = t;
        }
        m = m % n;
    }
    n
}

#[test]
fn test_gcd() {
    assert_eq!(gcd(7, 13), 1);
    assert_eq!(gcd(2 * 5, 3 * 5), 5);
}

const GCD_FORM: &str = r#"
<title>GCD Calculator</title>
<form action="/gcd" method="POST">
    <input type="text" name="n" />
    <input type="text" name="m" />
    <button type="submit">Compute GCD</button>
</form>
"#;
