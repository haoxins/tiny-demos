use axum::{http::StatusCode, response::IntoResponse, routing::get, Json, Router};
use serde::Serialize;
use tracing::info;
use tracing_subscriber;

#[tokio::main]
async fn main() {
    tracing_subscriber::fmt::init();

    let app = Router::new().route("/", get(handler));

    let listener = tokio::net::TcpListener::bind("127.0.0.1:3000")
        .await
        .unwrap();

    info!("listening on {}", listener.local_addr().unwrap());
    axum::serve(listener, app).await.unwrap();
}

#[derive(Debug, Serialize)]
struct Resp {
    message: String,
}

async fn handler() -> Result<impl IntoResponse, StatusCode> {
    let resp = Resp {
        message: "Hello, World!".to_string(),
    };
    Ok(Json(resp))
}
