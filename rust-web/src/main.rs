use axum::{
    routing::{get, post},
    Router,
};
use tracing::info;
use tracing_subscriber;

mod domain;
mod entity;
mod handler;
mod storage;

#[tokio::main]
async fn main() {
    tracing_subscriber::fmt::init();

    let db = storage::Db::default();

    let app = Router::new()
        .route("/accounts", get(handler::query_accounts))
        .route("/accounts/:id", get(handler::get_account))
        .route("/accounts", post(handler::create_account))
        .with_state(db);

    let listener = tokio::net::TcpListener::bind("127.0.0.1:8080")
        .await
        .unwrap();

    info!("listening on {}", listener.local_addr().unwrap());
    axum::serve(listener, app).await.unwrap();
}
