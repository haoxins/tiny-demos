use axum::{
    routing::{get, post},
    Router,
};
use sea_orm::Database;
// use tracing::info;
use tracing_subscriber;

mod domain;
mod entity;
mod handler;
mod repository;

const DATABASE_URL: &str = "sqlite::memory:";

#[tokio::main]
async fn main() {
    tracing_subscriber::fmt::init();

    let db = Database::connect(DATABASE_URL).await.unwrap();
    entity::setup_schema(&db).await;
    println!("schema applied");

    let app = Router::new()
        .route("/accounts", get(handler::account::query_accounts))
        .route("/accounts/:id", get(handler::account::get_account))
        .route("/accounts", post(handler::account::create_account))
        .with_state(db);

    let listener = tokio::net::TcpListener::bind("127.0.0.1:8080")
        .await
        .unwrap();

    println!("listening on {}", listener.local_addr().unwrap());
    axum::serve(listener, app).await.unwrap();
}
