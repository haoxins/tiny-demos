use axum::{
    routing::{get, post},
    Router,
};
use sea_orm::Database;
use tracing::info;
use tracing_subscriber;

mod domain;
mod entity;
mod handler;
mod storage;

const DATABASE_URL: &str = "sqlite::memory:";

#[tokio::main]
async fn main() {
    tracing_subscriber::fmt::init();

    let db = Database::connect(DATABASE_URL).await.unwrap();
    entity::setup_schema(&db).await;
    info!("schema applied");

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
