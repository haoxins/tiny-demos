use axum::{
    routing::{get, post},
    Router,
};
use sea_orm::Database;

mod account;
mod migration;

use account::handler::*;

const DATABASE_URL: &str = "sqlite::memory:";

#[tokio::main]
async fn main() {
    let db = Database::connect(DATABASE_URL).await.unwrap();
    migration::setup_schema(&db).await;
    println!("schema applied");

    let app = Router::new()
        .route("/accounts", get(query_accounts))
        .route("/accounts/:id", get(get_account))
        .route("/accounts", post(create_account))
        .with_state(db);

    let listener = tokio::net::TcpListener::bind("127.0.0.1:8080")
        .await
        .unwrap();

    println!("listening on {}", listener.local_addr().unwrap());
    axum::serve(listener, app).await.unwrap();
}
