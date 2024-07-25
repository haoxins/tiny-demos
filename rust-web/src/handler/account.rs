use axum::{
    extract::{Path, State},
    response::IntoResponse,
    Json,
};
use sea_orm::DatabaseConnection;
use tracing::info;

use crate::domain::account::*;
use crate::repository::account as repo;

pub async fn query_accounts(State(db): State<DatabaseConnection>) -> impl IntoResponse {
    let accounts = repo::query_accounts(&db).await.unwrap();
    QueryResponse::ok(accounts)
}

pub async fn get_account(
    Path(id): Path<i64>,
    State(db): State<DatabaseConnection>,
) -> impl IntoResponse {
    let account = repo::get_account(&db, id).await;
    let account = match account {
        Some(account) => account,
        None => {
            return GetResponse::not_found();
        }
    };

    GetResponse::ok(account)
}

pub async fn create_account(
    State(db): State<DatabaseConnection>,
    Json(payload): Json<CreatePayload>,
) -> impl IntoResponse {
    info!("creating account: {:?}", payload);
    let account = repo::create_account(&db, payload).await.unwrap();
    GetResponse::ok(account)
}
