use axum::{
    extract::{Path, State},
    response::IntoResponse,
    Json,
};
use uuid::Uuid;

use crate::domain;
use crate::storage::Db;

pub async fn query_accounts(State(db): State<Db>) -> impl IntoResponse {
    let accounts = db.read().unwrap();
    Json(accounts.clone())
}

pub async fn get_account(Path(id): Path<Uuid>, State(db): State<Db>) -> impl IntoResponse {
    let accounts = db.read().unwrap();
    let account = accounts.get(&id).unwrap();
    Json(account.clone())
}

pub async fn create_account(
    State(db): State<Db>,
    Json(payload): Json<domain::AccountPayload>,
) -> impl IntoResponse {
    // info!("creating account: {:?}", payload);
    let account = domain::Account {
        id: Uuid::now_v7(),
        name: payload.name,
        email: payload.email,
    };
    db.write().unwrap().insert(account.id, account.clone());
    Json(account)
}
