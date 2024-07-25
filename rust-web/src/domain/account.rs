use axum::{http::StatusCode, Json};
use serde::{Deserialize, Serialize};

use crate::entity::account::Model as Account;

#[derive(Debug, Deserialize)]
pub struct CreatePayload {
    pub name: String,
    pub email: String,
}

#[derive(Serialize)]
pub struct QueryResponse {
    pub message: String,
    pub accounts: Vec<Account>,
}

impl QueryResponse {
    pub fn ok(accounts: Vec<Account>) -> (StatusCode, Json<QueryResponse>) {
        (
            StatusCode::OK,
            Json(QueryResponse {
                message: "success".to_string(),
                accounts: accounts,
            }),
        )
    }
}

#[derive(Serialize)]
pub struct GetResponse {
    pub message: String,
    pub account: Option<Account>,
}

impl GetResponse {
    pub fn not_found() -> (StatusCode, Json<GetResponse>) {
        (
            StatusCode::NOT_FOUND,
            Json(GetResponse {
                message: "account not found".to_string(),
                account: None,
            }),
        )
    }

    pub fn ok(account: Account) -> (StatusCode, Json<GetResponse>) {
        (
            StatusCode::OK,
            Json(GetResponse {
                message: "success".to_string(),
                account: Some(account),
            }),
        )
    }
}
