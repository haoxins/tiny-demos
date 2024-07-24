use axum::{http::StatusCode, Json};
use serde::{Deserialize, Serialize};

use crate::entity::account::Model as Account;

#[derive(Debug, Deserialize)]
pub struct AccountPayload {
    pub name: String,
    pub email: String,
}

#[derive(Serialize)]
pub struct GetAccountResponse {
    pub message: String,
    pub account: Option<Account>,
}

impl GetAccountResponse {
    pub fn not_found() -> (StatusCode, Json<GetAccountResponse>) {
        (
            StatusCode::NOT_FOUND,
            Json(GetAccountResponse {
                message: "account not found".to_string(),
                account: None,
            }),
        )
    }

    pub fn ok(account: Account) -> (StatusCode, Json<GetAccountResponse>) {
        (
            StatusCode::OK,
            Json(GetAccountResponse {
                message: "success".to_string(),
                account: Some(account),
            }),
        )
    }
}
