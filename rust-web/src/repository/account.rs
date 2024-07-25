use sea_orm::error::DbErr;
use sea_orm::*;

use crate::domain::account::CreatePayload;
use crate::entity::account::{ActiveModel, Entity, Model};

pub async fn get_account(db: &DatabaseConnection, id: i64) -> Option<Model> {
    let account = Entity::find_by_id(id).one(db).await.unwrap();

    account
}

pub async fn query_accounts(db: &DatabaseConnection) -> Result<Vec<Model>, DbErr> {
    let accounts = Entity::find().all(db).await?;

    Ok(accounts)
}

pub async fn create_account(
    db: &DatabaseConnection,
    payload: CreatePayload,
) -> Result<Model, DbErr> {
    let account = ActiveModel {
        // id: Uuid::now_v7().to_string(),
        name: Set(payload.name),
        email: Set(payload.email),
        ..Default::default()
    };

    let account = account.insert(db).await?;

    Ok(account)
}
