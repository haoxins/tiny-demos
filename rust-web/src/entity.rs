use sea_orm::entity::prelude::*;
use serde::{Deserialize, Serialize};
use uuid::Uuid;

#[derive(Debug, Clone, Serialize, Deserialize)]
// #[sea_orm(table_name = "account")]
pub struct Account {
    // #[sea_orm(primary_key)]
    pub id: Uuid,
    pub name: String,
    pub email: String,
}

#[derive(Copy, Clone, Debug, EnumIter, DeriveRelation)]
pub enum Relation {}

// impl ActiveModelBehavior for ActiveModel {}
