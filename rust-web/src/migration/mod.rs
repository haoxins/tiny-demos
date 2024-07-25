use sea_orm::{DbBackend, DbConn, Schema};
use sea_orm_migration::prelude::*;
use tracing::info;

pub async fn setup_schema(db: &DbConn) {
    use crate::account::entity::Entity as AccountEntity;

    let schema = Schema::new(DbBackend::Sqlite);

    let stmt = schema.create_table_from_entity(AccountEntity);

    let result = db
        .execute(db.get_database_backend().build(&stmt))
        .await
        .unwrap();

    info!("setup_schema: {:?}", result);
}
