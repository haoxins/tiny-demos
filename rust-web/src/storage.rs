use std::{
    collections::HashMap,
    sync::{Arc, RwLock},
};
use uuid::Uuid;

use crate::entity::Account;

pub type Db = Arc<RwLock<HashMap<Uuid, Account>>>;
