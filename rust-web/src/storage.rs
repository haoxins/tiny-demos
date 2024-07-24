use std::{
    collections::HashMap,
    sync::{Arc, RwLock},
};
use uuid::Uuid;

use crate::entity::account::Model as Account;

pub type Db = Arc<RwLock<HashMap<Uuid, Account>>>;
