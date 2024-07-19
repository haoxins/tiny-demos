use std::{
    collections::HashMap,
    sync::{Arc, RwLock},
};
use uuid::Uuid;

use crate::domain;

pub type Db = Arc<RwLock<HashMap<Uuid, domain::Account>>>;
