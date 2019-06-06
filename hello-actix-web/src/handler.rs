use actix_web::{HttpRequest, Responder};
use bytes::Bytes;

pub fn greet((req): (HttpRequest)) -> impl Responder {
    let to = req.uri().query().unwrap();
    format!("Hello {}!", to)
}
