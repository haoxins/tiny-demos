use actix_web::{HttpRequest, Responder};
use bytes::Bytes;

pub fn greet((_body, req): (Bytes, HttpRequest)) -> impl Responder {
    let to = req.match_info().get("name").unwrap_or("World");
    format!("Hello {}!", to)
}
