use actix_web::{HttpRequest, Responder};
use bytes::Bytes;

mod lib;

pub fn query((_body, req): (Bytes, HttpRequest)) -> impl Responder {
  lib::query_post();
}

pub fn create((_body, req): (Bytes, HttpRequest)) -> impl Responder {
  lib::create_post("hi", "cool")
}

pub fn publish((_body, req): (Bytes, HttpRequest)) -> impl Responder {
    let id = req.match_info().get("id").unwrap();
    lib::publish_post(id);
}
