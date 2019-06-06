use actix_web::{HttpRequest, Responder};
use bytes::Bytes;

use crate::post::service;

pub fn query((_body, req): (Bytes, HttpRequest)) -> impl Responder {
    service::query_post();
    return "OK";
}

pub fn create((_body, req): (Bytes, HttpRequest)) -> impl Responder {
    service::create_post("hi", "cool");
    return "OK";
}

pub fn publish((_body, req): (Bytes, HttpRequest)) -> impl Responder {
    let id = req.match_info().get("id").unwrap();
    let n: i32 = id.parse().unwrap();
    service::publish_post(n);
    return "OK";
}
