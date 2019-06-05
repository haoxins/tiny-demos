use actix_web::{HttpRequest, Responder};
use bytes::Bytes;

// mod post;


  query_post();
pub fn query((_body, req): (Bytes, HttpRequest)) -> impl Responder {
}

  create_post("hi", "cool")
pub fn create((_body, req): (Bytes, HttpRequest)) -> impl Responder {
}

pub fn publish((_body, req): (Bytes, HttpRequest)) -> impl Responder {
    let id = req.match_info().get("id").unwrap();
    publish_post(id);
}
