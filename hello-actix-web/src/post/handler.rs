use actix_web::{HttpRequest, Responder};
use bytes::Bytes;

// mod post;
use self::hello_actix_web::*;


pub fn post_handler_query((_body, req): (Bytes, HttpRequest)) -> impl Responder {
  query_post();
}

pub fn post_handler_create((_body, req): (Bytes, HttpRequest)) -> impl Responder {
  create_post("hi", "cool")
}

pub fn post_handler_publish((_body, req): (Bytes, HttpRequest)) -> impl Responder {
    let id = req.match_info().get("id").unwrap();
    publish_post(id);
}
