extern crate actix_web;
use actix_web::{web, App, HttpRequest, HttpServer, Responder};
use bytes::Bytes;

fn greet((_body, req): (Bytes, HttpRequest)) -> impl Responder {
    let to = req.match_info().get("name").unwrap_or("World");
    format!("Hello {}!", to)
}

fn main() {
    HttpServer::new(|| {
        App::new()
            .service(web::resource("/").route(web::get().to(greet)))
            .service(web::resource("/{name}").route(web::get().to(greet)))
    })
    .bind("127.0.0.1:8000")
    .expect("Can not bind to port 8000")
    .run();
}
