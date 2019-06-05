extern crate actix_web;
use actix_web::{web, App, HttpRequest, HttpServer, Responder};

mod main_handler;
mod post;

fn main() {
    HttpServer::new(|| {
        App::new()
            .service(web::resource("/").route(web::get().to(main_handler::greet)))
            .service(web::resource("/{name}").route(web::get().to(main_handler::greet)))
    })
    .bind("127.0.0.1:8000")
    .expect("Can not bind to port 8000")
    .run();
}
