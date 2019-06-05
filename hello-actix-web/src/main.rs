extern crate actix_web;
use actix_web::{web, App, HttpRequest, HttpServer, Responder};

mod main_handler;
mod post;

fn main() {
    HttpServer::new(|| {
        App::new()
            .service(web::resource("/").route(web::get().to(main_handler::greet)))
            .service(
                web::resource("/posts")
                    .route(web::get().to(post::handler::query))
                    .route(web::post().to(post::handler::create)),
            )
            .service(
                web::resource("/posts/{id}/publish").route(web::patch().to(post::handler::publish)),
            )
    })
    .bind("127.0.0.1:8000")
    .expect("Can not bind to port 8000")
    .run();
}
