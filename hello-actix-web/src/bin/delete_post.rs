extern crate diesel;
extern crate hello_actix_web;

use self::diesel::prelude::*;
use self::hello_actix_web::*;
use std::env::args;

fn main() {
    use hello_actix_web::schema::posts::dsl::*;

    let target = args().nth(1).expect("Expected a target to match against");
    let pattern = format!("%{}%", target);

    let connection = establish_connection();
    let num_deleted = diesel::delete(posts.filter(title.like(pattern)))
        .execute(&connection)
        .expect("Error deleting posts");

    println!("Deleted {} posts", num_deleted);
}
