use diesel::pg::PgConnection;
use diesel::prelude::*;
use dotenv::dotenv;
use std::env;

use crate::post::models::{NewPost, Post};
use crate::post::schema::posts;
use crate::post::schema::posts::dsl::*;

pub fn establish_connection() -> PgConnection {
    dotenv().ok();

    let database_url = env::var("DATABASE_URL").expect("DATABASE_URL must be set");
    PgConnection::establish(&database_url).expect(&format!("Error connecting to {}", database_url))
}

pub fn create_post<'a>(title: &'a str, body: &'a str) -> Post {
    let conn = establish_connection();

    let new_post = NewPost {
        title: title,
        body: body,
    };

    diesel::insert_into(posts::table)
        .values(&new_post)
        .get_result(&conn)
        .expect("Error saving new post")
}

pub fn query_post<'a>() -> Vec<Post> {
    let conn = establish_connection();

    let results = posts
        .filter(published.eq(true))
        .limit(5)
        .load::<Post>(&conn)
        .expect("Error loading posts");

    return results;
}

pub fn publish_post<'a>(id: i32) -> Post {
    let conn = establish_connection();

    let post = diesel::update(posts.find(id))
        .set(published.eq(true))
        .get_result::<Post>(&conn)
        .expect("Unable to find post");

    return post;
}
