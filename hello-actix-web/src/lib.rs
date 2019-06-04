#[macro_use]
extern crate diesel;
extern crate dotenv;

use diesel::pg::PgConnection;
use diesel::prelude::*;
use dotenv::dotenv;
use std::env;

use self::models::{NewPost, Post};

pub mod models;
pub mod schema;

pub fn establish_connection() -> PgConnection {
    dotenv().ok();

    let database_url = env::var("DATABASE_URL").expect("DATABASE_URL must be set");
    PgConnection::establish(&database_url).expect(&format!("Error connecting to {}", database_url))
}

pub fn create_post<'a>(conn: &PgConnection, title: &'a str, body: &'a str) -> Post {
    use schema::posts;

    let new_post = NewPost {
        title: title,
        body: body,
    };

    diesel::insert_into(posts::table)
        .values(&new_post)
        .get_result(conn)
        .expect("Error saving new post")
}

pub fn query_post<'a>(conn: &PgConnection) -> Vec<Post> {
    use self::schema::posts::dsl::*;

    let results = posts
        .filter(published.eq(true))
        .limit(5)
        .load::<Post>(conn)
        .expect("Error loading posts");

    return results;
}

pub fn publish_post<'a>(conn: &PgConnection, id: i32) -> Post {
    use self::schema::posts::dsl::*;

    let post = diesel::update(posts.find(id))
        .set(published.eq(true))
        .get_result::<Post>(conn)
        .expect("Unable to find post");

    return post;
}
