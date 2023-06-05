use std::env;
use std::fs;
use std::path::Path;
use std::path::PathBuf;

fn get_sql_file_path(name: &str) -> PathBuf {
    let homedir = env::var("GITHUB_DIR").unwrap();
    Path::new(&homedir)
        .join("haoxins/tiny-demos/fusion/sql")
        .join(name)
        .with_extension("sql")
}

pub fn read_sql(name: &str) -> String {
    let path = get_sql_file_path(name);
    fs::read_to_string(path).unwrap().trim().to_string()
}

#[test]
fn test_get_sql_file_path() {
    let path = get_sql_file_path("test");
    assert!(path
        .to_str()
        .unwrap()
        .ends_with("haoxins/tiny-demos/fusion/sql/test.sql"));
}

#[test]
fn test_read_sql() {
    let sql = read_sql("test");
    assert_eq!(sql, "select * from test;");
}
