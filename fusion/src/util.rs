use std::env;
use std::path::Path;
use std::path::PathBuf;

pub fn get_file_path(s: &str) -> PathBuf {
    let homedir = env::var("GITHUB_DIR").unwrap();
    Path::new(&homedir)
        .join("haoxins/tiny-demos/testdata")
        .join(s)
}
