use std::rc::Rc;
use std::sync::{Arc, Mutex};

fn main() {
    let c = Rc::new(Box::new(30));
    let d = Arc::new(Mutex::new(40));

    let s1 = r"Hello!";
    let s2 = r#"Hello""#;
    let s3 = r#"Hello#"#;
}
