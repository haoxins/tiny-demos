use gpui::{prelude::*, App, AppContext, Model};

struct Counter {
    count: usize,
}

fn main() {
    App::new().run(|cx: &mut AppContext| {
        let counter: Model<Counter> = cx.new_model(|_cx| Counter { count: 0 });
        // ...
    });
}
