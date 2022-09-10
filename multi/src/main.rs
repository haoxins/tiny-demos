use std::{thread, time};

fn main() {
    let start = time::Instant::now();

    let handler_1 = thread::spawn(move || {
        let pause = time::Duration::from_millis(500);
        thread::sleep(pause.clone());
    });

    let handler_2 = thread::spawn(|| {
        let pause = time::Duration::from_millis(500);
        thread::sleep(pause);
    });

    handler_1.join().unwrap();
    handler_2.join().unwrap();

    let finished = time::Instant::now();

    println!("Finished in {:?}", finished.duration_since(start));
}
