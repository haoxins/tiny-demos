use std::{thread, time};

fn main() {
    for n in 1..=1000 {
        let mut handlers: Vec<thread::JoinHandle<()>> = Vec::with_capacity(n);

        let start = time::Instant::now();

        for m in 0..n {
            let handler = thread::spawn(|| {
                let pause = time::Duration::from_millis(20);
                thread::sleep(pause);
            });
            handlers.push(handler);
        }

        while let Some(handle) = handlers.pop() {
            handle.join();
        }

        let finished = time::Instant::now();
        println!("Finished in {:?}", finished.duration_since(start));
    }
}
