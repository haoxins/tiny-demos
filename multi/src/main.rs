use std::{thread, time};

fn main() {
    for n in 1..=1000 {
        let pause = time::Duration::from_millis(20);

        let mut handlers: Vec<thread::JoinHandle<()>> = Vec::with_capacity(n);

        let start = time::Instant::now();

        for _m in 0..n {
            let handler = thread::spawn(move || {
                let start = time::Instant::now();
                // thread::sleep(pause);
                while start.elapsed() < pause {
                    thread::yield_now();
                }
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
