use chrono::{DateTime, Local};

struct Clock;

impl Clock {
    fn get() -> DateTime<Local> {
        Local::now()
    }

    fn set() -> ! {
        unimplemented!();
        todo!();
    }
}

fn main() {
    let now = Clock::get();
    println!("{}", now);
}
