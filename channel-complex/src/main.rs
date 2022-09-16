#[macro_use]
extern crate crossbeam;

use crossbeam::channel::unbounded;
use std::thread;

use crate::ConnectivityCheck::*;

#[derive(Debug)]
enum ConnectivityCheck {
    Ping,
    Pong,
    Pang,
}

fn main() {
    let n_messages = 3;
    let (req_tx, req_rx) = unbounded();
    let (resp_tx, resp_rx) = unbounded();

    thread::spawn(move || loop {
        match req_rx.recv().unwrap() {
            Ping => resp_tx.send(Pong).unwrap(),
            Pong => eprintln!("unexpected Pong"),
            Pang => return,
        }
    });

    for _ in 0..n_messages {
        req_tx.send(Ping).unwrap();
    }

    req_tx.send(Pang).unwrap();

    for _ in 0..n_messages {
        select! {
            recv(resp_rx) -> msg => println!("{:?}", msg),
        }
    }
}
