pub struct Queue<T> {
    older: Vec<T>,
    younger: Vec<T>,
}

impl<T> Queue<T> {
    pub fn new() -> Self {
        Queue {
            older: Vec::new(),
            younger: Vec::new(),
        }
    }

    pub fn push(&mut self, t: T) {
        self.younger.push(t);
    }

    pub fn is_empty(&self) -> bool {
        self.older.is_empty() && self.younger.is_empty()
    }

    pub fn pop(&mut self) -> Option<T> {
        if self.older.is_empty() {
            use std::mem::swap;

            if self.younger.is_empty() {
                return None;
            }

            swap(&mut self.older, &mut self.younger);
            self.older.reverse();
        }

        self.older.pop()
    }
}

#[test]
fn test() {
    let mut q = Queue::new();

    q.push('*');
    assert_eq!(q.pop(), Some('*'));
    assert_eq!(q.pop(), None);

    q.push('0');
    q.push('1');
    assert_eq!(q.pop(), Some('0'));

    q.push('∞');
    assert_eq!(q.pop(), Some('1'));
    assert_eq!(q.pop(), Some('∞'));
    assert_eq!(q.pop(), None);

    assert!(q.is_empty());
    q.push('☉');
    assert!(!q.is_empty());
    q.pop();
    assert!(q.is_empty());

    let mut q = Queue::new();

    q.push('P');
    q.push('D');
    assert_eq!(q.pop(), Some('P'));
    q.push('X');
}

#[test]
fn test_generic() {
    let q = Queue::<char>::new();
    drop(q);

    let mut q = Queue::new();
    let mut r = Queue::new();

    q.push("CAD"); // apparently a Queue<&'static str>
    r.push(0.74); // apparently a Queue<f64>

    q.push("BTC");
    r.push(13764.0);
}
