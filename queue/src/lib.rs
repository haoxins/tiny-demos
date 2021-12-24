pub struct Queue {
    older: Vec<char>,
    younger: Vec<char>,
}

impl Queue {
    pub fn push(&mut self, c: char) {
        self.younger.push(c);
    }

    pub fn pop(&mut self) -> Option<char> {
        if self.older.is_empty() {
            if self.younger.is_empty() {
                return None;
            }

            use std::mem::swap;
            swap(&mut self.older, &mut self.younger);
            self.older.reverse();
        }

        self.older.pop()
    }
}

#[test]
fn test_push_pop() {
    let mut q = Queue {
        older: Vec::new(),
        younger: Vec::new(),
    };

    q.push('0');
    q.push('1');
    assert_eq!(q.pop(), Some('0'));

    q.push('∞');
    assert_eq!(q.pop(), Some('1'));
    assert_eq!(q.pop(), Some('∞'));
    assert_eq!(q.pop(), None);

    (&mut q).push('0');
    (&mut q).push('1');
    assert_eq!(q.pop(), Some('0'));
    assert_eq!(q.pop(), Some('1'));
}

impl Queue {
    pub fn is_empty(&self) -> bool {
        self.older.is_empty() && self.younger.is_empty()
    }
}

#[test]
fn test_is_empty() {
    let mut q = Queue {
        older: Vec::new(),
        younger: Vec::new(),
    };

    assert!(q.is_empty());
    q.push('☉');
    assert!(!q.is_empty());
    q.pop();
    assert!(q.is_empty());
}

impl Queue {
    pub fn split(self) -> (Vec<char>, Vec<char>) {
        (self.older, self.younger)
    }
}

#[test]
fn test_split() {
    let mut q = Queue {
        older: Vec::new(),
        younger: Vec::new(),
    };

    q.push('P');
    q.push('D');
    assert_eq!(q.pop(), Some('P'));
    q.push('X');

    let (older, younger) = q.split();
    // q is now uninitialized.
    assert_eq!(older, vec!['D']);
    assert_eq!(younger, vec!['X']);
}

impl Queue {
    pub fn new() -> Queue {
        Queue {
            older: Vec::new(),
            younger: Vec::new(),
        }
    }
}

#[test]
fn test_new() {
    let mut q = Queue::new();

    q.push('*');
    q.push('1');
    assert_eq!(q.pop(), Some('*'));

    q.push('∞');
    assert_eq!(q.pop(), Some('1'));
    assert_eq!(q.pop(), Some('∞'));
    assert_eq!(q.pop(), None);
}
