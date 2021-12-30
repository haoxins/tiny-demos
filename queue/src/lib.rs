pub struct Queue {
    older: Vec<char>,
    younger: Vec<char>,
}

impl Queue {
    pub fn push(&mut self, c: char) {
        self.younger.push(c);
    }

    pub fn pop(&mut self) -> Option<char> {
        if !self.older.is_empty() {
            return self.older.pop();
        }

        if self.younger.is_empty() {
            return None;
        }

        use std::mem::swap;
        println! {"Before swap: {:?}", self.younger}
        println! {"Before swap: {:?}", self.younger}

        swap(&mut self.older, &mut self.younger);
        println! {"After swap: {:?}", self.younger}
        println! {"After swap: {:?}", self.older}

        self.older.reverse();
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
