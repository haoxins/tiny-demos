#[derive(Debug, PartialEq)]
struct Interval<T> {
    lower: T,
    upper: T,
}

use std::cmp::{Ordering, PartialOrd};

impl<T: PartialOrd> PartialOrd<Interval<T>> for Interval<T> {
    fn partial_cmp(&self, other: &Interval<T>) -> Option<Ordering> {
        if self == other {
            Some(Ordering::Equal)
        } else if self.lower >= other.upper {
            Some(Ordering::Greater)
        } else if self.upper <= other.lower {
            Some(Ordering::Less)
        } else {
            None
        }
    }
}

#[test]
fn test() {
    assert!(Interval { lower: 7, upper: 8 } <= Interval { lower: 7, upper: 8 });
}
