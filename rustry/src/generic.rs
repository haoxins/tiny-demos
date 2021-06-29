use std::num::{Num, One, Zero};
use std::ops::Add;

pub fn find_min<T: Ord>(data: Vec<T>) -> Option<T> {
  let mut it = data.into_iter();
  let mut min = match it.next() {
    Some(item) => item,
    None => return None,
  };

  for item in it {
    if item < min {
      min = item;
    }
  }

  Some(min)
}

pub trait Counter {
  fn increment(&mut self);
  fn is_zero(&mut self) -> bool;
}

impl<T: Num + Add<T, T>> Counter for T {
  fn increment(&mut self) {
    *self = *self + One::one();
  }

  fn is_zero(&mut self) -> bool {
    *self == Zero::zero()
  }
}
