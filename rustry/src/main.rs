mod generic;

use generic::find_min;
use generic::Counter;

fn main() {
  let a = find_min(vec![1u32, 2, 3, 4]);
  let b = find_min(vec![10i16, 20, 30, 40]);
  println!("{:?} {:?}", a, b);
  //
  let mut data: Vec<Box<dyn Counter>> = Vec::new();
  data.push(Box::new(1u32));
  data.push(Box::new(0u16));
  data.push(Box::new(0i8));

  for x in data.iter_mut() {
    println!("{}", x.is_zero());
  }
  for x in data.iter_mut() {
    x.increment();
  }
  for x in data.iter_mut() {
    println!("{}", x.is_zero());
  }
}
