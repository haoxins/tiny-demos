from typing import List, NewType

def greeting(name: str) -> str:
  return 'hello ' + name

Vector = List[float]

def scale(scalar: float, vector: Vector) -> Vector:
  return [scalar * num for num in vector]

new_vector = scale(2.0, [1.0, -4.2, 5.4])

UserId = NewType('UserId', int)

some_id = UserId(524313)
