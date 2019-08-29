from dataclasses import dataclass

@dataclass
class Card:
  rank: str
  suit: str

q = Card('Q', 'Hearts')

print(q)
