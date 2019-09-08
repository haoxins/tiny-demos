import numpy as np

a = np.array([1, 2, 3, 4])
b = np.array([2, 3, 4, 5])

print(np.dot(a, b))

a = np.random.randn(5) # Bad
print(a)
print(a.shape)
print(a.T)
print(np.dot(a, a.T))
a = np.random.randn(5, 1) # Good
print(a.T)
