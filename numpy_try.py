import numpy as np

a = np.arange(15)
print(f"Shape: {a.shape}")
print(f"Reshaped: {a.reshape(3,5).shape}")
print(f"Item size: {a.itemsize}")

## Arithmetic operations
a = np.arange(4)
print(f"Array: {a}")

## Multiply by 2
print(f"Array * 2: {a * 2}")

## Raise to the power of 2
print(f"Array ** 2: {a ** 2}")