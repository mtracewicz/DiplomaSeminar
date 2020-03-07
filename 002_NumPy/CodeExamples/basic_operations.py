
import numpy as np

a = np.array([[1, 2],
              [3, 4]])

b = np.array([[4, 3],
              [2, 1]])

# Adding 1 to every element
print("Adding 1 to every element:", a + 1)

# Subtracting 2 from each element
print("\nSubtracting 2 from each element:", b - 2)

# sum of array elements
print("\nSum of all array "
      "elements: ", a.sum())

print("\nArray sum:\n", a + b)
print("\nArray multiplication:\n", a * b)
