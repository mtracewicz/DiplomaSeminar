import numpy as np

a = np.array([[1, 3, -5]])
b = np.array([
    [4],
    [-2],
    [-1]
])

print(f"{a}\n x \n{b} \n=\n {a @ b}")
