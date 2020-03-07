import numpy as np

a = np.array([
    [1, 2, 3],
    [4, 5, 6]
])

b = np.array([
    [7, 8],
    [9, 10],
    [11, 12]
])

print(f"\n{a} \n*\n {b} \n=\n {a@b}\n")

identity_matrix = np.zeros(4)
identity_matrix.shape = (2, 2)
identity_matrix[0, 0] = 1
identity_matrix[1, 1] = 1
print(f"{identity_matrix} \n*\n {a} \n=\n {identity_matrix@a}")
