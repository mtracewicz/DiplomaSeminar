import numpy as np

a = np.array([
    [1, 2],
    [3, 4],
    [5, 6]
])

print(f"A =\n{a}\n")
print(f"Elements at indexes (0,0),(1,1),(2,0)=\n{a[[0, 1, 2],[0, 1, 0]]}\n")
print(f"Elements which are from second or third row =\n{a[1:]}\n")
print(f"Elements which are bigger then 5) =\n{a[a > 3]}\n")

b = np.linspace(1, 10, 9)
b.shape = (3, 3)
print(f"B =\n{a}\n")
print(f"Elements which are from third column =\n{b[:,2:]}\n")

c = np.array([1, 2, np.nan, 4, 5])
print(f"C =\n{c}\n")
print(f"Elements which are not NaN =\n{c[~np.isnan(c)]}")

d = np.array([
    [1, 2, 3, 4],
    [5, 6, 7, 8],
    [1, 2, 3, 4],
    [5, 6, 7, 8]
])

print(f"D =\n{d}\n")
print(f"Upper left corner =\n{d[:2,:2]}\n")
