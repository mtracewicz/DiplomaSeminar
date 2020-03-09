import numpy as np
a = np.arange(0, 60, 5)
a = a.reshape(3, 4)
print(f'Original array is:\n{a}')
for x in np.nditer(a, op_flags=['readwrite']):
    x[...] = 2*x+1 # ... means in place assigment
print(f'Modified array is:\n{a}')
