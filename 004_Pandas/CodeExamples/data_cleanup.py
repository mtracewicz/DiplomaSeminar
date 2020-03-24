import numpy as np
import pandas as pd

# Preping data
df = np.random.random([1000])
# Rows x columns
df = df.reshape(200, 5)
labels = ['first', 'second', 'third', 'fourth', 'fifth']
df = pd.DataFrame(data=df, columns=labels)
print(df.head())

# Droping third column
df = df.drop('third', axis=1)
print(df.head())

# Droping last 50 rows
df = df.drop(df.index[150:])
print(df)

# For each row if value in first column is bigger then 0.75 set second
# column to 1 and fifth to 0
df.loc[df['first'] > 0.75, ['second', 'fifth']] = 1, 0
print(df)

# Creating new boolean value for each row representing if sum of all
# columns is bigger then 3
df["sum"] = np.where(df.sum(axis=1) > 2.00, 1, 0)
print(df)
