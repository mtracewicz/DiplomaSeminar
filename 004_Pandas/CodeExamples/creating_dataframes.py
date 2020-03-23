import numpy as np
import pandas as pd

#seeding random numer generator
np.random.seed(0)
#creating array (size of 12)  of random numers from [0,100]
arr = np.random.randint(101, size = 12)
#reshaping the array to have 4 columns and 3 rows
arr = arr.reshape(3,4)
#prepering column names
my_column_names = ['Eleanor', 'Chidi','Tahani' ,'Jason']
#creating dataframe from array and column names
my_dataframe = pd.DataFrame(data=arr, columns=my_column_names)
#printing whole dataframe
print(my_dataframe)
#printing cell, column "Eleanor", row 1
print(my_dataframe["Eleanor"][1])
#creating new colum "Janet" which cell in each row is sum of cells from colums "Jason" and "Tahani" in the same row
my_dataframe["Jannet"] =  my_dataframe["Tahani"] + my_dataframe["Jason"] 
print(my_dataframe)
