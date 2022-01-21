# reads smiley.txt and converts the  

import csv
import numpy as np

pattern = []

with open('smiley.txt', newline='\n') as pattern_file:
    pattern_data = csv.reader(pattern_file, delimiter=' ')
    for row in pattern_data:
        pattern.append([int(numeric_string) for numeric_string in row])

print("\nFile read as a list")
print(pattern)
        
row_num = pattern_data.line_num

rows = int(np.array(pattern[0]))
cols = int(np.array(pattern[1]))
dim_pattern = rows * cols 

pattern = np.ndarray((1,dim_pattern),dtype=int,buffer=np.array(pattern[2:]))

pattern=pattern[0];

print("\nConverted pattern")
print (pattern)

# Printing type of arr object
print("Array is of type: ", type(pattern))
 
# Printing array dimensions (axes)
print("No. of dimensions: ", pattern.ndim)
 
# Printing shape of array
print("Shape of array: ", pattern.shape)
 
# Printing size (total number of elements) of array
print("Size of array: ", pattern.size)

