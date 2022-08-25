# Importing numpy library

import numpy as np
import sys

# Initializing a 1-Dimensional Array
a = np.array([1, 2, 3, 4])
print(a)

# Initializing a 2-Dimensional Array
b = np.array([[1.0, 2.3, 4.1], [3.5, 6.7, 9.2]])
print(b)

# Initializing a 3-Dimensional Array
c = np.array([[[3.0, 4.0, 5.0], [1.0, 4.0], [2.0, 8.0], [3.0]],[7, 10, 12]], dtype=object)
print(c)

# Getting the dimensions of the Array
print(a.ndim)
print(b.ndim)

# Method to get the shape of the Array (rows, columns)
print(b.shape)

# Method to get the data type of the Array
print(a.dtype)
print(c.dtype)

# Method to get the size of the items in byte (1 byte = 8 bits)
print(a.itemsize)

# Method to get the total number of elements in the Array
print(a.size)

print("Thus the total size of the array occupying the memory is :",a.itemsize * a.size,"bits")

a1 = np.array([[2, 4, 6, 8, 10, 12], [1, 3, 5, 7, 9, 11]], dtype="int32")
#Accessing the elements always goes by the notation of [row][column]

# Accessing an entire row
print(a1[0, :])
# Accessing an entire column
print(a1[:, 1])
# Accessing via [start index : end index+1 : step size]
print(a1[1, 1:6:2])
print(a1[0, 1:-1:1])

# Changing elements in the Array 
a1[1][5] = 13                   # At a particular location
a1[:, 2] = 3                    # Of a specific row or column
a1[:, 3] = [50, 52]             # Of a specific row or column with a unique values
print(a1)

b1 = np.array([[[1, 2], [3, 4]], [[3, 8], [6, 9]]], dtype="int16")
print(b1.ndim)
print(b1)

# In 3-Dimensional Arrays we have notation as [matrix, row, column]
b1[0][1][1] = 6
print(b1[0][1][1])
# Changing values of entire row
b1[:, 1, :] = 4
print(b1)

# Initializing different types of Arrays
#    Array of Zeroes
z = np.zeros(5, dtype="int16")               # A 1-D Array
z1 = np.zeros((2, 3), dtype="int16")         # A 2-D Array
z2 = np.zeros((3, 3, 3), dtype="int16")      # A 3-D Array
print(z)
print(z1)
print(z2)

# Array of Ones
one = np.ones((2, 2, 2), dtype="int32")
print(one)

# Array of any other number
num = np.full((3, 4), 18)
print(num)

# Array of Random numbers
ran = np.random.rand(3, 3)
print(ran)
# Array of random numbers of shape of any desired pre-defined array
ran1 = np.random.random_sample(a1.shape)
print(ran1)
# Random integer values beginning from 3 to 10 with matrix of size 3 x 3
ran2 = np.random.randint(3, 10, size=(3, 3))
print(ran2)

# Printing an Identity Square matrix
iden = np.identity(4, dtype="int32")
print(iden)

# Copying the elements of an array n times and storing it in an array
ar = np.repeat(a, 3, axis=0)
print(ar)

# Mathematical Operations on the Arrays
a11 = np.array([1, 3, 4, 6, 8, 11])
# Operations on Arrays by the use of constants
a11 = a11 + 2
print(a11)
a11 = a11 - 2
a11 = a11 * 4
a11 = a11 / 2
a11 = a11 ** 2   # Exponential
b11 = np.array([1, 1, 2, 1, 3, 11])
# Operations on Arrays by the use of another arrays.. The Operations are performed on the basis of index
a11 = a11 / b11
a11 = a11 + b11
a11 = a11 - b11
a11 = a11 * b11
# Trigonometric functions
print(np.cos(a11))

m = np.array([[1 ,2, 3], [4, 5, 6]], dtype="int32")
print(m.shape)
n = np.array([[2, 4], [6, 8], [10, 6]], dtype="int32")
print(n.shape)
# Method for Matrix Multiplication
mn = np.matmul(m, n)
print(mn)

# Evaluating determinant
d = np.identity(3)
# linalg represents the linear functions in numpy... used for eigenvalues, vectors, determinant, Inverse, Matrix norm, etc.
print(np.linalg.det(d))

# Statistical functions
stats = np.array([[1, 2, 3], [4, 5, 6]])
print(np.min(stats, axis = 1))  # Axis 1 represents columns
print(np.max(stats, axis = 0))  # Axis 0 represents rows
print(np.max(stats))
print(np.sum(stats, axis = 0))  # Evaluating sum of each row

# Reshaping an Array into a same dimension but of different shape
be = np.array([[1, 2, 3, 4, 5, 6, 7, 8], [20, 19, 18, 17, 16, 15, 14, 13]], dtype="int32")
print(be.shape)
# Reshaping the array and storing it into another array
af = be.reshape((4, 4))
print(af.shape)

# Vectors
v1 = np.array([1, 2, 3, 4], dtype="int16")
v2 = np.array([5, 6, 7, 8], dtype="int16")
# Creating an Array who stacks vectors
print(np.vstack((v1, v2)))          # vstack refers to vertical stack
# Creating an Array of Stacking vectors and which can also perform Operations on the vectors
print(np.vstack((v1, v2, v2, v1, v2-v1)))
print(np.hstack((v1, v2, v1-v2)))   # hstack refers to horizontal stack

# Method to read data from a text file... It can also be used for a excel or a pdf file... as well only the extension needs to be changed for the different file types...
data = np.genfromtxt('./Numpy Library/data.txt', delimiter=',')
print(data.shape)
print(data)
# After getting the source file, one can easily modify data...
data1 = data.reshape(4, 10)
print(data1.shape)
print(data)
# Type Casting of data file using astype method
file = data.astype('int32')
print(file)
# Boolean Masking and Advanced Array Indexing to get the boolean result of a condition where condition is applied on every element of the array
print(file > 50)      # Here > 50 condition is checked for every array element
print(file[file > 50])       # To get the elements as well which are > 50

# Getting the elements as a list when their indices are passed in the array as a list
p = np.array([1, 3, 6, 8, 10, 13, 17, 2, 18, 22])
print(p[[1, 3, 7, 4]])      # indices passed as a list