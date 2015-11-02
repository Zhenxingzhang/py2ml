__author__ = 'zhenxingzhang'

'''
high-performance multidimensional array object and tools for working with these arrays.
'''

'''Numpy'''

'''
Array
A grid of values, all of the same type, indexed by a tuple of nonnegative integers

The number of dimensions is the rank of the array

the shape of an array is a tuple of integers giving the size of the array along each dimension
'''

import  numpy as np

a = np.array([1,2,3])
b = np.array([1,2,3])
b.shape=(3,1)
print a*b

array= [1, 2, 3]
print type(array)
a = np.array([1,2,3])
print type(a)
print a.shape

print a[0], a[1], a[2]

a[0] = 5

print a

b = np.array([[1,2,3],[4,5,6]])
print b.shape
print b[0, 0], b[0, 1], b[1, 0]

'''
Array indexing
'''
a = np.array([[1,2,3,4], [5,6,7,8], [9,10,11,12]])
b= a[:2, 1:3]

b[0,0]=77

print a[0,1]

row_r1 = a[1, :]   #rank 1 view of the second row of a
row_r2 = a[1:2, :] #rank 2 view of the second row of a

print row_r1, row_r1.shape
print row_r2, row_r2.shape

#the same distinction when accessing columns of an array
col_r1 = a[:, 1]  #rank 1 view of the second column of a
col_r2 = a[:, 1:2]#rank 2 view of the second column of a

print col_r1, col_r1.shape #one dimensional array
print col_r1[1]
print col_r2, col_r2.shape #still two dimensional array
print col_r2[1][0]

#slicing index just create the reference to the data
#integer index will copy the data
a = np.array([[1,2], [3, 4], [5, 6]])
b=a[[0, 1, 2], [0, 1, 0]]
b[0] = 100
print a[0][0], b[0]

c=a[:2,:2]
c[0][0] = 100
print a[0][0], c[0]

'''
Array math
Basic mathematical function operate elementwise on arrays, and are available both as operator overload and as functions in the numpy module
'''
x = np.array([[1,2],[3,4]], dtype=np.float64)
y = np.array([[5,6],[7,8]], dtype=np.float64)

# Elementwise sum; both produce the array
# [[ 6.0  8.0]
#  [10.0 12.0]]
print x + y
print np.add(x, y)

# Elementwise difference; both produce the array
# [[-4.0 -4.0]
#  [-4.0 -4.0]]
print x - y
print np.subtract(x, y)

# Elementwise product; both produce the array
# [[ 5.0 12.0]
#  [21.0 32.0]]
print x * y
print np.multiply(x, y)

# Elementwise division; both produce the array
# [[ 0.2         0.33333333]
#  [ 0.42857143  0.5       ]]
print x / y
print np.divide(x, y)

# Elementwise square root; produces the array
# [[ 1.          1.41421356]
#  [ 1.73205081  2.        ]]
print np.sqrt(x)

'''
dot function compute inner product of vectors, to multiply a vector by a matrix, and to multiply matrices
'''
x = np.array([[1,2],[3,4]])
y = np.array([[5,6],[7,8]])

v = np.array([9,10])
w = np.array([11, 12])

print x.dot(v)
print np.dot(x, v)

# Matrix / matrix product; both produce the rank 2 array
# [[19 22]
#  [43 50]]
print x.dot(y)
print np.dot(x, y)

'''
sum
'''
x = np.array([[1,2],[3,4]])

print np.sum(x)  # Compute sum of all elements; prints "10"
print np.sum(x, axis=0) #column-wise [4,6]
print np.sum(x, axis=1) #row-wise[3,7]

'''
broadcasting
'''
x = np.array([[1,2,3], [4,5,6], [7,8,9], [10, 11, 12]])
v = np.array([[1], [0], [1], [0]])
print v.shape
y = x + v  # Add v to each row of x using broadcasting
print y