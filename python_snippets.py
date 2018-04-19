[ENVIRONMENT]::SETENVIRONMENTVARIABLE("PATH", "$ENV:PATH;C:\Program Files (x86)\ProgramData\Anaconda3", "USER")

! pip install --user wikipedia

##################### NumPy #####################

>>> x = np.arange(10)
>>> x[2]
2
>>> x[-2]
8

>>> x.shape = (2,5) # now x is 2-dimensional
>>> x[1,3]
8
>>> x[1,-1]
9

>>> x[0]
array([0, 1, 2, 3, 4])

>>> x[0][2]
2

x[0,2] = x[0][2] though the second case is more inefficient as a new temporary array is created after the first index that is subsequently indexed by 2.

>>> x = np.arange(10)
>>> x[2:5]
array([2, 3, 4])
>>> x[:-7]
array([0, 1, 2])
>>> x[1:7:2]
array([1, 3, 5])

>>> y = np.arange(35).reshape(5,7)
>>> y[1:5:2,::3]
array([[ 7, 10, 13], [21, 24, 27]])

>>> x = np.arange(10,1,-1)
>>> x
array([10,  9,  8,  7,  6,  5,  4,  3,  2])
>>> x[np.array([3, 3, 1, 8])]
array([7, 7, 9, 2])

>>> x = np.arange(10,1,-1)
>>> x
array([10,  9,  8,  7,  6,  5,  4,  3,  2])
>>> x[np.array([3, 3, 1, 8])]
array([7, 7, 9, 2])

>>> b = y>20
>>> y[b]
array([21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34])

import numpy as np
y = np.arange(35).reshape(5,7)
b = y > 20
print(y[b])

import numpy as np
y = np.arange(35).reshape(5,7)
print("y = ",y)
print("y[np.array([0,2,4]),1:3] = ", y[np.array([0,2,4]),1:3])

y.shape


>>> x = np.arange(10)
>>> x[2:7] = 1

y = np.arange(35).reshape(5,7)
y[2:3,2:5]=0

y = np.arange(35).reshape(5,7)
y[2:3,2:5] += 100
y

#Upload data
# csv = comma separated values
data = np.genfromtxt("your_file_here.csv", delimiter=",")

#####################  #####################