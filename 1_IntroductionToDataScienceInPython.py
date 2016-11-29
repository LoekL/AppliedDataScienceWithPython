
== 1 - Introduction to Data Science with Python ==

-- Week 1

[The Python Programming Language: Functions]

x = 1
y = 2
x + y

> 3

x

> 1

# add_numbers is a function that takes two numbers and adds them together.

def add_numbers(x, y):
    return x + y
​
add_numbers(1, 2)

> 3

# add_numbers updated to take an optional 3rd parameter. Using print allows printing of multiple expressions within a single cell.

def add_numbers(x,y,z=None):
    if (z==None):
        return x+y
    else:
        return x+y+z
​
print(add_numbers(1, 2))
print(add_numbers(1, 2, 3))

> 3
> 6

# add_numbers updated to take an optional flag parameter.

def add_numbers(x, y, z=None, flag=False):
    if (flag):
        print('Flag is true!')
    if (z==None):
        return x + y
    else:
        return x + y + z

print(add_numbers(1, 2, flag=True))

> Flag is true!
> 3

# Assign function add_numbers to variable a.

def add_numbers(x,y):
    return x+y
​
a = add_numbers
a(1,2)

> 3

[The Python Programming Language: Types and Sequences]

# Use type to return the object`s type.

type('This is a string')

> str

type(None)

> NoneType

type(1)

> int

type(1.0)

> float

type(add_numbers)

> function

# Tuples are an immutable data structure (cannot be altered).

x = (1, 'a', 2, 'b')
type(x)

> tuple

# Lists are a mutable data structure.

x = [1, 'a', 2, 'b']
type(x)

> list

# Use append to append an object to a list.

x.append(3.3)
print(x)

> [1, 'a', 2, 'b', 3.3]

# This is an example of how to loop through each item in the list.

for item in x:
    print(item)

> 1
> a
> 2
> b
> 3.3

# Or using the indexing operator:

i = 0
while(i != len(x)):
    print(x[i])
    i = i + 1

> 1
> a
> 2
> b
> 3.3

# Use + to concatenate lists.

[1,2] + [3,4]

> [1, 2, 3, 4]

# Use * to repeat lists.

[1] * 3

> [1, 1, 1]

# Use the in operator to check if something is inside a list.

1 in [1, 2, 3]

> True

# Now let's look at strings. Use bracket notation to slice a string.

x = 'This is a string'
print(x[0]) #first character
print(x[0:1]) #first character, but we have explicitly set the end character
print(x[0:2]) #first two characters
​
> T
> T
> Th

# This will return the last element of the string.

x[-1]

> 'g'

# This will return the slice starting from the 4th element from the end and stopping before the 2nd element from the end.

x[-4:-2]

> 'ri'

# This is a slice from the beginning of the string and stopping before the 3rd element.

x[:3]

> 'Thi'

# And this is a slice starting from the 3rd element of the string and going all the way to the end.

x[3:]

> 's is a string'

firstname = 'Christopher'
lastname = 'Brooks'
​
print(firstname + ' ' + lastname)
print(firstname * 3)
print('Chris' in firstname)
​
> Christopher Brooks
> ChristopherChristopherChristopher
> True

# split returns a list of all the words in a string, or a list split on a specific character.

firstname = 'Christopher Arthur Hansen Brooks'.split(' ')[0] # [0] selects the first element of the list
lastname = 'Christopher Arthur Hansen Brooks'.split(' ')[-1] # [-1] selects the last element of the list
print(firstname)
print(lastname)

> Christopher
> Brooks

# Make sure you convert objects to strings before concatenating.

'Chris' + 2

> ---------------------------------------------------------------------------
> TypeError                                 Traceback (most recent call last)
> <ipython-input-27-1623ac76de6e> in <module>()
> ----> 1 'Chris' + 2
>
> TypeError: Can`t convert 'int' object to str implicitly

'Chris' + str(2)

> 'Chris2'

# Dictionaries associate keys with values.

x = {'Christopher Brooks': 'brooksch@umich.edu', 'Bill Gates': 'billg@microsoft.com'}
x['Christopher Brooks'] # Retrieve a value by using the indexing operator
​
> 'brooksch@umich.edu'

x['Kevyn Collins-Thompson'] = None
x['Kevyn Collins-Thompson']

> None

# Iterate over all of the keys:

for name in x:
    print(x[name])

> billg@microsoft.com
> brooksch@umich.edu
> None

# Iterate over all of the values:

for email in x.values():
    print(email)

> billg@microsoft.com
> brooksch@umich.edu
> None

# Iterate over all of the items in the list:

for name, email in x.items():
    print(name)
    print(email)

> Bill Gates
> billg@microsoft.com
> Christopher Brooks
> brooksch@umich.edu
> Kevyn Collins-Thompson
> None

# You can unpack a sequence into different variables:

x = ('Christopher', 'Brooks', 'brooksch@umich.edu')
fname, lname, email = x
fname

> 'Christopher'

lname

> 'Brooks'

# Make sure the number of values you are unpacking matches the number of variables being assigned.

x = ('Christopher', 'Brooks', 'brooksch@umich.edu', 'Ann Arbor')
fname, lname, email = x

> ---------------------------------------------------------------------------
> ValueError                                Traceback (most recent call last)
> <ipython-input-105-9ce70064f53e> in <module>()
>       1 x = ('Christopher', 'Brooks', 'brooksch@umich.edu', 'Ann Arbor')
> ----> 2 fname, lname, email = x
>
> ValueError: too many values to unpack (expected 3)

# Python has a built in method for convenient string formatting.

sales_record = {
'price': 3.24,
'num_items': 4,
'person': 'Chris'}
​
sales_statement = '{} bought {} item(s) at a price of {} each for a total of {}'
​
print(sales_statement.format(sales_record['person'],
                             sales_record['num_items'],
                             sales_record['price'],
                             sales_record['num_items']*sales_record['price']))
​
> Chris bought 4 item(s) at a price of 3.24 each for a total of 12.96

[Reading and Writing CSV files]

# Let's import our datafile mpg.csv, which contains fuel economy data for 234 cars.

# mpg : miles per gallon
# class : car classification
# cty : city mpg
# cyl : # of cylinders
# displ : engine displacement in liters
# drv : f = front-wheel drive, r = rear wheel drive, 4 = 4wd
# fl : fuel (e = ethanol E85, d = diesel, r = regular, p = premium, c = CNG)
# hwy : highway mpg
# manufacturer : automobile manufacturer
# model : model of car
# trans : type of transmission
# year : model year

import csv
​
# Set precision to 2 decimals

%precision 2 # Specific to iPython/Jupyter
​
with open('mpg.csv') as csvfile:
    mpg = list(csv.DictReader(csvfile)) # generate a dictionary per record

mpg[:2] # The first two dictionaries in our list.
Out[109]:
[{'': '1',
  'class': 'compact',
  'cty': '18',
  'cyl': '4',
  'displ': '1.8',
  'drv': 'f',
  'fl': 'p',
  'hwy': '29',
  'manufacturer': 'audi',
  'model': 'a4',
  'trans': 'auto(l5)',
  'year': '1999'},
 {'': '2',
  'class': 'compact',
  'cty': '21',
  'cyl': '4',
  'displ': '1.8',
  'drv': 'f',
  'fl': 'p',
  'hwy': '29',
  'manufacturer': 'audi',
  'model': 'a4',
  'trans': 'manual(m5)',
  'year': '1999'}]

# csv.Dictreader has read in each row of our csv file as a dictionary.
# len shows that our list is comprised of 234 dictionaries.

len(mpg)

> 234

# keys gives us the column names of our csv.

mpg[0].keys()

> dict_keys(['', 'cty', 'class', 'model', 'drv', 'hwy', 'trans', 'cyl', 'displ', 'manufacturer', 'fl', 'year'])

# This is how to find the average cty fuel economy across all cars. All values in the dictionaries are strings,
# so we need to convert to float.

sum(float(d['cty']) for d in mpg) / len(mpg)

> 16.86

# Similarly this is how to find the average hwy fuel economy across all cars.

sum(float(d['hwy']) for d in mpg) / len(mpg)

> 23.44

# Use set to return the unique values for the number of cylinders the cars in our dataset have.

cylinders = set(d['cyl'] for d in mpg)
cylinders

> {'4', '5', '6', '8'}

# Here's a more complex example where we are grouping the cars by number of cylinder,
# and finding the average cty mpg for each group.

CtyMpgByCyl = []
​
for c in cylinders: # iterate over all the cylinder levels
    summpg = 0
    cyltypecount = 0
    for d in mpg: # iterate over all dictionaries
        if d['cyl'] == c: # if the cylinder level type matches,
            summpg += float(d['cty']) # add the cty mpg
            cyltypecount += 1 # increment the count
    CtyMpgByCyl.append((c, summpg / cyltypecount)) # append the tuple ('cylinder', 'avg mpg')
​
CtyMpgByCyl.sort(key=lambda x: x[0]) # key takes a function, in this case it simply retrieves the first element of the tuple, hence it is ordered by c(ylinder)
CtyMpgByCyl

> [('4', 21.01), ('5', 20.50), ('6', 16.22), ('8', 12.57)]

# Use set to return the unique values for the class types in our dataset.

vehicleclass = set(d['class'] for d in mpg) # what are the class types
vehicleclass

> {'2seater', 'compact', 'midsize', 'minivan', 'pickup', 'subcompact', 'suv'}

# And here's an example of how to find the average hwy mpg for each class of vehicle in our dataset.

HwyMpgByClass = []
​
for t in vehicleclass: # iterate over all the vehicle classes
    summpg = 0
    vclasscount = 0
    for d in mpg: # iterate over all dictionaries
        if d['class'] == t: # if the cylinder amount type matches,
            summpg += float(d['hwy']) # add the hwy mpg
            vclasscount += 1 # increment the count
    HwyMpgByClass.append((t, summpg / vclasscount)) # append the tuple ('class', 'avg mpg')
​
HwyMpgByClass.sort(key=lambda x: x[1]) # now we sort by the second element, summpg
HwyMpgByClass

> [('pickup', 16.88),
>  ('suv', 18.13),
>  ('minivan', 22.36),
>  ('2seater', 24.80),
>  ('midsize', 27.29),
>  ('subcompact', 28.14),
>  ('compact', 28.30)]

# The Python Programming Language: Dates and Times

import datetime as dt
import time as tm

# time returns the current time in seconds since the Epoch. (January 1st, 1970)

tm.time()

> 1478202359.44 # EPOCH time / 1970-01-01

# Convert the timestamp to datetime.

dtnow = dt.datetime.fromtimestamp(tm.time())
dtnow

> datetime.datetime(2016, 11, 3, 19, 46, 1, 99402)

# Handy datetime attributes:

dtnow.year, dtnow.month, dtnow.day, dtnow.hour, dtnow.minute, dtnow.second # get year, month, day, etc. from a datetime

> (2016, 11, 3, 19, 46, 1)

# timedelta is a duration expressing the difference between two dates.

delta = dt.timedelta(days = 100) # create a timedelta of 100 days
delta

> datetime.timedelta(100)

# date.today returns the current local date.

today = dt.date.today()
today - delta # the date 100 days ago

> datetime.date(2016, 7, 26)

today > today - delta # compare dates

> True

[The Python Programming Language: Objects and map()]

# An example of a class in python:

class Person:
    department = 'School of Information' # a class variable
​
    def set_name(self, new_name): # a method
        self.name = new_name
    def set_location(self, new_location):
        self.location = new_location

person = Person()
person.set_name('Christopher Brooks')
person.set_location('Ann Arbor, MI, USA')
print('{} live in {} and works in the department {}'.format(person.name, person.location, person.department))

> Christopher Brooks live in Ann Arbor, MI, USA and works in the department School of Information

# Here's an example of mapping the min function between two lists.

store1 = [10.00, 11.00, 12.34, 2.34]
store2 = [9.00, 11.10, 12.34, 2.01]
cheapest = map(min, store1, store2)
cheapest

> <map at 0x7fc87011b940>

# Now let's iterate through the map object to see the values.

for item in cheapest:
    print(item)

> 9.0
> 11.0
> 12.34
> 2.01

[The Python Programming Language: Lambda and List Comprehensions]

# Here's an example of lambda that takes in three parameters and adds the first two.

my_function = lambda a, b, c : a + b
my_function(1, 2, 3)

> 3

# Let's iterate from 0 to 999 and return the even numbers.

my_list = []
for number in range(0, 1000):
    if number % 2 == 0:
        my_list.append(number)
my_list

> [0,
>  2,
>  4,
>  ...
>  996,
>  998]

# Now the same thing but with list comprehension.

my_list = [number for number in range(0,1000) if number % 2 == 0]
my_list

> [0,
>  2,
>  4,
>  ...
>  996,
>  998]

[The Python Programming Language: Numerical Python (NumPy)]

import numpy as np

# Creating Arrays
# Create a list and convert it to a numpy array

mylist = [1, 2, 3]
x = np.array(mylist)
x

> array([1, 2, 3])

# Or just pass in a list directly

y = np.array([4, 5, 6])
y

> array([4, 5, 6])

# Pass in a list of lists to create a multidimensional array.

m = np.array([[7, 8, 9], [10, 11, 12]])
m

> array([[7,  8,  9],
>       [10, 11, 12]])

# Use the shape method to find the dimensions of the array. (rows, columns)

m.shape

> (2, 3)

# arange returns evenly spaced values within a given interval.

n = np.arange(0, 30, 2) # start at 0 count up by 2, stop before 30
n

> array([0,  2,  4,  6,  8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28])

# reshape returns an array with the same data with a new shape.

n = n.reshape(3, 5) # reshape array to be 3x5
n

> array([[ 0,  2,  4,  6,  8],
>        [10, 12, 14, 16, 18],
>        [20, 22, 24, 26, 28]])

# linspace returns evenly spaced numbers over a specified interval.

o = np.linspace(0, 4, 9) # return 9 evenly spaced values from 0 to 4
o

> array([0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0])

# resize changes the shape and size of array in-place (you do not need to specifically assign it like .reshape).

o.resize(3, 3)
o

> array([[0.0, 0.5, 1.0],
>        [1.5, 2.0, 2.5],
>        [3.0, 3.5, 4.0]])

# ones returns a new array of given shape and type, filled with ones.

np.ones((3, 2))

> array([[1.0, 1.0],
>        [1.0, 1.0],
>        [1.0, 1.0]])

# zeros returns a new array of given shape and type, filled with zeros.

np.zeros((2, 3))

> array([[0.0, 0.0, 0.0],
>        [0.0, 0.0, 0.0]])

# eye returns a 2-D array with ones on the diagonal and zeros elsewhere.

np.eye(3)

> array([[ 1.0, 0.0, 0.0],
>        [ 0.0, 1.0, 0.0],
>        [ 0.0, 0.0, 1.0]])

# diag extracts a diagonal or constructs a diagonal array.

np.diag(y) # y = np.array([4, 5, 6])

> array([[4, 0, 0],
>        [0, 5, 0],
>        [0, 0, 6]])

# Create an array using repeating list (or see np.tile)

np.array([1, 2, 3] * 3)

> array([1, 2, 3, 1, 2, 3, 1, 2, 3])

# Repeat elements of an array using repeat.

np.repeat([1, 2, 3], 3)

> array([1, 1, 1, 2, 2, 2, 3, 3, 3])

# Combining Arrays

p = np.ones([2, 3], int)
p

> array([[1, 1, 1],
>        [1, 1, 1]])

# Use vstack to stack arrays in sequence vertically (row wise).

np.vstack([p, 2 * p])

> array([[1, 1, 1],
>        [1, 1, 1],
>        [2, 2, 2],
>        [2, 2, 2]])

# Use hstack to stack arrays in sequence horizontally (column wise).

np.hstack([p, 2 * p])

> array([[1, 1, 1, 2, 2, 2],
>        [1, 1, 1, 2, 2, 2]])

# Operations
# Use +, -, *, / and ** to perform element wise addition, subtraction, multiplication, division and power.

print(x + y) # elementwise addition     [1 2 3] + [4 5 6] = [5  7  9]
print(x - y) # elementwise subtraction  [1 2 3] - [4 5 6] = [-3 -3 -3]

> [5, 7, 9]
> [-3, -3, -3]

print(x * y) # elementwise multiplication  [1 2 3] * [4 5 6] = [4  10  18]
print(x / y) # elementwise divison         [1 2 3] / [4 5 6] = [0.25  0.4  0.5]

> [4, 10, 18]
> [0.25, 0.4, 0.5]

print(x ** 2) # elementwise power  [1 2 3] ^2 =  [1 4 9]

> [1, 4, 9]

# Dot Product:
# [x1, x2, x3] * [y1, y2, y3] = x1*y1 + x2*y2 + x3*y3

# x = [1, 2, 3]
# y = [4, 5, 6]
x.dot(y) # dot product  1*4 + 2*5 + 3*6

> 32

z = np.array([y, y**2])
print(len(z)) # number of rows of array
2

# Let's look at transposing arrays. Transposing permutes the dimensions of the array.

z = np.array([y, y**2])
z

> array([[ 4,  5,  6],
>        [16, 25, 36]])

# The shape of array z is (2,3) before transposing.

z.shape

> (2, 3)

# Use .T to get the transpose.

z.T

> array([[4, 16],
>        [5, 25],
>        [6, 36]])

# The number of rows has swapped with the number of columns.

z.T.shape

> (3, 2)

# Use .dtype to see the data type of the elements in the array.

z.dtype

> dtype('int64')

# Use .astype to cast to a specific type.

z = z.astype('f')
z.dtype

> dtype('float32')

# Math Functions
# Numpy has many built in math functions that can be performed on arrays.

a = np.array([-4, -2, 1, 3, 5])

a.sum()

> 3

a.max()

> 5

a.min()

> -4

a.mean()

> 0.59999999999999998

a.std()

> 3.2619012860600183

# argmax and argmin return the index of the maximum and minimum values in the array.

a.argmax()

> 4

a.argmin()

> 0

# Indexing / Slicing

s = np.arange(13)**2
s

> array([0,   1,   4,   9,  16,  25,  36,  49,  64,  81, 100, 121, 144])

# Use bracket notation to get the value at a specific index. Remember that indexing starts at 0.

s[0], s[4], s[-1]

> (0, 16, 144)

# Use : to indicate a range. array[start:stop]
# Leaving start or stop empty will default to the beginning/end of the array.

s[1:5]

> array([1,  4,  9, 16])

# Use negatives to count from the back.

s[-4:]

> array([81, 100, 121, 144])

# A second : can be used to indicate step-size. array[start:stop:stepsize]
# Here we are starting 5th element from the end, and counting backwards by 2 until the beginning of the array is reached.

s[-5::-2] # same as -5:stop:-2 -> -2 makes you go to 0, +2 would go up to 144

> array([64, 36, 16,  4,  0])

# Let's look at a multidimensional array.

r = np.arange(36)
r.resize((6, 6))
r

> array([[ 0,  1,  2,  3,  4,  5],
>        [ 6,  7,  8,  9, 10, 11],
>        [12, 13, 14, 15, 16, 17],
>        [18, 19, 20, 21, 22, 23],
>        [24, 25, 26, 27, 28, 29],
>        [30, 31, 32, 33, 34, 35]])

# Use bracket notation to slice: array[row, column]

r[2, 2]

> 14

# And use : to select a range of rows or columns

r[3, 3:6]

> array([21, 22, 23])

# Here we are selecting all the rows up to (and not including) row 2, and all the columns up to (and not including) the last column.

r[:2, :-1]

> array([[ 0,  1,  2,  3,  4],
>        [ 6,  7,  8,  9, 10]])

# This is a slice of the last row, and only every other element.

r[-1, ::2] # all columns (:) but with steps of 2

> array([30, 32, 34])

# We can also perform conditional indexing. Here we are selecting values from the array that are greater than 30.
# (Also see np.where)

r[r > 30]

> array([31, 32, 33, 34, 35])

# Here we are assigning all values in the array that are greater than 30 to the value of 30.

r[r > 30] = 30
r

> array([[ 0,  1,  2,  3,  4,  5],
>        [ 6,  7,  8,  9, 10, 11],
>        [12, 13, 14, 15, 16, 17],
>        [18, 19, 20, 21, 22, 23],
>        [24, 25, 26, 27, 28, 29],
>        [30, 30, 30, 30, 30, 30]])

# Copying Data
# Be careful with copying and modifying arrays in NumPy!
# r2 is a slice of r

r2 = r[:3,:3]
r2

> array([[ 0,  1,  2],
>        [ 6,  7,  8],
>        [12, 13, 14]])

# Set this slice's values to zero ([:] selects the entire array)

r2[:] = 0
r2

> array([[0, 0, 0],
>        [0, 0, 0],
>        [0, 0, 0]])

# r has also been changed!

r

> array([[ 0,  0,  0,  3,  4,  5],
>        [ 0,  0,  0,  9, 10, 11],
>        [ 0,  0,  0, 15, 16, 17],
>        [18, 19, 20, 21, 22, 23],
>        [24, 25, 26, 27, 28, 29],
>        [30, 30, 30, 30, 30, 30]])

# To avoid this, use r.copy to create a copy that will not affect the original array

r_copy = r.copy()
r_copy

> array([[ 0,  0,  0,  3,  4,  5],
>        [ 0,  0,  0,  9, 10, 11],
>        [ 0,  0,  0, 15, 16, 17],
>        [18, 19, 20, 21, 22, 23],
>        [24, 25, 26, 27, 28, 29],
>        [30, 30, 30, 30, 30, 30]])

# Now when r_copy is modified, r will not be changed.

r_copy[:] = 10 # setting all elements to 10
print(r_copy, '\n')
print(r)

> [[10 10 10 10 10 10]
>  [10 10 10 10 10 10]
>  [10 10 10 10 10 10]
>  [10 10 10 10 10 10]
>  [10 10 10 10 10 10]
>  [10 10 10 10 10 10]]
>
> [[ 0  0  0  3  4  5]
>  [ 0  0  0  9 10 11]
>  [ 0  0  0 15 16 17]
>  [18 19 20 21 22 23]
>  [24 25 26 27 28 29]
>  [30 30 30 30 30 30]]

# Iterating Over Arrays
# Let's create a new 4 by 3 array of random numbers 0-9.

test = np.random.randint(0, 10, (4,3))
test

> array([[1, 1, 3],
>        [8, 6, 6],
>        [7, 1, 3],
>        [5, 9, 1]])

# Iterate by row:

for row in test:
    print(row)

> [1 1 3]
> [8 6 6]
> [7 1 3]
> [5 9 1]

Iterate by index:

for i in range(len(test)):
    print(test[i])

> [1 1 3]
> [8 6 6]
> [7 1 3]
> [5 9 1]

# Iterate by row and index:

for i, row in enumerate(test):
    print('row', i, 'is', row)

> row 0 is [1 1 3]
> row 1 is [8 6 6]
> row 2 is [7 1 3]
> row 3 is [5 9 1]

# Use zip to iterate over multiple iterables.

test2 = test**2
test2

> array([[ 1,  1,  9],
>        [64, 36, 36],
>        [49,  1,  9],
>        [25, 81,  1]])

for i, j in zip(test, test2):
    print(i, '+', j, '=', i + j)

> [1 1 3] + [1 1 9] = [ 2  2 12]
> [8 6 6] + [64 36 36] = [72 42 42]
> [7 1 3] + [49  1  9] = [56  2 12]
> [5 9 1] + [25 81  1] = [30 90  2]

[Wk1 - Lecture Quizzes]

people = ['Dr. Christopher Brooks', 'Dr. Kevyn Collins-Thompson', 'Dr. VG Vinod Vydiswaran', 'Dr. Daniel Romero']

def split_title_and_name(person):
    title = person.split()[0]
    lastname = person.split()[-1]
    return '{} {}'.format(title, lastname)

list(map(split_title_and_name, people))

> ['Dr. Brooks', 'Dr. Collins-Thompson', 'Dr. Vydiswaran', 'Dr. Romero']

def split_title_and_name(person):
    return person.split()[0] + ' ' + person.split()[-1]

func = lambda person: person.split()[0] + ' ' + person.split()[-1]

# option 1
for person in people:
    print(split_title_and_name(person) == func(person))

> True

# option 2
list(map(split_title_and_name, people)) == list(map(func, people))

> True

def times_tables():
    lst = []
    for i in range(10):
        for j in range (10):
            lst.append(i * j)
    return lst

# Same as

times_tables() == [i * j for i in range(10) for j in range(10)]

> True

# Generate all possible uids that can take the format char-char-num-num, e.g. 'aa12'

lowercase = 'abcdefghijklmnopqrstuvwxyz'
digits = '0123456789'

answer = [a + b + c + d for a in lowercase for b in lowercase for c in digits for d in digits]
correct_answer == answer

> True

[Quiz Wk1]

import numpy as np

r = np.arange(36)
r.resize((6, 6))
r

> array([[ 0,  1,  2,  3,  4,  5],
>        [ 6,  7,  8,  9, 10, 11],
>        [12, 13, 14, 15, 16, 17],
>        [18, 19, 20, 21, 22, 23],
>        [24, 25, 26, 27, 28, 29],
>        [30, 31, 32, 33, 34, 35]])

r.reshape(36)[::7]
> array([0,  7, 14, 21, 28, 35]) # True

r[::7]

# Now it looks at the list of rows, returns all starting at 0 with steps of 7
# Hence only the first row is returned

> array([[0, 1, 2, 3, 4, 5]])

# First 6 rows, only colums from start to end with steps of -7, none match other than 0, hence only the 0 column is returned

r[0:6,::-7]

> array([[ 5],
>        [11],
>        [17],
>        [23],
>        [29],
>        [35]])

r[:,::7]

# Same as above but now the first column moving in the other direction

> array([[ 0],
>        [ 6],
>        [12],
>        [18],
>        [24],
>        [30]])

-- Week 2

[The Series Data Structure]

import pandas as pd

pd.Series? # Gives description about panda's Series object

animals = ['Tiger', 'Bear', 'Moose']
pd.Series(animals)

> 0    Tiger
> 1     Bear
> 2    Moose
> dtype: object

numbers = [1, 2, 3]
pd.Series(numbers)

> 0    1
> 1    2
> 2    3
>  dtype: int64

animals = ['Tiger', 'Bear', None]
pd.Series(animals)

> 0    Tiger
> 1     Bear
> 2     None
> dtype: object

numbers = [1, 2, None]
pd.Series(numbers)

> 0    1.0
> 1    2.0
> 2    NaN
> dtype: float64

# Missing strings are None, missing numbers are NaN

import numpy as np

# NaN != None

np.nan == None

> False

# NaN != NaN

np.nan == np.nan

> False

# You need to use methods like isnan():

np.isnan(np.nan)

> True

sports = {'Archery': 'Bhutan',
          'Golf': 'Scotland',
          'Sumo': 'Japan',
          'Taekwondo': 'South Korea'}

s = pd.Series(sports)
s

> Archery           Bhutan
> Golf            Scotland
> Sumo               Japan
> Taekwondo    South Korea
> dtype: object

s.index

> Index(['Archery', 'Golf', 'Sumo', 'Taekwondo'], dtype='object')

s = pd.Series(['Tiger', 'Bear', 'Moose'], index=['India', 'America', 'Canada'])
s

> India      Tiger
> America     Bear
> Canada     Moose
> dtype: object

sports = {'Archery': 'Bhutan',
          'Golf': 'Scotland',
          'Sumo': 'Japan',
          'Taekwondo': 'South Korea'}

s = pd.Series(sports, index=['Golf', 'Sumo', 'Hockey'])
s

> Golf      Scotland
> Sumo         Japan
> Hockey         NaN
> dtype: object

[Querying a Series]

sports = {'Archery': 'Bhutan',
          'Golf': 'Scotland',
          'Sumo': 'Japan',
          'Taekwondo': 'South Korea'}

s = pd.Series(sports)
s

> Archery           Bhutan
> Golf            Scotland
> Sumo               Japan
> Taekwondo    South Korea
> dtype: object

s.iloc[3] # by integer index

> 'South Korea'

s.loc['Golf'] # by label index

> 'Scotland'

s[3] # this works because there is no ambiguity between numerical index & label index

> 'South Korea'

s['Golf']

> 'Scotland'

sports = {99: 'Bhutan',
          100: 'Scotland',
          101: 'Japan',
          102: 'South Korea'}

s = pd.Series(sports)

# Ambiguity when keys are ints
s[0] # This won't call s.iloc[0] as one might expect, it generates an error instead

> ---------------------------------------------------------------------------
> KeyError                                  Traceback (most recent call last)
> <ipython-input-19-a5f43d492595> in <module>()
> ...
> KeyError: 0

s = pd.Series([100.00, 120.00, 101.00, 3.00])
s

> 0    100.0
> 1    120.0
> 2    101.0
> 3      3.0
> dtype: float64

total = 0
for item in s:
    total += item
print(total)

> 324.0

import numpy as np
​
total = np.sum(s)
print(total)

> 324.0

# this creates a big series of random numbers
s = pd.Series(np.random.randint(0, 1000, 10000))
s.head()

> 0    279
> 1    281
> 2    818
> 3    598
> 4    373
> dtype: int64

len(s)

> 10000

# magic functions start with %
# %% indicates a cellular magic function: only modify code in the current jupyter cell

%%timeit -n 100
summary = 0
for item in s:
    summary += item

> 100 loops, best of 3: 1.72 ms per loop

%%timeit -n 100
summary = np.sum(s)

> 100 loops, best of 3: 166 µs per loop

s += 2 # adds two to each item in s using broadcasting
s.head()

> 0    281
> 1    283
> 2    820
> 3    600
> 4    375
> dtype: int64

for label, value in s.iteritems():
    s.set_value(label, value+2)
s.head()

> 0    283
> 1    285
> 2    822
> 3    602
> 4    377
> dtype: int64

%%timeit -n 10
s = pd.Series(np.random.randint(0, 1000, 10000))
for label, value in s.iteritems():
    s.loc[label] = value + 2

> 10 loops, best of 3: 1.6 s per loop

%%timeit -n 10
s = pd.Series(np.random.randint(0, 1000, 10000))
s += 2
​
> 10 loops, best of 3: 421 µs per loop

s = pd.Series([1, 2, 3])
s.loc['Animal'] = 'Bears'
s

> 0             1
> 1             2
> 2             3
> Animal    Bears
> dtype: object

original_sports = pd.Series({'Archery': 'Bhutan',
                             'Golf': 'Scotland',
                             'Sumo': 'Japan',
                             'Taekwondo': 'South Korea'})
cricket_loving_countries = pd.Series(['Australia',
                                      'Barbados',
                                      'Pakistan',
                                      'England'],
                                   index=['Cricket',
                                          'Cricket',
                                          'Cricket',
                                          'Cricket'])
all_countries = original_sports.append(cricket_loving_countries)

original_sports

> Archery           Bhutan
> Golf            Scotland
> Sumo               Japan
> Taekwondo    South Korea
> dtype: object

cricket_loving_countries

> Cricket    Australia
> Cricket     Barbados
> Cricket     Pakistan
> Cricket      England
> dtype: object

all_countries

> Archery           Bhutan
> Golf            Scotland
> Sumo               Japan
> Taekwondo    South Korea
> Cricket        Australia
> Cricket         Barbados
> Cricket         Pakistan
> Cricket          England
> dtype: object

all_countries.loc['Cricket']

> Cricket    Australia
> Cricket     Barbados
> Cricket     Pakistan
> Cricket      England
> dtype: object

[The DataFrame Data Structure]

import pandas as pd

purchase_1 = pd.Series({'Name': 'Chris',
                        'Item Purchased': 'Dog Food',
                        'Cost': 22.50})
purchase_2 = pd.Series({'Name': 'Kevyn',
                        'Item Purchased': 'Kitty Litter',
                        'Cost': 2.50})
purchase_3 = pd.Series({'Name': 'Vinod',
                        'Item Purchased': 'Bird Seed',
                        'Cost': 5.00})

df = pd.DataFrame([purchase_1, purchase_2, purchase_3], index=['Store 1', 'Store 1', 'Store 2'])
df.head()

> Cost  Item Purchased  Name
> Store 1 22.5  Dog Food  Chris
> Store 1 2.5 Kitty Litter  Kevyn
> Store 2 5.0 Bird Seed Vinod

df.loc['Store 2']

> Cost                      5
> Item Purchased    Bird Seed
> Name                  Vinod
> Name: Store 2, dtype: object

type(df.loc['Store 2'])

> pandas.core.series.Series

df.loc['Store 1']

> Cost  Item Purchased  Name
> Store 1 22.5  Dog Food  Chris
> Store 1 2.5 Kitty Litter  Kevyn

df.loc['Store 1', 'Cost']

> Store 1    22.5
> Store 1     2.5
> Name: Cost, dtype: float64

df.T

> Store 1 Store 1 Store 2
> Cost  22.5  2.5 5
> Item Purchased  Dog Food  Kitty Litter  Bird Seed
> Name  Chris Kevyn Vinod

df.T.loc['Cost']

> Store 1    22.5
> Store 1     2.5
> Store 2       5
> Name: Cost, dtype: object

df['Cost']

> Store 1    22.5
> Store 1     2.5
> Store 2     5.0
> Name: Cost, dtype: float64

df.loc['Store 1']['Cost']

> Store 1    22.5
> Store 1     2.5
> Name: Cost, dtype: float64

df.loc[:,['Name', 'Cost']]

> Name  Cost
> Store 1 Chris 22.5
> Store 1 Kevyn 2.5
> Store 2 Vinod 5.0

# copy_df.drop(label, {0,1}) -> 0 is default (row) can be set to 1 (column) as second arg.

df.drop('Store 1')

> Cost  Item Purchased  Name
> Store 2 5.0 Bird Seed Vinod

df

> Cost  Item Purchased  Name
> Store 1 22.5  Dog Food  Chris
> Store 1 2.5 Kitty Litter  Kevyn
> Store 2 5.0 Bird Seed Vinod

# You need to assign specifically to make .drop() stick

copy_df = df.copy()
copy_df = copy_df.drop('Store 1')
copy_df

> Cost  Item Purchased  Name
> Store 2 5.0 Bird Seed Vinod

copy_df.drop?

# del sticks instantly

del copy_df['Name']
copy_df

> Cost  Item Purchased
> Store 2 5.0 Bird Seed

df['Location'] = None
df

> Cost  Item Purchased  Name  Location
> Store 1 22.5  Dog Food  Chris None
> Store 1 2.5 Kitty Litter  Kevyn None
> Store 2 5.0 Bird Seed Vinod None

[Dataframe Indexing and Loading]

costs = df['Cost']
costs

> Store 1    22.5
> Store 1     2.5
> Store 2     5.0
> Name: Cost, dtype: float64

costs += 2
costs

> Store 1    24.5
> Store 1     4.5
> Store 2     7.0
> Name: Cost, dtype: float64

# Now the cost column in df is also updated!

df

> Cost  Item Purchased  Name  Location
> Store 1 24.5  Dog Food  Chris None
> Store 1 4.5 Kitty Litter  Kevyn None
> Store 2 7.0 Bird Seed Vinod None

# !cat olympics.csv --> all statements starting with ! are passed to the systems operating shell for processing

!cat olympics.csv

> 0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15
> ,№ Summer,01 !,02 !,03 !,Total,№ Winter,01 !,02 !,03 !,Total,№ Games,01 !,02 !,03 !,Combined total
> Afghanistan (AFG),13,0,0,2,2,0,0,0,0,0,13,0,0,2,2
> Algeria (ALG),12,5,2,8,15,3,0,0,0,0,15,5,2,8,15
> ...
> Zambia (ZAM) [ZAM],12,0,1,1,2,0,0,0,0,0,12,0,1,1,2
> Zimbabwe (ZIM) [ZIM],12,3,4,1,8,1,0,0,0,0,13,3,4,1,8
> Mixed team (ZZX) [ZZX],3,8,5,4,17,0,0,0,0,0,3,8,5,4,17
> Totals,27,4809,4775,5130,14714,22,959,958,948,2865,49,5768,5733,6078,17579

df = pd.read_csv('olympics.csv')
df.head()

> 0 1 2 3 4 5 6 7 8 9 10  11  12  13  14  15
> 0 NaN № Summer  01 !  02 !  03 !  Total № Winter  01 !  02 !  03 !  Total № Games 01 !  02 !  03 !  Combined total
> 1 Afghanistan (AFG) 13  0 0 2 2 0 0 0 0 0 13  0 0 2 2
> 2 Algeria (ALG) 12  5 2 8 15  3 0 0 0 0 15  5 2 8 15
> 3 Argentina (ARG) 23  18  24  28  70  18  0 0 0 0 41  18  24  28  70
> 4 Armenia (ARM) 5 1 2 9 12  6 0 0 0 0 11  1 2 9 12

# set the index column specifically & header using skiprows

df = pd.read_csv('olympics.csv', index_col = 0, skiprows = 1)
df.head()

> № Summer  01 !  02 !  03 !  Total № Winter  01 !.1  02 !.1  03 !.1  Total.1 № Games 01 !.2  02 !.2  03 !.2  Combined total
> Afghanistan (AFG) 13  0 0 2 2 0 0 0 0 0 13  0 0 2 2
> Algeria (ALG) 12  5 2 8 15  3 0 0 0 0 15  5 2 8 15
> Argentina (ARG) 23  18  24  28  70  18  0 0 0 0 41  18  24  28  70
> Armenia (ARM) 5 1 2 9 12  6 0 0 0 0 11  1 2 9 12
> Australasia (ANZ) [ANZ] 2 3 4 5 12  0 0 0 0 0 2 3 4 5 12

df.columns

> Index(['№ Summer', '01 !', '02 !', '03 !', 'Total', '№ Winter', '01 !.1',
>        '02 !.1', '03 !.1', 'Total.1', '№ Games', '01 !.2', '02 !.2', '03 !.2',
>        'Combined total'],
>       dtype='object')

for col in df.columns:
    if col[:2] == '01':
        # inplace=True -> Modify the DataFrame in place (do not create a new object)
        df.rename(columns = {col:'Gold' + col[4:]}, inplace = True)
    if col[:2] == '02':
        df.rename(columns = {col:'Silver' + col[4:]}, inplace = True)
    if col[:2] == '03':
        df.rename(columns = {col:'Bronze' + col[4:]}, inplace = True)
    if col[:1] == '№':
        df.rename(columns = {col:'#' + col[1:]}, inplace = True)
​
df.head()

> # Summer  Gold  Silver  Bronze  Total # Winter  Gold.1  Silver.1  Bronze.1  Total.1 # Games Gold.2  Silver.2  Bronze.2  Combined total
> Afghanistan (AFG) 13  0 0 2 2 0 0 0 0 0 13  0 0 2 2
> Algeria (ALG) 12  5 2 8 15  3 0 0 0 0 15  5 2 8 15
> Argentina (ARG) 23  18  24  28  70  18  0 0 0 0 41  18  24  28  70
> Armenia (ARM) 5 1 2 9 12  6 0 0 0 0 11  1 2 9 12
> Australasia (ANZ) [ANZ] 2 3 4 5 12  0 0 0 0 0 2 3 4 5 12

[Querying a DataFrame]

df['Gold'] > 0

> Afghanistan (AFG)                               False
> Algeria (ALG)                                    True
> Argentina (ARG)                                  True
> ...
> Zimbabwe (ZIM) [ZIM]                             True
> Mixed team (ZZX) [ZZX]                           True
> Totals                                           True
> Name: Gold, dtype: bool

only_gold = df.where(df['Gold'] > 0)
only_gold.head()

> # Summer  Gold  Silver  Bronze  Total # Winter  Gold.1  Silver.1  Bronze.1  Total.1 # Games Gold.2  Silver.2  Bronze.2  Combined total
> Afghanistan (AFG) NaN NaN NaN NaN NaN NaN NaN NaN NaN NaN NaN NaN NaN NaN NaN
> Algeria (ALG) 12.0  5.0 2.0 8.0 15.0  3.0 0.0 0.0 0.0 0.0 15.0  5.0 2.0 8.0 15.0
> Argentina (ARG) 23.0  18.0  24.0  28.0  70.0  18.0  0.0 0.0 0.0 0.0 41.0  18.0  24.0  28.0  70.0
> Armenia (ARM) 5.0 1.0 2.0 9.0 12.0  6.0 0.0 0.0 0.0 0.0 11.0  1.0 2.0 9.0 12.0
> Australasia (ANZ) [ANZ] 2.0 3.0 4.0 5.0 12.0  0.0 0.0 0.0 0.0 0.0 2.0 3.0 4.0 5.0 12.0

# Only returns rows where Gold > 0

only_gold['Gold'].count()

> 100

# Returns Boolean mask of all rows, where true equals Gold > 0

df['Gold'].count()

> 147

only_gold = only_gold.dropna()
only_gold.head()

> # Summer  Gold  Silver  Bronze  Total # Winter  Gold.1  Silver.1  Bronze.1  Total.1 # Games Gold.2  Silver.2  Bronze.2  Combined total
> Algeria (ALG) 12.0  5.0 2.0 8.0 15.0  3.0 0.0 0.0 0.0 0.0 15.0  5.0 2.0 8.0 15.0
> Argentina (ARG) 23.0  18.0  24.0  28.0  70.0  18.0  0.0 0.0 0.0 0.0 41.0  18.0  24.0  28.0  70.0
> Armenia (ARM) 5.0 1.0 2.0 9.0 12.0  6.0 0.0 0.0 0.0 0.0 11.0  1.0 2.0 9.0 12.0
> Australasia (ANZ) [ANZ] 2.0 3.0 4.0 5.0 12.0  0.0 0.0 0.0 0.0 0.0 2.0 3.0 4.0 5.0 12.0
> Australia (AUS) [AUS] [Z] 25.0  139.0 152.0 177.0 468.0 18.0  5.0 3.0 4.0 12.0  43.0  144.0 155.0 181.0 480.0

# You do not necessarily need to use the .where() method.
# By using the below syntax you apply where, and automatically drop None / NaN rows.

only_gold = df[df['Gold'] > 0]
only_gold.head()

> # Summer  Gold  Silver  Bronze  Total # Winter  Gold.1  Silver.1  Bronze.1  Total.1 # Games Gold.2  Silver.2  Bronze.2  Combined total
> Algeria (ALG) 12  5 2 8 15  3 0 0 0 0 15  5 2 8 15
> Argentina (ARG) 23  18  24  28  70  18  0 0 0 0 41  18  24  28  70
> Armenia (ARM) 5 1 2 9 12  6 0 0 0 0 11  1 2 9 12
> Australasia (ANZ) [ANZ] 2 3 4 5 12  0 0 0 0 0 2 3 4 5 12
> Australia (AUS) [AUS] [Z] 25  139 152 177 468 18  5 3 4 12  43  144 155 181 480

# You can use boolean operators to chain conditions:

len(df[(df['Gold'] > 0) | (df['Gold.1'] > 0)])

> 101

df[(df['Gold.1'] > 0) & (df['Gold'] == 0)]

> # Summer  Gold  Silver  Bronze  Total # Winter  Gold.1  Silver.1  Bronze.1  Total.1 # Games Gold.2  Silver.2  Bronze.2  Combined total
> Liechtenstein (LIE) 16  0 0 0 0 18  2 2 5 9 34  2 2 5 9

[Indexing Dataframes]

df.head()

> # Summer  Gold  Silver  Bronze  Total # Winter  Gold.1  Silver.1  Bronze.1  Total.1 # Games Gold.2  Silver.2  Bronze.2  Combined total
> Afghanistan (AFG) 13  0 0 2 2 0 0 0 0 0 13  0 0 2 2
> Algeria (ALG) 12  5 2 8 15  3 0 0 0 0 15  5 2 8 15
> Argentina (ARG) 23  18  24  28  70  18  0 0 0 0 41  18  24  28  70
> Armenia (ARM) 5 1 2 9 12  6 0 0 0 0 11  1 2 9 12
> Australasia (ANZ) [ANZ] 2 3 4 5 12  0 0 0 0 0 2 3 4 5 12

df['country'] = df.index
df = df.set_index('Gold')
df.head()

> # Summer  Silver  Bronze  Total # Winter  Gold.1  Silver.1  Bronze.1  Total.1 # Games Gold.2  Silver.2  Bronze.2  Combined total  country
> Gold
> 0 13  0 2 2 0 0 0 0 0 13  0 0 2 2 Afghanistan (AFG)
> 5 12  2 8 15  3 0 0 0 0 15  5 2 8 15  Algeria (ALG)
> 18  23  24  28  70  18  0 0 0 0 41  18  24  28  70  Argentina (ARG)
> 1 5 2 9 12  6 0 0 0 0 11  1 2 9 12  Armenia (ARM)
> 3 2 4 5 12  0 0 0 0 0 2 3 4 5 12  Australasia (ANZ) [ANZ]

df = df.reset_index()
df.head()

> Gold  # Summer  Silver  Bronze  Total # Winter  Gold.1  Silver.1  Bronze.1  Total.1 # Games Gold.2  Silver.2  Bronze.2  Combined total  country
> 0 0 13  0 2 2 0 0 0 0 0 13  0 0 2 2 Afghanistan (AFG)
> 1 5 12  2 8 15  3 0 0 0 0 15  5 2 8 15  Algeria (ALG)
> 2 18  23  24  28  70  18  0 0 0 0 41  18  24  28  70  Argentina (ARG)
> 3 1 5 2 9 12  6 0 0 0 0 11  1 2 9 12  Armenia (ARM)
> 4 3 2 4 5 12  0 0 0 0 0 2 3 4 5 12  Australasia (ANZ) [ANZ]

df = pd.read_csv('census.csv')
df.head()

> SUMLEV  REGION  DIVISION  STATE COUNTY  STNAME  CTYNAME CENSUS2010POP ESTIMATESBASE2010 POPESTIMATE2010 ...
> 0 40  3 6 1 0 Alabama Alabama 4779736 4780127 4785161 ...
> 1 50  3 6 1 1 Alabama Autauga County  54571 54571 54660 ...
> 2 50  3 6 1 3 Alabama Baldwin County  182265  182265  183193  ...
> 3 50  3 6 1 5 Alabama Barbour County  27457 27457 27341 ...
> 4 50  3 6 1 7 Alabama Bibb County 22915 22919 22861 ...
> 5 rows × 100 columns

df['SUMLEV'].unique()

> array([40, 50])

df = df[df['SUMLEV'] == 50]
df.head()

> SUMLEV  REGION  DIVISION  STATE COUNTY  STNAME  CTYNAME CENSUS2010POP ESTIMATESBASE2010 POPESTIMATE2010 ...
> 1 50  3 6 1 1 Alabama Autauga County  54571 54571 54660 ...
> 2 50  3 6 1 3 Alabama Baldwin County  182265  182265  183193  ...
> 3 50  3 6 1 5 Alabama Barbour County  27457 27457 27341 ...
> 4 50  3 6 1 7 Alabama Bibb County 22915 22919 22861 ...
> 5 50  3 6 1 9 Alabama Blount County 57322 57322 57373 ...
> 5 rows × 100 columns

columns_to_keep = ['STNAME',
                   'CTYNAME',
                   'BIRTHS2010',
                   'BIRTHS2011',
                   'BIRTHS2012',
                   'BIRTHS2013',
                   'BIRTHS2014',
                   'BIRTHS2015',
                   'POPESTIMATE2010',
                   'POPESTIMATE2011',
                   'POPESTIMATE2012',
                   'POPESTIMATE2013',
                   'POPESTIMATE2014',
                   'POPESTIMATE2015']

df = df[columns_to_keep]
df.head()

> STNAME  CTYNAME BIRTHS2010  BIRTHS2011  BIRTHS2012  BIRTHS2013  BIRTHS2014  BIRTHS2015  POPESTIMATE2010 POPESTIMATE2011 POPESTIMATE2012 POPESTIMATE2013 POPESTIMATE2014 POPESTIMATE2015
> 1 Alabama Autauga County  151 636 615 574 623 600 54660 55253 55175 55038 55290 55347
> 2 Alabama Baldwin County  517 2187  2092  2160  2186  2240  183193  186659  190396  195126  199713  203709
> 3 Alabama Barbour County  70  335 300 283 260 269 27341 27226 27159 26973 26815 26489
> 4 Alabama Bibb County 44  266 245 259 247 253 22861 22733 22642 22512 22549 22583
> 5 Alabama Blount County 183 744 710 646 618 603 57373 57711 57776 57734 57658 57673

df = df.set_index(['STNAME', 'CTYNAME'])
df.head()

> BIRTHS2010  BIRTHS2011  BIRTHS2012  BIRTHS2013  BIRTHS2014  BIRTHS2015  POPESTIMATE2010 POPESTIMATE2011 POPESTIMATE2012 POPESTIMATE2013 POPESTIMATE2014 POPESTIMATE2015
> STNAME  CTYNAME
> Alabama Autauga County  151 636 615 574 623 600 54660 55253 55175 55038 55290 55347
> Baldwin County  517 2187  2092  2160  2186  2240  183193  186659  190396  195126  199713  203709
> Barbour County  70  335 300 283 260 269 27341 27226 27159 26973 26815 26489
> Bibb County 44  266 245 259 247 253 22861 22733 22642 22512 22549 22583
> Blount County 183 744 710 646 618 603 57373 57711 57776 57734 57658 57673

df.loc['Michigan', 'Washtenaw County']

> BIRTHS2010            977
> BIRTHS2011           3826
> BIRTHS2012           3780
> BIRTHS2013           3662
> BIRTHS2014           3683
> BIRTHS2015           3709
> POPESTIMATE2010    345563
> POPESTIMATE2011    349048
> POPESTIMATE2012    351213
> POPESTIMATE2013    354289
> POPESTIMATE2014    357029
> POPESTIMATE2015    358880
> Name: (Michigan, Washtenaw County), dtype: int64

df.loc[[ ('Michigan', 'Washtenaw County'),
         ('Michigan', 'Wayne County') ]]

> BIRTHS2010  BIRTHS2011  BIRTHS2012  BIRTHS2013  BIRTHS2014  BIRTHS2015  POPESTIMATE2010 POPESTIMATE2011 POPESTIMATE2012 POPESTIMATE2013 POPESTIMATE2014 POPESTIMATE2015
> STNAME  CTYNAME
> Michigan  Washtenaw County  977 3826  3780  3662  3683  3709  345563  349048  351213  354289  357029  358880
> Wayne County  5918  23819 23270 23377 23607 23586 1815199 1801273 1792514 1775713 1766008 1759335

[Missing values]

df = pd.read_csv('log.csv')
df

> time  user  video playback position paused  volume
> 0 1469974424  cheryl  intro.html  5 False 10.0
> 1 1469974454  cheryl  intro.html  6 NaN NaN
> 2 1469974544  cheryl  intro.html  9 NaN NaN
> ...
> 30  1469974664  cheryl  intro.html  13  NaN NaN
> 31  1469977694  bob intro.html  1 NaN NaN
> 32  1469977724  bob intro.html  1 NaN NaN

df.fillna?

df = df.set_index('time')
df = df.sort_index()
df

> user  video playback position paused  volume
> time
> 1469974424  cheryl  intro.html  5 False 10.0
> 1469974424  sue advanced.html 23  False 10.0
> 1469974454  cheryl  intro.html  6 NaN NaN
> ...
> 1469977664  bob intro.html  1 NaN NaN
> 1469977694  bob intro.html  1 NaN NaN
> 1469977724  bob intro.html  1 NaN NaN

df = df.reset_index()
df = df.set_index(['time', 'user'])
df

> video playback position paused  volume
> time  user
> 1469974424  cheryl  intro.html  5 False 10.0
>             sue advanced.html 23  False 10.0
> 1469974454  cheryl  intro.html  6 NaN NaN
>             sue advanced.html 24  NaN NaN
> 1469974484  cheryl  intro.html  7 NaN NaN
> ...
> 1469977664  bob intro.html  1 NaN NaN
> 1469977694  bob intro.html  1 NaN NaN
> 1469977724  bob intro.html  1 NaN NaN

# ffil == forward fill (copy the previous value), bfil would take the next value

df = df.fillna(method='ffill')
df.head()

> video playback position paused  volume
> time  user
> 1469974424  cheryl  intro.html  5 False 10.0
>             sue advanced.html 23  False 10.0
> 1469974454  cheryl  intro.html  6 False 10.0
>             sue advanced.html 24  False 10.0
> 1469974484  cheryl  intro.html  7 False 10.0

[Wk2 - Lecture Quizzes]

# For the purchase records from the pet store, how would you get a list of all items which had been
# purchased (regardless of where they might have been purchased, or by whom)?

import pandas as pd

purchase_1 = pd.Series({'Name': 'Chris',
                        'Item Purchased': 'Dog Food',
                        'Cost': 22.50})
purchase_2 = pd.Series({'Name': 'Kevyn',
                        'Item Purchased': 'Kitty Litter',
                        'Cost': 2.50})
purchase_3 = pd.Series({'Name': 'Vinod',
                        'Item Purchased': 'Bird Seed',
                        'Cost': 5.00})

df = pd.DataFrame([purchase_1, purchase_2, purchase_3], index=['Store 1', 'Store 1', 'Store 2'])

df['Item Purchased'] # Correct

# For the purchase records from the pet store, how would you update the DataFrame, applying a discount
# of 20% across all the values in the 'Cost' column?

purchase_1 = pd.Series({'Name': 'Chris',
                        'Item Purchased': 'Dog Food',
                        'Cost': 22.50})
purchase_2 = pd.Series({'Name': 'Kevyn',
                        'Item Purchased': 'Kitty Litter',
                        'Cost': 2.50})
purchase_3 = pd.Series({'Name': 'Vinod',
                        'Item Purchased': 'Bird Seed',
                        'Cost': 5.00})

df = pd.DataFrame([purchase_1, purchase_2, purchase_3], index=['Store 1', 'Store 1', 'Store 2'])

# Your answer here

df['Cost'] = df['Cost'] * 0.8 # Correct

# Or:

df['Cost'] *= 0.8
print(df)

# Write a query to return all of the names of people who bought products worth more than $3.00.

purchase_1 = pd.Series({'Name': 'Chris',
                        'Item Purchased': 'Dog Food',
                        'Cost': 22.50})
purchase_2 = pd.Series({'Name': 'Kevyn',
                        'Item Purchased': 'Kitty Litter',
                        'Cost': 2.50})
purchase_3 = pd.Series({'Name': 'Vinod',
                        'Item Purchased': 'Bird Seed',
                        'Cost': 5.00})

df = pd.DataFrame([purchase_1, purchase_2, purchase_3], index=['Store 1', 'Store 1', 'Store 2'])

# Instantly reduce to only records where Costs > 3

df[df['Cost'] > 3.00]['Name']

# Or applying a Boolean mask on df['Name']

df['Name'][df['Cost'] > 3]

# Re-index the purchase records DataFrame to be indexed hierarchically, first by store, then by person.
# Name these indexes 'Location' and 'Name'. Then add a new entry to it with the value of:
# Name: 'Kevyn', Item Purchased: 'Kitty Food', Cost: 3.00, Location: 'Store 2'

purchase_1 = pd.Series({'Name': 'Chris',
                        'Item Purchased': 'Dog Food',
                        'Cost': 22.50})
purchase_2 = pd.Series({'Name': 'Kevyn',
                        'Item Purchased': 'Kitty Litter',
                        'Cost': 2.50})
purchase_3 = pd.Series({'Name': 'Vinod',
                        'Item Purchased': 'Bird Seed',
                        'Cost': 5.00})

df = pd.DataFrame([purchase_1, purchase_2, purchase_3], index=['Store 1', 'Store 1', 'Store 2'])

# Your answer here

df['Location'] = df.index
df = df.set_index(['Location', 'Name'])
df[('Store 2', 'Kevyn')]['Item Purchased', 'Cost'] = ['Kitty Food', 3.00]
df

df = df.set_index([df.index, 'Name'])
df.index.names = ['Location', 'Name']
df = df.append(pd.Series(data={'Cost': 3.00, 'Item Purchased': 'Kitty Food'}, name=('Store 2', 'Kevyn')))
df

-- Week 3

[Merging Dataframes]

import pandas as pd
​
df = pd.DataFrame([{'Name': 'Chris', 'Item Purchased': 'Sponge', 'Cost': 22.50},
                   {'Name': 'Kevyn', 'Item Purchased': 'Kitty Litter', 'Cost': 2.50},
                   {'Name': 'Filip', 'Item Purchased': 'Spoon', 'Cost': 5.00}],
                  index=['Store 1', 'Store 1', 'Store 2'])
df

> Cost  Item Purchased  Name
> Store 1 22.5  Sponge  Chris
> Store 1 2.5 Kitty Litter  Kevyn
> Store 2 5.0 Spoon Filip

df['Date'] = ['December 1', 'January 1', 'mid-May']
df

> Cost  Item Purchased  Name  Date
> Store 1 22.5  Sponge  Chris December 1
> Store 1 2.5 Kitty Litter  Kevyn January 1
> Store 2 5.0 Spoon Filip mid-May

df['Delivered'] = True
df

> Cost  Item Purchased  Name  Date  Delivered
> Store 1 22.5  Sponge  Chris December 1  True
> Store 1 2.5 Kitty Litter  Kevyn January 1 True
> Store 2 5.0 Spoon Filip mid-May True

# We specifically declare the second Feedback value to be None

df['Feedback'] = ['Positive', None, 'Negative']
df

> Cost  Item Purchased  Name  Date  Delivered Feedback
> Store 1 22.5  Sponge  Chris December 1  True  Positive
> Store 1 2.5 Kitty Litter  Kevyn January 1 True  None
> Store 2 5.0 Spoon Filip mid-May True  Negative

adf = df.reset_index()

# Date in row 1 is now automatically set to NaN

adf['Date'] = pd.Series({0: 'December 1', 2: 'mid-May'})
adf

> index Cost  Item Purchased  Name  Date  Delivered Feedback
> 0 Store 1 22.5  Sponge  Chris December 1  True  Positive
> 1 Store 1 2.5 Kitty Litter  Kevyn NaN True  None
> 2 Store 2 5.0 Spoon Filip mid-May True  Negative

staff_df = pd.DataFrame([{'Name': 'Kelly', 'Role': 'Director of HR'},
                         {'Name': 'Sally', 'Role': 'Course liasion'},
                         {'Name': 'James', 'Role': 'Grader'}])
staff_df = staff_df.set_index('Name')
student_df = pd.DataFrame([{'Name': 'James', 'School': 'Business'},
                           {'Name': 'Mike', 'School': 'Law'},
                           {'Name': 'Sally', 'School': 'Engineering'}])
student_df = student_df.set_index('Name')

print(staff_df.head())
print()
print(student_df.head())

>                  Role
> Name
> Kelly  Director of HR
> Sally  Course liasion
> James          Grader

>             School
> Name
> James     Business
> Mike           Law
> Sally  Engineering

# Missing values are NaN

pd.merge(staff_df, student_df, how = 'outer', left_index = True, right_index = True)

> Role  School
> Name
> James Grader  Business
> Kelly Director of HR  NaN
> Mike  NaN Law
> Sally Course liasion  Engineering

pd.merge(staff_df, student_df, how = 'inner', left_index = True, right_index = True)

> Role  School
> Name
> James Grader  Business
> Sally Course liasion  Engineering

pd.merge(staff_df, student_df, how = 'left', left_index = True, right_index = True)

> Role  School
> Name
> Kelly Director of HR  NaN
> Sally Course liasion  Engineering
> James Grader  Business

pd.merge(staff_df, student_df, how = 'right', left_index = True, right_index = True)

> Role  School
> Name
> James Grader  Business
> Mike  NaN Law
> Sally Course liasion  Engineering

staff_df = staff_df.reset_index()
student_df = student_df.reset_index()
pd.merge(staff_df, student_df, how = 'left', left_on = 'Name', right_on = 'Name')

> Name  Role  School
> 0 Kelly Director of HR  NaN
> 1 Sally Course liasion  Engineering
> 2 James Grader  Business

staff_df = pd.DataFrame([{'Name': 'Kelly', 'Role': 'Director of HR', 'Location': 'State Street'},
                         {'Name': 'Sally', 'Role': 'Course liasion', 'Location': 'Washington Avenue'},
                         {'Name': 'James', 'Role': 'Grader', 'Location': 'Washington Avenue'}])
student_df = pd.DataFrame([{'Name': 'James', 'School': 'Business', 'Location': '1024 Billiard Avenue'},
                           {'Name': 'Mike', 'School': 'Law', 'Location': 'Fraternity House #22'},
                           {'Name': 'Sally', 'School': 'Engineering', 'Location': '512 Wilson Crescent'}])

# Join on column name, instead of index

pd.merge(staff_df, student_df, how = 'left', left_on = 'Name', right_on = 'Name')

> Location_x  Name  Role  Location_y  School
> 0 State Street  Kelly Director of HR  NaN NaN
> 1 Washington Avenue Sally Course liasion  512 Wilson Crescent Engineering
> 2 Washington Avenue James Grader  1024 Billiard Avenue  Business

# Join on multiple keys!
staff_df = pd.DataFrame([{'First Name': 'Kelly', 'Last Name': 'Desjardins', 'Role': 'Director of HR'},
                         {'First Name': 'Sally', 'Last Name': 'Brooks', 'Role': 'Course liasion'},
                         {'First Name': 'James', 'Last Name': 'Wilde', 'Role': 'Grader'}])
student_df = pd.DataFrame([{'First Name': 'James', 'Last Name': 'Hammond', 'School': 'Business'},
                           {'First Name': 'Mike', 'Last Name': 'Smith', 'School': 'Law'},
                           {'First Name': 'Sally', 'Last Name': 'Brooks', 'School': 'Engineering'}])

pd.merge(staff_df, student_df, how = 'inner', left_on = ['First Name','Last Name'], right_on = ['First Name','Last Name'])

> First Name  Last Name Role  School
> 0 Sally Brooks  Course liasion  Engineering

[Idiomatic Pandas: Making Code Pandorable]

import pandas as pd
df = pd.read_csv('census.csv')
df

> SUMLEV  REGION  DIVISION  STATE COUNTY  STNAME  CTYNAME CENSUS2010POP ESTIMATESBASE2010 POPESTIMATE2010 ...
> 0 40  3 6 1 0 Alabama Alabama 4779736 4780127 4785161 ...
> 1 50  3 6 1 1 Alabama Autauga County  54571 54571 54660 ...
> ...
> 3191  50  4 8 56  43  Wyoming Washakie County 8533  8533  8545 ...
> 3192  50  4 8 56  45  Wyoming Weston County 7208  7208  7181 ...
> 3193 rows × 100 columns

# Method Chaining
# In python if you write a statement within brackets you can span it across multiple lines

(df.where(df['SUMLEV'] == 50)
    .dropna()
    .set_index(['STNAME', 'CTYNAME'])
    .rename(columns = {'ESTIMATESBASE2010': 'Estimates Base 2010'}))

> SUMLEV  REGION  DIVISION  STATE COUNTY  CENSUS2010POP Estimates Base 2010 POPESTIMATE2010 POPESTIMATE2011 POPESTIMATE2012 ...
> STNAME  CTYNAME
> Alabama Autauga County  50.0  3.0 6.0 1.0 1.0 54571.0 54571.0 54660.0 55253.0 55175.0 ...
> Baldwin County  50.0  3.0 6.0 1.0 3.0 182265.0  182265.0  183193.0  186659.0  190396.0 ...
> ...
> Washakie County 50.0  4.0 8.0 56.0  43.0  8533.0  8533.0  8545.0  8469.0  8443.0 ...
> Weston County 50.0  4.0 8.0 56.0  45.0  7208.0  7208.0  7181.0  7114.0  7065.0 ...
> 3142 rows × 98 columns

# Similar code but not pandorable
df = df[df['SUMLEV'] == 50]
df.set_index(['STNAME', 'CTYNAME'], inplace = True)
df.rename(columns = {'ESTIMATESBASE2010': 'Estimates Base 2010'})

> SUMLEV  REGION  DIVISION  STATE COUNTY  CENSUS2010POP Estimates Base 2010 POPESTIMATE2010 POPESTIMATE2011 POPESTIMATE2012 ...
> STNAME  CTYNAME
> Alabama Autauga County  50  3 6 1 1 54571 54571 54660 55253 55175 ...
> Baldwin County  50  3 6 1 3 182265  182265  183193  186659  190396 ...
> ...
> Washakie County 50  4 8 56  43  8533  8533  8545  8469  8443 ...
> Weston County 50  4 8 56  45  7208  7208  7181  7114  7065 ...
> 3142 rows × 98 columns

import numpy as np

def min_max(row):
    data = row[['POPESTIMATE2010',
                'POPESTIMATE2011',
                'POPESTIMATE2012',
                'POPESTIMATE2013',
                'POPESTIMATE2014',
                'POPESTIMATE2015']]
    # return a series with a min and max column
    return pd.Series({'min': np.min(data), 'max': np.max(data)})

# apply over rows -> axis = 1
df.apply(min_max, axis = 1)

> max min
> STNAME  CTYNAME
> Alabama Autauga County  55347.0 54660.0
> Baldwin County  203709.0  183193.0
> Washakie County 8545.0  8316.0
> Weston County 7234.0  7065.0
> 3142 rows × 2 columns

import numpy as np

def min_max(row):
    data = row[['POPESTIMATE2010',
                'POPESTIMATE2011',
                'POPESTIMATE2012',
                'POPESTIMATE2013',
                'POPESTIMATE2014',
                'POPESTIMATE2015']]
    # append a max and min column to row (df)
    row['max'] = np.max(data)
    row['min'] = np.min(data)
    return row

df.apply(min_max, axis = 1)

> SUMLEV  REGION  DIVISION  STATE COUNTY  CENSUS2010POP ESTIMATESBASE2010 POPESTIMATE2010 POPESTIMATE2011 POPESTIMATE2012 ... RDOMESTICMIG2013  RDOMESTICMIG2014  RDOMESTICMIG2015  RNETMIG2011 RNETMIG2012 RNETMIG2013 RNETMIG2014 RNETMIG2015 max min
> STNAME  CTYNAME
> Alabama Autauga County  50.0  3.0 6.0 1.0 1.0 54571.0 54571.0 54660.0 55253.0 55175.0 ... -3.012349 2.265971  -2.530799 7.606016  -2.626146 -2.722002 2.592270  -2.187333 55347.0 54660.0
> Baldwin County  50.0  3.0 6.0 1.0 3.0 182265.0  182265.0  183193.0  186659.0  190396.0  ... 21.845705 19.243287 17.197872 15.844176 18.559627 22.727626 20.317142 18.293499 203709.0  183193.0
> ...
> Washakie County 50.0  4.0 8.0 56.0  43.0  8533.0  8533.0  8545.0  8469.0  8443.0  ... -2.013502 -17.781491  1.682288  -11.990126  -1.182592 -2.250385 -18.020168  1.441961  8545.0  8316.0
> Weston County 50.0  4.0 8.0 56.0  45.0  7208.0  7208.0  7181.0  7114.0  7065.0  ... 12.372583 1.533635  6.935294  -12.032179  -8.040059 12.372583 1.533635  6.935294  7234.0  7065.0
> 3142 rows × 100 columns

rows = ['POPESTIMATE2010',
        'POPESTIMATE2011',
        'POPESTIMATE2012',
        'POPESTIMATE2013',
        'POPESTIMATE2014',
        'POPESTIMATE2015']

df.apply(lambda x: np.max(x[rows]), axis = 1)

> STNAME     CTYNAME
> Alabama    Autauga County         55347.0
>            Baldwin County        203709.0
> ...
> Wyoming    Albany County          37956.0
>            Big Horn County        12022.0
> dtype: float64

[Group by]

import pandas as pd
import numpy as np

df = pd.read_csv('census.csv')
df = df[df['SUMLEV'] == 50]
df

> SUMLEV  REGION  DIVISION  STATE COUNTY  STNAME  CTYNAME CENSUS2010POP ESTIMATESBASE2010 POPESTIMATE2010 ...
> 1 50  3 6 1 1 Alabama Autauga County  54571 54571 54660 ...
> 2 50  3 6 1 3 Alabama Baldwin County  182265  182265  183193 ...
> 3191  50  4 8 56  43  Wyoming Washakie County 8533  8533  8545 ...
> 3192  50  4 8 56  45  Wyoming Weston County 7208  7208  7181 ...
> 3142 rows × 100 columns

%%timeit -n 10
for state in df['STNAME'].unique():
    avg = np.average(df.where(df['STNAME'] == state).dropna()['CENSUS2010POP'])
    print('Counties in state ' + state + ' have an average population of ' + str(avg))

> Counties in state Alabama have an average population of 71339.3432836
> Counties in state Alaska have an average population of 24490.7241379
> ...
> Counties in state Wisconsin have an average population of 78985.9166667
> Counties in state Wyoming have an average population of 24505.4782609
> 10 loops, best of 3: 1.1 s per loop

%%timeit -n 10
# Also possible to group by a list of columns
for group, frame in df.groupby('STNAME'):
    avg = np.average(frame['CENSUS2010POP'])
    print('Counties in state ' + group + ' have an average population of ' + str(avg))

> Counties in state Alabama have an average population of 71339.3432836
> Counties in state Alaska have an average population of 24490.7241379
> ...
> Counties in state Wisconsin have an average population of 78985.9166667
> Counties in state Wyoming have an average population of 24505.4782609
# Much better performance
> 10 loops, best of 3: 24.5 ms per loop

df.head()

> SUMLEV  REGION  DIVISION  STATE COUNTY  STNAME  CTYNAME CENSUS2010POP ESTIMATESBASE2010 POPESTIMATE2010 ...
> 1 50  3 6 1 1 Alabama Autauga County  54571 54571 54660 ...
> 2 50  3 6 1 3 Alabama Baldwin County  182265  182265  183193 ...
> ...
> 4 50  3 6 1 7 Alabama Bibb County 22915 22919 22861 ...
> 5 50  3 6 1 9 Alabama Blount County 57322 57322 57373 ...
> 5 rows × 100 columns

df = df.set_index('STNAME')
​
def fun(item):
    if item[0] < 'M':
        return 0
    if item[0] < 'Q':
        return 1
    return 2
​
# Grouping by a hash function: fun is applied over the index!
for group, frame in df.groupby(fun):
    print('There are ' + str(len(frame)) + ' records in group ' + str(group) + ' for processing.')
​
> There are 1177 records in group 0 for processing.
> There are 1134 records in group 1 for processing.
> There are 831 records in group 2 for processing.

df = pd.read_csv('census.csv')
df = df[df['SUMLEV'] == 50]

df.groupby('STNAME').agg({'CENSUS2010POP': np.average})

> CENSUS2010POP
> STNAME
> Alabama 71339.343284
> Alaska  24490.724138
> ...
> Wisconsin 78985.916667
> Wyoming 24505.478261

# Series vs DF groupby object

print(type(df.groupby(level = 0)['POPESTIMATE2010','POPESTIMATE2011']))
print(type(df.groupby(level = 0)['POPESTIMATE2010']))

> <class 'pandas.core.groupby.DataFrameGroupBy'>
> <class 'pandas.core.groupby.SeriesGroupBy'>

# level = 0 == group by index / first column (?)
(df.set_index('STNAME').groupby(level = 0)['CENSUS2010POP']
    # return 'avg' column using np.average, etc.
    .agg({'avg': np.average, 'sum': np.sum}))

> avg sum
> STNAME
> Alabama 71339.343284  4779736
> Alaska  24490.724138  710231
> ...
> Wisconsin 78985.916667  5686986
> Wyoming 24505.478261  563626

(df.set_index('STNAME').groupby(level = 0)['POPESTIMATE2010','POPESTIMATE2011']
    .agg({'avg': np.average, 'sum': np.sum}))

>               avg                             sum
> POPESTIMATE2010 POPESTIMATE2011 POPESTIMATE2010 POPESTIMATE2011
> STNAME
> Alabama 71420.313433  71658.328358  4785161 4801108
> Alaska  24621.413793  24921.379310  714021  722720
> ...
> Wisconsin 79030.611111  79301.666667  5690204 5709720
> Wyoming 24544.173913  24685.565217  564516  567768

# Now you do not get the hierarchical structure as above, but it applies
# average only to POPESTIMATE2010 and sum only to POPESTIMATE 2011
(df.set_index('STNAME').groupby(level = 0)['POPESTIMATE2010','POPESTIMATE2011']
    .agg({'POPESTIMATE2010': np.average, 'POPESTIMATE2011': np.sum}))

> POPESTIMATE2011 POPESTIMATE2010
> STNAME
> Alabama 4801108 71420.313433
> Alaska  722720  24621.413793
> ...
> Wisconsin 5709720 79030.611111
> Wyoming 567768  24544.173913

[Scales]

# - Ratio scale
#   + Units are equally spaced
#   + Mathematical operations of +-/* are all valid
#   + E.g. height and weight
# - Interval scale
#   + Units are equally spaced, but there is no true zero
# - Ordinal scale
#   + The order of the units is important, but not evenly spaced
#   + Letter grades such as A+, A are good examples
# - Nominal scale
#   + Categories of data, but the categories have no order with respect to one another
#   + E.g. teams of a sport

df = pd.DataFrame(['A+', 'A', 'A-', 'B+', 'B', 'B-', 'C+', 'C', 'C-', 'D+', 'D'],
                  index = ['excellent', 'excellent', 'excellent', 'good', 'good', 'good', 'ok', 'ok', 'ok', 'poor', 'poor'])
df.rename(columns={0: 'Grades'}, inplace = True)
df

> Grades
> excellent A+
> excellent A
> ...
> poor  D+
> poor  D

df['Grades'].astype('category').head()

> excellent    A+
> excellent     A
> excellent    A-
> good         B+
> good          B
> Name: Grades, dtype: category
> Categories (11, object): [A, A+, A-, B, ..., C+, C-, D, D+]

grades = df['Grades'].astype('category',
                             categories = ['D', 'D+', 'C-', 'C', 'C+', 'B-', 'B', 'B+', 'A-', 'A', 'A+'],
                             ordered = True)
grades.head()

> excellent    A+
> excellent     A
> excellent    A-
> good         B+
> good          B
> Name: Grades, dtype: category
> Categories (11, object): [D < D+ < C- < C ... B+ < A- < A < A+]

# Else lexographically C- would also be greater than C

grades > 'C'

> excellent     True
> ...
> poor         False
> Name: Grades, dtype: bool

df = pd.read_csv('census.csv')
df = df[df['SUMLEV'] == 50]
df = df.set_index('STNAME').groupby(level=0)['CENSUS2010POP'].agg({'avg': np.average})

# Cut into 10 categories

pd.cut(df['avg'], 10)

# > STNAME
# > Alabama                  (11706.0871, 75333.413]
# > Alaska                   (11706.0871, 75333.413]
# > ...
# > Wisconsin                (75333.413, 138330.766]
# > Wyoming                  (11706.0871, 75333.413]
# > Name: avg, dtype: category
# > Categories (10, object): [(11706.0871, 75333.413] < (75333.413, 138330.766] < (138330.766, 201328.118] < (201328.118, 264325.471] ...

[Pivot Tables]

# http://open.canada.ca/data/en/dataset/98f1a129-f628-4ce4-b24d-6f16bf24dd64

df = pd.read_csv('cars.csv')
df.head()

> YEAR  Make  Model Size  (kW)  Unnamed: 5  TYPE  CITY (kWh/100 km) HWY (kWh/100 km)  COMB (kWh/100 km) CITY (Le/100 km)  HWY (Le/100 km) COMB (Le/100 km)  (g/km)  RATING  (km)  TIME (h)
> 0 2012  MITSUBISHI  i-MiEV  SUBCOMPACT  49  A1  B 16.9  21.4  18.7  1.9 2.4 2.1 0 n/a 100 7
> 1 2012  NISSAN  LEAF  MID-SIZE  80  A1  B 19.3  23.0  21.1  2.2 2.6 2.4 0 n/a 117 7
> ...
> 4 2013  NISSAN  LEAF  MID-SIZE  80  A1  B 19.3  23.0  21.1  2.2 2.6 2.4 0 n/a 117 7

df.pivot_table(values='(kW)', index='YEAR', columns='Make', aggfunc=np.mean)

> Make  BMW CHEVROLET FORD  KIA MITSUBISHI  NISSAN  SMART TESLA
> YEAR
> 2012  NaN NaN NaN NaN 49.0  80.0  NaN NaN
> ...
> 2016  125.0 104.0 107.0 81.0  49.0  80.0  35.0  409.700000

df.pivot_table(values='(kW)', index='YEAR', columns='Make', aggfunc=[np.mean,np.min], margins=True)

>                             mean                                                   amin
> Make  BMW CHEVROLET FORD  KIA MITSUBISHI  NISSAN  SMART TESLA All BMW CHEVROLET FORD  KIA MITSUBISHI  NISSAN  SMART TESLA All
> YEAR
> 2012  NaN NaN NaN NaN 49.0  80.0  NaN NaN 64.500000 NaN NaN NaN NaN 49.0  80.0  NaN NaN 49.0
> ...
> 2016  125.0 104.0 107.0 81.0  49.0  80.0  35.0  409.700000  252.263158  125.0 104.0 107.0 81.0  49.0  80.0  35.0  283.0 35.0
> All 125.0 104.0 107.0 81.0  49.0  80.0  35.0  345.478261  190.622642  125.0 104.0 107.0 81.0  49.0  80.0  35.0  225.0 35.0

[Date Functionality in Pandas]

import pandas as pd
import numpy as np

[Timestamp]

pd.Timestamp('9/1/2016 10:05AM')

> Timestamp('2016-09-01 10:05:00')

[Period]

pd.Period('1/2016')

> Period('2016-01', 'M')

pd.Period('3/5/2016')

> Period('2016-03-05', 'D')

[DatetimeIndex]

t1 = pd.Series(list('abc'), [pd.Timestamp('2016-09-01'), pd.Timestamp('2016-09-02'), pd.Timestamp('2016-09-03')])
t1

> 2016-09-01    a
> 2016-09-02    b
> 2016-09-03    c
> dtype: object

type(t1.index)

> pandas.tseries.index.DatetimeIndex

[PeriodIndex]

t2 = pd.Series(list('def'), [pd.Period('2016-09'), pd.Period('2016-10'), pd.Period('2016-11')])
t2

> 2016-09    d
> 2016-10    e
> 2016-11    f
> Freq: M, dtype: object

type(t2.index)

> pandas.tseries.period.PeriodIndex

[Converting to Datetime]

d1 = ['2 June 2013', 'Aug 29, 2014', '2015-06-26', '7/12/16']
ts3 = pd.DataFrame(np.random.randint(10, 100, (4,2)), index=d1, columns=list('ab'))
ts3

> a b
> 2 June 2013 32  50
> Aug 29, 2014  86  86
> 2015-06-26  79  94
> 7/12/16 55  69

ts3.index = pd.to_datetime(ts3.index)
ts3

# All dates now in the same format
> a b
> 2013-06-02  32  50
> 2014-08-29  86  86
> 2015-06-26  79  94
> 016-07-12  55  69

# Read European time
pd.to_datetime('4.7.12', dayfirst = True)

> Timestamp('2012-07-04 00:00:00')

[Timedeltas]

# Timestamp - Timestamp
pd.Timestamp('9/3/2016') - pd.Timestamp('9/1/2016')

> Timedelta('2 days 00:00:00')

# Timestamp + Timedelta
pd.Timestamp('9/2/2016 8:10AM') + pd.Timedelta('12D 3H')

> Timestamp('2016-09-14 11:10:00')

[Working 'with' Dates in a Dataframe]

dates = pd.date_range('10-01-2016', periods = 9, freq = '2W-SUN')
dates

> DatetimeIndex(['2016-10-02', '2016-10-16', '2016-10-30', '2016-11-13',
>                '2016-11-27', '2016-12-11', '2016-12-25', '2017-01-08',
>                '2017-01-22'],
>               dtype='datetime64[ns]', freq='2W-SUN')

df = pd.DataFrame({'Count 1': 100 + np.random.randint(-5, 10, 9).cumsum(),
                  'Count 2': 120 + np.random.randint(-5, 10, 9)}, index=dates)
df

> Count 1 Count 2
> 2016-10-02  108 121
> 2016-10-16  114 123
> ...
> 2017-01-08  107 117
> 2017-01-22  110 118

df.index.weekday_name

array(['Sunday', 'Sunday', 'Sunday', 'Sunday', 'Sunday', 'Sunday',
       'Sunday', 'Sunday', 'Sunday'], dtype = object)

# Get the differences between subsequent days (therefore first row is NaN - cannot compute as there is no prev day)

df.diff()

> Count 1 Count 2
> 2016-10-02  NaN NaN
> 2016-10-16  6.0 2.0
> ...
> 2017-01-08  -2.0  -9.0
> 2017-01-22  3.0 1.0

# Get the mean by month 'M'

df.resample('M').mean()

> Count 1 Count 2
> 2016-10-31  111.666667  120.0
> 2016-11-30  113.000000  119.5
> 2016-12-31  109.000000  122.0
> 2017-01-31  108.500000  117.5

df['2017']

> Count 1 Count 2
> 2017-01-08  107 117
> 2017-01-22  110 118

df['2016-12']

> Count 1 Count 2
> 2016-12-11  109 118
> 2016-12-25  109 126

df['2016-12':]

> Count 1 Count 2
> 2016-12-11  109 118
> 2016-12-25  109 126
> 2017-01-08  107 117
> 2017-01-22  110 118

# Go from bi-weekly to weekly, use forward fill to fill up missing values

df.asfreq('W', method='ffill')

> Count 1 Count 2
> 2016-10-02  108 121
> 2016-10-09  108 121
> ...
> 2017-01-15  107 117
> 2017-01-22  110 118

import matplotlib.pyplot as plt
%matplotlib inline
​
df.plot()

<matplotlib.axes._subplots.AxesSubplot at 0x7f7b84332518>

[Wk 3 - Lecture Quizzes]

answer = pd.merge(products, invoices, how = 'inner', left_index = True, right_on = 'Product ID') # Correct

# Suppose we are working on a DataFrame that holds information on our equipment for an upcoming backpacking trip.
# Can you use method chaining to modify the DataFrame df in one statement to drop any entries where 'Quantity' is 0 and rename the
# column 'Weight' to 'Weight (oz.)'?

print(df.drop(df[df['Quantity'] == 0].index).rename(columns={'Weight': 'Weight (oz.)'}))

# Using apply over grouped data

print(df.groupby('Category').apply(lambda df: sum(df['Weight (oz.)'] * df['Quantity'])))

# Try casting this series to categorical with the ordering Low < Medium < High

d = pd.Series(['Low', 'Low', 'High', 'Medium', 'Low', 'High', 'Low'])
s = d.astype('category', categories = ['Low', 'Medium', 'High'], ordered = True)
print(s > 'Low')

# Suppose we have a series that holds height data for jacket wearers. Use pd.cut to bin this data into 3 bins.

s = pd.Series([168, 180, 174, 190, 170, 185, 179, 181, 175, 169, 182, 177, 180, 171])
pd.cut(s, 3)

# You can also add labels for the sizes [Small < Medium < Large].

pd.cut(s, 3, labels = ['Small', 'Medium', 'Large'])

# Suppose we have a DataFrame with price and ratings for different bikes, broken down by manufacturer and type of bicycle.
# Create a pivot table that shows the mean price and mean rating for every 'Manufacturer' / 'Bike Type' combination.

print(pd.pivot_table(Bikes, index=['Manufacturer','Bike Type']))

-- Week 4

[Distributions in Pandas]

# - Distribution: Set of all possible random variables
# - Example:
#   + Flipping Coins for heads and tails
#     * a binomial distribution (two possible outcomes)
#     * discrete (categories of heads and tails, no real numbers)
#     * evenly weighted (heads are just as likely as tails)
#   + Tornado events in Ann Arbor
#     * A binomial distribution
#     * Discrete
#     * Unevenly weighted (tornadoes are rare events)

import pandas as pd
import numpy as np

np.random.binomial(1, 0.5)

> 0

np.random.binomial(1000, 0.5)/1000

> 0.477

chance_of_tornado = 0.01/100
np.random.binomial(100000, chance_of_tornado)

> 11

chance_of_tornado = 0.01
​
tornado_events = np.random.binomial(1, chance_of_tornado, 1000000)

two_days_in_a_row = 0
for j in range(1,len(tornado_events)-1):
    if tornado_events[j] == 1 and tornado_events[j-1] == 1:
        two_days_in_a_row += 1
​
print('{} tornadoes back to back in {} years'.format(two_days_in_a_row, 1000000/365))

> 100 tornadoes back to back in 2739.72602739726 years

# - Uniform Distribution: just a squared shape
# - Normal (Gaussian) Distribution
#   + Mean: a measure of central tendency (as are median & mode -> where is the bulk of the probability in the distribution)
#   + Stdev: a measure of variability

# Generate 5 values between 0 & 1, each of which is equally likely to occur.

np.random.uniform(low = 0, high = 1, size = 5)

> array([ 0.06682767,  0.8859055 ,  0.51923987,  0.14208689,  0.73670251])

# Generate 5 values with mean 1 and stdev 0.75

np.random.normal(loc = 1, scale = 0.75, size = 5)

> array([-0.73921202,  2.10630117,  0.38773682,  1.4042543 ,  0.13697617])

# Formula for standard deviation
# $$\sqrt{\frac{1}{N} \sum_{i=1}^N (x_i - \overline{x})^2}$$

distribution = np.random.normal(loc = 0.75, scale = 1, size = 1000)

np.sqrt(np.sum((distribution - np.mean(distribution)) ** 2) / len(distribution))

> 0.98432578097823831

np.std(distribution)

> 0.98432578097823831

import scipy.stats as stats
​
# Positive values indicate a more peaked distribution
# Negative values indicate a less peaked distribution
​
stats.kurtosis(distribution)

> -0.09188174208709388

stats.skew(distribution)

> -0.0023729460337239766

# Positive skew: distribution pushed to the positive/high side (usually the right side of the graph)
# Negative skew: distribution pushed to the negative/low side (usually the left side of the graph)

chi_squared_df2 = np.random.chisquare(df = 2, size = 10000)
stats.skew(chi_squared_df2)

> 1.9650104221773115

chi_squared_df5 = np.random.chisquare(5, size = 10000)
stats.skew(chi_squared_df5)

> 1.3098079383565786

[Hypothesis Testing in Python]

# - Hypothesis: A statement we can test
#   + Alternative hypothesis: Our idea, e.g. there is a difference between groups
#   + Null hypothesis: The alternative of our idea, e.g. there is no difference between groups
# - Critical Value alpha (α)
#   + The threshold as to how much chance you are willing to accept
#   + Typical values in social sciences are 0.1, 0.05, 0.01

df = pd.read_csv('grades.csv')

df.head()

> student_id  assignment1_grade assignment1_submission  assignment2_grade ...
> 0 B73F2C11-70F0-E37D-8B10-1D20AFED50B1  92.733946 2015-11-02 06:55:34.282000000 83.030552 2015-11-09 02:22:58.938000000 67.164441 ...
> 1 98A0FAE0-A19A-13D2-4BB5-CFBFD94031D1  86.790821 2015-11-29 14:57:44.429000000 86.290821 2015-12-06 17:41:18.449000000 69.772657 ...
> 2 D0F62040-CEB0-904C-F563-2F8620916C4E  85.512541 2016-01-09 05:36:02.389000000 85.512541 2016-01-09 06:39:44.416000000 68.410033 ...
> 3 FFDF2B2C-F514-EF7F-6538-A6A53518E9DC  86.030665 2016-04-30 06:50:39.801000000 68.824532 2016-04-30 17:20:38.727000000 61.942079 ...
> 4 5ECBEEB6-F1CE-80AE-3164-E45E99473FB4  64.813800 2015-12-13 17:06:10.750000000 51.491040 2015-12-14 12:25:12.056000000 41.932832 ...

len(df)

> 2315

early = df[df['assignment1_submission'] <= '2015-12-31']
late = df[df['assignment1_submission'] > '2015-12-31']

early.mean()

> assignment1_grade    74.972741
> assignment2_grade    67.252190
> ...
> assignment5_grade    48.634643
> assignment6_grade    43.838980
> dtype: float64

late.mean()

> assignment1_grade    74.017429
> assignment2_grade    66.370822
> ...
> assignment5_grade    48.599402
> assignment6_grade    43.844384
> dtype: float64

from scipy import stats
stats.ttest_ind?

stats.ttest_ind(early['assignment1_grade'], late['assignment1_grade'])

> Ttest_indResult(statistic=1.400549944897566, pvalue=0.16148283016060577)

stats.ttest_ind(early['assignment2_grade'], late['assignment2_grade'])

> Ttest_indResult(statistic=1.3239868220912567, pvalue=0.18563824610067967)

stats.ttest_ind(early['assignment3_grade'], late['assignment3_grade'])

> Ttest_indResult(statistic=1.7116160037010733, pvalue=0.087101516341556676)

# - P-hacking, or Dredging
#   + Doing many tests until you find one which is of statistical significance
#   + At a confidence level of 0.05, we expect to find one positive result 1 time out of 20 tests
#   + Remedies:
#     * Bonferroni correction
#     * Hold-out sets
#     * Investigation pre-registration

# -- Helpful R code to disect a T-Test:

# g1 <- sleep$extra[1 : 10]; g2 <- sleep$extra[11 : 20]

# n1 <- length(g1); n2 <- length(g2)
# sp <- sqrt(((n1-1) * sd(g1)^2 + (n2-1) * sd(g2)^2) / (n1 + n2 - 2))
# md <- mean(g2) - mean(g1)
# semd <- sp * sqrt(1 / n1 + 1 / n2) # this is the standard error (of the mean difference)!
# rbind(
#   md + c(-1, 1) * qt(0.975, n1 + n2 - 2) * semd,
#   t.test(g2, g1, paired = FALSE, var.equal = TRUE),
#   t.test(g2, g1, paired = TRUE)$conf
# )

[Wk4 - Lecture Quizzes]

# Suppose we want to simulate the probability of flipping a fair coin 20 times, and getting a number
# greater than or equal to 15. Use np.random.binomial(n, p, size) to do 10000 simulations of flipping
# a fair coin 20 times, then see what proportion of the simulations are 15 or greater.

import numpy as np
size = 10000
np.sum(np.random.binomial(20, 0.5, size) >= 15) / (size * 1.0) # 0.022
# or: (np.random.binomial(20, 0.5, size) >= 15).mean()
