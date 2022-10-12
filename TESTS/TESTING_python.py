### This file is for testing/learning python functions and how to apply them correctly

### Python OOP Tutorial 3 - 4.10.22
import random
from collections import deque, namedtuple
import gym
import numpy as np

env = gym.make('LunarLander-v2')

class Chemical:
    danger_lvl = 0

    def __init__(self,name, atomic_weight ):    # __init__ defines the internal structure of a class. one can set its paramerters with variables
        self.name = name
        self.atomic_weight = atomic_weight

    @classmethod
    def set_all_danger_lvl(cls, level):
        cls.danger_lvl = level

    @staticmethod
    def calc_multiply(x,y):
        return x*y

    def __repr__(self):     # meant for debugging/developers
        return "Level %s chemical (%s %s)" % (self.danger_lvl, self.name, self.atomic_weight)

    def __str__(self):      # meant for end user. supposed to be easily understandable. overrides __repr__
        return "This chemical is %s. It has atomic weight %s and has a danger level %s" % (self.name, self.atomic_weight, self.danger_lvl)

    # def __add__(self, x): # defines how addition of 2 instances should be executed. For other methods for artihmatic (and more) see documentation

# ozone = Chemical('Ozone', 16*3)
# water = Chemical('Water', 18)

### regular method
# print(ozone.name, ozone.atomic_weight)
# print('Danger levels of water: %s and ozone: %s' % (water.danger_lvl, ozone.danger_lvl))

### class method
# Chemical.set_all_danger_lvl(2)
# print('Danger level of all chemicals set to %s' % Chemical.danger_lvl)
# print('Danger levels of water: %s and ozone: %s' % (water.danger_lvl, ozone.danger_lvl))

### static method
# they pass neither instance/self nor class. Behaviour of a normal function
# print(ozone.calc_multiply(3,2))

### magic/dunder methods
# print(ozone)
# print(ozone.__repr__())

### property decorators
# methods are like functions and have to be used with parenthesis -> instance.method()
# attribute are also functions but parenthesis are not needed -> instance.attribute. attributes cannot change self.variables
# getter    - @property above the method. define the property. definition turns it into an attribute.
# setter    - @methodname.setter. define setter method
# deleter   - @methodname.deleter. define deleter method. use as $del instance.methodname

##### collections datatype. Are extenstions and more specialized versions of general containers (such as list, tuple). imported from 'collections'
### deque() - list of a fixed length with overridable arguments. Can be overwritten with new data like RAM
# If appending new arg would increase the size over the limit ( len(deque)>deque.maxlength ) then append new arg and delete the last one ( del(deque[0]) )

# print('\nTesiting deque:')
# card_deck = deque(maxlen=4)
# for i in range(16):
#     card_deck.append(i)
#     print(card_deck)

### namedtuple() - tuple which can be accessed by given field names instead of indicies
# define new object/tuple: namedtuple( objectname, (attribute1, attribute2, ...) )
# assign newly created object/tuple to a var. This var is now of type objectname with attributes listed in the tuple

# print('\nTesting namedtuple():')
# animal = namedtuple('Animal',('name','size','species') )
# lion = animal('Latin name','Big','Average lion')
# print(lion)

# point_3d = namedtuple('Point',('x','y','z'))
# for i in range(6):
#     p = point_3d(i,2*i,3*i)
#     print(p, random.sample(p,2))    # from library random use sample method
                                      # sample(obj, size) - from obj return a list ( len=size ) with random attributes of obj

### Math
# print(20%4)     # x%y returns the 'rest' of division
# print(20%7)     # x//y returns the max amount of times that y fits in x
# print(20//4)
# print(20//6)
# print(20/7)
# print(20/6)

### *args
# def sum(*args):
#     x = 0
#     for i in args:
#         x += i
#         print(x)
#
# sum(1,2,3,4)

# for i in range(10):
#     print(random.randint(0,3))

### Time module
import time
# print( time.asctime(time.localtime(time.time())) )
# t0 = time.process_time()
# for i in range(10000):
#     print(i)
# t1 = time.process_time()
# print( t1-t0 )
# t1 = time.process_time()
# for i in range(120000):
#     print(i)
# t2 = time.process_time()
# print( t2-t1 )

print( time.localtime(time.time()).tm_mday )

### np.random.choice()
# memory = deque(maxlen=100)
# mem = namedtuple('Memory',('x','y'))
#
# for i in range(100):
#     memory.append((i,i))
#
# print(memory)
# print(np.random.choice(memory, 10, replace=False))

### zip()
a = ['A','B']
b = [1,2]
x = zip(a,b)
print(list(x))