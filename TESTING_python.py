##### Python Classes
### Python OOP Tutorial 3 - 4.10.22
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

##### deque()
###