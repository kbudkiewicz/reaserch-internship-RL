import matplotlib.pyplot as mpl
import math
import numpy as np

### This file is for testing/learning matplotlib functions and how to apply them correctly
# A lot of (plot) functions from Matlab have the same names in matplotlib and/or numpy.
# For more commands see Module 'Grundlagen Numerik und Simulation'

x = np.linspace(0,10,11)
y = [1,1,1,5,6,120,130,140,150,200,250]
# y = np.random.randint(0,100,11)
plot1 = mpl.figure(1)
mpl.plot(x,y)
plot2 = mpl.figure(2)
mpl.semilogy(x,y)
mpl.show()                          # show -> make a figure visible