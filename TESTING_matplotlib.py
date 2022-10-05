import matplotlib.pyplot as mpl
import math
import numpy as np

# A lot of (plot) functions from Matlab have the same names in matplotlib and/or numpy.
# For more commands see Module 'Grundlagen Numerik und Simulation'

x = np.linspace(0,10,20)
y = np.random.randint(0,100,20)
plot = mpl.plot(x,y)
mpl.show()                          # show -> make a figure visible