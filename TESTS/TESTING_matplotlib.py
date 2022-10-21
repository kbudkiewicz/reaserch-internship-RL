import matplotlib.pyplot as mpl
import math
import numpy as np

# mpl.rcParams.update({'text.usetex' : True})

### This file is for testing/learning matplotlib functions and how to apply them correctly
# A lot of (plot) functions from Matlab have the same names in matplotlib and/or numpy.
# For more commands see Module 'Grundlagen Numerik und Simulation'

# x = np.linspace(0,10,11)
# y = [1,1,1,5,6,120,130,140,150,200,250]
# # y = np.random.randint(0,100,11)
# plot1 = mpl.figure(1)
# mpl.plot(x,y)
# mpl.title('Lambda')
# plot2 = mpl.figure(2)
# mpl.semilogy(x,y)
# mpl.show()                          # show -> make a figure visible

### tests for final plot
# l1 = [1,2,3,4]
# l2 = [5,5,5]
#
# def leng(*args):
#     print( len(args) )
#     for i in range( len(args) ):
#         print(args[i])
#
# leng(l1,l2,l1)
#
# M = np.eye(4)
# M[0,:] = l1
# print(M)

### ploting errorbars
# li1 = [10,12,14]
# sdevs = [0.2,0.3,0.4]
# x = np.linspace(0, len(li1), len(li1))
# mpl.figure()
# mpl.errorbar(x, li1, yerr=sdevs)
# mpl.show()

### fill_between() - allows for plotting areas (e.g. between average and variance lines)
# x = [0,1,2,3]
# y = [12.,13.,15.,21.]
# y1 = [12.1,13.1,15.1,21.1]
# y2 = [11.9,12.9,14.9,20.9]
#
# j1 = [20.2,21.2,24.2,26.2]
# j2 = [19.8,20.8,23.8,25.8]
# j = [20,21,24,26]
#
# mpl.fill_between(x,y1,y2, linewidth=0, alpha=0.5)
# mpl.plot(x,y)
# mpl.fill_between(x,j1,j2, alpha=0.5)
# mpl.plot(x,j)
# mpl.show()

### testing calculations with None Type
# l = 1+None
# print(l)
# m = np.mean(l)
# print(m)

lis = [1,2,None,6]
stack = []
for i in lis:
    if i == None:
        continue
    else:
        stack.append(i)
print(stack)
print( np.mean(stack) )