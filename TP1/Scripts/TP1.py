# -*- coding: utf-8 -*-
import numpy as np
from scipy import optimize
import matplotlib.pyplot as plt
import random

# generate data
x =...
y = ...
#check
# plot the results
plt.figure(figsize = (10,8))
plt.plot(x, y, 'b.')
plt.xlabel('x')
plt.ylabel('y')
plt.title('Observed data')
plt.show()


    
plt.hist(beta2, range = (1,3), bins = 100, color = 'yellow',
         edgecolor = 'red')
plt.xlabel('beta2')
plt.ylabel('nombre occurences')
plt.title('histogramme beta2')
# 