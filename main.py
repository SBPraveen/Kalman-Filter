# -*- coding: utf-8 -*-
"""
Created on Fri Mar  6 09:34:00 2020

@author: Praveen
"""

import numpy as np
import matplotlib.pyplot as plt
from kf import KF
plt.ion()#Turn the interactive mode on
plt.figure()#use plt.figure() when we want to tweak the size of the figure and 
#when we want to add multiple Axes objects in a single figure.

real_x=0.0#encoder angle
meas_variance=0.1**2#simulate noise in our measurement
real_v=0.5

kf=KF(initial_x=0.0,initial_v=1.0,accel_variance=0.1)

DT=0.1
NUM_STEPS=1000
MEAS_EVERY_STEP = 20

mus=[]#creating a list for storing the mean
covs=[]#creating a list for storing the covariance
real_xs = []
real_vs = []


for step in range(NUM_STEPS):
    covs.append(kf.cov)
    mus.append(kf.mean)

    
    real_x = real_x + DT * real_v
    
    kf.predict(dt=DT)
    if step != 0 and step%MEAS_EVERY_STEP == 0:
        kf.update(meas_value = real_x + np.random.randn()*np.sqrt(meas_variance) , meas_variance = meas_variance)
    real_xs.append(real_x)
    real_vs.append(real_v)
"""     
plt.subplot(2,1,1)
plt.title("Position")
plt.plot([mu[0] for mu in mus],'r')#'r'--> denotes the red color of the line
plt.plot(real_xs,'b')
#FOr creating disturbances
plt.plot([mu[0]-2*np.sqrt(cov[0,0])for mu,cov in zip(mus,covs)],'r--')#np.sqrt is used to find the square root of a list
#'r--" denotes the red color dotted line
plt.plot([mu[0]+2*np.sqrt(cov[0,0])for mu,cov in zip(mus,covs)],'r--')

"""
"""
The zip() function returns a zip object, which is an iterator of tuples where the first item in each passed iterator is paired together, and then the second item in each passed iterator are paired together etc.

If the passed iterators have different lengths, the iterator with the least items decides the length of the new iterator.
eg:
    a = ("John", "Charles", "Mike","Praveen")
    b = ("Jenny", "Christy", "Monica")

    x = zip(a, b)
o/p:(('John', 'Jenny'), ('Charles', 'Christy'), ('Mike', 'Monica'))

"""


plt.subplot(2,1,2)
plt.title("ANGLE")
plt.plot(real_vs,'b')
plt.plot([mu[1] for mu in mus],'r')
plt.plot([mu[1]-2*np.sqrt(cov[1,1])for mu,cov in zip(mus,covs)],'r--')#np.sqrt is used to find the square root of a list
#'r--" denotes the red color dotted line
plt.plot([mu[1]+2*np.sqrt(cov[1,1])for mu,cov in zip(mus,covs)],'r--')

plt.show
#plt.ginput(1)