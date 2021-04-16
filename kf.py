# -*- coding: utf-8 -*-
"""
Created on Fri Mar  6 10:59:41 2020

@author: Praveen
"""

import numpy as np
class KF:
    def __init__(self,initial_x:float,initial_v:float,accel_variance:float)->None:
        #Type Annotations: used to inform someone reading the code what the type of a variable should be
            #type annotations do not affect the runtime of the program in any way. These hints are ignored by the interpreter and are used solely to increase the readability 
                #for other programmers and yourself.None denotes the return type of the function
            
        """
        what is a GRV??
        
        A random variable with a Gaussian distribution is said to be normally distributed and is called a 
        normal deviate. Normal distributions are important in statistics and are often used in the natural and 
        social sciences to represent real-valued random variables whose distributions are not known.
        
        https://www.researchgate.net/post/When_is_guassian_random_variable_or_gaussian_distribution_used
        1.The noise can be modelled as various distributions...gaussian distribution helps analyse the intricacies
        involved due to the presence of noise. In fact, any real signal has noise which is quite random and not 
        exactly gaussian, but can be enveloped by a gaussian distribution. 
        2.The summary that i got is that Gaussian random variable is the sum of a large number of small random 
        variables where the deviation from the true value is random. That is why as the noise consist of 
        different small components and in the presence of noise the actual signal deviates randomly from the 
        true value that is whywe model it as Gaussian. If the sample set is large we use Gaussian otherwise we 
        use poisson.
        3.Well, mostly, while modeling the noise in channel, we say the noise is white noise, we use gaussian 
        random variable, that is because the gaussian variable with its gaussian distribution is mostly similar
        to the noise performance    
        """
        #mean of the state of gaussian random variable(GRV)
        self._x=np.array([initial_x,initial_v])
        self._accel_variance=accel_variance
        #covariance of the state of gaussian random variable(GRV)
        self._P=np.eye(2) #identity matrix
    def predict(self,dt:float)->None:#time evolution of the discussion random variable as the time passes
        #X=F x #new x after dt =X
        #P=F P Ft + G Gt a #Ft --> F transpose ; Gt--> G transpose
        F=np.array([[1,dt],[0,1]])
        new_x=F.dot(self._x)#matrix  0multiplication or vector multiplication dont use " * " symbol
        G=np.array([0.5*dt**2,dt]).reshape((2,1))
        new_P = F.dot(self._P).dot(F.T)+G.dot(G.T)*self._accel_variance
        self._P = new_P
        self._x = new_x
    
    def update(self,meas_value: float,meas_variance: float):#meas_value: measurement value 
        # y = z - H x
        # S = H P Ht + R
        # K = P Ht S(inverse)
        # x = x + K y
        # P = (I - K H)* P
        H=np.array([1,0]).reshape((1,2))
        z = np.array([meas_value])
        R = np.array([meas_variance])
        y = z - H.dot(self._x)
        S = H.dot(self._P).dot(H.T) + R
        K = self._P.dot(H.T).dot(np.linalg.inv(S))
        new_x=self._x + K.dot(y)
        new_P=(np.eye(2)-K.dot(H)).dot(self._P)
        self._P = new_P
        self._x = new_x
    @property# x is private
    def cov(self)->np.array:#cov-->covariance
        return self._P
    @property# P is private
    def mean(self)->np.array:
        return self._x    
    @property
    def pos(self)->float:
        return self._x[0]
    @property
    def vel(self)->float:
        return self._x[1]
