# -*- coding: utf-8 -*-
"""
Created on Mon Nov  9 16:33:00 2020

@author: peter fazekas
"""
import sympy
from sympy import symbols, diff
import numpy as np
import math as m
import random
import matplotlib.pyplot as plt
np.random.seed(3)
def sigmoid(x):
    return 1.0/(1.0+np.exp(-x))
                
w1 = np.array([[-8.69413149,9.35492699,6.83860075],
       [ 7.20170118,7.03754429,-4.91852026]])
w2 = np.array([[ 11.92907,-11.92453376],
       [ 11.503676 ,-11.50324274],
       [-12.56603613,12.57001863]])
b1 = np.array([[-6.17767189 ,-6.36609515, -8.83107425]])
       
b2 = np.array([[-5.57217888, 5.57042239]])

x=0
y=0.25
f=sigmoid(w2[0][0]*sigmoid(x*w1[0][0]+y*w1[1][0]+b1[0][0])+w2[1][0]*sigmoid(x*w1[0][1]+y*w1[1][1]+b1[0][1])+w2[2][0]*sigmoid(x*w1[0][2]+y*w1[1][2]+b1[0][2])+b2[0][0])-sigmoid(w2[0][1]*sigmoid(x*w1[0][0]+y*w1[1][0]+b1[0][0])+w2[1][1]*sigmoid(x*w1[0][1]+y*w1[1][1]+b1[0][1])+w2[2][1]*sigmoid(x*w1[0][2]+y*w1[1][2]+b1[0][2])+b2[0][1])
print(f)