# -*- coding: utf-8 -*-
"""
Created on Mon Nov  9 16:33:00 2020

@author: peter fazekas
"""
import sympy
from sympy.functions.elementary.exponential import exp
from sympy import symbols, diff
import numpy as np
import math as m

import matplotlib.pyplot as plt
np.random.seed(3)
def sigmoid(x):
    return 1.0/(1.0+exp(-x))
                
w1 = np.array([[-8.69413149,9.35492699,6.83860075],
       [ 7.20170118,7.03754429,-4.91852026]])
w2 = np.array([[ 11.92907,-11.92453376],
       [ 11.503676 ,-11.50324274],
       [-12.56603613,12.57001863]])
b1 = np.array([[-6.17767189 ,-6.36609515, -8.83107425]])
       
b2 = np.array([[-5.57217888, 5.57042239]])

x = symbols("x")
y=symbols("y")

def f(x,y):
    return sigmoid(w2[0][0]*sigmoid(x*w1[0][0]+y*w1[1][0]+b1[0][0])+w2[1][0]*sigmoid(x*w1[0][1]+y*w1[1][1]+b1[0][1])+w2[2][0]*sigmoid(x*w1[0][2]+y*w1[1][2]+b1[0][2])+b2[0][0])-sigmoid(w2[0][1]*sigmoid(x*w1[0][0]+y*w1[1][0]+b1[0][0])+w2[1][1]*sigmoid(x*w1[0][1]+y*w1[1][1]+b1[0][1])+w2[2][1]*sigmoid(x*w1[0][2]+y*w1[1][2]+b1[0][2])+b2[0][1])

fprime=diff(f(x,y),y,x)

#calculation of adversarial point from [0,0.25] in the helper.py function
y_adv=0.25+0.46549999999994873
x_adv=0
print(fprime.evalf(subs={x:x_adv,y:y_adv}))

hessian =[[diff(f(x,y),x,x).evalf(subs={x:x_adv,y:y_adv}),diff(f(x,y),x,y).evalf(subs={x:x_adv,y:y_adv})],
          [diff(f(x,y),y,x).evalf(subs={x:x_adv,y:y_adv}),diff(f(x,y),y,y).evalf(subs={x:x_adv,y:y_adv})]]
print("the hessian is: ",hessian)