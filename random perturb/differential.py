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
from sympy.matrices import Matrix, eye, zeros, ones, diag, GramSchmidt,MatAdd
import matplotlib.pyplot as plt
np.random.seed(3)
def sigmoid(x):
    return 1.0/(1.0+sympy.exp(-x))
                
w1 = Matrix([[-8.69413149,9.35492699,6.83860075],
             [ 7.20170118,7.03754429,-4.91852026]])

    
w2 = Matrix([[ 11.92907,-11.92453376],
             [ 11.503676 ,-11.50324274],
             [-12.56603613,12.57001863]])
    
b1 = Matrix([[-6.17767189 ,-6.36609515, -8.83107425]])  

b2 = Matrix([[-5.57217888, 5.57042239]])
x = Matrix([symbols("x%d"%i)for i in range(2)])
def fn(data,weight1,weight2,bias1,bias2):
    l1 = weight1.dot(data)
    l1=Matrix(l1)+b1.T
    e1=l1.applyfunc(sigmoid)
    l2 = weight2.dot(e1)
    l2=Matrix(l2)+b2.T
    e2 = l2.applyfunc(sigmoid)
    return e2[0]-e2[1]
print("value of function is:",fn(x,w1,w2,b1,b2).evalf(subs={x[0]:0,x[1]:0.25}))
fprime=diff(fn(x,w1,w2,b1,b2),x[0])
#calculated as the adversarial distance to the boundary
rtilde=0.46549999999994873
y_adv=0.25+rtilde
x_adv=0

alpha_x=np.linalg.norm(np.array([diff(fn(x,w1,w2,b1,b2),x[0]).evalf(subs={x[0]:x_adv,x[1]:y_adv}),diff(fn(x,w1,w2,b1,b2),x[1]).evalf(subs={x[0]:x_adv,x[1]:y_adv})],dtype=np.float64))/rtilde

hessian =[[diff(fn(x,w1,w2,b1,b2),x[0],x[0]).evalf(subs={x[0]:x_adv,x[1]:y_adv}),diff(fn(x,w1,w2,b1,b2),x[0],x[1]).evalf(subs={x[0]:x_adv,x[1]:y_adv})],
          [diff(fn(x,w1,w2,b1,b2),x[1],x[0]).evalf(subs={x[0]:x_adv,x[1]:y_adv}),diff(fn(x,w1,w2,b1,b2),x[1],x[1]).evalf(subs={x[0]:x_adv,x[1]:y_adv})]]
print("the hessian is: ",hessian)
from numpy import linalg as LA
m=np.array(hessian, dtype=float)
v=np.linalg.eigvals(m)
print("eigenvalues are:",v)

lambd=max(v)
kappa =1/(lambd*alpha_x)
print("kappa is:",kappa)
tressian_x=np.array([[diff(fn(x,w1,w2,b1,b2),x[0],x[0],x[0]).evalf(subs={x[0]:x_adv,x[1]:y_adv}),diff(fn(x,w1,w2,b1,b2),x[0],x[1],x[0]).evalf(subs={x[0]:x_adv,x[1]:y_adv})],
                      [diff(fn(x,w1,w2,b1,b2),x[1],x[0],x[0]).evalf(subs={x[0]:x_adv,x[1]:y_adv}),diff(fn(x,w1,w2,b1,b2),x[1],x[1],x[0]).evalf(subs={x[0]:x_adv,x[1]:y_adv})]],dtype=float)
tressian_y=np.array([[diff(fn(x,w1,w2,b1,b2),x[0],x[0],x[1]).evalf(subs={x[0]:x_adv,x[1]:y_adv}),diff(fn(x,w1,w2,b1,b2),x[0],x[1],x[1]).evalf(subs={x[0]:x_adv,x[1]:y_adv})],
                      [diff(fn(x,w1,w2,b1,b2),x[1],x[0],x[1]).evalf(subs={x[0]:x_adv,x[1]:y_adv}),diff(fn(x,w1,w2,b1,b2),x[1],x[1],x[1]).evalf(subs={x[0]:x_adv,x[1]:y_adv})]],dtype=float)
eigen_x=max(np.linalg.eigvals(tressian_x))
eigen_y=max(np.linalg.eigvals(tressian_y))
eigen_tress=max(eigen_x,eigen_y)
print("max eigen value of 3rd order tensor is: ",eigen_tress)
kappa_p=1/(eigen_tress*alpha_x)
print("kappa prime is ",kappa_p)

