# -*- coding: utf-8 -*-
"""
Created on Mon Nov  9 14:03:15 2020

@author: peter fazekas
"""

from sympy import symbols, diff
import numpy as np
import math as m
import random
import matplotlib.pyplot as plt
np.random.seed(3)
training_input, training_output = datasets.make_moons(200,noise = 0.1)
color = training_output
def sigmoid(x):
    return 1.0/(1.0+np.exp(-x))

w1n = np.array([[-8.69413149,9.35492699,6.83860075],
       [ 7.20170118,7.03754429,-4.91852026]])
w2n = np.array([[ 11.92907,-11.92453376],
       [ 11.503676 ,-11.50324274],
       [-12.56603613,12.57001863]])
b1n = np.array([[-6.17767189 ,-6.36609515, -8.83107425]])
       
b2n = np.array([[-5.57217888, 5.57042239]])
def predict(data,weight1,weight2,bias1,bias2):
    l1 = data.dot(weight1)+bias1
    e1 = sigmoid(l1)
    l2 = e1.dot(weight2)+bias2
    
    e2 = sigmoid(l2)
    return e2
test = np.array([[0,0.25]])
def adversarial(test):
    r=0
    x =test[0][0]
    y=test[0][1]
    x_t =test[0][0]
    y_t=test[0][1]
    value =predict(test,w1n,w2n,b1n,b2n)[0][1]-predict(test,w1n,w2n,b1n,b2n)[0][0]
    print(value)
    delta=0.0001
    while(abs(value)>0.01):
        y_t+=delta
        
        test=np.array([[x_t,y_t]])
        value =predict(test,w1n,w2n,b1n,b2n)[0][1]-predict(test,w1n,w2n,b1n,b2n)[0][0]
        r=y_t-y
        #print([r,value])
    return r
r=adversarial(test)
print("adversarial: ",r)
def uniform(test,r):
    circle_x = test[0][0]
    circle_y = test[0][1]
    alpha =2*m.pi*random.random()
    x = r * m.cos(alpha) + circle_x
    y = r * m.sin(alpha) + circle_y
    return np.array([[x,y]])
zeros=0
ones=0
plot_arrx =[]
plot_arry =[]
delta=-1000
radius=0
while(delta<0.1):
    zeros=0
    ones=0
    plot_arrx =[]
    plot_arry =[]
    
    for i in range(500):
        data=uniform(test,radius)
        output =predict(data,w1n,w2n,b1n,b2n)
        plot_arrx.append(data[0][0])
        plot_arry.append(data[0][1])
        output_val=np.argmax(output,axis=1)
        if(output_val==0):
            zeros+=1
        else:
            ones+=1
    delta=zeros/(ones+zeros)
    radius+=0.01

def predict_argm(data,weight1,weight2,bias1,bias2):
    l1 = data.dot(weight1)+bias1
    e1 = sigmoid(l1)
    l2 = e1.dot(weight2)+bias2
    e2 = sigmoid(l2)
    return np.argmax(e2,axis = 1)
plt.scatter(training_input[:,0],training_input[:,1],c = color)
plt.scatter(test[0][0],test[0][1],c = "r")
def plot_dec_bound():
    cmap='Paired'
    cmap = plt.get_cmap(cmap)

    h = 1000  # step size in the mesh
    #create a mesh to plot in
    #x_min, x_max = training_input[:, 0].min()-1 , training_input[:, 0].max()+1 
    #y_min, y_max = training_input[:, 1].min()-1 , training_input[:, 1].max()+1 
    xx, yy = np.meshgrid(np.linspace(-1, 2, h),
                       np.linspace(-1,2, h))
    data1 = np.c_[xx.ravel(), yy.ravel()]

    Z = predict_argm(data1,w1n,w2n,b1n,b2n)
    Z = Z.reshape(xx.shape)
    plt.contour(xx,yy,Z,alpha = 0.2)
    

plot_dec_bound()
plt.scatter(plot_arrx,plot_arry,s=1)
plt.show()

print("probability of misclassification (delta) is: ",delta)    
print("radius is:",radius)



