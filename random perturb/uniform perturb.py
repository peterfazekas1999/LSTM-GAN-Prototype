# -*- coding: utf-8 -*-
"""
Created on Mon Nov  9 13:54:41 2020

@author: peter fazekas
"""

import numpy as np
from sklearn import datasets
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

def sigmoid(x):
    return 1.0/(1.0+np.exp(-x))


def d_sigmoid(x):
    return sigmoid(x)*(1.0-sigmoid(x))

#training_input = np.array([[0,0,1],
                           #[1,1,1],
                           #[1,0,1],
                           #[0,1,1]])

#training_output = np.array([0,1,1,0])

#training_output = training_output.reshape(4,1)
np.random.seed(3)
training_input, training_output = datasets.make_moons(200,noise = 0.1)
color = training_output
#print(training_input)
#print(color)
#print(type(color))

#print(training_input)
#print(training_output)
np.random.seed(0)

w1 = np.random.randn(2,3)/np.sqrt(2)
w2 = np.random.randn(3,2)/np.sqrt(3)

epoch = 3000
loss_x = []
error_arr = []
b1 = np.zeros((1,3))
b2 = np.zeros((1,2))
X = training_input
lr = 0.01
reg = 0.01
for i in range(epoch):
    z1 = X.dot(w1) + b1
    a1 = sigmoid(z1)
    z2 = a1.dot(w2) + b2
    a2 =sigmoid(z2)
    del2 = a2
    del2[range(len(training_input)),training_output] -= 1
    #del2 = np.multiply(del2,a2*(1-a2))
    dw2 = (a1.T).dot(del2)
    db2 = np.sum(del2,axis = 0,keepdims = True)
    del1 = del2.dot(w2.T)*(1-a1)*a1
    dw1 = (X.T).dot(del1)
    
    db1 = np.sum(del1,axis = 0)
    #dw1 += reg*w1
    #dw2 += reg*w2
    w1 = w1-lr*dw1
    b1 = b1-lr*db1
    w2 = w2-lr*dw2
    b2 = b2-lr*db2

    
    print(del2.sum())
w1n = w1
w2n = w2
b1n = b1
b2n = b2

print(b1n)
print(b2n)
print("\n")
print("\n")
def predict(data,weight1,weight2,bias1,bias2):
    l1 = data.dot(weight1)+bias1
    e1 = sigmoid(l1)
    l2 = e1.dot(weight2)+bias2
    e2 = sigmoid(l2)
    print(e2)
    return np.argmax(e2,axis = 1)
test = np.array([[1,0],
                [0,0.25]])
prediction = predict(test,w1n,w2n,b1n,b2n)
print(prediction)
plt.scatter(training_input[:,0],training_input[:,1],c = color)
plt.scatter(test[0],test[1],c = "r")
#plt.show()
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
    Z = predict(data1,w1n,w2n,b1n,b2n)
    Z = Z.reshape(xx.shape)
    plt.contour(xx,yy,Z,alpha = 0.2)
    plt.show()

plot_dec_bound()