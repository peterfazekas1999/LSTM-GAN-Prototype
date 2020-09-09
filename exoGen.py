import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt

np.random.seed(1)

print('Loading train and test data...')
df=pd.read_csv('exoTrain.csv')
#print(df1)
df2=pd.read_csv('exoTest.csv')
#print(df2)


"""
In the traning data, We have light intensites of stars measured at 3198 time instances. 
The training data has the flux sequenc for 5087 stars while the test data has the flux sequences for 570 stars. 
If the value in LABEL column is 2, it is an exoplanet host star and if it is 1, it is not an exoplanet host star.

"""

#print(train_data)
test_data=np.array(df2,dtype=np.float32)
#print(test_data)


Y=df.loc[:,df.columns=="LABEL"]
X=df.loc[:,df.columns!="LABEL"]
ytest=test_data[:,0]
Xtest=test_data[:,1:]
# print(ytrain,'\n',Xtrain)
# print(ytest,'\n',Xtest)

m=1 # A chosen exoplanet host star's index for plott
n=100 # A chosen non-exoplanet host star's index

#print('Shape of Xtrain:',np.shape(Xtrain),'\nShape of ytrain:',np.shape(ytrain))

Xtrain =X
Y =Y.to_numpy()

# plt.plot(X.iloc[m],'r')
# plt.title('Light intensity vs time (for an exoplanet star)')
# plt.xlabel('Time index')
# plt.ylabel('Light intensity')
# plt.show()


###  Applying Fourier Transform

from scipy.fftpack import fft,ifft

print('Applying Fourier Transform...')

Xtrain=Xtrain.to_numpy()
plt.figure(5)
plt.plot(Xtrain[1])
Xtrain=np.abs(fft(Xtrain,n=len(Xtrain[1]),axis=1))
# Xtest=np.abs(fft(Xtest,n=len(Xtest[1]),axis=1))

# print(Xtrain,Xtrain.shape)

Xtrain=Xtrain[:,:1+int((len(Xtrain[0])-1)/2)]

# print('\n\n',Xtrain,Xtrain.shape)
#print('Shape of Xtrain:',np.shape(Xtrain),'\nShape of ytrain:',np.shape(ytrain))

Xtest=Xtest[:,:1+int((len(Xtest[0])-1)/2)]
plt.figure(6)
plt.plot(Xtrain[1],'r')
plt.title('After FFT (for an exoplanet star)')
plt.xlabel('Frequency')
plt.ylabel('Feature value')
plt.show()

# plt.plot(Xtrain[n],'b')
# plt.title('After FFT (for a non exoplanet star)')
# plt.xlabel('Frequency')
# plt.ylabel('Feature value')
# plt.show()


#### Normalizing

from sklearn.preprocessing import normalize

print('Normalizing...')
Xtrain=normalize(Xtrain)
Xtest=normalize(Xtest)
plt.figure(7)
plt.plot(Xtrain[m],'r')
plt.title('After FFT,Normalization (for an exoplanet star)')
plt.xlabel('Frequency')
plt.ylabel('Feature value')
plt.show()




#### Applying Gaussian Filter

from scipy import ndimage

print('Applying Gaussian filter...')
Xtrain=ndimage.filters.gaussian_filter(Xtrain,sigma=10)
Xtest=ndimage.filters.gaussian_filter(Xtest,sigma=10)
plt.figure(8)
plt.plot(Xtrain[m],'r')
plt.title('After FFT,Normalization and Gaussian filtering (for an exoplanet star)')
plt.xlabel('Frequency')
plt.ylabel('Feature value')
plt.show()




#### Scaling down the data

from sklearn.preprocessing import MinMaxScaler

print('Applying MinMaxScaler...')
scaler=MinMaxScaler(feature_range=(0,1))
Xtrain=scaler.fit_transform(Xtrain)
Xtest=scaler.fit_transform(Xtest)
plt.figure(8)
plt.plot(Xtrain[m],'r')
plt.title('After FFT,Normalization, Gaussian filtering and scaling (for an exoplanet star)')
plt.xlabel('Frequency')
plt.ylabel('Feature value')
plt.show()



#model deployment

# print(Xtrain.shape)
# Xtrain = np.reshape(Xtrain,(Xtrain.shape[0],Xtrain.shape[1],1))
# print(Xtrain.shape)
# print(Xtrain)
X_tester = Xtrain
xtrain_t = []
ytrain_t = []
for j in range(1,5):
    for i in range(30,len(X_tester[0])):
        xtrain_t.append(X_tester[j][i-30:i])
        ytrain_t.append(X_tester[j][i])
xtrain_t = np.array(xtrain_t)
ytrain_t = np.array(ytrain_t)
print(xtrain_t.shape)
# print(xtrain_t)
xtrain_t = np.reshape(xtrain_t,(xtrain_t.shape[0],xtrain_t.shape[1],1))
# ytrain_t = np.reshape(ytrain_t,(ytrain_t.shape[0],1))
print(xtrain_t)
print(xtrain_t.shape)
    

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM,SimpleRNN
from keras.layers import Dropout,GRU,Flatten
from keras.layers import TimeDistributed
from keras.wrappers.scikit_learn import KerasClassifier
from keras.models import Sequential, load_model
import tensorflow as tf
from keras.optimizers import Adam,Adamax,Nadam
adam = Adam(learning_rate = 3E-4)
model = Sequential()
model.add(LSTM(units =256,input_shape=(xtrain_t.shape[1],1),return_sequences = True))
model.add(LSTM(units =128))
model.add(Dense(units =1))
model.compile(loss='binary_crossentropy',optimizer="rmsprop",metrics = ["accuracy"])
model.summary()
model.fit(xtrain_t,ytrain_t,epochs =5,batch_size = 500)
model.save("trial1.h")
model = load_model("trial1.h")


limit =300
i =0
trial = []
trial_t = xtrain_t[2000:2001]

for i in range(limit):
    yt = model.predict(trial_t)
    trial.append([yt[0][0]])
    trial_t = np.reshape(trial_t,(trial_t.shape[1]))
    trial_t =np.append(trial_t,yt[0][0])
    trial_t = trial_t[1:]
    trial_t = np.reshape(trial_t,(1,trial_t.shape[0],1))
     
 

plt.figure(10)
plt.plot(trial)


xtester = model.predict(xtrain_t)
print(trial)
print(xtester)

plt.figure(1)
plt.plot(xtester[2000:2700],color ="r")
plt.figure(2)
plt.plot(ytrain_t[2000:2700],color = "b")
plt.show()
    
    



    

