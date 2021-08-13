################# Imports #######################################
import pandas_datareader as pdr
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from numpy import array
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM
import tensorflow as fr
import math
from sklearn.metrics import mean_squared_error

################# To Collect Data Of A Stock #################### 
#df = pdr.get_data_tingo('',api_key = 'f18647627f0a8629c5de70bc99a7f33de70deb7f')
#df.to_csv('AAPL.csv')


################# To Insert Data ############################
df = pd.read_csv('TCS.csv')
##de = df.tail()
##print(de)
df1= df.reset_index()['Close']
##print(df1)


################ To Plot Data ####################
plt.plot(df1)
##plt.show()

################ Long Term Short Memory With Help Of MinMax Scaler #####################
scaler = MinMaxScaler(feature_range=(0,1))
df1 = scaler.fit_transform(np.array(df1).reshape(-1,1))
df1.shape
##print(df1.shape)

############### Dividing Data Set In Training And Test Set ##########################3
training_size = int(len(df1)*0.65)
test_size = len(df1)-training_size
train_data, test_data = df1[0:training_size,:], df1[training_size:len(df1),:1]
##print(training_size,test_size)



############# Conversion Of Array Values To Dataset Matrix #################
def create_dataset(dataset,time_step=1):
    dataX, dataY = [],[]
    for i in range(len(dataset)-time_step-1):
        a = dataset[i:(i+time_step),0]   ###i=0,  0,1,2,3_______,99
        dataX.append(a)
        dataY.append(dataset[i+time_step,0])
    return np.array(dataX), np.array(dataY)


############# Reshape into X=t,t+1,t+2,t+3 and Y=t+4############
time_step = 100
X_train ,Y_train=create_dataset(train_data,time_step)
X_test,Y_test = create_dataset(test_data,time_step)
##print(X_train.shape)
##print(Y_train.shape)

############ Reshape Input to be ########################
X_train = X_train.reshape(X_train.shape[0],X_train.shape[1],1)
X_test = X_test.reshape(X_test.shape[0],X_test.shape[1],1)

model = Sequential()
model.add(LSTM(50,return_sequences = True,input_shape = (100,1)))
model.add(LSTM(50,return_sequences=True))
model.add(LSTM(50))
model.add(Dense(1))
model.compile(loss='mean_squared_error',optimizer='adam')

##model.summary()

##print(X_train[:100])

model.fit(X_train,Y_train,validation_data=(X_test,Y_test),epochs=100,batch_size=64,verbose=1)

############### Do Predicton And Check Performance Metrices ######################
train_predict = model.predict(X_train)
test_predict = model.predict(X_test)

train_predict = scaler.inverse_transform(train_predict)
test_predict = scaler.inverse_transform(test_predict)

############## Calculate RMSE Performance Metrices #########33
math.sqrt(mean_squared_error(Y_train,train_predict))
math.sqrt(mean_squared_error(Y_test,test_predict))

############## Plotting ############33
look_back= 100
trainPredictPlot = np.empty_like(df1)
trainPredictPlot[:, :] = np.nan
trainPredictPlot[look_back:len(train_predict)+look_back,:] = train_predict
testPredictPlot = np.empty_like(df1)
testPredictPlot[:,:] = np.nan
testPredictPlot[len(train_predict)+(look_back*2)+1:len(df1)-1,:]=test_predict
plt.plot(scaler.inverse_transform(df1))
plt.plot(trainPredictPlot)
plt.plot(testPredictPlot)
plt.show()




x_input = test_data[1313:].reshape(1,-1)
x_input.shape
temp_input=list(x_input)
temp_input[0].tolist()


############### Prediction For Next 10 Days ######################
lst_output=[]
n_steps=100
i=0
while(i<30):
    if(len(temp_input)>100):
        x_input=np.array(temp_input[1:])
        print('{} day input {}'.format(i,x_input))
        x_input = x_input.reshape(1,-1)
        x_input = x_input.reshape((1,n_steps , 1))
        yhat = model.predict(x_input,verbose=0)
        print('{} day output {}'.format(i,yhat))
        temp_input.extend(yhat[0].tolist())
        temp_input= temp_input[1:]
        lst_output.extend(yhat.tolist())
        i=i+1
    else:
        x_input = x_input.reshape((1,n_steps,1))
        yhat = model.predict(x_input,verbose=0)
        print(yhat[0])
        temp_input.extend(yhat[0].tolist())
        print(len(temp_input))
        lst_output.extend(yhat.tolist())
        i=i+1
print(lst_output)


day_new = np.arange(1,101)
day_pred=np.arange(101,131)
df3 = d1.tolist()
df3.extend(lst_output)
plt.plot(day_new,scaler.inverse_transform(df1[3937:]))
plt.plot(day_pred,scaler.inverse_transform(lst_output))
df3=df1.tolist()
df3.extend(lst_output)
plt.plot(df3[4100:])
df3 = scaler.inverse_transform(df3).tolist()
plt.plot(df3)
