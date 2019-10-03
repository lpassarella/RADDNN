import numpy as np 
from sklearn.preprocessing import StandardScaler
from scipy.stats import pearsonr, zscore
import tensorflow as tf
import keras.backend as kb
import keras.backend as K
import os

os.environ["CUDA_VISIBLE_DEVICES"] = '1' #use GPU with ID=1
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.5 # maximun alloc gpu50% of MEM
config.gpu_options.allow_growth = True #allocate dynamically

import numpy as np
rad_out=np.load('rad_out.npy')
rad_in=np.load('rad_in.npy')
print(np.shape(rad_out))
rad_out=rad_out[:,:,1:]
rad_in=rad_in[:,:,1:]
rad_out=np.reshape(rad_out,(rad_out.shape[0]*rad_out.shape[1],rad_out.shape[2]))
import pandas as pd
rad_in=np.reshape(rad_in,(rad_in.shape[0]*rad_in.shape[1],rad_in.shape[2]))
print(np.shape(rad_out))
rad_out=pd.DataFrame(rad_out)
rad_in=pd.DataFrame(rad_out)

from sklearn.model_selection import train_test_split
train_X, test_X, train_Y, test_Y = train_test_split(rad_in,rad_out, test_size=0.2, random_state=42)

train_X=train_X.values
train_Y = train_Y.values
test_X=test_X.values
test_Y=test_Y.values

XMEAN=np.max(train_X,axis=0)
XSTDD=np.min(train_X,axis=0)

for j in range(train_X.shape[1]):
 if((XMEAN[j]-XSTDD[j])== 0):
  train_X[:,j] = 0.0
  test_X[:,j] = 0.0
 else:
  train_X[:,j]=(train_X[:,j]-XSTDD[j])/(XMEAN[j]-XSTDD[j])
  test_X[:,j]=(test_X[:,j]-XSTDD[j])/(XMEAN[j]-XSTDD[j])

YMEAN=np.max(train_Y,axis=0)
YSTDD=np.min(train_Y,axis=0)

for j in range(train_Y.shape[1]):
 if((YMEAN[j]-YSTDD[j])== 0):
  train_Y[:,j] = 0.0
  test_Y[:,j] = 0.0
 else:
  train_Y[:,j]=(train_Y[:,j]-YSTDD[j])/(YMEAN[j]-YSTDD[j])
  test_Y[:,j]=(test_Y[:,j]-YSTDD[j])/(YMEAN[j]-YSTDD[j])



print(train_X.shape, test_X.shape)
print(train_Y.shape, test_Y.shape)

n_cols = train_X.shape[1]
out_n_cols = train_Y.shape[1]
import keras
from keras.models import Sequential
from keras.layers import Dense

classifier= Sequential()
seed = 7
np.random.seed(seed)
#input layer and first hidden layer
classifier.add(Dense(64,input_shape=(n_cols,),kernel_initializer='uniform',
                     activation='relu'))
#second hidden layer
classifier.add(Dense(64,kernel_initializer='uniform',
                     activation='relu'))

#output layer
classifier.add(Dense(out_n_cols,kernel_initializer='uniform',
                     activation='sigmoid'))

#compile whole artifical network
classifier.compile(optimizer='adam',loss='mean_squared_error',metrics=['accuracy'])


classifier.fit(train_X,train_Y, validation_split=0.1,batch_size=1024,epochs=100)

score  = classifier.predict(test_X)
print(score.shape)
for j in range(out_n_cols):
 score[:,j]=score[:,j]*(YMEAN[j]-YSTDD[j])+YSTDD[j]
classifier.save('my_model.h5')
#summary = model.summary()
W_Input_Hidden0 = classifier.layers[0].get_weights()[0];
biases0  = classifier.layers[0].get_weights()[1];
W_Input_Hidden1 = classifier.layers[1].get_weights()[0];
biases1  = classifier.layers[1].get_weights()[1];
W_Input_Hidden2 = classifier.layers[2].get_weights()[0];
biases2  = classifier.layers[2].get_weights()[1];


np.save('SWHidden001.npy',W_Input_Hidden0)
np.save('SWbiases001.npy',biases0)
np.save('SWHidden011.npy',W_Input_Hidden1)
np.save('SWbiases011.npy',biases1)
np.save('SWHidden021.npy',W_Input_Hidden2)
np.save('SWbiases021.npy',biases2)
np.save('SWX_test1.npy', X_test)
np.save('SWY_test1.npy', Y_test)
np.save('SWScore1.npy', score)


print(history.history.keys())
train_loss = history.history['loss']
val_loss   = history.history['val_loss']
acc_train=history.history['acc']
acc_val=history.history['val_acc']
np.save('train_loss.npy',train_loss)
np.save('val_loss.npy',val_loss)
np.save('train_acc.npy',acc_train)
np.save('val_acc.npy',acc_val)


