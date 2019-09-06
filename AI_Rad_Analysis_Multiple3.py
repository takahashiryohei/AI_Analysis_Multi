
import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense, Activation
from datetime import datetime
from keras.optimizers import SGD, Nadam
import keras.backend as K

#####################################
#        Main Routine Start         #
#####################################

INPUT_NUM = 4095
LAYER1_UNIT_NUM = 170 #Default 170
LAYER2_UNIT_NUM =  140 #Default 40
LERNING_RATE = 0.01 #Default 0.01
DECAY_RATE = 0.0 #Default 1e-4
MOMENTUM_COE = 0.0 #Default 0.9
OUTPUT_NUM = 7
BATCH_SIZE = 128
EPOCH_NO = 50

#np.random.seed(0) # 乱数を固定値で初期化し再現性を持たせる
'''
#ここから乱数を固定
import os
import numpy as np
import random as rn
import tensorflow as tf

os.environ['PYTHONHASHSEED'] = '0'
np.random.seed(7)
rn.seed(7)

session_conf = tf.ConfigProto(
    intra_op_parallelism_threads=1,
    inter_op_parallelism_threads=1
)

from keras import backend as K

tf.set_random_seed(7)
sess = tf.Session(graph=tf.get_default_graph(), config=session_conf)
K.set_session(sess)
#ここまで乱数固定
'''

def mean_hot(y_true,y_pred):#二値化制度
	return K.mean(K.equal(K.sum(K.clip(y_true*10000000000,0,1),axis=1)+K.sum(K.clip(y_true*10000000000,0,1)*10*K.clip(y_true*10000000000,0,1)*10,axis=-1),K.sum(K.clip(y_pred*10000000000,0,1),axis=-1)+K.sum(K.clip(y_pred*10000000000,0,1)*10*K.clip(y_true*10000000000,0,1)*10,axis=-1)))

def mean(y_true,y_pred):#評価値が許容誤差いないかの制度
	return K.mean(K.greater(K.ones_like(K.max(K.abs((y_true-y_pred)/(y_true+0.01)),axis=-1))-0.7,K.max(K.abs((y_true-y_pred)/(y_true+0.01)),axis=-1)))


model = Sequential()
model.add(Dense(input_dim=INPUT_NUM, output_dim=LAYER1_UNIT_NUM, activation='relu'))
model.add(Dense(output_dim=LAYER2_UNIT_NUM, activation='relu'))
model.add(Dense(output_dim=OUTPUT_NUM,activation='relu'))


model.compile(loss = 'mse', optimizer=Nadam(lr=0.002, beta_1=0.9, beta_2=0.999, epsilon=None, schedule_decay=0.004), metrics=['acc',mean_hot])

print(' ')
print('Training Data Loading from ' + datetime.now().strftime('%Y/%m/%d %H:%M:%S'))
#Open train and test data
df_train = pd.read_csv('Multiple_train_shuffle_pro.csv',sep=',')
df_test = pd.read_csv('Multiple_test_shuffle_pro.csv',sep=',')
print('Training Data Loadinng finish at ' + datetime.now().strftime('%Y/%m/%d %H:%M:%S'))
print(' ')



### Training Data Loading ###
temp_array = df_train.values
Train_Meas_Time = temp_array[:,7]
Train_Activity_Set = temp_array[:,30:37] 
Train_Spectra = temp_array[:,68:4163]
del df_train
del temp_array

### Test Data Loading ###
temp_array = df_test.values
Test_Meas_Time = temp_array[:,7]
Test_Activity_Set = temp_array[:,30:37] 
Test_Spectra = temp_array[:,68:4163]
del df_test
del temp_array

#############################################

############ Preprocessing Start ############

#Train_Spectra = Train_Spectra + 1  #後藤さんのアイデア。log前に1を足す。
Train_Spectra = np.square(Train_Spectra)
#Train_Spectra = np.log(Train_Spectra) #スペクトルの対数をとる
Train_Spectra_Sum = np.sum(Train_Spectra, axis=1)+1
Train_Spectra = Train_Spectra.T / Train_Spectra_Sum #スペクトルの積分値を1に規格化
Train_Spectra = Train_Spectra.T #計算の都合上転置していたので元に戻す


#Test_Spectra = Test_Spectra + 1  #後藤さんのアイデア。log前に1を足す。
Test_Spectra = np.square(Test_Spectra)
#Test_Spectra = np.log(Test_Spectra) #スペクトルの対数をとる
Test_Spectra_Sum = np.sum(Test_Spectra, axis=1)+1
Test_Spectra = Test_Spectra.T / Test_Spectra_Sum #スペクトルの積分値を1に規格化
Test_Spectra = Test_Spectra.T #計算の都合上転置していたので元に戻す

############ Preprocessing  Finish ############


############ Training Start ###################
history = model.fit(Train_Spectra, Train_Activity_Set, epochs=EPOCH_NO, validation_data=(Test_Spectra, Test_Activity_Set))

predict = model.predict(Test_Spectra,batch_size=BATCH_SIZE)
print(predict)

np.savetxt('test.txt',predict)

json_string = model.to_json()
open('gamma_analysis_model.json', 'w').write(json_string)
model.save_weights('my_model_weights.h5')

#Accuracy
acc = history.history['acc']
val_acc = history.history['val_acc']
epochs = range(1, len(acc)+1) #エポック数が０から始まるので＋１ accが#(1,epochsの数)の大きさだから合わせている  range(3)=[0, 1, 2]
plt.plot(epochs, acc, 'b')  #b=blue line
plt.plot(epochs, val_acc, 'r') #r=red line
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')#凡例
plt.show()
#loss
loss = history.history['loss']
val_loss = history.history['val_loss']
plt.plot(epochs, loss, 'b')
plt.plot(epochs, val_loss, 'r')
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper right')
plt.show()


#####################################
#        Main Routine END           #
#####################################

