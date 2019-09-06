import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense, Activation
from datetime import datetime
from keras.optimizers import SGD, Nadam
import keras.backend as K
'''ここから　後で利用するやつ
def Separate_Train_Data_Line(str_line):
    all_data = []
    Activity_Set = []
    Spectrum = []

    all_data = str_line.split(',')
    Ene_Cal_Num = all_data[1]
    FWHN_Num = all_data[2]
    Meas_Time = all_data[7]
    Activity_Set = all_data[8:27]
    Spectrum = all_data[58:4154]

    tmp_list = [Ene_Cal_Num,FWHN_Num,Meas_Time,Activity_Set, Spectrum]

    return (tmp_list)

def Pre_Processing(list_data):

    list_data = np.log(list_data)
    fact = np.sum(list_data)
    list_data = list_data/fact

    return fact
'''#ここまで後で利用するやつ
#####################################
#        Main Routine Start         #
#####################################

INPUT_NUM = 4095
LAYER1_UNIT_NUM = 100 #Default 170
LAYER2_UNIT_NUM = 50 #Default 40
LAYER3_UNIT_NUM = 20 
LERNING_RATE = 0.01 #Default 0.01
DECAY_RATE = 0.0 #Default 1e-4
MOMENTUM_COE = 0.0 #Default 0.9
OUTPUT_NUM = 7
BATCH_SIZE = 128
EPOCH_NO = 30

##### log file directory for TensorBoard ######
log_dir = './tmp'
#If log directory exists, remove and recreate
if tf.gfile.Exists(log_dir):
    tf.gfile.DeleteRecursively(log_dir)
tf.gfile.MakeDirs(log_dir)
###############################################

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
'''
#二値化制度の失敗
def mean_hot(y_true,y_pred):
	return K.mean(K.equal(K.sum(K.greater(y_true,K.zeros_like(y_true)),axis=-1),K.sum(K.get_value(K.greater(y_true,K.zeros_like(y_true)),K.greater(y_pred,K.zeros_like(y_pred))),axis=-1)))
'''

def mean_hot(y_true,y_pred):#二値化制度
	return K.mean(K.equal(K.sum(K.clip(y_true*10000000000,0,1),axis=1)+K.sum(K.clip(y_true*10000000000,0,1)*10*K.clip(y_true*10000000000,0,1)*10,axis=-1),K.sum(K.clip(y_pred*10000000000,0,1),axis=-1)+K.sum(K.clip(y_pred*10000000000,0,1)*10*K.clip(y_true*10000000000,0,1)*10,axis=-1)))


def mean_pred(y_true,y_pred):#評価値
	return K.abs(y_true-y_pred)/(y_true+0.001)

def mean(y_true,y_pred):#評価値が許容誤差いないかの制度
	return K.mean(K.greater(K.ones_like(K.max(K.abs((y_true-y_pred)/(y_true+0.01)),axis=-1))-0.7,K.max(K.abs((y_true-y_pred)/(y_true+0.01)),axis=-1)))

model = Sequential()
model.add(Dense(input_dim=INPUT_NUM, output_dim=LAYER1_UNIT_NUM, activation='relu'))
model.add(Dense(output_dim=LAYER2_UNIT_NUM, activation='relu'))
model.add(Dense(output_dim=LAYER3_UNIT_NUM,activation='relu'))
model.add(Dense(output_dim=OUTPUT_NUM,activation='relu'))

#model.compile(loss = 'categorical_crossentropy', optimizer=SGD(lr=LERNING_RATE, momentum=MOMENTUM_COE, decay=DECAY_RATE, nesterov=True), metrics=['acc'])
#model.compile(loss = 'mse', optimizer=SGD(lr=0.05, momentum=0.9, decay=1e-4, nesterov=True), metrics=['acc',mean_pred])
model.compile(loss = 'mse', optimizer=Nadam(lr=0.002, beta_1=0.9, beta_2=0.999, epsilon=None, schedule_decay=0.004), metrics=['acc',mean,mean_hot])
#model.compile(loss = 'categorical_crossentropy', optimizer='Adam', metrics=['acc'])
###model.compile(loss = 'mse', optimizer=Nadam(lr=0.002, beta_1=0.9, beta_2=0.999, epsilon=None, schedule_decay=0.004), metrics=['acc'])


#Visualization test of NN model
graph = tf.get_default_graph()
with tf.summary.FileWriter(log_dir) as writer:
    writer.add_graph(graph)
###データの読み込み
print(' ')
print('Training Data Loading from ' + datetime.now().strftime('%Y/%m/%d %H:%M:%S'))
#Open train and test data
df_train = pd.read_csv('Multiple_train_shuffle_pro.csv',sep=',')
df_test = pd.read_csv('Multiple_test_shuffle_pro.csv',sep=',')
print('Training Data Loadinng finish at ' + datetime.now().strftime('%Y/%m/%d %H:%M:%S'))
print(' ')

#読み込んだデータの処理
#df_test = df_test[df_test['sec']>30]
#df_train = df_train[df_train['sec']>30]
#df_test = df_test[df_test['Cs-137']==0.352764]


###読み込んだデータを更に処理
temp_array = df_train.values
Train_Meas_Time = temp_array[:,7]
#print(Train_Meas_Time)
Train_Activity_Set = temp_array[:,30:37] 
#print(Train_Activity_Set)
Train_Spectra = temp_array[:,68:4163]
#print(Train_Spectra)
#print(temp_array)
#print('Meas. Time: ' + str(Train_Meas_Time))
#print('Activity: ' + str(Train_Activity_Set))
#print('Spectrum: ' + str(Train_Spectra))
del df_train
del temp_array

###読み込んだデータを更に処理
temp_array = df_test.values
Test_Meas_Time = temp_array[:,7]
Test_Activity_Set = temp_array[:,30:37] 
Test_Spectra = temp_array[:,68:4163]
#print(temp_array)
#print('Meas. Time: ' + str(Test_Meas_Time))
#print('Activity: ' + str(Test_Activity_Set))
#print('Spectrum: ' + str(Test_Spectra))
del df_test
del temp_array


###データ処理
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

#Train_Activity_Set[Train_Activity_Set>0] = 1
#Train_Activity_Set = Train_Activity_Set.T / 10000 # 10kBq = 1として線形補間
#Train_Activity_Set = Train_Activity_Set.T 
#print(Train_Activity_Set)

#Test_Activity_Set[Test_Activity_Set>0] = 1
#Test_Activity_Set = Test_Activity_Set.T / 10000 # 10kBq = 1として線形補間
#Test_Activity_Set = Test_Activity_Set.T 



############ Training Start ###################
fit = model.fit(Train_Spectra, Train_Activity_Set, epochs=EPOCH_NO, validation_data=(Test_Spectra, Test_Activity_Set))

predict = model.predict(Test_Spectra,batch_size=BATCH_SIZE)
print(predict)

np.savetxt('test.txt',predict)#出力の保存

json_string = model.to_json()#重みの保存
open('gamma_analysis_model.json', 'w').write(json_string)
model.save_weights('my_model_weights.h5')

# ----------------------------------------------
# Some plots
# ----------------------------------------------
fig, (axL, axR) = plt.subplots(ncols=2, figsize=(10,4))

# loss
def plot_history_loss(fit):
    # Plot the loss in the history
    axL.plot(fit.history['loss'],label="loss for training")
    axL.plot(fit.history['val_loss'],label="loss for validation")
    axL.set_title('model loss')
    axL.set_xlabel('epoch')
    axL.set_ylabel('loss')
    axL.legend(loc='upper right')

# acc
def plot_history_acc(fit):
    # Plot the loss in the history
    axR.plot(fit.history['acc'],label="acc for training")
    axR.plot(fit.history['val_acc'],label="acc for validation")
    axR.set_title('model accuracy')
    axR.set_xlabel('epoch')
    axR.set_ylabel('accuracy')
    axR.legend(loc='lower left')

plot_history_loss(fit)
plot_history_acc(fit)
fig.savefig('./test.png')
plt.close()

#このあたりはメモリにデータを展開しない方法（あとでやる）
#train_f = open('train_data.csv','r')
#train_data_line = train_f.readline() #Dispose 1 line of header
#for i_epoch in range(EPOCH_NO):
#    for i_batch in range(BATCH_SIZE):
#train_data
#Test of train Data Slice
#train_data_line = train_f.readline()
#separated_train_data = Separate_train_Data_Line(train_data_line)
#print(separated_train_data[4])
#train_f.close()


#####################################
#        Main Routine END           #
#####################################

