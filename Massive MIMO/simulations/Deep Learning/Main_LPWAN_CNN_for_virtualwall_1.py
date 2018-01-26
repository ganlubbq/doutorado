# -*- coding: utf-8 -*-
#"""
#Created on Fri Nov 24 16:57:20 2017thor: Adnan Shahid
#"""

# -*- coding: utf-8 -*-
#"""
#Created on Wed Nov 22 11:26:52 2017
#@author: Adnan Shahid
#"""

# -*- coding: utf-8 -*-
#"""
#Created on Thu Nov  9 16:22:13 2017

#@author: Adnan Shahid
#"""

#from __future__ import print_function

import keras
import os,random
import pickle
os.environ["KERAS_BACKEND"] = "tensorflow"
os.environ["TENSORFLOW_FLAGS"]  = "device=gpu%d"%(1)
import keras
from keras.models import Sequential
import scipy.io as sio
import array 
import numpy as np
from matplotlib import pyplot as plt
from keras import losses
from keras.layers.convolutional import Conv2D, MaxPooling2D, ZeroPadding2D
from keras.layers.core import Reshape,Dense,Dropout,Activation,Flatten
from keras.optimizers import RMSprop
from IPython.display import clear_output
from keras import optimizers
from keras.models import load_model

numclass=10
in_dim=[2,500]
#snrs=[-70,-60,-50,-40,-30,-20,-10,0,10,20]
snrs=[-70,-60,-50,-40,-30,-20,-10,0,10,20,30,40]
#loading sensing-shots
# For training -------------------------------------------------------------------------
load_data = sio.loadmat('testing_training_combined')
X = load_data['X_combined']
X_label=load_data['X_combined_label']
lbl=load_data['label']
X_labeld = keras.utils.to_categorical(X_label,numclass)
[r_comb,c_comb]=X.shape
X=np.reshape(X,(int(r_comb),2,int(c_comb/2))) 

np.random.seed(2016)
n_examples = X.shape[0]
n_train = n_examples * 0.6
train_idx = np.random.choice(range(0,n_examples), size=int(n_train), replace=False)
test_idx = list(set(range(0,n_examples))-set(train_idx))
X_train = X[train_idx]
X_test =  X[test_idx]
Y_train_labeld=X_labeld[train_idx]
Y_test_labeld=X_labeld[test_idx]

dr = 0.5
model = Sequential()
model.add(Reshape(in_dim+[1],input_shape=in_dim))
model.add(ZeroPadding2D((2, 2)))
model.add(Conv2D(64, (2, 3), name='conv1', padding='valid', activation='relu', kernel_initializer='glorot_uniform'))
model.add(MaxPooling2D(pool_size=(2, 2), strides=1, padding='valid', data_format=None))
model.add(Dropout(dr))
model.add(Conv2D(40, (2, 3), name='conv2', padding='valid', activation='relu', kernel_initializer='glorot_uniform'))
model.add(MaxPooling2D(pool_size=(2, 2), strides=2, padding='valid', data_format=None))
model.add(Dropout(dr))
model.add(Flatten())
model.add(Dense(200, activation='relu', init='he_normal', name="dense1"))
model.add(Dropout(dr))
model.add(Dense(numclass, init='he_normal', name="dense2" ))
model.add(Activation('softmax'))
model.add(Reshape([numclass]))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
#model.compile(loss='sparse_categorical_crossentropy', optimizer='adam',metrics=['accuracy'])
model.summary()

nb_epoch = 200
batch_size = 1024
#filepath = 'LPWANrecnets_CNN2_0.5.wts.h5'
history = model.fit(X_train, Y_train_labeld,
                    batch_size=batch_size,
                    epochs=nb_epoch,
                    verbose=1,
                    validation_split=0.2,
                    validation_data=(X_test, Y_test_labeld))

#model.load_weights(filepath)

score = model.evaluate(X_test, Y_test_labeld, verbose=0)

print('Test loss:', score[0])
print('Test accuracy:', score[1])
model.save('LPWAN_epochs_200_batchsize_1024_dropout_0.5_classes_10_Split_60n40_conv_3layers.h5') 

  
## Plot confusion matrix
classes=['Sigfox - ch1', 'Sigfox - ch180', 'Sigfox - ch400', 'IEEE 802.15.4 (subGHz)', 'LoRA-SF7','LoRA-SF8','LoRA-SF9','LoRA-SF10','LoRA-SF11','LoRA-SF12']
test_Y_hat = model.predict(X_test, batch_size=batch_size)
conf = np.zeros([len(classes),len(classes)])
confnorm1 = np.zeros([len(classes),len(classes)])
for i in range(0,X_test.shape[0]):
    j = list(Y_test_labeld[i,:]).index(1)
    k = int(np.argmax(test_Y_hat[i,:]))
    conf[j,k] = conf[j,k] + 1
for i in range(0,len(classes)):
    confnorm1[i,:] = conf[i,:] / np.sum(conf[i,:])
#plot_confusion_matrix(confnorm1, labels=classes)
    
# Plot confusion matrix
acc = {}
conf_tot={}
for snr in snrs:

    # extract classes @ SNR
    test_SNRs = list(map(lambda x: lbl[x][1], test_idx))
    temp=np.where(np.array(test_SNRs)==snr)
    test_X_i = X_test[temp[0]]
    test_Y_i = Y_test_labeld[temp[0]]    

    # estimate classes
    test_Y_i_hat = model.predict(test_X_i)
    conf = np.zeros([len(classes),len(classes)])
    confnorm = np.zeros([len(classes),len(classes)])
    for i in range(0,test_X_i.shape[0]):
        j = list(test_Y_i[i,:]).index(1)
        k = int(np.argmax(test_Y_i_hat[i,:]))
        conf[j,k] = conf[j,k] + 1
    for i in range(0,len(classes)):
        confnorm[i,:] = conf[i,:] / np.sum(conf[i,:])
    conf_tot[snr]=confnorm  
#    plt.figure()
#    plot_confusion_matrix(confnorm, labels=classes, title="ConvNet Confusion Matrix (SNR=%d)"%(snr))
    
    cor = np.sum(np.diag(conf))
    ncor = np.sum(conf) - cor
    print("Overall Accuracy: ", cor / (cor+ncor))
    acc[snr] = 1.0*cor/(cor+ncor)


# saving variables
f_myfile = open('LPWAN_epochs_200_batchsize__1024_dropout_0.5_classes_10_Split_60n40_conv_3layers.pickle', 'wb')
pickle_objects={'snr':snrs,'confusion_mart_snrs':conf_tot,'confusion_matrix_SNR':confnorm1,'classification_acc_with_SNR':acc, 'training_loss':history.history['loss'],'validation_loss':history.history['val_loss'],'validation_accuracy':history.history['val_acc'],'training_accuracy':history.history['acc']}
pickle.dump(pickle_objects, f_myfile)
f_myfile.close()

# file = open('LPWAN_epochs_200_batchsize_400_dropout_0.0_classes_8_Split_60n40_with_fft.pickle', 'rb')
# object = pickle.load(file)
# 
    
   
