import sys
import os
# Generate dummy data
import numpy as np
import scipy.io as sio
import time

t = time.localtime()
timestamp = time.strftime('%d%Y%H%M%S', t)
filename = 'mse_ls_cnn_1d_%s.txt' % (timestamp)

#-----------------------------------------------------------------------

M = 70
L = 7
K = 10
N = K
batch_size = 100
number_of_epochs = 200


input_layer_size = M*2
second_layer_size = M*2
output_layer_size = M*2
kernel_size = 1
optimizer = 'sgd'
cmd = 'python deep_learning_mmimo_channel_estimation_loss_mse_ls_cnn_1d.py %d %d %d %d %d %d %d %d %d %d %s %s&' % (M,L,K,N,batch_size,number_of_epochs,input_layer_size,second_layer_size,output_layer_size,kernel_size,filename,optimizer)
os.system(cmd)                         

