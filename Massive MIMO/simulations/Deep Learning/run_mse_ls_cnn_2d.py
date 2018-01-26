import sys
import os
# Generate dummy data
import numpy as np
import scipy.io as sio
import time

t = time.localtime()
timestamp = time.strftime('%d%Y%H%M%S', t)
filename = 'mse_ls_cnn_2d_v2_%s.txt' % (timestamp)

#-----------------------------------------------------------------------

M = 70
L = 7
K = 10
N = K
batch_size = 100
number_of_epochs = 100

input_layer_size = M
second_layer_size = M
output_layer_size = 2*M
kernel_size_1st_layer = 4
kernel_size_2nd_layer = 4
optimizer = 'sgd'
cmd = 'python deep_learning_mmimo_channel_estimation_loss_mse_ls_cnn_2d.py %d %d %d %d %d %d %d %d %d %d %d %s %s&' % (M,L,K,N,batch_size,number_of_epochs,input_layer_size,second_layer_size,output_layer_size,kernel_size_1st_layer,kernel_size_2nd_layer,filename,optimizer)
os.system(cmd)

input_layer_size = 2*M
second_layer_size = 2*M
output_layer_size = 2*M
kernel_size_1st_layer = 4
kernel_size_2nd_layer = 2
optimizer = 'sgd'
cmd = 'python deep_learning_mmimo_channel_estimation_loss_mse_ls_cnn_2d.py %d %d %d %d %d %d %d %d %d %d %d %s %s&' % (M,L,K,N,batch_size,number_of_epochs,input_layer_size,second_layer_size,output_layer_size,kernel_size_1st_layer,kernel_size_2nd_layer,filename,optimizer)
os.system(cmd)

input_layer_size = 2*M
second_layer_size = 2*M
output_layer_size = 2*M
kernel_size_1st_layer = 4
kernel_size_2nd_layer = 4
optimizer = 'sgd'
cmd = 'python deep_learning_mmimo_channel_estimation_loss_mse_ls_cnn_2d.py %d %d %d %d %d %d %d %d %d %d %d %s %s&' % (M,L,K,N,batch_size,number_of_epochs,input_layer_size,second_layer_size,output_layer_size,kernel_size_1st_layer,kernel_size_2nd_layer,filename,optimizer)
os.system(cmd)

