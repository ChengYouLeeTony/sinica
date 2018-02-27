"""

Created on 2016.10.22

@author: Jen-Cheng Hou

CNN for Audio-Visual Speech Enhancement
Input:Noisy spectrum and Lips motion(RGB)
Output:Clean spectrum and Lips motion(RGB)

"""
from __future__ import print_function
from keras.models import Sequential, Model, load_model
from keras.layers import Dense, Dropout, Activation, Flatten, Merge, merge
from keras.layers import Convolution2D, MaxPooling2D, Input
from keras.optimizers import SGD, Adam, RMSprop, Adagrad, Nadam, Adamax, Adadelta
# import matplotlib.pyplot as plt
import scipy.io
import time
import numpy as np
import theano
import h5py
import os
from keras.callbacks import Callback, ModelCheckpoint, EarlyStopping
# from keras.utils.visualize_util import plot
from keras.layers.normalization import BatchNormalization
import load_data_during_training_in_car
import random
import gc

list_path = "/home/tony180004/sinica"
gen_list_name_main = "gpu1/data_concat_list/in_car_for_concat_"
model_name = 'model_01'
start_time = time.time()
n_batch_size = 100 #1000 # max:~9k
num_lst = 50
get_train_data_loops = 100 #2000 # multiplier of num_lst max:~10k
get_test_data_loops = 100 #500 # multiplier of num_lst max:~3.5k
n_epoch = 5
n_loop = 3 #40
patience = n_epoch
# optimizer = RMSprop(lr=0.0001, rho=0.9, epsilon=1e-08, decay=0.0)
loss_weights = {'main_output':np.array(1), 'aux_output':np.array(1)}

freq_resolution = 257
dynamic_frame = 5
img_row  = 80
img_col = 24
img_chan = 3

main_input = Input(shape=(freq_resolution,dynamic_frame,1), dtype='float32', name='main_input')
auxiliary_input = Input(shape=(img_row,img_col,img_chan), name='aux_input')

x1 = Convolution2D(10, 12, 2, border_mode='valid', dim_ordering='tf', input_shape=(freq_resolution,dynamic_frame,1))(main_input)
x1 = BatchNormalization(axis=1)(x1)
x1 = MaxPooling2D(pool_size=(2,1), dim_ordering='tf')(x1)
x1 = Convolution2D(4, 5, 1, border_mode='valid', dim_ordering='tf')(x1)
x1 = BatchNormalization(axis=1)(x1)
x1 = Flatten()(x1)

x2 = Convolution2D(12, 15, 2, border_mode='valid', dim_ordering='tf', input_shape=(img_row,img_col,img_chan))(auxiliary_input)
x2 = BatchNormalization(axis=1)(x2)
x2 = Convolution2D(10, 7, 2, border_mode='valid')(x2)
x2 = BatchNormalization(axis=1)(x2)
x2 = Convolution2D(6, 3, 2, border_mode='valid')(x2)
x2 = BatchNormalization(axis=1)(x2)
x2 = Flatten()(x2)

x = merge([x1, x2], mode='concat')
x = Dense(1000, activation='sigmoid', init='glorot_uniform')(x)
x = Dropout(0.10)(x) # Fraction of the input units to drop.
x = BatchNormalization(axis=-1)(x)
x = Dense(800, activation='sigmoid', init='glorot_uniform')(x)
x = Dropout(0.10)(x) # Fraction of the input units to drop.
x = BatchNormalization(axis=-1)(x)

main_hidden = Dense((600), activation='sigmoid', init='glorot_uniform', name='main_hidden')(x)
main_hidden = BatchNormalization(axis=-1)(main_hidden)
auxiliary_hidden = Dense((1500), activation='sigmoid', init='glorot_uniform', name='auxiliary_hidden')(x)
auxiliary_hidden = BatchNormalization(axis=-1)(auxiliary_hidden)

main_output = Dense((freq_resolution*dynamic_frame), activation='linear', init='glorot_uniform', name='main_output')(main_hidden)
auxiliary_output = Dense((img_row*img_col*img_chan), activation='linear', init='glorot_uniform', name='aux_output')(auxiliary_hidden)

model = Model(input=[main_input, auxiliary_input], output=[main_output, auxiliary_output])

# model.compile(optimizer=optimizer, loss='mse', loss_weights=loss_weights)

nfold_weights_path = "gpu1/weights.best-{loss:.4f}.hdf5"
checkpointer = ModelCheckpoint(nfold_weights_path, monitor='loss',save_best_only=False, mode='min')
early_stopping = EarlyStopping(monitor='val_main_output_loss', patience=patience)

for i in range(0, n_loop):

	print('prepare training data at loop ' + str(i) + '/' + str(n_loop))

	num_key_error = 0
	start_time_tmp = time.time()
	#generate num_lst (e.g. 200) lists for faster concatenating
	speaker = 'speaker01'
	in_car = 'in_car_14'
	phase = 'train'
	np_noise_type_sets = np.linspace(101,191,num=91,dtype='int16')
	SNR_sets = [10,6,2,-2,-6]
	np_SNR_sets = np.array(SNR_sets)
	SNR_car_sets = [10,6,2,-2,-6]
	np_SNR_car_sets = np.array(SNR_car_sets)
	gen_list_name_pre = gen_list_name_main + phase
	load_data_during_training_in_car.gen_list(gen_list_name_pre, speaker, in_car, phase, np_noise_type_sets, np_SNR_sets, np_SNR_car_sets, get_train_data_loops, num_lst)

	for m in range(0,num_lst):
		list_full_path = list_path + os.sep + gen_list_name_pre + "%03d" % m + ".list"
		# print("Reading " + list_full_path)
		N, C, V = load_data_during_training_in_car.av_pair_with_list(list_full_path) # concatenated data in single list
		try:
			N_all = np.concatenate((N_all, N),axis=0)
			C_all = np.concatenate((C_all, C),axis=0)
			V_all = np.concatenate((V_all, V),axis=0)
			del N, C, V
			gc.collect()
		except NameError:
			N_all, C_all, V_all = N, C, V
			del N, C, V
			gc.collect()

	end_time_tmp = time.time()
	# print('                      concatenating training data in this loop ran for %.2fm' % ((end_time_tmp - start_time_tmp) / 60.))

	X_train_audio = N_all
	y_train_audio = C_all

	train_frames = X_train_audio.shape[0] #e.g: 313305 (=62661x5(SNRs))

	X_train_audio = np.reshape(X_train_audio,(train_frames, freq_resolution, dynamic_frame, 1))
	y_train_audio = np.reshape(y_train_audio,(train_frames, freq_resolution*dynamic_frame))

	X_train_lips = V_all
	y_train_lips = np.reshape(V_all,(train_frames, img_row*img_col*img_chan))

	del N_all, C_all, V_all
	gc.collect()

	print('prepare testing data at loop ' + str(i) + '/' + str(n_loop))

	num_key_error = 0
	start_time_tmp = time.time()
	#generate num_lst (e.g. 200) lists list for faster concatenating
	speaker = 'speaker01'
	in_car = 'in_car_02'
	phase = 'test'
	noise_type_sets = [1, 4, 5, 7, 8, 9, 10, 11, 12, 13]
	np_noise_type_sets = np.array(noise_type_sets)
	SNR_sets = [5,0,-5]
	np_SNR_sets = np.array(SNR_sets)
	SNR_car_sets = [5,0,-5]
	np_SNR_car_sets = np.array(SNR_car_sets)
	gen_list_name_pre = gen_list_name_main + phase
	load_data_during_training_in_car.gen_list(gen_list_name_pre, speaker, in_car, phase, np_noise_type_sets, np_SNR_sets, np_SNR_car_sets, get_test_data_loops, num_lst)

	for m in range(0,num_lst):
		list_full_path = list_path + os.sep + gen_list_name_pre + "%03d" % m + ".list"
		# print("Reading " + list_full_path)
		N, C, V = load_data_during_training_in_car.av_pair_with_list(list_full_path) # concatenated data in single list
		try:
			N_all = np.concatenate((N_all, N),axis=0)
			C_all = np.concatenate((C_all, C),axis=0)
			V_all = np.concatenate((V_all, V),axis=0)
			del N, C, V
			gc.collect()

		except NameError:
			N_all, C_all, V_all = N, C, V
			del N, C, V
			gc.collect()

	end_time_tmp = time.time()
	# print('                      concatenating testing data in this loop ran for %.2fm' % ((end_time_tmp - start_time_tmp) / 60.))

	X_test_audio = N_all
	y_test_audio = C_all

	test_frames = X_test_audio.shape[0] #e.g: 313305 (=62661x5(SNRs))

	X_test_audio = np.reshape(X_test_audio,(test_frames, freq_resolution, dynamic_frame, 1))
	y_test_audio = np.reshape(y_test_audio,(test_frames, freq_resolution*dynamic_frame))

	X_test_lips = V_all
	y_test_lips = np.reshape(V_all,(test_frames, img_row*img_col*img_chan))

	del N_all, C_all, V_all
	gc.collect()

	# print('data is prepared for training at loop ' + str(i) + '/' + str(n_loop))
	print('START to train at loop ' + str(i) + '/' + str(n_loop))

	lrs = [0.0001, 0.00001, 0.000001]
	for l in lrs:
		print("learning rate at " + str(l))
		optimizer = RMSprop(lr=l, rho=0.9, epsilon=1e-08, decay=0.0)
		model.compile(optimizer=optimizer, loss='mse', loss_weights=loss_weights)
		model.fit([X_train_audio, X_train_lips], [y_train_audio, y_train_lips], nb_epoch=n_epoch, batch_size=n_batch_size, verbose=1, shuffle=True, callbacks = [checkpointer, early_stopping], validation_data=([X_test_audio, X_test_lips], [y_test_audio, y_test_lips]))

	del X_train_audio, X_train_lips, y_train_audio, y_train_lips
	del X_test_audio, X_test_lips, y_test_audio, y_test_lips
	gc.collect()

model.save('./gpu1/' + model_name + '.h5')  # creates a HDF5 file

end_time = time.time()
print('The code for this file ran for %.2fm' % ((end_time - start_time) / 60.))