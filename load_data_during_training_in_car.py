## Please run CNN_AV_SE_train_in_car_list.py instead --> faster in concatenating data

from __future__ import print_function
import os
import numpy as np
import scipy.io
import random
import time

top_path_parent = "/mnt/gv0/user_jchou/Projects/AudioVisual_SE/Preprocess/Matlab"  # server 242
# top_path_parent = "/home/coolkiu/Projects/AudioVisual_SE/Preprocess/Matlab" # server 147

context = 2
ratio = 0.4
nor = 3

def av_pair(speaker, in_car, phase, noise_type, SNR, SNR_car):
	
	noisy_path_parent = top_path_parent + os.sep + "Audio/Preprocess/context_feature/direct_concat/context" + str(context) + os.sep + "noisy" + os.sep + speaker + os.sep + in_car 
	if phase == 'train':
		noise_type_str = "%03d" % noise_type
	else:
		noise_type_str = "%02d" % noise_type
	noisy_sub_path = "noisy" + noise_type_str + os.sep + phase + os.sep + str(SNR) + "dB/car" + os.sep + str(SNR_car) + "dB"
	noisy_path = noisy_path_parent + os.sep + noisy_sub_path
	list_name = os.listdir(noisy_path) 
	id = np.random.permutation(len(list_name))[0]
	fname = list_name[id]
	tmp_N = scipy.io.loadmat(noisy_path + os.sep + fname) 
	N = tmp_N['Spectrum_out_nor_ctx'] 
	clean_path = top_path_parent + os.sep + "Audio/Preprocess/context_feature/direct_concat/context" + str(context) + os.sep + "clean" + os.sep + speaker + os.sep + "ratio_" + str(ratio)
	tmp_C = scipy.io.loadmat(clean_path + os.sep + fname) 
	C = tmp_C['Spectrum_out_nor_ctx'] 
	visual_path = top_path_parent + os.sep + "Mouth/Preprocess/context_feature/direct_concat/context" + str(context) + os.sep + "nor" + str(nor) + os.sep + speaker
	tmp_V = scipy.io.loadmat(visual_path + os.sep + fname[:-4] + "_mouth.mat") 
	V = tmp_V['subImage_resize_sentence_nor_ctx'] 
	
	return N, C, V

def av_pair_assigned_sentence(speaker, in_car, phase, noise_type, SNR, SNR_car, fname):
	
	noisy_path_parent = top_path_parent + os.sep + "Audio/Preprocess/context_feature/direct_concat/context" + str(context) + os.sep + "noisy" + os.sep + speaker + os.sep + in_car 
	if phase == 'train':
		noise_type_str = "%03d" % noise_type
	else:
		noise_type_str = "%02d" % noise_type
	noisy_sub_path = "noisy" + noise_type_str + os.sep + phase + os.sep + str(SNR) + "dB/car" + os.sep + str(SNR_car) + "dB"
	noisy_path = noisy_path_parent + os.sep + noisy_sub_path
	tmp_N = scipy.io.loadmat(noisy_path + os.sep + fname) 
	N = tmp_N['Spectrum_out_nor_ctx'] 
	clean_path = top_path_parent + os.sep + "Audio/Preprocess/context_feature/direct_concat/context" + str(context) + os.sep + "clean" + os.sep + speaker + os.sep + "ratio_" + str(ratio)
	tmp_C = scipy.io.loadmat(clean_path + os.sep + fname) 
	C = tmp_C['Spectrum_out_nor_ctx'] 
	visual_path = top_path_parent + os.sep + "Mouth/Preprocess/context_feature/direct_concat/context" + str(context) + os.sep + "nor" + str(nor) + os.sep + speaker
	tmp_V = scipy.io.loadmat(visual_path + os.sep + fname[:-4] + "_mouth.mat") 
	V = tmp_V['subImage_resize_sentence_nor_ctx'] 
	
	return N, C, V

def gen_list(gen_list_name_pre, speaker, in_car, phase, np_noise_type_sets, np_SNR_sets, np_SNR_car_sets, get_train_data_loops, num_lst):

	for i in range(0, num_lst):
		
		# print('		generating list ' + str(i))
		gen_list_name = gen_list_name_pre + ("%03d" % i) + ".list"
		text_file = open(gen_list_name, "w")
		
		for j in range(0, get_train_data_loops/num_lst):
			# if j % 100 == 0:
			# 	print('		iteration: ' + str(j))
			noise_type = random.choice(np_noise_type_sets)
			SNR = random.choice(np_SNR_sets)
			SNR_car = random.choice(np_SNR_car_sets)
			noisy_path_parent = top_path_parent + os.sep + "Audio/Preprocess/context_feature/direct_concat/context" + str(context) + os.sep + "noisy" + os.sep + speaker + os.sep + in_car 
			if phase == 'train':
				noise_type_str = "%03d" % noise_type
			else:
				noise_type_str = "%02d" % noise_type
			noisy_sub_path = "noisy" + noise_type_str + os.sep + phase + os.sep + str(SNR) + "dB/car" + os.sep + str(SNR_car) + "dB"
			noisy_path = noisy_path_parent + os.sep + noisy_sub_path
			list_name = os.listdir(noisy_path) 
			id = np.random.permutation(len(list_name))[0]
			fname = list_name[id]
			noisy_full_path = noisy_path + os.sep + fname
			# print(noisy_full_path)
			text_file.write(noisy_full_path+"\n")

		text_file.close()

def av_pair_with_list(list_full_path):
	sentence = 0
	num_key_error = 0
	with open(list_full_path, 'r') as f:
		for line in f:
			# if sentence % 10 == 0:
				# print("sentence at %d" % sentence)
			speaker = line.split('/')[-8]
			in_car = line.split('/')[-7]
			phase = line.split('/')[-5]
			noise_type = int(line.split('/')[-6][5:])
			SNR = int(line.split('/')[-4][:-2])
			SNR_car = int(line.split('/')[-2][:-2])
			fname = line.split('/')[-1][:-1]
			# print(speaker, in_car, phase, noise_type, SNR, SNR_car, fname)

			try:
				N, C, V = av_pair_assigned_sentence(speaker, in_car, phase, noise_type, SNR, SNR_car, fname)
			except KeyError:
				# print(("KeyError at " + speaker + os.sep + in_car +  os.sep + "noisy" + "%03d" % noise_type + os.sep + phase + os.sep +
				    # str(SNR) + "dB" + os.sep + "car" + os.sep + str(SNR_car) + "dB" + os.sep + fname ))
				# num_key_error = num_key_error + 1
				# print("Number of KeyError (training data): " + str(num_key_error))
				continue
			except ValueError:
				# print(("ValueError at " + speaker + os.sep + in_car +  os.sep + "noisy" + "%03d" % noise_type + os.sep + phase + os.sep +
				    # str(SNR) + "dB" + os.sep + "car" + os.sep + str(SNR_car) + "dB" + os.sep + fname ))
				continue
				
			try:
				N_all = np.concatenate((N_all, N),axis=0)
				C_all = np.concatenate((C_all, C),axis=0)
				V_all = np.concatenate((V_all, V),axis=0)
			except NameError:
				N_all, C_all, V_all = N, C, V
				
			sentence = sentence + 1 

	return N_all, C_all, V_all