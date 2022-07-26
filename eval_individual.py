import sys
import utils
import numpy as np
import torch
from PIL import Image
import utils
import h5py
from sklearn import preprocessing
import time
import random as r
import math
from collections import defaultdict
import os
import csv
import shutil
import random

device = 'cuda:0'

def get_pair_dist(pair_idx, embeddings, batch_size=3000):
	'''
	Get the cosine similarity of normalized pair embeddings
	'''
	pair_num = pair_idx.shape[0]
	dist = torch.zeros(pair_num).float().to(device)
	cur_idx = 0
	batch_idx = 0
	embeddings_cuda = torch.Tensor(embeddings).float().to(device)
	pair_idx_cuda = torch.Tensor(pair_idx).long().to(device)
	while cur_idx<pair_num:
		process_num = min(batch_size, pair_num-cur_idx)
		idx1 = pair_idx_cuda[cur_idx:cur_idx+process_num, 0]
		idx2 = pair_idx_cuda[cur_idx:cur_idx+process_num, 1]
		embedding1 = embeddings_cuda[idx1,:]
		embedding2 = embeddings_cuda[idx2,:]
		process_dist = torch.sum(embedding1*embedding2, axis=1)
		dist[cur_idx:cur_idx+process_num] = process_dist
		cur_idx += process_num
		batch_idx += 1
		print('calculating pair distance, processed %d/%d batch' \
				%(batch_idx, (pair_num+batch_size-1)//batch_size))
	return dist.cpu().numpy()

def get_train_index(pair_idx, pos_pairs, neg_pairs):
	'''
	Get the negative pair indexes
	'''
	train_idx = np.array(range(neg_pairs))
	train_idx = train_idx + pos_pairs
	return train_idx

def get_topk_neg_dist(dist, k=1):
	'''
	Find the topk negative similarities by batches, and using torch.topk(dist) seems to be slower
	'''
	batch_size = 8000000 # should be larger than (FPR * neg_pair_num)
	merge_time = (len(dist) + batch_size - 1) // batch_size
	dist = torch.Tensor(dist).to(device)
	old_sims = -1 * torch.ones(batch_size).to(device)
	cur_idx = 0
	for i in range(merge_time):
		batch_num = min(batch_size, len(dist)-cur_idx)
		batch_sims = dist[cur_idx:cur_idx+batch_num]
		merged_dist = torch.cat((old_sims, batch_sims)) # concatenate with new batch
		old_sims, _ = torch.topk(merged_dist, k=k) # find new topk
		cur_idx += batch_num
		print('sorted %d/%d batches'%(i+1, merge_time))
	return old_sims.cpu().numpy()

def sort_dist(dist, descending=True):
	dist = torch.Tensor(dist).float().to(device)
	if descending:
		dist = -1 * dist
	dist, _ = torch.sort(dist)
	if descending:
		dist = -1 * dist
	return dist.cpu().numpy()

def get_fpr_threshold(fpr, dist):
	print('find the fpr threshold...')
	pair_num = len(dist)
	#sorted_dist = sort_dist(dist, descending=True)
	top_num = math.ceil(fpr * pair_num)
	sorted_dist = get_topk_neg_dist(dist, k=top_num)
	threshold = sorted_dist[top_num-1]
	return threshold

def get_pos_num(id_list, id_img_num):
	'''
	Get the number of all positive pairs
	'''
	pos_pairs = 0
	for img_num in id_img_num.values():
		pos_pairs += img_num * (img_num - 1)//2
	return pos_pairs

def get_neg_num(id_list, id_img_num):
	'''
	Get the number of all negative pairs
	'''
	neg_pairs = 0
	total_img_num = sum(list(id_img_num.values()))
	for i,id1 in enumerate(id_list):
		for id2 in id_list[i+1:]:
			neg_pairs += id_img_num[id1] * id_img_num[id2]
	return neg_pairs

def get_id_img_start_idx(id_list, id_img_num):
	'''
	Get id image start indexes
	'''
	accumulated = np.zeros(len(id_list), dtype=np.int32)
	for i in range(1, len(id_list)):
		accumulated[i] = accumulated[i-1] + id_img_num[id_list[i-1]]
	return accumulated

def get_accumulated_positive_idxes(id_list, id_img_num):
	'''
	Get id-accumulated positive pair indexes
	'''
	positive_idxes = {}
	cur_idx = 0
	for i,id_ in enumerate(id_list):
		img_num = id_img_num[id_list[i]]
		pos_num = img_num * (img_num - 1)//2
		positive_idxes[id_] = np.array(range(pos_num), dtype=np.int32) + cur_idx
		cur_idx += pos_num
	return positive_idxes

def get_accumulated_negative_idxes(id_list, id_img_num):
	'''
	Get id-accumulated negatuve pair indexes
	'''
	negative_idxes = defaultdict(list)
	cur_idx = get_pos_num(id_list, id_img_num)
	for i,id1 in enumerate(id_list):
		for id2 in id_list[i+1:]:
			
			negative_idxes[id1].append(np.array(range(id_img_num[id1]*id_img_num[id2]), dtype=np.int32)+cur_idx)
			negative_idxes[id2].append(np.array(range(id_img_num[id1]*id_img_num[id2]), dtype=np.int32)+cur_idx)
			cur_idx += id_img_num[id1] * id_img_num[id2]
		#break
	print('Max accumulated idx: %d'%(cur_idx))
	new_idxes = {}
	for id_,idxes in negative_idxes.items():
		new_idxes[id_] = np.concatenate(idxes)
	# print(list(negative_idxes.values())[0].shape)
	return new_idxes

def get_pair_idxes(total_pairs, id_list, id_img_num):
	'''
	Get the image index of the corresponding pair
	'''
	print('getting pair image idexes...')
	pair_idxes = np.zeros((total_pairs,2),dtype=np.int32)
	id_num = len(id_list)
	cur_idx = 0
	
	id_img_start_idxes = get_id_img_start_idx(id_list, id_img_num)
	print('getting positive pair idexes...')
	for i in range(id_num):
		for j in range(id_img_num[id_list[i]]):
			for k in range(j+1, id_img_num[id_list[i]]):
				pair_idxes[cur_idx, 0] = id_img_start_idxes[i] + j
				pair_idxes[cur_idx, 1] = id_img_start_idxes[i] + k
				cur_idx += 1

	print('getting negative pair idexes...')
	for i in range(id_num):
		for j in range(i+1, id_num):
			id1 = id_list[i]
			id2 = id_list[j]
			idx1 = np.array(range(id_img_num[id1]),dtype=np.int32) + id_img_start_idxes[i]
			idx1 = idx1.reshape(-1, 1).repeat(id_img_num[id2], 1).reshape(-1)
			idx2 = np.array(range(id_img_num[id2]),dtype=np.int32) + id_img_start_idxes[j]
			idx2 = idx2.reshape(1, -1).repeat(id_img_num[id1], 0).reshape(-1)
			pair_idxes[cur_idx:cur_idx+id_img_num[id1] * id_img_num[id2], 0] = idx1
			pair_idxes[cur_idx:cur_idx+id_img_num[id1] * id_img_num[id2], 1] = idx2
			cur_idx += id_img_num[id1] * id_img_num[id2]
	print('pair number is : %d'%(cur_idx))
	return pair_idxes

def get_id_metric(
					model_name, net_depth, train_set, dist, labels, fpr_thresholds,
					img2idx, idx2img, pair_idx, id_positive_idxes, id_negative_idxes,
					whole_fpr=[1e-5], device='cuda:1'):
	print('getting id metrics...')
	tprs = defaultdict(float)
	fprs = defaultdict(float)

	dist = torch.Tensor(dist).to(device)
	labels = torch.Tensor(labels).to(device)
	#print('%d pair distances'%(dist.shape[0]))
	id_fprs = defaultdict(list)
	id_tprs = defaultdict(list)

	depth = net_depth
	
	for i,threshold in enumerate(fpr_thresholds):
		predicted_issame = torch.ge(dist, threshold)
		actual_issame = torch.ge(labels, 0.5)
		
		for k,id_ in enumerate(id_positive_idxes.keys()):
			
			id_pos_idx = torch.from_numpy(id_positive_idxes[id_]).to(device).long()
			id_neg_idx = torch.from_numpy(id_negative_idxes[id_]).to(device).long()
			id_idxes = torch.cat((id_pos_idx, id_neg_idx), 0)
			#print(torch.max(id_neg_idx))
			tp = torch.sum(torch.logical_and(
												predicted_issame[id_idxes],
												actual_issame[id_idxes]
											)
							).data.item()
			fp = torch.sum(torch.logical_and(
												predicted_issame[id_idxes],
												torch.logical_not(actual_issame[id_idxes])
											)
							).data.item()
			tn = torch.sum(torch.logical_and(
												torch.logical_not(predicted_issame[id_idxes]),
												torch.logical_not(actual_issame[id_idxes])
											)
							).data.item()
			fn = torch.sum(torch.logical_and(
												torch.logical_not(predicted_issame[id_idxes]),
												actual_issame[id_idxes])
							).data.item()


			id_fprs[id_].append(0 if (fp+tn==0) else float(fp) / float(fp+tn))
			id_tprs[id_].append(0 if (tp+fn==0) else float(tp) / float(tp+fn))
			print('processed %d id'%(k+1))

	csv_file = open('./results/individual_result_%s_%s_%s.csv'%(train_set, depth, model_name), 'w', newline='')
	csv_writer = csv.writer(csv_file)
	head = ['id'] + [str(fpr) for fpr in whole_fpr] + [str(fpr) for fpr in whole_fpr]
	csv_writer.writerow(head)
	fpr_array = np.zeros((len(id_positive_idxes), len(fpr_thresholds)))
	tpr_array = np.zeros((len(id_positive_idxes), len(fpr_thresholds)))
	for k,id_ in enumerate(id_positive_idxes.keys()):
		row = [id_] + id_fprs[id_] + id_tprs[id_]
		csv_writer.writerow(row)
		for fpr_idx,id_fpr in enumerate(id_fprs[id_]):
			fpr_array[k,fpr_idx] = id_fpr
		for tpr_idx,id_tpr in enumerate(id_tprs[id_]):
			tpr_array[k,tpr_idx] = id_tpr
	mean_row = ['Mean']
	for fpr_idx,fpr in enumerate(whole_fpr):
		fpr_mean = np.mean(fpr_array[:,fpr_idx]) / fpr
		mean_row.append('%.4f'%(fpr_mean))
	for fpr_idx,fpr in enumerate(whole_fpr):
		tpr_mean = np.mean(tpr_array[:,fpr_idx])*100
		mean_row.append('%.2f'%(tpr_mean))
	std_row = ['Std']
	for fpr_idx,fpr in enumerate(whole_fpr):
		fpr_std = np.std(fpr_array[:,fpr_idx], ddof=1) / fpr
		std_row.append('%.2f'%(fpr_std))
	for fpr_idx,fpr in enumerate(whole_fpr):
		tpr_std = np.std(tpr_array[:,fpr_idx], ddof=1)*100
		std_row.append('%.2f'%(tpr_std))
	csv_writer.writerow(mean_row)
	csv_writer.writerow(std_row)
	csv_file.close()

def test_model(
				backbone, model_name, net_depth, train_set,
				individual_id, individual_img, batch_size=64):
	
	pair_idx = []
	labels = []
	
	img_list = []
	img2idx = {}
	idx2img = {}
	id_list = individual_id
	id_img_num = defaultdict(int)
	idx2id = defaultdict(tuple)
	country_path = './data/satisfied_countries.txt'
	candidate_country = utils.load_candidate_countries(country_path)

	print('id number is : %d'%(len(id_list)))
	img_nums = []
	for id_ in id_list:
		img_list += individual_img[id_]
		id_img_num[id_] = len(individual_img[id_])
		img_nums.append(id_img_num[id_])
	print('img number is : %d'%(len(img_list)))
	for i,img in enumerate(img_list):
		#print(img)
		img2idx[img] = i
		idx2img[i] = img
	
	img_embeddings = utils.get_img_embeddings(img_list, backbone, batch_size, device=device)
	img_embeddings = preprocessing.normalize(img_embeddings)
	id_num = len(id_list)
	
	print('start test')
	time_begin = time.time()
	
	pos_pairs = get_pos_num(id_list, id_img_num)
	neg_pairs = get_neg_num(id_list, id_img_num)
	total_pairs = pos_pairs + neg_pairs
	total_img_num = len(img_list)

	pos_labels = np.ones(pos_pairs, dtype=np.uint8)
	neg_labels = np.zeros(neg_pairs, dtype=np.uint8)
	print('total positive pairs: %d'%(pos_pairs))
	print('total negative pairs: %d'%(neg_pairs))
	
	labels = np.concatenate((pos_labels,neg_labels))

	fprs = [1e-6,1e-5,1e-4,1e-3,1e-2]
	pair_idx = get_pair_idxes(total_pairs, id_list, id_img_num)
	
	pair_dist = get_pair_dist(pair_idx, img_embeddings, batch_size=30000)
	train_idx = get_train_index(pair_idx, pos_pairs, neg_pairs)
	torch.cuda.empty_cache()
	id_pos_idxes = get_accumulated_positive_idxes(id_list, id_img_num)
	id_neg_idxes = get_accumulated_negative_idxes(id_list, id_img_num)
	
	fpr_thresholds = [get_fpr_threshold(fpr, pair_dist[train_idx]) for fpr in fprs]
	print(fpr_thresholds)
	
	get_id_metric(
					model_name, net_depth, train_set, pair_dist, labels, fpr_thresholds,
					img2idx, idx2img,pair_idx, id_pos_idxes, id_neg_idxes, whole_fpr=fprs, device=device)
	time_end = time.time()
	
	print('finish test, consuming time %.6f s'%(time_end-time_begin))

if __name__ == '__main__':

	model_name = 'icffr'
	net_depth = '34'
	train_set = 'ba'

	cfg = utils.load_test_config()
	backbone = utils.get_backbone(model_name, net_depth)
	backbone_path = cfg[model_name]['ir'+net_depth][train_set]

	utils.load_backbone(backbone, backbone_path, device)

	nfw_root = '.'
	individual_id, individual_img = utils.load_individual_data(nfw_root)
	test_model(backbone, model_name, net_depth, train_set, individual_id, individual_img, batch_size=64)
