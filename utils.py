import torch
import numpy as np
from PIL import Image
from model_irse import IR_34
from model_irse import IR_50
import yaml
import pickle

def create_backbone(depth):
	if depth=='34':
		return IR_34([112,112])
	elif depth=='50':
		return IR_50([112,112])

def get_backbone(model_name, depth):
	if model_name == 'icffr':
		return create_backbone(depth)

def load_backbone(backbone, backbone_path, device):
	pretrained_dict = torch.load(backbone_path)
	backbone.load_state_dict(pretrained_dict)
	backbone.to(device)
	backbone.eval()

def read_img_data(img_path):
	img = Image.open(img_path)
	img = np.array(img)
	img = (img/255.0-0.5)/0.5
	img = img.transpose(2,0,1).reshape(1,3,112,112)
	return img

def get_img_embeddings(img_list, backbone, batch_size=64, device='cuda:0', show_detail=True):
	img_num = len(img_list)
	img_data = []
	for i in range(img_num):
		img_path = img_list[i]
		img_id = img_path.split('_')[0]
		local_path = './data/images/id_' + img_id + '/' + img_path
		img_data.append(read_img_data(local_path))

	embeddings = np.zeros((img_num,512))
	cur_idx = 0
	batch_idx = 0
	while cur_idx < img_num:
		input_num = min(batch_size, img_num-cur_idx)
		input_data = np.concatenate(img_data[cur_idx:cur_idx+input_num], axis=0)
		input_data = torch.Tensor(input_data).to(device).float()
		embedding = backbone(input_data).detach().cpu().numpy()
		embeddings[cur_idx:cur_idx+input_num,:] = embedding
		cur_idx += input_num
		batch_idx += 1
		if show_detail:
			print('processing batch %d/%d'%(batch_idx, (img_num+batch_size-1)//batch_size))
	return embeddings

def load_test_config():
	with open('./test_config.yaml', 'r') as ifs:
		cfg = yaml.safe_load(ifs)
	return cfg

def load_individual_data(nfw_root):
	with open(nfw_root+'/data/individual_id.bin', 'rb') as fr:
		individual_id = pickle.load(fr)

	with open(nfw_root+'/data/individual_img.bin', 'rb') as fr:
		individual_img = pickle.load(fr)

	return individual_id, individual_img

def load_candidate_countries(country_path):
	'''
	Load candidate countries with enough identities
	'''
	candiadate_countries = []
	with open(country_path, 'r') as fr:
		for line in fr.readlines():
			country = line.strip('\n')
			candiadate_countries.append(country)
	return candiadate_countries