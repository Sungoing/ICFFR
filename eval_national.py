import os
import argparse
import sys
import numpy as np
from scipy import misc
from numpy import linalg as line
from sklearn.model_selection import KFold
from scipy import interpolate
import sklearn
from sklearn import preprocessing
import cv2
import math
import datetime
import pickle
from easydict import EasyDict as edict
from sklearn.decomposition import PCA
import torch
import torch.nn as nn
import time
from PIL import Image
from collections import defaultdict
import csv
import shutil
import random
import utils

batch_size = 64
nfolds = 10
img_size = 112
device = 'cuda:0'

class LFold:
    def __init__(self, n_splits = 2, shuffle = False):
        self.n_splits = n_splits
        if self.n_splits>1:
            self.k_fold = KFold(n_splits = n_splits, shuffle = shuffle)

    def split(self, indices):
        if self.n_splits>1:
            return self.k_fold.split(indices)
        else:
            return [(indices, indices)]


def calculate_roc(thresholds, embeddings1, embeddings2, actual_issame, nrof_folds=10, pca = 0):
    assert(embeddings1.shape[0] == embeddings2.shape[0])
    assert(embeddings1.shape[1] == embeddings2.shape[1])
    nrof_pairs = min(len(actual_issame), embeddings1.shape[0])
    nrof_thresholds = len(thresholds)
    k_fold = LFold(n_splits=nrof_folds, shuffle=False)

    tprs = np.zeros((nrof_folds,nrof_thresholds))
    fprs = np.zeros((nrof_folds,nrof_thresholds))
    accuracy = np.zeros((nrof_folds))
    indices = np.arange(nrof_pairs)
    #print('pca', pca)

    if pca==0:
        # diff = np.subtract(embeddings1, embeddings2)
        # dist = np.sum(np.square(diff),1)
        
        #veclist = np.concatenate((embeddings1,embeddings2),axis=0)
        #meana = np.mean(veclist,axis=0)
        #embeddings1 -= meana
        #embeddings2 -= meana
        dist = np.sum(embeddings1 * embeddings2,axis=1)
        # print(embeddings1.shape, embeddings2.shape)
        # dist = dist / line.norm(embeddings1,axis=1) / line.norm(embeddings2,axis=1)
        # print(np.max(dist[:3000]),' ',np.min(dist[:3000]),' ',np.sum(dist[:3000]))
        # print(np.max(dist[-3000:]),' ',np.min(dist[-3000:]),' ',np.sum(dist[-3000:]))
    for fold_idx, (train_set, test_set) in enumerate(k_fold.split(indices)):
        # print('train_set', train_set)
        # print('test_set', test_set)
        if pca>0:
            print('doing pca on', fold_idx)
            embed1_train = embeddings1[train_set]
            embed2_train = embeddings2[train_set]
            _embed_train = np.concatenate( (embed1_train, embed2_train), axis=0 )
            #print(_embed_train.shape)
            pca_model = PCA(n_components=pca)
            pca_model.fit(_embed_train)
            embed1 = pca_model.transform(embeddings1)
            embed2 = pca_model.transform(embeddings2)
            embed1 = sklearn.preprocessing.normalize(embed1)
            embed2 = sklearn.preprocessing.normalize(embed2)
            #print(embed1.shape, embed2.shape)
            diff = np.subtract(embed1, embed2)
            dist = np.sum(np.square(diff),1)
        
        acc_train = np.zeros((nrof_thresholds))
        for threshold_idx, threshold in enumerate(thresholds):
            _, _, acc_train[threshold_idx] = calculate_accuracy(threshold, dist[train_set], actual_issame[train_set])
        best_threshold_index = np.argmax(acc_train)
        
        for threshold_idx, threshold in enumerate(thresholds):
            tprs[fold_idx,threshold_idx], fprs[fold_idx,threshold_idx], _ = calculate_accuracy(threshold, dist[test_set], actual_issame[test_set])
        _, _, accuracy[fold_idx] = calculate_accuracy(thresholds[best_threshold_index], dist[test_set], actual_issame[test_set])
    tpr = np.mean(tprs,0)
    fpr = np.mean(fprs,0)
    return tpr*2, fpr*2, accuracy

def calculate_accuracy(threshold, dist, actual_issame):
    predict_issame = np.greater_equal(dist, threshold)
    actual_issame = np.greater_equal(actual_issame, 0.5)
    tp = np.sum(np.logical_and(predict_issame, actual_issame))
    fp = np.sum(np.logical_and(predict_issame, np.logical_not(actual_issame)))
    tn = np.sum(np.logical_and(np.logical_not(predict_issame), np.logical_not(actual_issame)))
    fn = np.sum(np.logical_and(np.logical_not(predict_issame), actual_issame))
    # print('threshold: %.3f, tp: %d, tn: %d'%(threshold, tp, tn))
    tpr = 0 if (tp+fn==0) else float(tp) / float(tp+fn)
    fpr = 0 if (fp+tn==0) else float(fp) / float(fp+tn)
    # print(dist.size)
    acc = float(tp+tn)/dist.size
    return tpr, fpr, acc

def calculate_val(thresholds, embeddings1, embeddings2, actual_issame, far_target, nrof_folds=10):
    assert(embeddings1.shape[0] == embeddings2.shape[0])
    assert(embeddings1.shape[1] == embeddings2.shape[1])
    nrof_pairs = min(len(actual_issame), embeddings1.shape[0])
    nrof_thresholds = len(thresholds)
    k_fold = LFold(n_splits=nrof_folds, shuffle=False)

    val = np.zeros(nrof_folds)
    far = np.zeros(nrof_folds)

    dist = np.sum(embeddings1 * embeddings2,axis=1)
    
    #dist = dist / line.norm(embeddings1,axis=1) / line.norm(embeddings2,axis=1)
    indices = np.arange(nrof_pairs)

    for fold_idx, (train_set, test_set) in enumerate(k_fold.split(indices)):

        # Find the threshold that gives FAR = far_target
        far_train = np.zeros(nrof_thresholds)
        for threshold_idx, threshold in enumerate(thresholds):
            _, far_train[threshold_idx] = calculate_val_far(threshold, dist[train_set], actual_issame[train_set])
        if np.max(far_train)>=far_target:
            f = interpolate.interp1d(far_train, thresholds, kind='slinear')
            threshold = f(far_target)
        else:
            threshold = 0.0

        val[fold_idx], far[fold_idx] = calculate_val_far(threshold, dist[test_set], actual_issame[test_set])

    val_mean = np.mean(val)
    far_mean = np.mean(far)
    val_std = np.std(val)
    return val_mean, val_std, far_mean


def calculate_val_far(threshold, dist, actual_issame):
    predict_issame = np.greater_equal(dist, threshold)
    true_accept = np.sum(np.logical_and(predict_issame, actual_issame))
    false_accept = np.sum(np.logical_and(predict_issame, np.logical_not(actual_issame)))
    n_same = np.sum(actual_issame)
    n_diff = np.sum(np.logical_not(actual_issame))
    
    val = float(true_accept) / float(n_same+1e-6)
    far = float(false_accept) / float(n_diff+1e-6)
    return val, far

def evaluate(embeddings, actual_issame, nrof_folds=10, pca = 0):
    # Calculate evaluation metrics
    
    thresholds = np.arange(-1,1,0.001)
    embeddings1 = embeddings[0::2]
    embeddings2 = embeddings[1::2]
    #print(actual_issame)
    tpr, fpr, accuracy = calculate_roc(thresholds, embeddings1, embeddings2,
        np.asarray(actual_issame), nrof_folds=nrof_folds, pca = pca)
    # thresholds = np.arange(0, 4, 0.001)
    thresholds = np.arange(-1,1,0.001)
    val, val_std, far = calculate_val(thresholds, embeddings1, embeddings2,
        np.asarray(actual_issame), 1e-3, nrof_folds=nrof_folds)
    return tpr, fpr, accuracy, val, val_std, far

def load_data(country_name, image_size=[112, 112], nfw_root='.'):
    data_list = []
    issame_list = []
    pair_num = 6000
    images = np.empty((pair_num*2, 112, 112, 3))
    issame_list = np.empty(pair_num, dtype=np.int32)
    
    label_cnt = 0
    img_cnt = 0
    image_root = nfw_root+'/data/images/'
    with open(nfw_root+'/data/national_pospairs/'+country_name+'.txt', 'r') as f:
        for pair_line in f.readlines():
            info = pair_line.strip('\n').split(' ')
            img_path1 = image_root + info[0]
            img_path2 = image_root + info[1]

            img_data1 = np.array(Image.open(img_path1))
            img_data2 = np.array(Image.open(img_path2))
            images[img_cnt] = img_data1
            images[img_cnt+1] = img_data2
            issame_list[label_cnt] = int(info[2])
            img_cnt += 2
            label_cnt += 1
    
    with open(nfw_root+'/data/national_negpairs/'+country_name+'.txt', 'r') as f:
        for pair_line in f.readlines():
            info = pair_line.strip('\n').split(' ')
            img_path1 = image_root + info[0]
            img_path2 = image_root + info[1]

            img_data1 = np.array(Image.open(img_path1))
            img_data2 = np.array(Image.open(img_path2))
            images[img_cnt] = img_data1
            images[img_cnt+1] = img_data2
            issame_list[label_cnt] = int(info[2])
            img_cnt += 2
            label_cnt += 1
    
    images = images.transpose(0, 3, 1, 2)
    data_list.append(images)
    return (data_list, issame_list)

def test(data_set, model, batch_size, nfolds=10):
    #print('testing verification..')
    
    data_list = data_set[0]
    issame_list = data_set[1]
    embeddings_list = []
    time_consumed = 0.0
    with torch.no_grad():
        for i in range(len(data_list)):
            data = data_list[i]
            embeddings = None
            ba = 0
            while ba < data.shape[0]:
                bb = min(ba+batch_size, data.shape[0])
                count = bb-ba
                _data = torch.tensor(data[bb-batch_size:bb, ...]).float().to(device)
                time0 = datetime.datetime.now()
                outputs = model(_data)
                _embeddings = outputs.cpu().numpy()
                time_now = datetime.datetime.now()
                diff = time_now - time0
                time_consumed+=diff.total_seconds()
                if embeddings is None:
                    embeddings = np.zeros((data.shape[0], _embeddings.shape[1]))
                embeddings[ba:bb,:] = _embeddings[(batch_size-count):,:]
                ba = bb
            embeddings_list.append(embeddings)

    _xnorm = 0.0
    _xnorm_cnt = 0
    for embed in embeddings_list:
        for i in range(embed.shape[0]):
            _em = embed[i]
            _norm=np.linalg.norm(_em)
            #print(_em.shape, _norm)
            _xnorm+=_norm
            _xnorm_cnt+=1
    _xnorm /= _xnorm_cnt

    acc1 = 0.0
    std1 = 0.0
    embeddings = embeddings_list[0]
    embeddings = sklearn.preprocessing.normalize(embeddings)
    
    tpr, fpr, accuracy, val, val_std, far = evaluate(embeddings, issame_list, nrof_folds=nfolds)
    print(accuracy)
    acc2, std2 = np.mean(accuracy), np.std(accuracy)
    # acc2, std2 = np.max(accuracy), np.std(accuracy)
    return tpr, fpr, acc1, std1, acc2, std2, _xnorm, embeddings_list

def save_tpr_fpr(tpr, fpr, nfw_root, save_name):
    roc_dict = defaultdict(list)
    roc_dict['tpr'] = list(tpr)
    roc_dict['fpr'] = list(fpr)
    with open('%s/national_roc_%s.bin'%(nfw_root, save_name), 'wb') as fw:
        pickle.dump(roc_dict, fw)
    print('National roc of %s is saved.'%(save_name))

if __name__ == '__main__':
    
    model_name = 'icffr'
    net_depth = '34'
    train_set = 'ba'

    cfg = utils.load_test_config()
    backbone = utils.get_backbone(model_name, net_depth)
    backbone_path = cfg[model_name]['ir'+net_depth][train_set]
    utils.load_backbone(backbone, backbone_path, device)

    accuracies = []
    candidate_countries = []
    nfw_root = '.'

    with open('%s/data/satisfied_countries.txt'%(nfw_root), 'r') as f:
        for country_line in f.readlines():
            country_name = country_line.strip('\n')
            candidate_countries.append(country_name)

    acc_dict = {}
    mean_tpr = None
    mean_fpr = None
    for country in candidate_countries:

        ver_list = load_data(country, nfw_root=nfw_root)
        ver_list[0][0] = (ver_list[0][0]/255. - 0.5) / 0.5
        
        results = []
        print('testing country '+country)
        tpr, fpr, acc1, std1, acc2, std2, xnorm, embeddings_list = test(ver_list, backbone, nfolds)
        if mean_tpr is None:
            mean_tpr = tpr
        else:
            mean_tpr = mean_tpr + tpr
        if mean_fpr is None:
            mean_fpr = fpr
        else:
            mean_fpr = mean_fpr + fpr
        print('[%s]XNorm: %f' % (country, xnorm))
        print('[%s]Accuracy: %1.5f+-%1.5f' % (country, acc1, std1))
        print('[%s]Accuracy-Flip: %1.5f+-%1.5f' % (country, acc2, std2))
        results.append(acc2)
        print('Max of [%s] is %1.5f' % (country, np.max(results)))
        accuracies.append(np.max(results))
        acc_dict[country] = np.max(results)*100
    mean_tpr = mean_tpr / len(candidate_countries)
    mean_fpr = mean_fpr / len(candidate_countries)

    depth = net_depth
    #save_tpr_fpr(mean_tpr, mean_fpr, './results', '%s_%s_%s'%(train_set, depth, model_name))
    accuracies = np.array(accuracies)*100
    acc_mean = np.mean(accuracies)
    acc_std = np.std(accuracies, ddof=1)

    csv_file = open('%s/national_result_%s_%s_%s.csv'%('./results', train_set, depth, model_name), 'w', newline='')
    csv_writer = csv.writer(csv_file)
    head = [country for country in candidate_countries].append('Acc')
    for country_name,acc in acc_dict.items():
        row = [country_name, '%.2f'%(acc)]
        csv_writer.writerow(row)
    csv_writer.writerow(['Mean', '%.2f'%(acc_mean)])
    csv_writer.writerow(['Std', '%.2f\n'%(acc_std)])
    csv_file.close()
    print('Mean of accurary: {}, Std of accuracy: {}'.format(acc_mean, acc_std))
