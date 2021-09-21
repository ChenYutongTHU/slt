# define project dependency
from numpy.random import SeedSequence
from models_3d.CoCLR.utils.utils import batch_denorm
import _init_paths
from functools import partial
import os, glob, pickle, random
import numpy as np
from tqdm import tqdm 

import torch
import torch.nn as nn
from torch.utils import data 
from torchvision import transforms
import torch.distributed as dist
import torchvision.utils as vutils
import torch.nn.functional as F
import sys
import utils.augmentation as A
import utils.transforms as T
from utils.utils import neq_load_customized
from dataload_optim import PersistentDataLoader, SoftwarePipeline

from lmdb_dataset import *
from index_dataset import *

pre_task = {
    's3d': 'coclr',
    's3dt': 'milnce',
    's3ds': 'actioncls',
    'i3d': 'glosscls'
}

class ChannelSwap:
    def __init__(self):
        kk = 0
    def __call__(self, tensor_4d):
        # [RGB,t,h,w] -> [BGR,t,h,w]
        return tensor_4d[[2,1,0],:,:,:]


def get_data_transform(mode, dataset_info):
    ## preprocess data (PIL-image list) before batch binding
    if mode == 'train':
        ops = [A.RandomSizedCrop(size=224, consistent=True, bottom_area=0.2),]
        if dataset_info['aug_hflip']:
            ops.append(A.RandomHorizontalFlip())
        ops.append(A.Scale(dataset_info['img_size']))
        if dataset_info['color_jitter']:
            ops.append(A.ColorJitter(0.4, 0.4, 0.4, 0.1, p=0.3, consistent=True))
    elif mode == 'val' or mode == 'test':
        ops = [A.CenterCrop(size=224, consistent=True), A.Scale(dataset_info['img_size']),]
    else:
        raise NotImplementedError

    ## define extra ops per to the pretrained model setting
    ## A.ToTensor(): conver PIL-image to tensor(c,h,w) and normalize to [0,1] by list
    ## from list[img(h,w,c)] to tensor(c,t,h,w)
    if dataset_info['model'] == 's3d' and dataset_info['pretask'] == 'coclr':
        ops.extend([
            A.ToTensor(),
            T.Stack(dim=1),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], channel=0),
        ])
        data_transform = transforms.Compose(ops)
    elif dataset_info['model'] == 's3dt' and dataset_info['pretask'] == 'milnce':
        ## desired data range [0,1]
        ops.extend([
            A.ToTensor(),
            T.Stack(dim=1) #C,T,H,W
        ])
        data_transform = transforms.Compose(ops)
    elif dataset_info['model'] == 's3ds' and dataset_info['pretask'] == 'actioncls':
        ## desired data range [-1,1] and RGB -> BRG
        ops.extend([
            A.ToTensor(),
            T.Stack(dim=1),
            T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], channel=0),
            ChannelSwap(),
        ])
        data_transform = transforms.Compose(ops)
    elif dataset_info['model'] == 'i3d' and dataset_info['pretask'] == 'glosscls':
        ## desired data range [-1,1] and RGB -> BRG
        ops.extend([
            A.ToTensor(),
            T.Stack(dim=1),
            T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], channel=0),
            ChannelSwap(),
        ])
        data_transform = transforms.Compose(ops)
    else:
        print('[Error] no specified data loader is built (check your <model> & <pretask>).')
        raise NotImplementedError
    return data_transform


### data loader settings ---------------------------------------------------
def build_dataloader(dataset_info, split='test', mode='test', gpu_num=1, rank=0, is_mpdist=False, pad_method='zero'):
    '''
    <<dataset_info structure>>:
    'model':
    'pretask':
    'dataset':
    'data_dir':
    'seq_len':
    'use_cache':
    'sampler': sampling 'seq_num' frames from the input video when loaded in 'train' or 'val' mode
    'img_size':
    'aug_hflip':
    'batch_size':
    'num_workers:
    '''
    data_transform = get_data_transform(mode, dataset_info)
    dataset = get_data(dataset_info['dataset'], dataset_info['data_dir'], dataset_info['seq_len'],
                       dataset_info['sampler'], dataset_info['use_cache'], data_transform, split, mode) #return Video
    data_loader = get_dataloader(dataset, mode, dataset_info['batch_size'], dataset_info['num_workers'], gpu_num, rank,
                                 is_mpdist, pad_method)
    print('-created dataset [split:%s | mode:%s]' % (split, mode))
    return data_loader

def padding2maxlength(x, pad_method):
    #x [[seq_len,str,valid_len],[],[],...,[]]
    max_length = max([e[2] for e in x])
    #padding to max_length
    video_inputs, folder_name, valid_len = [], [], []
    for seq, fn, vl in x:
        assert seq.shape[1]==vl, (seq.shape, vl)  #C,T,H,W
        C,_,H,W = seq.shape
        if vl<max_length:
            if pad_method=='zero':#channel1!
                padding = torch.zeros([C,max_length-vl,H,W], dtype=seq.dtype, device=seq.device)
                padded_video_inputs = torch.cat([seq, padding], dim=1) 
            elif pad_method=='replicate':
                padding = seq[:,-1:,:,:] #C,1,H,W
                padding = torch.tile(padding, (1,max_length-vl,1,1)) #C,MAX_LENGTH-VL,H,W
                padded_video_inputs = torch.cat([seq, padding], dim=1)
            else:
                raise ValueError
        else:
            padded_video_inputs = seq
        assert padded_video_inputs.shape == torch.Size([C,max_length,H,W])
        video_inputs.append(padded_video_inputs)
        folder_name.append(fn)
        valid_len.append(vl)
    video_inputs = torch.stack(video_inputs, dim=0) #B,C,T,H,W
    return video_inputs, folder_name, valid_len 

def get_dataloader(dataset, mode, batch_size, num_workers, gpu_num=1, rank=0, is_mpdist=False, pad_method='zero'):
    print("dataset of size %d was created" % (len(dataset)))
    need_shuffle = True if mode == 'train' else False
    #batch_size = 1 if mode == 'test' else batch_size
    drop_last = False if mode == 'test' else True
    collate_fn = lambda x: padding2maxlength(x, pad_method=pad_method)
    if is_mpdist:
        train_sampler = torch.utils.data.distributed.DistributedSampler(dataset, \
            num_replicas=gpu_num, rank=rank, shuffle=need_shuffle)
        dataloader = torch.utils.data.DataLoader(dataset,
                                                 batch_size=batch_size,
                                                 sampler=train_sampler,
                                                 shuffle=False,
                                                 num_workers=num_workers,
                                                 pin_memory=True,
                                                 collate_fn=partial(padding2maxlength, pad_method=pad_method),
                                                 drop_last=drop_last)
        '''
        dataloader = SoftwarePipeline(PersistentDataLoader(dataset,
                                                 batch_size=batch_size,
                                                 sampler=train_sampler,
                                                 shuffle=False,
                                                 num_workers=num_workers,
                                                 pin_memory=True,
                                                 drop_last=drop_last))
        '''
    else:
        dataloader = torch.utils.data.DataLoader(dataset,
                                                 batch_size=batch_size,
                                                 shuffle=need_shuffle,
                                                 num_workers=num_workers,
                                                 collate_fn=collate_fn,
                                                 drop_last=drop_last)
    return dataloader


def get_data(name, data_dir, seq_len, use_sampling, use_cache, transform, split, mode):
    #print('Loading dataset %s for "%s" mode' % (args.dataset, split))
    # if name == 'msasl1000':
    #     frame_root = os.path.join(data_dir, 'MSASL', 'frames1')
    #     label_file = os.path.join(data_dir, 'ASL_dataset', 'DataIndex', 'MSASL1000', '%s.csv'%split)
    #     dataset = VideoDataset(frame_root, label_file, transform, mode, use_sampling, seq_len, use_cache)
    # elif name == 'wlasl2000':
    #     frame_root = os.path.join(data_dir, 'WLASL', 'start_kit', 'frames')
    #     label_file = os.path.join(data_dir, 'ASL_dataset', 'DataIndex', 'WLASL2000', '%s.csv'%split)
    #     dataset = VideoDataset(frame_root, label_file, transform, mode, use_sampling, seq_len, use_cache)
    if name == 'how2sign': #to-do (by yutong)
        frame_root = os.path.join(data_dir, 'WLASL', 'start_kit', 'frames')
        label_file = os.path.join(data_dir, 'ASL_dataset', 'DataIndex', 'WLASL2000', '%s.csv'%split)
        dataset = VideoDataset_2Clip(frame_root, label_file, transform, mode, use_sampling, seq_len, use_cache)
    elif name == 'phoenix':
        frame_root = os.path.join(data_dir, split) #train dev test
        dataset = VideoDataset(frame_root, transform, mode, use_sampling, seq_len, use_cache) #use_sampling
    else:
        raise NotImplementedError
    return dataset


def get_data_lmdb(name, data_dir, seq_len, use_sampling, transform, split, mode):
    #print('Loading dataset %s for "%s" mode' % (args.dataset, mode))
    if name == 'wlasl2000':
        lmdb_root = os.path.join(data_dir, 'ASL_dataset', 'Database_lmdb', 'WLASL2000')
        root = os.path.join(lmdb_root, 'wlasl2000')
        db_path = os.path.join(lmdb_root, 'wlasl2000_frame.lmdb')
        dataset = MSASL1000LMDB(root=root, db_path=db_path, split=split, mode=mode,
            transform=transform, num_frames=seq_len, use_window=use_sampling)
    elif name == 'msasl1000':
        lmdb_root = os.path.join(data_dir, 'ASL_dataset', 'Database_lmdb', 'MSASL1000')
        root = os.path.join(lmdb_root, 'msasl1000')
        db_path = os.path.join(lmdb_root, 'msasl1000_frame.lmdb')
        dataset = MSASL1000LMDB(root=root, db_path=db_path, split=split, mode=mode,
            transform=transform, num_frames=seq_len, use_window=use_sampling)
    else: 
        raise NotImplementedError
    return dataset 


### model loader settings ---------------------------------------------------
def get_premodel_weight(network, pretask, model_without_dp, model_path):
    if not os.path.exists(model_path):
        print("[Error] no pretrained model found at:%s"%model_path)
        return False

    if network == 's3d' and pretask == 'coclr':
        filename = glob.glob(os.path.join(model_path, '*.pth.tar'))
        checkpoint = torch.load(filename[0], map_location='cpu')
        state_dict = checkpoint['state_dict']
        new_dict = {}
        for k,v in state_dict.items():
            k = k.replace('encoder_q.0.', 'backbone.')
            new_dict[k] = v
        state_dict = new_dict
        try: model_without_dp.load_state_dict(state_dict)
        except: neq_load_customized(model_without_dp, state_dict, verbose=False)
    elif network == 's3dt' and pretask == 'milnce':
        filename = glob.glob(os.path.join(model_path, '*.pth'))
        checkpoint = torch.load(filename[0], map_location='cpu')
        state_dict = checkpoint
        new_dict = {}
        for k,v in state_dict.items():
            k = 'backbone.' + k
            new_dict[k] = v
        state_dict = new_dict
        try: model_without_dp.load_state_dict(state_dict)
        except: neq_load_customized(model_without_dp, state_dict, verbose=True)
    elif network == 's3ds' and pretask == 'actioncls':
        filename = glob.glob(os.path.join(model_path, '*.pt'))
        checkpoint = torch.load(filename[0], map_location='cpu')
        state_dict = checkpoint
        new_dict = {}
        for k,v in state_dict.items():
            k = k.replace('module.', 'backbone.')
            new_dict[k] = v
        state_dict = new_dict
        try: model_without_dp.load_state_dict(state_dict)
        except: neq_load_customized(model_without_dp, state_dict, verbose=True)
    elif network == 's3ds' and pretask in ['phoenix', 'k400_phoenix']:
        filename = glob.glob(os.path.join(model_path, '*.ckpt'))
        checkpoint = torch.load(filename[0], map_location='cpu')
        state_dict = checkpoint['model_state']
        new_dict = {}
        for k, v in state_dict.items():
            if 'tokenizer' in k:
                k = k.replace('tokenizer.backbone.', 'backbone.')
                new_dict[k] = v
        state_dict = new_dict
        try:
            model_without_dp.load_state_dict(state_dict)
        except:
            neq_load_customized(model_without_dp, state_dict, verbose=True)
    elif network == 's3ds' and pretask in ['glosscls']:
        filename = glob.glob(os.path.join(model_path, '*.pth.tar'))
        checkpoint = torch.load(filename[0], map_location='cpu')
        state_dict = checkpoint['state_dict']
        try:
            model_without_dp.load_state_dict(state_dict)
        except:
            neq_load_customized(model_without_dp, state_dict, verbose=True)
        print('Succesful load s3ds_glosscls')
    elif network == 'i3d' and pretask == 'glosscls':
        #filename = glob.glob(os.path.join(model_path, '*.pt'))
        #checkpoint = torch.load(filename[0], map_location='cpu')
        ## temp usage: ft from k400 weight
        filename = os.path.join(model_path, 'i3d_kinectics', 'rgb_imagenet.pt')
        checkpoint = torch.load(filename, map_location='cpu')
        state_dict = checkpoint
        new_dict = {}
        for k,v in state_dict.items():
            k = 'backbone.' + k
            new_dict[k] = v
        state_dict = new_dict
        try: model_without_dp.load_state_dict(state_dict)
        except: neq_load_customized(model_without_dp, state_dict, verbose=True)
    else:
        raise NotImplementedError
    return True


def adjust_learning_rate(optimizer, epoch, schedule_milestone=[80,100], lr_scheduler=None):
    '''Decay the learning rate based on schedule'''
    if lr_scheduler:
        lr_scheduler.step()
    else:
        # stepwise lr schedule
        ratio = 0.1 if epoch in schedule_milestone else 1.
        for param_group in optimizer.param_groups:
            param_group['lr'] = param_group['lr'] * ratio


def load_msasl2wlasl(data_dir):
    path = os.path.join(data_dir, 'ASL_dataset', 'msasl2wlasl.txt')
    with open(path, 'r') as f:
        content = f.readlines()
    content = np.array([int(i.strip()) for i in content], np.int16)
    return content


def measure_feature_nmi(feat_list, label_list):
    gt_labels = np.array(label_list, np.int16).squeeze()
    samples = np.array(feat_list, np.float32)
    vec_dim = samples.shape[1]
    print("-clustering via Kmeans (%d samples)..."%len(label_list))
    def kmeans_clutering(samples, dimension, num_clusters):
        samples = np.reshape(samples, [-1,dimension])
        #Estimate bandwidth
        km = KMeans(n_clusters=num_clusters, n_jobs=-1)
        pred = km.fit_predict(samples)
        labels = km.labels_
        return labels

    clusters_labels = kmeans_clutering(samples, dimension=vec_dim, num_clusters=2000)
    score = normalized_mutual_info_score(gt_labels, clusters_labels)
    return score


def load_model_weight(path, mode, model_without_dp, optimizer=None):
    if mode == 'resume':
        if os.path.isfile(path):
            checkpoint = torch.load(path, map_location='cpu')
            start_epoch = checkpoint['epoch']+1
            best_acc = checkpoint['best_acc']
            state_dict = checkpoint['state_dict']
            try: model_without_dp.load_state_dict(state_dict)
            except:
                print('[WARNING] resuming training with different weights')
                neq_load_customized(model_without_dp, state_dict, verbose=True)
            print("@resumed checkpoint (epoch {})".format(checkpoint['epoch']))
            try:
                optimizer.load_state_dict(checkpoint['optimizer'])
            except:
                print('[WARNING] failed to load optimizer state, initialize optimizer')
            return start_epoch, best_acc
        else:
            print("[Warning] no checkpoint found at '{}' and exit...".format(path))
            sys.exit(0)
    elif mode == 'test':
        if not os.path.exists(path):
            print("[Warning] no specified checkpoint found, exiting...")
            print('---%s'%path)
            sys.exit(0)
        checkpoint = torch.load(path, map_location='cpu')
        state_dict = checkpoint['state_dict']
        try: model_without_dp.load_state_dict(state_dict)
        except:
            print('[Error] loaded weight Not Fully matched with specified model.')
            sys.exit(0)
        print('@testing checkpoint loaded.')
    else:
        raise NotImplementedError

    
def set_path(args, gpu_no=0):
    output_path = os.path.join(args.save_dir, 'log_%s_%s_%s-%s'%(args.model, args.pretask, args.eval, args.exp_name))
    if not os.path.exists(output_path) and gpu_no == 0:
        os.makedirs(output_path)
    img_path = os.path.join(output_path, 'img')
    model_path = os.path.join(output_path, 'model')
    if not os.path.exists(img_path) and gpu_no == 0:
        os.makedirs(img_path)
    if not os.path.exists(model_path) and gpu_no == 0:
        os.makedirs(model_path)
    return img_path, model_path, output_path


def sum_reduce_tensor(tensor):
    if not dist.is_available() or \
        not dist.is_initialized():
        return tensor
    rt = tensor.clone()
    dist.all_reduce(rt, op=dist.ReduceOp.SUM)
    return rt

def sum_reduce_scalar(value):
    spawn_tenor = torch.Tensor([value]).cuda()
    spawn_tenor = sum_reduce_tensor(spawn_tenor)
    return spawn_tenor.item()

def mean_reduce_tensor(tensor):
    if not dist.is_available() or \
        not dist.is_initialized():
        return tensor
    rt = tensor.clone()
    dist.all_reduce(rt, op=dist.ReduceOp.SUM)
    world_size = dist.get_world_size()
    rt /= world_size
    return rt

def mean_reduce_scalar(value):
    spawn_tenor = torch.Tensor([value]).cuda()
    spawn_tenor = mean_reduce_tensor(spawn_tenor)
    return spawn_tenor.item()
