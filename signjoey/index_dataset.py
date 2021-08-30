import os, sys, glob, random, time
import importlib
from typing import ValuesView
import numpy as np
import pandas as pd
from PIL import Image
import cv2, pickle
from torch.nn.functional import fold
from tqdm import tqdm
from joblib import delayed, Parallel
import torch
from torch.utils.data import Dataset, DataLoader


class VideoDataset(Dataset):
    def __init__(self, data_dir, 
                transform=None,
                mode='test',
                use_window=True,
                seq_num=32,
                use_cache=True,
                shrink_method='truncate'):
        """
        Args:
            data_dir (string): directory of video frame folders
            label_file: sample items <path, lable> of corresponding dataset split
            transform (callable, optional): Optional transform to be applied on a sample.
            shrink_method: truncate or subsample (cut video frames to length of seq_num)
        """
        self.data_dir = data_dir
        if not os.path.exists(data_dir):
            print('[Error]: dataset NOT exist...')
            print('>>>>> data_dir:%s'%data_dir)
            return
        self.sub_folders = [f for f in os.listdir(self.data_dir) if os.path.isdir(os.path.join(self.data_dir,f))]
        # !sort
        self.sub_folders = sorted(self.sub_folders, key=lambda x: len(os.listdir(os.path.join(self.data_dir, x))))
        self.transform = transform
        self.use_window = use_window
        assert use_window==False, 'only support use_window=False'
        self.shrink_method = shrink_method
        self.num_frames = seq_num
        self.mode = mode
        self.cache_dir = None
        if use_cache:
            root_dir = os.path.abspath(os.path.join(data_dir, '..'))
            self.cache_dir = os.path.join(root_dir, data_dir.split('/')[-1] + '_cache')
            print('>> the cache directory is:%s'%self.cache_dir)
            if not os.path.exists(self.cache_dir):
                os.makedirs(self.cache_dir)

    def __len__(self):
        return len(self.sub_folders)

    def load_frames(self, folder_name, selected_indexs=None):
        def read_img(path):
            #rgb_im = np.array(Image.open(path).convert("RGB"), np.float32)
            rgb_im = Image.open(path).convert("RGB")
            return rgb_im
        sample_path = os.path.join(self.data_dir, folder_name)
        file_list = glob.glob(os.path.join(sample_path, '*.png'))
        file_list.sort()    #! Your are the Evil !!!!! !!!! !!! !! !
        if selected_indexs is None:
            selected_indexs = np.arange(len(file_list))
        rgb_imgs = [read_img(file_list[i]) for i in selected_indexs]
        return rgb_imgs

    def load_frames_cache(self, folder_name, selected_indexs):
        def refresh_load():
            rgb_imgs = self.load_frames(folder_name)
            with open(cache_file, 'wb') as fid:
                pickle.dump(rgb_imgs, fid, pickle.HIGHEST_PROTOCOL)
            return rgb_imgs
        cache_file = os.path.join(self.cache_dir, '%s.pkl'%folder_name)
        if os.path.exists(cache_file):
            with open(cache_file, 'rb') as fid:
                try: rgb_imgs = pickle.load(fid)
                except:
                    rgb_imgs = refresh_load()
        else:
            rgb_imgs = refresh_load()
        frame_seq = [rgb_imgs[i] for i in selected_indexs]
        return frame_seq

    def frame_sampler(self, vlen):
        total = vlen
        MAX_LEN = self.num_frames*4
        if (self.mode == 'test'): # half overlap - 1
            if total-self.num_frames <= 0: # pad left, only sample once
                sequence = np.arange(self.num_frames)
                seq_idx = np.zeros_like(sequence)
                sequence = sequence[sequence < total]
                seq_idx[-len(sequence)::] = sequence
            else:
                available = total-self.num_frames
                start = np.expand_dims(np.arange(0, available+1, self.num_frames//2-1), 1)
                seq_idx = np.expand_dims(np.arange(self.num_frames), 0) + start # [test_sample, num_frames]
                seq_idx = seq_idx.flatten('C')
            valid_len = len(seq_idx)
            if len(seq_idx) > MAX_LEN:
                seq_idx = seq_idx[:MAX_LEN]
            while len(seq_idx) < MAX_LEN:
                seq_idx = np.append(seq_idx, seq_idx[-self.num_frames:])
            assert len(seq_idx) == MAX_LEN
        else: # train or val
            if total-self.num_frames <= 0: # pad left
                sequence = np.arange(self.num_frames) + np.random.choice(range(1),1) # 0
                seq_idx = np.zeros_like(sequence)
                sequence = sequence[sequence < total]
                seq_idx[-len(sequence)::] = sequence
            else:
                start = np.random.choice(range(total-self.num_frames), 1)
                seq_idx = np.arange(self.num_frames) + start
            valid_len = len(seq_idx)
        return seq_idx, valid_len
        
    def get_selected_indexs(self, vlen): 
        if self.use_window:
            frame_index, valid_len = self.frame_sampler(vlen)
        else: #no padding!
            #move padding operation to collate_fn
            if vlen < self.num_frames:
                frame_index = np.arange(vlen)
                valid_len = vlen
            else:
                if self.shrink_method=='truncate':
                    sequence = np.arange(vlen)
                    an = (vlen - self.num_frames)//2
                    en = vlen - self.num_frames - an
                    frame_index = sequence[an: -en]
                    valid_len = self.num_frames
                elif self.shrink_method=='subsample':
                    #stride = vlen//self.num_frames
                    valid_len = vlen
                    assert None, 'only support truncate now!'
                else:
                    raise ValueError
        assert len(frame_index) == valid_len
        return frame_index, valid_len

    def __getitem__(self, idx):
        sub_folder_name =self.sub_folders[idx]
        folder_name = sub_folder_name#os.path.join(self.data_dir,sub_folder_name)
        file_list = glob.glob(os.path.join(self.data_dir, folder_name, '*.png'))
        selected_indexs, valid_len = self.get_selected_indexs(len(file_list))
        if self.cache_dir is None:
            frame_seq = self.load_frames(folder_name, selected_indexs)
        else:
            frame_seq = self.load_frames_cache(folder_name, selected_indexs) 
        #return rgb imgs†
        if self.transform is not None: frame_seq = self.transform(frame_seq)

        #padding C,T,H,W


        #arrays = np.stack([np.array(frame_seq[i], np.float32) for i in frame_index], axis=0)
        if self.mode == 'test':
            return frame_seq, sub_folder_name, valid_len
        return frame_seq, sub_folder_name


class VideoDataset_2Clip(VideoDataset):
    def __init__(self, data_dir, label_file,
                transform=None,
                mode='test',
                use_window=True,
                seq_num=32,
                use_cache=True):
        super(VideoDataset_2Clip, self).__init__(data_dir, label_file, 
                transform,
                mode,
                use_window,
                seq_num,
                use_cache)
        
    def __getitem__(self, idx):
        gloss_id = self.sample2label['label'][idx]
        folder_name = '%05d'%self.sample2label['sample'][idx]
        file_list = glob.glob(os.path.join(self.data_dir, folder_name, '*.png'))
        selected_indexs, valid_len = self.get_selected_indexs(len(file_list))
        if self.cache_dir is None:
            frame_seq = self.load_frames(folder_name, selected_indexs)
        else:
            frame_seq = self.load_frames_cache(folder_name, selected_indexs)
        ## double the frames in the list
        frame_seq.extend(frame_seq)
        if self.transform is not None: frame_seq = self.transform(frame_seq)
        ## convert list to ndarray here: considering the transforms in 'pretrain_infonce.py'
        frame_seq = torch.stack(frame_seq, 1)
        #arrays = np.stack([np.array(frame_seq[i], np.float32) for i in frame_index], axis=0)
        if self.mode == 'test':
            return frame_seq, gloss_id, valid_len
        #print('get_item shape:', frame_seq.shape)
        return frame_seq, gloss_id