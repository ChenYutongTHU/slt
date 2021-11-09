# coding: utf-8
from logging import error
import math, os
import random
import torch
import numpy as np
import torchtext
import _init_paths
from PIL import Image
import torchvision
from utils_3d import get_data_transform, pre_task

def scale_function(x):
    return x*255

transform_dict = {
    'byol': torchvision.transforms.Compose([
        torchvision.transforms.Resize(256),
        torchvision.transforms.CenterCrop(224),
        # Converts a PIL Image or numpy.ndarray (H x W x C) in the range [0, 255] to a torch.FloatTensor of shape (C x H x W) in the range [0.0, 1.0]
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Lambda(scale_function),
        torchvision.transforms.Normalize(mean=[123.675, 116.28, 103.53],
                                         std=[58.395, 57.12, 57.375])
    ]),
    'sup': torchvision.transforms.Compose([
        torchvision.transforms.Resize(256),
        torchvision.transforms.CenterCrop(224),
        # Converts a PIL Image or numpy.ndarray (H x W x C) in the range [0, 255] to a torch.FloatTensor of shape (C x H x W) in the range [0.0, 1.0]
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
    ])
}

        
class Batch:
    """Object for holding a batch of data with mask during training.
    Input is a batch from a torch text iterator.
    """

    def __init__(
        self,
        torch_batch,
        txt_pad_index,
        sgn_dim,
        is_train: bool = False,
        use_cuda: bool = False,
        frame_subsampling_ratio: int = None,
        random_frame_subsampling: bool = None,
        random_frame_masking_ratio: float = None,
    ):
        """
        Create a new joey batch from a torch batch.
        This batch extends torch text's batch attributes with sgn (sign),
        gls (gloss), and txt (text) length, masks, number of non-padded tokens in txt.
        Furthermore, it can be sorted by sgn length.

        :param torch_batch:
        :param txt_pad_index:
        :param sgn_dim:
        :param is_train:
        :param use_cuda:
        :param random_frame_subsampling
        """

        # Sequence Information
        self.sequence = torch_batch.sequence
        self.signer = torch_batch.signer
        # Sign
        self.sgn, self.sgn_lengths = torch_batch.sgn

        # Here be dragons
        if frame_subsampling_ratio:
            tmp_sgn = torch.zeros_like(self.sgn)
            tmp_sgn_lengths = torch.zeros_like(self.sgn_lengths)
            for idx, (features, length) in enumerate(zip(self.sgn, self.sgn_lengths)):
                features = features.clone()
                if random_frame_subsampling and is_train:
                    init_frame = random.randint(0, (frame_subsampling_ratio - 1))
                else:
                    init_frame = math.floor((frame_subsampling_ratio - 1) / 2)

                tmp_data = features[: length.long(), :]
                tmp_data = tmp_data[init_frame::frame_subsampling_ratio]
                tmp_sgn[idx, 0 : tmp_data.shape[0]] = tmp_data
                tmp_sgn_lengths[idx] = tmp_data.shape[0]

            self.sgn = tmp_sgn[:, : tmp_sgn_lengths.max().long(), :]
            self.sgn_lengths = tmp_sgn_lengths

        if random_frame_masking_ratio and is_train:
            tmp_sgn = torch.zeros_like(self.sgn)
            num_mask_frames = (
                (self.sgn_lengths * random_frame_masking_ratio).floor().long()
            )
            for idx, features in enumerate(self.sgn):
                features = features.clone()
                mask_frame_idx = np.random.permutation(
                    int(self.sgn_lengths[idx].long().numpy())
                )[: num_mask_frames[idx]]
                features[mask_frame_idx, :] = 1e-8
                tmp_sgn[idx] = features
            self.sgn = tmp_sgn

        self.sgn_dim = sgn_dim
        self.sgn_mask = (self.sgn != torch.zeros(sgn_dim))[..., 0].unsqueeze(1)

        # Text
        self.txt = None
        self.txt_mask = None
        self.txt_input = None
        self.txt_lengths = None

        # Gloss
        self.gls = None
        self.gls_lengths = None

        # Other
        self.num_txt_tokens = None
        self.num_gls_tokens = None
        self.use_cuda = use_cuda
        self.num_seqs = self.sgn.size(0)

        if hasattr(torch_batch, "txt"):
            txt, txt_lengths = torch_batch.txt
            # txt_input is used for teacher forcing, last one is cut off
            self.txt_input = txt[:, :-1]
            self.txt_lengths = txt_lengths
            # txt is used for loss computation, shifted by one since BOS
            self.txt = txt[:, 1:]
            # we exclude the padded areas from the loss computation
            self.txt_mask = (self.txt_input != txt_pad_index).unsqueeze(1)
            self.num_txt_tokens = (self.txt != txt_pad_index).data.sum().item()

        if hasattr(torch_batch, "gls"):
            self.gls, self.gls_lengths = torch_batch.gls
            self.num_gls_tokens = self.gls_lengths.sum().detach().clone().numpy()

        if use_cuda:
            self._make_cuda()

    def _make_cuda(self):
        """
        Move the batch to GPU

        :return:
        """
        if self.sgn!=None:
            self.sgn = self.sgn.cuda()
        else:
            if self.sgn_img!=None:
                self.sgn_img = self.sgn_img.cuda()
        if self.sgn_mask!=None:
            self.sgn_mask = self.sgn_mask.cuda()

        if self.txt_input is not None:
            self.txt = self.txt.cuda()
            self.txt_mask = self.txt_mask.cuda()
            self.txt_input = self.txt_input.cuda()
        
        if self.input_data == 'gloss':
            self.gls = self.gls.cuda()
            self.gls_mask = self.gls_mask.cuda()

    def sort_by_sgn_lengths(self):
        """
        Sort by sgn length (descending) and return index to revert sort

        :return:
        """

        if self.input_data == 'image':
            rev_index = list(range(0, len(self.sgn_lengths)))
            return rev_index
        if self.input_data == 'gloss':
            rev_index = list(range(0, len(self.gls_lengths)))
            return rev_index
        if self.input_data == 'feature':
            rev_index = list(range(0, len(self.gls_lengths)))
            return rev_index
            #don't sort

        _, perm_index = self.sgn_lengths.sort(0, descending=True)
        rev_index = [0] * perm_index.size(0)
        for new_pos, old_pos in enumerate(perm_index.cpu().numpy()):
            rev_index[old_pos] = new_pos

        self.sgn = self.sgn[perm_index]
        self.sgn_mask = self.sgn_mask[perm_index]
        self.sgn_lengths = self.sgn_lengths[perm_index]

        self.signer = [self.signer[pi] for pi in perm_index]
        self.sequence = [self.sequence[pi] for pi in perm_index]

        if self.gls is not None:
            self.gls = self.gls[perm_index]
            self.gls_lengths = self.gls_lengths[perm_index]

        if self.txt is not None:
            self.txt = self.txt[perm_index]
            self.txt_mask = self.txt_mask[perm_index]
            self.txt_input = self.txt_input[perm_index]
            self.txt_lengths = self.txt_lengths[perm_index]

        if self.use_cuda:
            self._make_cuda()

        return rev_index


class Batch_from_examples(Batch):
    def __init__(
        self,
        example_list,
        txt_pad_index,
        gls_pad_index,
        sgn_dim,
        dataset, 
        input_data: str = 'feature',
        img_path: str = None,
        img_transform: str = None,
        tokenizer_type: str = None, #s3dt ..
        downsample: int=1, 
        max_num_frames: int=400,
        split: str = None,
        is_train: bool = False,
        use_cuda: bool = False,
        frame_subsampling_ratio: int = None,
        random_frame_subsampling: bool = None,
        random_frame_masking_ratio: float = None,
        transform_mode: str='train',
        data_cfg: dict=None,
    ):
        # Sequence Information
        torch_batch = torchtext.data.Batch(data=example_list, dataset=dataset, device=None)
        self.sequence = torch_batch.sequence
        self.signer = torch_batch.signer
        self.input_data = input_data
        self.max_num_frames = max_num_frames
        self.tokenizer_type = tokenizer_type
        self.downsample = downsample

        self.input_data = input_data
        if input_data == 'feature':
            self.sgn, self.sgn_lengths = torch_batch.sgn
            # Here be dragons
            if frame_subsampling_ratio:
                tmp_sgn = torch.zeros_like(self.sgn)
                tmp_sgn_lengths = torch.zeros_like(self.sgn_lengths)
                for idx, (features, length) in enumerate(zip(self.sgn, self.sgn_lengths)):
                    features = features.clone()
                    if random_frame_subsampling and is_train:
                        init_frame = random.randint(
                            0, (frame_subsampling_ratio - 1))
                    else:
                        init_frame = math.floor(
                            (frame_subsampling_ratio - 1) / 2)

                    tmp_data = features[: length.long(), :]
                    tmp_data = tmp_data[init_frame::frame_subsampling_ratio]
                    tmp_sgn[idx, 0: tmp_data.shape[0]] = tmp_data
                    tmp_sgn_lengths[idx] = tmp_data.shape[0]

                self.sgn = tmp_sgn[:, : tmp_sgn_lengths.max().long(), :]
                self.sgn_lengths = tmp_sgn_lengths

            if random_frame_masking_ratio and is_train:
                tmp_sgn = torch.zeros_like(self.sgn)
                num_mask_frames = (
                    (self.sgn_lengths * random_frame_masking_ratio).floor().long()
                )
                for idx, features in enumerate(self.sgn):
                    features = features.clone()
                    mask_frame_idx = np.random.permutation(
                        int(self.sgn_lengths[idx].long().numpy())
                    )[: num_mask_frames[idx]]
                    features[mask_frame_idx, :] = 1e-8
                    tmp_sgn[idx] = features
                self.sgn = tmp_sgn

            self.sgn_dim = sgn_dim
            self.sgn_mask = (self.sgn != torch.zeros(
                sgn_dim))[..., 0].unsqueeze(1)
        elif input_data == 'image' and self.tokenizer_type=='cnn':
            assert split != None, (split)
            if split == 'train':
                assert is_train
            else:
                assert is_train == False
        
            assert img_transform in [
                'byol', 'sup'], 'unsupported img_transform={}'.format(img_transform)
            self.transform = transform_dict[img_transform]
            self.sgn = None
            self.sgn_lengths = []
            self.sgn_img = []
            assert os.path.isdir(img_path), (img_path)
            for idx, name in enumerate(self.sequence):
                seq_folder = os.path.join(img_path, name)
                assert os.path.isdir(seq_folder), seq_folder
                image_path_list = [ss for ss in sorted(os.listdir(seq_folder)) if ss[-4:]=='.png']
                if self.downsample>1:
                    image_path_list = image_path_list[::self.downsample]
                self.sgn_lengths.append(len(image_path_list))  # l0,l1,l2,l3,l4
                for p in image_path_list:
                    p_ = os.path.join(seq_folder, p)
                    pil_img = Image.open(p_).convert('RGB')
                    pil_img_transformed = self.transform(pil_img)  # C,H,W
                    self.sgn_img.append(pil_img_transformed)

            # create sgn_mask according to max_length & sgn_lengths
            self.max_length = max(self.sgn_lengths)
            self.sgn_mask = []
            for length in self.sgn_lengths:
                self.sgn_mask.append([1]*length+[0]*(self.max_length-length))

            self.sgn_lengths = torch.tensor(
                self.sgn_lengths, dtype=torch.long)  # B,
            self.sgn_mask = torch.tensor(self.sgn_mask, dtype=torch.bool).unsqueeze(1)
            self.sgn_img = torch.stack(self.sgn_img, dim=0) #(l1+l2+l3+..l4), C,H,W
        elif input_data == 'image':
            assert downsample==1, (downsample)
            # 3d preprocess, adapted from Menghan's code
            dataset_info = dict()
            dataset_info['model'], dataset_info['pretask'] = tokenizer_type, pre_task[tokenizer_type]
            dataset_info['img_size'] = 224
            dataset_info['aug_hflip'] = False
            dataset_info['use_cache'] = False
            dataset_info['color_jitter'] = data_cfg.get('color_jitter',True)
            dataset_info['dataset_name'] = data_cfg.get('dataset_name','phoenix')
            dataset_info['bottom_area'] = data_cfg.get('bottom_area',0.2)
            self.transform = get_data_transform(
                mode=transform_mode if is_train else 'test', 
                dataset_info=dataset_info)
            self.sgn = None
            self.sgn_lengths = []
            self.sgn_img = []
            unpadded_seq = []

            dat_min=1
            dat_max=1
            mode=transform_mode if is_train else 'test'
            if 'temporal_augmentation' in data_cfg and mode=='train':
                dat_min = data_cfg['temporal_augmentation'].get('min', 1)
                dat_max = data_cfg['temporal_augmentation'].get('max', 1)
            assert os.path.isdir(img_path), (img_path)
            for idx, name in enumerate(self.sequence):
                num_tokens = torch_batch.gls[0][idx].shape[0]
                seq_folder = os.path.join(img_path, name)
                assert os.path.isdir(seq_folder), seq_folder
                image_path_list = [os.path.join(seq_folder, ss) for ss in sorted(
                    os.listdir(seq_folder)) if ss[-4:] in ['.png','.jpg']]
                selected_indexs, valid_len = self.get_selected_indexs(
                    len(image_path_list), tmin=dat_min, tmax=dat_max,
                    level=data_cfg['temporal_augmentation'].get('level','sentence') if 'temporal_augmentation' in data_cfg else 'sentence',
                    num_tokens=num_tokens)
                self.sgn_lengths.append(valid_len)  # l0,l1,l2,l3,l4
                frame_seq = self.load_frames(image_path_list, selected_indexs, 
                    dataset_name=dataset_info['dataset_name'])
                if self.transform is not None: frame_seq = self.transform(frame_seq) #c,t,h,w
                unpadded_seq.append(frame_seq)
            #padding to maxmimum length   
            self.sgn_img = self.padding2maxlength(unpadded_seq) 


            # create sgn_mask according to max_length & sgn_lengths
            self.max_length = max(self.sgn_lengths)
            self.sgn_mask = None
            self.sgn_lengths = torch.tensor(
                self.sgn_lengths, dtype=torch.long)  # B,

        # Text
        self.txt = None
        self.txt_mask = None
        self.txt_input = None
        self.txt_lengths = None

        # Gloss
        self.gls = None
        self.gls_lengths = None

        # Other
        self.num_txt_tokens = None
        self.num_gls_tokens = None
        self.use_cuda = use_cuda

        if hasattr(torch_batch, "txt"):
            txt, txt_lengths = torch_batch.txt
            # txt_input is used for teacher forcing, last one is cut off
            self.txt_input = txt[:, :-1]
            self.txt_lengths = txt_lengths
            # txt is used for loss computation, shifted by one since BOS
            self.txt = txt[:, 1:]
            # we exclude the padded areas from the loss computation
            self.txt_mask = (self.txt_input != txt_pad_index).unsqueeze(1)
            self.num_txt_tokens = (self.txt != txt_pad_index).data.sum().item()

        if hasattr(torch_batch, "gls"):
            self.gls, self.gls_lengths = torch_batch.gls
            self.num_gls_tokens = self.gls_lengths.sum().detach().clone().numpy()

        if input_data == 'gloss':
            self.sgn, self.sgn_mask = None, None
            self.sgn_img = None
            self.gls_mask = (self.gls != gls_pad_index).unsqueeze(1)
            # print(self.gls)  #B,L (pad with 2)
            # print(self.gls_lengths) #B
        self.num_seqs = self.gls_lengths.size(0)
        # if use_cuda:
        #     self._make_cuda()

    def get_selected_indexs(self, vlen, tmin=1, tmax=1, level='sequence', num_tokens=1):
        #num_tokens is ignored when level is set as 'sequence'
        if tmin==1 and tmax==1:
            #output deterministic results
            if vlen <= self.max_num_frames:
                frame_index = np.arange(vlen)
                valid_len = vlen
            else:
                sequence = np.arange(vlen)
                an = (vlen - self.max_num_frames)//2
                en = vlen - self.max_num_frames - an
                frame_index = sequence[an: -en]
                valid_len = self.max_num_frames

            assert len(frame_index) == valid_len, (frame_index, valid_len)
            return frame_index, valid_len
        
        if level=='sentence' or vlen<=num_tokens:
            min_len = int(tmin*vlen)
            max_len = min(self.max_num_frames, int(tmax*vlen))
            selected_len = np.random.randint(min_len, max_len+1)
            if selected_len<=vlen: #speed up
                selected_index = sorted(np.random.permutation(np.arange(vlen))[:selected_len])
            else: #slow down boring video we need to copy some frames
                copied_index = np.random.randint(0,vlen,selected_len-vlen)
                selected_index = sorted(np.concatenate([np.arange(vlen), copied_index]))

            if selected_len <= self.max_num_frames:
                frame_index = selected_index
                valid_len = selected_len
            else:
                assert False, (vlen, selected_len, min_len, max_len)
                sequence = np.arange(vlen)
                an = (vlen - self.max_num_frames)//2
                en = vlen - self.max_num_frames - an
                frame_index = sequence[an: -en]
                valid_len = self.max_num_frames
        elif level=='token':
            frame_index = []
            frames_per_token = vlen//num_tokens
            valid_len = 0
            #print('num_tokens ', num_tokens, 'num_frames ', vlen)
            for i in range(num_tokens):
                s, t = frames_per_token*i, frames_per_token*(i+1)
                if i==num_tokens-1:
                    t = vlen
                token_vlen = t-s
                #print('token ',i, s,t)
                min_len = int(token_vlen*tmin)
                max_len = int(token_vlen*tmax)
                selected_len = np.random.randint(min_len, max_len+1)
                if selected_len <= token_vlen: #speed up
                    selected_index = sorted(np.random.permutation(np.arange(token_vlen))[:selected_len])
                else:
                    copied_index = np.random.randint(0,token_vlen,selected_len-token_vlen)
                    selected_index = sorted(np.concatenate([np.arange(token_vlen), copied_index])) 
                selected_index = np.array(selected_index)+s   
                #print('select  ', selected_index)
                valid_len += selected_len
                frame_index.append(selected_index)    
            frame_index = np.concatenate(frame_index, axis=0)
            #print('finally selected ', frame_index)
            if valid_len>self.max_num_frames:
                #randomly drop some frames
                frame_index = sorted(np.random.permutation(frame_index)[:self.max_num_frames])
                valid_len = self.max_num_frames

        else:
            raise ValueError
        assert len(frame_index) == valid_len, (frame_index, valid_len)
        return frame_index, valid_len

    def load_frames(self, file_list, selected_indexs=None, dataset_name='phoenix'):
        def read_img(path):
            #rgb_im = np.array(Image.open(path).convert("RGB"), np.float32)
            rgb_im = Image.open(path).convert("RGB")
            if dataset_name=='csl':
                rgb_im = rgb_im.crop((0,80,512,512))
            return rgb_im
        if selected_indexs is None:
            selected_indexs = np.arange(len(file_list))
        rgb_imgs = [read_img(file_list[i]) for i in selected_indexs]
        return rgb_imgs

    def padding2maxlength(self, x, pad_method='zero'):
        #x [[seq_len,str,valid_len],[],[],...,[]]
        max_length = max([e.shape[1] for e in x])
        #padding to max_length
        video_inputs = []
        for seq in x:
            C,vl,H,W = seq.shape
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
            assert padded_video_inputs.shape == torch.Size(
                [C, max_length, H, W]), (padded_video_inputs.shape, [C, max_length, H, W])
            video_inputs.append(padded_video_inputs)

        video_inputs = torch.stack(video_inputs, dim=0) #B,C,T,H,W
        return video_inputs 
