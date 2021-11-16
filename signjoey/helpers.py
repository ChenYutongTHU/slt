# coding: utf-8
"""
Collection of helper functions
"""
import copy
import glob
import os, math
import os.path
import sys


sys.path.append(os.getcwd())#slt dir
from signjoey import batch
import errno
import shutil
import random
import logging
from sys import platform
from logging import Logger
from typing import Callable, Optional
import numpy as np

import torch
from torch import nn, Tensor
from torchtext.data import Dataset
import yaml
from signjoey.vocabulary import GlossVocabulary, TextVocabulary
import tensorflow as tf
from itertools import groupby

def ctc_decode_func(tf_gloss_probabilities, batch, recognition_beam_size, gloss_scores):
    assert recognition_beam_size > 0
    ctc_decode, _ = tf.nn.ctc_beam_search_decoder(
        inputs=tf_gloss_probabilities,
        sequence_length=batch.sgn_lengths.cpu().detach().numpy(),
        beam_width=recognition_beam_size,
        top_paths=1,
    )
    ctc_decode = ctc_decode[0]
    # Create a decoded gloss list for each sample
    tmp_gloss_sequences = [[] for i in range(gloss_scores.shape[0])]
    for (value_idx, dense_idx) in enumerate(ctc_decode.indices):
        tmp_gloss_sequences[dense_idx[0]].append(
            ctc_decode.values[value_idx].numpy() + 1
        )
    decoded_gloss_sequences = []
    for seq_idx in range(0, len(tmp_gloss_sequences)):
        decoded_gloss_sequences.append(
            [x[0] for x in groupby(tmp_gloss_sequences[seq_idx])]
        )
    return decoded_gloss_sequences

def shift_tokens_right(input_ids: torch.Tensor, pad_token_id: int, ignore_index: int=-100):
    """
    Shift input ids one token to the right, and wrap the last non pad token (the <LID> token) Note that MBart does not
    have a single `decoder_start_token_id` in contrast to other Bart-like models.
    """
    prev_output_tokens = input_ids.clone()

    assert pad_token_id is not None, "self.model.config.pad_token_id has to be defined."
    # replace possible -100 values in labels by `pad_token_id`
    prev_output_tokens.masked_fill_(prev_output_tokens == -100, pad_token_id)
    index_of_eos = (prev_output_tokens.ne(pad_token_id).sum(dim=1) - 1).unsqueeze(-1)
    for ii,ind in enumerate(index_of_eos.squeeze(-1)):
        input_ids[ii, ind:] = ignore_index
    decoder_start_tokens = prev_output_tokens.gather(1, index_of_eos).squeeze()
    prev_output_tokens[:, 1:] = prev_output_tokens[:, :-1].clone()
    prev_output_tokens[:, 0] = decoder_start_tokens

    return prev_output_tokens,input_ids


def traverse(results, cur_res, cur_probs, sp, cur_id):
    if cur_id >= len(sp):
        results.append([cur_res, cur_probs])
        return 
    else:
        for sid, gls_prob in sp[cur_id]:
            cur_res_extend = cur_res +[sid]
            cur_probs_extend = cur_probs + [gls_prob]
            traverse(results, cur_res_extend, cur_probs_extend, sp, cur_id+1)
#traverse
def get_all_combinations(gls_prob):
    t = gls_prob.shape[0]
    sort_idx = torch.argsort(gls_prob, dim=1, descending=True) #T,d
    gls_pred = torch.argmax(gls_prob, axis=1) #T
    num = torch.sum(torch.argmax(gls_prob, axis=1) != 0)
    if not num:
        return [[[i for i in range(t)],[1]*t]]
    spans = []
    i,j = 0,0
    while i<t:
        span_id = []
        while i < t and sort_idx[i, 0] == 0:
            i += 1
        if i >= t:
            break
        j = i
        cur_pred = sort_idx[i, 0]
        while j < t and sort_idx[j, 0] == cur_pred:
            span_id.append([j, gls_prob[j][cur_pred].item()])  # top1
            j += 1
        spans.append(span_id)
        i = j
    
    results = []
    traverse(results, [], [], spans, 0)

    return results

from collections import OrderedDict


def get_spans(sort_idx,topk='top2'):
    pred2spans = OrderedDict()
    i, j = 0, 0
    t = sort_idx.shape[0]
    while i < t:
        span_id = []
        while i < t and sort_idx[i, 0] == 0:
            i += 1
        if i >= t:
            break
        j = i
        cur_pred = sort_idx[i, 0]
        while j < t and sort_idx[j, 0] == cur_pred:
            span_id.append(j)  # top1
            j += 1
        if topk == 'top1':
            pass
        elif topk == 'top2':
            #left span (top2)
            i -= 1
            while i >= 0 and sort_idx[i, 0] == 0 and sort_idx[i, 1] == cur_pred:
                span_id.append(i)
                i -= 1
            #right span
            while j < t and sort_idx[j, 0] == 0 and sort_idx[j, 1] == cur_pred:
                span_id.append(j)
                j += 1
        else:
            raise ValueError
        i = j
        pred2spans[cur_pred] = sorted(span_id)
    return pred2spans
def get_spans_sequence(sort_idx):
    t = sort_idx.shape[0] # T,V
    pred2spans = get_spans(sort_idx)

    spans_pred = []
    pt = 0
    for pred, spans in pred2spans.items():
        if spans[0]>pt:
            spans_pred.append([pt, spans[0],0]) # [start,end), type
            pt = spans[-1]+1
        spans_pred.append([spans[0],spans[-1]+1,pred.item()])
    if pt<t:
        spans_pred.append([pt,t,0])
    return spans_pred
    
def sparse_sample(batch_enc_op, batch_gls_prob, batch_mask, 
    select_strategy='all', return_pred_gls=False,
    return_empty_flag=False):
    #enc_op B,T,D
    #gls_prob B,T,C
    #batch_mask B,1,T
    #select_strategy format ['all','random_num_top1', 'top1/2_mean/max/random/all']
    batch_size = batch_enc_op.shape[0]
    if select_strategy == 'all':
        outputs = (batch_enc_op, batch_mask)
        if return_pred_gls:
            batch_pred_gls = []
            for b in range(batch_size):
                length = torch.sum(batch_mask[b])
                gls_prob = batch_gls_prob[b, :length]
                gls_pred = torch.argmax(gls_prob, axis=1)
                batch_pred_gls.append([gls_pred[i].item() for i in range(length)])
            outputs += (batch_pred_gls,)
        if return_empty_flag:
            outputs += ([],)
        return outputs

    empty_ids=[]
    #assert batch_size == 1, 'currently only support batch_size=1!'
    batch_selected_op = []
    batch_selected_op_len = []
    batch_selected_pred_gls = []
    for b in range(batch_size):
        length = torch.sum(batch_mask[b])
        gls_prob, enc_op = batch_gls_prob[b, :length], batch_enc_op[b, :length]
        selected_op = []
        selected_op_pred_gls = []
        t, v = gls_prob.shape
        t, d = enc_op.shape
        sort_idx = torch.argsort(gls_prob, dim=1, descending=True)  #T D -> T D
        gls_pred = torch.argmax(gls_prob, axis=1)

        num = torch.sum(torch.argmax(gls_prob, axis=1) != 0)
        if select_strategy == 'random_num_top1':
            selected_op_id = np.sort(np.random.permutation(t)[:num])
            # print(selected_op_id)
            selected_op = [enc_op[i, :] for i in selected_op_id]
            selected_op_pred_gls = [gls_pred[i] for i in selected_op_id]
        else:
            topk, agg = select_strategy.split('_')
            assert topk in ['top1', 'top2'], topk
            assert agg in ['mean', 'maxprob', 'random', 'all'], agg
            i, j = 0, 0
            while i < t:
                span_id = []
                while i < t and sort_idx[i, 0] == 0:
                    i += 1
                if i >= t:
                    break
                j = i
                cur_pred = sort_idx[i, 0]
                while j < t and sort_idx[j, 0] == cur_pred:
                    span_id.append(j)  # top1
                    j += 1
                if topk == 'top1':
                    pass
                elif topk == 'top2':
                    #left span (top2)
                    i -= 1
                    while i >= 0 and sort_idx[i, 0] == 0 and sort_idx[i, 1] == cur_pred:
                        span_id.append(i)
                        i -= 1
                    #right span
                    while j < t and sort_idx[j, 0] == 0 and sort_idx[j, 1] == cur_pred:
                        span_id.append(j)
                        j += 1
                else:
                    raise ValueError
                if agg == 'mean':
                    ops = torch.stack([enc_op[id_, :]
                                      for id_ in span_id], dim=0)
                    ops = torch.mean(ops, dim=0)
                    selected_op.append(ops)
                    selected_op_pred_gls.append(cur_pred.item())
                elif agg == 'random':
                    id_ = np.random.choice(span_id)
                    selected_op.append(enc_op[id_, :])
                    selected_op_pred_gls.append(cur_pred.item())
                elif agg == 'all':
                    selected_op += [enc_op[id_, :] for id_ in span_id]
                    selected_op_pred_gls += [cur_pred.item() for id_ in span_id] 
                elif agg == 'maxprob':
                    sorted_span_id = sorted(
                        span_id, key=lambda x: gls_prob[x, cur_pred])[::-1]
                    id_ = sorted_span_id[0]
                    selected_op.append(enc_op[id_, :])
                    selected_op_pred_gls.append(cur_pred.item())
                else:
                    raise ValueError
                i = j
        if selected_op == []:
            selected_op = enc_op
            selected_op_pred_gls = gls_pred.detach().cpu().numpy().tolist() #T,
        else:
            selected_op = torch.stack(selected_op, dim=0)
            #selected_op_pred_gls = torch.tensor(selected_op_pred_gls, dtype=torch.long, device=selected_op.device) #T
        batch_selected_op.append(selected_op)  # T,D
        batch_selected_op_len.append(selected_op.shape[0])
        batch_selected_pred_gls.append(selected_op_pred_gls) #T,

    #padding
    max_len = max(batch_selected_op_len)
    padded_ops = []
    new_mask = torch.zeros([batch_size, 1, max_len],
                           dtype=batch_mask.dtype,
                           device=batch_mask.device)
    for oi, op in enumerate(batch_selected_op):
        #ops_ Td
        op_len = op.shape[0]
        new_mask[oi, :, :op_len] = True
        if op_len < max_len:
            padded = torch.zeros(
                [max_len-op_len, op.shape[1]],
                dtype=op.dtype,
                device=op.device)
            op = torch.cat(
                [op, padded],
                dim=0)  # T',D
        padded_ops.append(op)
    batch_selected_op = torch.stack(padded_ops, dim=0)  # B,T,D
    #I don't think we need to pad pred_gls
    outputs = (batch_selected_op, new_mask)
    if return_pred_gls:
        outputs += (batch_selected_pred_gls,) #list of list
    if return_empty_flag:
        outputs += (empty_ids,)
    return outputs



def get_distributed_sampler_index(total_len, batch_size, rank, world_size):
    indices = list(range(total_len))
    num_replicas = world_size
    num_samples = math.ceil(total_len / num_replicas)
    total_size = num_samples * num_replicas
    padding_size = total_size - len(indices)
    padding_indices = []
    if padding_size <= len(indices):
        indices += [-1]*padding_size
    else:
        indices += [-1]*padding_size
    assert len(indices) == total_size, (len(indices),total_size)
    indices = indices[rank:total_size:num_replicas]
    assert len(indices) == num_samples, (len(indices),num_samples)
    indices = [ind for ind in indices if not ind==-1]
    return indices

def make_model_dir(model_dir: str, overwrite: bool = False) -> str:
    """
    Create a new directory for the model.

    :param model_dir: path to model directory
    :param overwrite: whether to overwrite an existing directory
    :return: path to model directory
    """
    if os.path.isdir(model_dir):
        # if not overwrite:
        #     raise FileExistsError("Model directory exists and overwriting is disabled.")
        # delete previous directory to start with empty dir again
        if overwrite:
            print("overwrite previous model_dir={}".format(model_dir))
            shutil.rmtree(model_dir)
    os.makedirs(model_dir, exist_ok=True)
    return model_dir


def make_logger(model_dir: str, log_file: str = "train.log") -> Logger:
    """
    Create a logger for logging the training process.

    :param model_dir: path to logging directory
    :param log_file: path to logging file
    :return: logger object
    """
    logger = logging.getLogger(__name__)
    if not logger.handlers:
        logger.setLevel(level=logging.DEBUG)
        fh = logging.FileHandler("{}/{}".format(model_dir, log_file))
        fh.setLevel(level=logging.DEBUG)
        logger.addHandler(fh)
        formatter = logging.Formatter("%(asctime)s %(message)s")
        fh.setFormatter(formatter)
        if platform == "linux":
            sh = logging.StreamHandler()
            if not is_main_process():
                sh.setLevel(logging.ERROR)
            sh.setFormatter(formatter)
            logging.getLogger("").addHandler(sh)
        logger.info("Hello! This is Joey-NMT.")
        return logger


def log_cfg(cfg: dict, logger: Logger, prefix: str = "cfg"):
    """
    Write configuration to log.

    :param cfg: configuration to log
    :param logger: logger that defines where log is written to
    :param prefix: prefix for logging
    """
    for k, v in cfg.items():
        if isinstance(v, dict):
            p = ".".join([prefix, k])
            log_cfg(v, logger, prefix=p)
        else:
            p = ".".join([prefix, k])
            logger.info("{:34s} : {}".format(p, v))


def clones(module: nn.Module, n: int) -> nn.ModuleList:
    """
    Produce N identical layers. Transformer helper function.

    :param module: the module to clone
    :param n: clone this many times
    :return cloned modules
    """
    return nn.ModuleList([copy.deepcopy(module) for _ in range(n)])


def subsequent_mask(size: int) -> Tensor:
    """
    Mask out subsequent positions (to prevent attending to future positions)
    Transformer helper function.

    :param size: size of mask (2nd and 3rd dim)
    :return: Tensor with 0s and 1s of shape (1, size, size)
    """
    mask = np.triu(np.ones((1, size, size)), k=1).astype("uint8")
    return torch.from_numpy(mask) == 0


def set_seed(seed: int):
    """
    Set the random seed for modules torch, numpy and random.

    :param seed: random seed
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


def log_data_info(
    train_data: Dataset,
    valid_data: Dataset,
    test_data: Dataset,
    gls_vocab: GlossVocabulary,
    txt_vocab: TextVocabulary,
    logging_function: Callable[[str], None],
):
    """
    Log statistics of data and vocabulary.

    :param train_data:
    :param valid_data:
    :param test_data:
    :param gls_vocab:
    :param txt_vocab:
    :param logging_function:
    """
    logging_function(
        "Data set sizes: \n\ttrain {:d},\n\tvalid {:d},\n\ttest {:d}".format(
            len(train_data),
            len(valid_data),
            len(test_data) if test_data is not None else 0,
        )
    )

    logging_function(
        "First training example:\n\t[GLS] {}\n\t[TXT] {}".format(
            " ".join(vars(train_data[0])["gls"]), " ".join(vars(train_data[0])["txt"])
        )
    )

    logging_function(
        "First 10 words (gls): {}".format(
            " ".join("(%d) %s" % (i, t) for i, t in enumerate(gls_vocab.itos[:10]))
        )
    )
    logging_function(
        "First 10 words (txt): {}".format(
            " ".join("(%d) %s" % (i, t) for i, t in enumerate(txt_vocab.itos[:10]))
        )
    )

    logging_function("Number of unique glosses (types): {}".format(len(gls_vocab)))
    logging_function("Number of unique words (types): {}".format(len(txt_vocab)))


def load_config(path="configs/default.yaml") -> dict:
    """
    Loads and parses a YAML configuration file.

    :param path: path to YAML configuration file
    :return: configuration dictionary
    """
    with open(path, "r", encoding="utf-8") as ymlfile:
        cfg = yaml.safe_load(ymlfile)
    from model_3d import BLOCK2SIZE
    from resnet import LAYER2SIZE
    if cfg['data'].get('input_data','feature') == 'image':

        if 'sentence_frames-512x512' in cfg['data']['img_path']:
            assert cfg['data']['dataset_name']=='csl', (cfg['data']['dataset_name'], 'we need to crop for 512x512')
        elif 'sentence_frames-' in cfg['data']['img_path']:
            assert cfg['data']['dataset_name']!='csl', ('we cannot crop images for ',cfg['data']['dataset_name'])
        if cfg['model']['tokenizer']['architecture'] == 'cnn':
            use_layer = cfg['model']['cnn'].get('use_layer', 4)
            if cfg['data'].get('feature_size', 0) != LAYER2SIZE[use_layer]:
                cfg['data']['feature_size'] = LAYER2SIZE[use_layer]
                print('tokenizer={}, use_layer={} Rewrite feature_size to {}'.format(
                    cfg['model']['tokenizer']['architecture'],
                    use_layer,
                    LAYER2SIZE[use_layer]
                ))
        elif cfg['model']['tokenizer']['architecture'] in ['i3d','s3d','s3ds']:
            use_block=cfg['model']["tokenizer"].get('use_block', 5)
            if cfg['data'].get('feature_size', 1024) != BLOCK2SIZE[use_block]:
                cfg['data']['feature_size'] = BLOCK2SIZE[use_block]
                print('tokenizer={}, use_block={} Rewrite feature_size to {}'.format(
                    cfg['model']['tokenizer']['architecture'],
                    use_block,
                    BLOCK2SIZE[use_block]
                ))
        elif cfg['model']['tokenizer']['architecture'] == 'bntin':
            if cfg['data'].get('feature_size', 1024) != 512:
                cfg['data']['feature_size'] = 512
                print('tokenizer={}, Rewrite feature_size to {}'.format(
                    cfg['model']['tokenizer']['architecture'],
                    512))            
    elif cfg['data'].get('input_data','feature') == 'gloss':
        cfg['training']['recognition_loss_weight'] = 0
        cfg['training']['translation_loss_weight'] = 1
        if cfg['model']['type'] == 'gpt2':
            os.environ['TOKENIZERS_PARALLELISM'] = 'false'
    else:
        name = cfg['data']['train']
        ind = name.find('block')
        if ind!=-1:
            use_block = int(name[ind+5])
            cfg['data']['feature_size'] = BLOCK2SIZE[use_block]
            print('use_block={} Rewrite feature_size to {}'.format(
                use_block,
                BLOCK2SIZE[use_block]
            ))
    # if cfg['training'].get('distillation_loss_weight',0.0)>0:
    #     assert cfg['model'].get('sample_strategy','all')!='all', 'Please use sparse sample strategy when doing distillation'
    return cfg


def bpe_postprocess(string) -> str:
    """
    Post-processor for BPE output. Recombines BPE-split tokens.

    :param string:
    :return: post-processed string
    """
    return string.replace("@@ ", "")


def get_latest_checkpoint(ckpt_dir: str) -> Optional[str]:
    """
    Returns the latest checkpoint (by time) from the given directory.
    If there is no checkpoint in this directory, returns None

    :param ckpt_dir:
    :return: latest checkpoint file
    """
    list_of_files = glob.glob("{}/*.ckpt".format(ckpt_dir))
    latest_checkpoint = None
    if list_of_files:
        latest_checkpoint = max(list_of_files, key=os.path.getctime)
    return latest_checkpoint


def load_checkpoint(path: str, map_location: str='cpu') -> dict:
    """
    Load model from saved checkpoint.

    :param path: path to checkpoint
    :param use_cuda: using cuda or not
    :return: checkpoint (dict)
    """
    assert os.path.isfile(path), "Checkpoint %s not found" % path
    checkpoint = torch.load(path, map_location=map_location)
    return checkpoint


# from onmt
def tile(x: Tensor, count: int, dim=0) -> Tensor:
    """
    Tiles x on dimension dim count times. From OpenNMT. Used for beam search.

    :param x: tensor to tile
    :param count: number of tiles
    :param dim: dimension along which the tensor is tiled
    :return: tiled tensor
    """
    if isinstance(x, tuple):
        h, c = x
        return tile(h, count, dim=dim), tile(c, count, dim=dim)

    perm = list(range(len(x.size())))
    if dim != 0:
        perm[0], perm[dim] = perm[dim], perm[0]
        x = x.permute(perm).contiguous()
    out_size = list(x.size())
    out_size[0] *= count
    batch = x.size(0)
    x = (
        x.view(batch, -1)
        .transpose(0, 1)
        .repeat(count, 1)
        .transpose(0, 1)
        .contiguous()
        .view(*out_size)
    )
    if dim != 0:
        x = x.permute(perm).contiguous()
    return x


def freeze_params(module: nn.Module):
    """
    Freeze the parameters of this module,
    i.e. do not update them during training

    :param module: freeze parameters of this module
    """
    for _, p in module.named_parameters():
        p.requires_grad = False


def symlink_update(target, link_name):
    os.system('cp {} {}'.format(target, link_name))
    # try:
    #     os.symlink(target, link_name)
    # except FileExistsError as e:
    #     if e.errno == errno.EEXIST:
    #         os.remove(link_name)
    #         os.symlink(target, link_name)
    #     else:
    #         raise e

def is_main_process():
    return 'WORLD_SIZE' not in os.environ or os.environ['WORLD_SIZE']=='1' or os.environ['LOCAL_RANK']=='0'


def visualize_bn(model, writer, step):
    for name, param in model.named_parameters():
        #print(name)
        if 'bn' in name or 'sgn_embed.norm' in name:
            #print(name, param.size())
            writer.add_histogram(name, param, step)

    for name, param in model.named_buffers():
        #print(name)
        if 'bn' in name or 'sgn_embed.norm' in name:
            writer.add_histogram(name, param, step)
