#!/usr/bin/env python
import argparse
from collections import defaultdict
from math import pi
from helpers import get_distributed_sampler_index, make_model_dir
import os
from signjoey import data
import torch
from torch.nn.parallel import distributed
from torch.nn.parallel import DistributedDataParallel as DDP, distributed
torch.backends.cudnn.deterministic = True
from tqdm import tqdm
import logging
import numpy as np
import pickle as pickle
import time
import torch.nn as nn

from typing import List
from torchtext.data import Dataset
from tqdm import tqdm
import sys
sys.path.append(os.getcwd())#slt dir
from signjoey.loss import XentLoss
from signjoey.helpers import (
    bpe_postprocess,
    load_config,
    get_latest_checkpoint,
    load_checkpoint,
    make_logger,
    set_seed
)
from signjoey.metrics import bleu, chrf, rouge, wer_list
from signjoey.model import build_model, SignModel, get_loss_for_batch
from signjoey.gloss2text_model import build_gloss2text_model
from signjoey.batch import Batch, Batch_from_examples
from signjoey.data import load_data, make_data_iter
from signjoey.vocabulary import PAD_TOKEN, SIL_TOKEN
from signjoey.phoenix_utils.phoenix_cleanup import (
    clean_phoenix_2014,
    clean_phoenix_2014_trans,
)

def save_immediate_results_fun(
    data,
    output_file_format,
    **kwargs
):
    for k,v in kwargs.items():
        if v==None:
            continue
        output_file = output_file_format.format(k)
        sgn2result = {}
        for seq,  res in zip(data.sequence, 
            v):
            sgn2result[seq] = res
        with open(output_file,'wb') as f:
            pickle.dump(sgn2result,f)  
        print('save {} as {}'.format(k, output_file))

# pylint: disable=too-many-arguments,too-many-locals,no-member
def validate_on_data(
    model: SignModel,
    data: Dataset,
    cfg: dict,
    split: str,
    batch_size: int,
    use_cuda: bool,
    sgn_dim: int,
    do_recognition: bool,
    recognition_loss_function: torch.nn.Module,
    recognition_loss_weight: int,
    do_translation: bool,
    translation_loss_function: torch.nn.Module,
    translation_loss_weight: int,
    translation_max_output_length: int,
    level: str,
    txt_pad_index: int,
    gls_pad_index: int,
    recognition_beam_size: int = 1, # or list
    translation_beam_size: int = 1,
    translation_beam_alpha: int = -1,
    batch_type: str = "sentence",
    dataset_version: str = "phoenix_2014_trans",
    frame_subsampling_ratio: int = None,
    use_amp: bool=False,
    output_attention: bool=False,
    output_feature: bool=False
) -> (
    float,
    float,
    float,
    List[str],
    List[List[str]],
    List[str],
    List[str],
    List[List[str]],
    List[np.array],
):
    """
    Generate translations for the given data.
    If `loss_function` is not None and references are given,
    also compute the loss.

    :param model: torch.nn.parallel.DistributedDataParallel
    :param data: dataset for validation
    :param batch_size: validation batch size
    :param use_cuda: if True, use CUDA
    :param translation_max_output_length: maximum length for generated hypotheses
    :param level: segmentation level, one of "char", "bpe", "word"
    :param translation_loss_function: translation loss function (XEntropy)
    :param recognition_loss_function: recognition loss function (CTC)
    :param recognition_loss_weight: CTC loss weight
    :param translation_loss_weight: Translation loss weight
    :param txt_pad_index: txt padding token index
    :param sgn_dim: Feature dimension of sgn frames
    :param recognition_beam_size: beam size for validation (recognition, i.e. CTC).
        If 0 then greedy decoding (default). or list !
    :param translation_beam_size: beam size for validation (translation).
        If 0 then greedy decoding (default).
    :param translation_beam_alpha: beam search alpha for length penalty (translation),
        disabled if set to -1 (default).
    :param batch_type: validation batch type (sentence or token)
    :param do_recognition: flag for predicting glosses
    :param do_translation: flag for predicting text
    :param dataset_version: phoenix_2014 or phoenix_2014_trans
    :param frame_subsampling_ratio: frame subsampling ratio

    :return:
        - current_valid_score: current validation score [eval_metric],
        - valid_loss: validation loss,
        - valid_ppl:, validation perplexity,
        - valid_sources: validation sources,
        - valid_sources_raw: raw validation sources (before post-processing),
        - valid_references: validation references,
        - valid_hypotheses: validation_hypotheses,
        - decoded_valid: raw validation hypotheses (before post-processing),
        - valid_attention_scores: attention scores for validation hypotheses
    """
    assert type(model) == torch.nn.parallel.DistributedDataParallel
    if type(recognition_beam_size)==list:
        assert do_recognition and not do_translation
    if output_attention:
        assert int(os.environ['WORLD_SIZE'])==1, os.environ['WORLD_SIZE']
    input_data = cfg['data'].get('input_data', 'feature')
    if input_data in ['feature','gloss']:
        tokenizer_type = None
    else:
        tokenizer_type = cfg['model']['tokenizer']['architecture']

    valid_iter, valid_sampler = make_data_iter(
        dataset=data,
        collate_fn=lambda x: Batch_from_examples(
            is_train=False,
            example_list=x,
            txt_pad_index=txt_pad_index,
            gls_pad_index=gls_pad_index,
            sgn_dim=sgn_dim,
            dataset=data,
            input_data=cfg['data'].get('input_data','feature'),
            img_path=cfg['data'].get('img_path', None),
            img_transform=cfg['model']['cnn']['type']
            if tokenizer_type=='cnn'
            else None,
            tokenizer_type=tokenizer_type,
            downsample = cfg['data'].get('downsample', 1),
            max_num_frames=cfg['data']['max_sent_length'],
            split=split,
            use_cuda=use_cuda,
            frame_subsampling_ratio=frame_subsampling_ratio,
            data_cfg=cfg['data']
        ),
        batch_size=batch_size,
        batch_type=batch_type,
        shuffle=False,
        distributed=True, #currently only support distributed!
    )

    # disable dropout
    if cfg['data'].get('eval_mode','eval')=='eval':
        model.module.set_eval()
    else:
        print('Set module.set_train()')
        model.module.set_train()

    model_name = model.module.__class__.__name__
    # don't track gradients during validation
    with torch.no_grad():
        all_gls_outputs = [] if (type(recognition_beam_size)==int or do_recognition==False) else defaultdict(list)
        all_gls_probs = []
        all_txt_outputs = []
        all_attention_scores = [] #generated by run_batch
        if output_attention:
            all_self_attention_scores = [] # generated by CTC encoder (self_attention)
        if output_feature:
            all_sgn_feature = []
            all_encoder_outputs = []
        total_recognition_loss = 0
        total_translation_loss = 0
        total_num_txt_tokens = 0
        total_num_gls_tokens = 0
        total_num_seqs = 0
        split_gls = []
        split_txt = []
        for batch in tqdm(iter(valid_iter), disable=os.environ['WORLD_SIZE']!='1'):
            split_gls.append(batch.gls)
            split_txt.append(batch.txt)

            batch._make_cuda()
            sort_reverse_index = batch.sort_by_sgn_lengths() 
            with torch.cuda.amp.autocast(enabled=use_amp):
                batch_recognition_loss, batch_translation_loss, attention, encoder_outputs = get_loss_for_batch(
                    model=model,
                    batch=batch,
                    recognition_loss_function=recognition_loss_function
                    if do_recognition
                    else None,
                    translation_loss_function=translation_loss_function
                    if do_translation
                    else None,
                    recognition_loss_weight=recognition_loss_weight
                    if do_recognition
                    else None,
                    translation_loss_weight=translation_loss_weight
                    if do_translation
                    else None,
                    input_data=cfg['data'].get('input_data','feature'),
                    output_attention=output_attention
                )
                if output_attention:
                    assert attention!=None #tensor B,Headsize, L, L
                    all_self_attention_scores += [attention[si].detach().cpu().numpy() 
                        for si in sort_reverse_index]

                if output_feature:
                    # T,D
                    all_sgn_feature += [batch.sgn[si].cpu().numpy() for si in sort_reverse_index]
                    all_encoder_outputs += [encoder_outputs[si].cpu().numpy() for si in sort_reverse_index] #T,D

                (
                    batch_gls_predictions,
                    batch_txt_predictions, #already decoded in mBart
                    batch_attention_scores,
                    batch_gls_prob #B, T, C
                ) = model.module.run_batch(
                    batch=batch,
                    recognition_beam_size=recognition_beam_size if do_recognition else None,
                    translation_beam_size=translation_beam_size if do_translation else None,
                    translation_beam_alpha=translation_beam_alpha
                    if do_translation
                    else None,
                    translation_max_output_length=translation_max_output_length
                    if do_translation
                    else None,
                    output_gloss_prob=do_recognition
                )
                
            if do_recognition:
                total_recognition_loss += batch_recognition_loss.item()
                total_num_gls_tokens += batch.num_gls_tokens
            if do_translation:
                total_translation_loss += batch_translation_loss.item()
                total_num_txt_tokens += batch.num_txt_tokens


            total_num_seqs += batch.num_seqs
            # sort outputs back to original order
            if do_recognition:
                if type(all_gls_outputs)==list:
                    all_gls_outputs.extend(
                        [batch_gls_predictions[sri] for sri in sort_reverse_index]
                    )
                else:
                    for k, bp in batch_gls_predictions.items():
                        all_gls_outputs[k].extend(
                            [bp[sri] for sri in sort_reverse_index]
                        )
                all_gls_probs.extend([batch_gls_prob[sri] for sri in sort_reverse_index])# (batch_gls_prob)
            if do_translation:
                if model_name.lower() in ['huggingface_transformer', 'transformer_spm']:
                    for si in sort_reverse_index:
                        all_txt_outputs.append(batch_txt_predictions[si])
                else:
                    all_txt_outputs.extend(batch_txt_predictions[sort_reverse_index])
            if batch_attention_scores:
                all_attention_scores.extend(
                    [batch_attention_scores[si] for si in sort_reverse_index]
                )
    #torch.distributed.barrier()
    torch.cuda.empty_cache()
    #data -> data_split
    split_indices = get_distributed_sampler_index(
        total_len=len(data), batch_size=batch_size,
        rank=int(os.environ['LOCAL_RANK']), world_size=int(os.environ['WORLD_SIZE']))
    valid_scores = {}
    results = {} # return
    if do_recognition:
        def get_recognition_results(all_gls_outputs, results, valid_scores):
            all_gls_outputs = all_gls_outputs[:len(split_indices)]
            assert len(all_gls_outputs) == len(split_indices), (len(all_gls_outputs), len(split_indices))
            if (
                recognition_loss_function is not None
                and recognition_loss_weight != 0
                and total_num_gls_tokens > 0
            ):
                valid_recognition_loss = total_recognition_loss
            else:
                valid_recognition_loss = -1
            # decode back to symbols
            decoded_gls = model.module.gls_vocab.arrays_to_sentences(arrays=all_gls_outputs)

            # Gloss clean-up function
            if dataset_version == "phoenix_2014_trans":
                gls_cln_fn = clean_phoenix_2014_trans
            elif dataset_version == "phoenix_2014":
                gls_cln_fn = clean_phoenix_2014
            else:
                raise ValueError("Unknown Dataset Version: " + dataset_version)

            # Construct gloss sequences for metrics
            gls_ref = [gls_cln_fn(" ".join(t)) for ti, t in enumerate(data.gls) if ti in split_indices]
            gls_hyp = [gls_cln_fn(" ".join(t)) for t in decoded_gls]
            assert len(gls_ref) == len(gls_hyp)

            # GLS Metrics
            gls_wer_score = wer_list(hypotheses=gls_hyp, references=gls_ref)
            valid_scores["wer_scores"] = {k:torch.tensor(s, device='cuda') for k,s in gls_wer_score.items()}
            results["valid_recognition_loss"] = valid_recognition_loss
            results["decoded_gls"] = decoded_gls
            results["gls_ref"] = gls_ref
            results["gls_hyp"] = gls_hyp
            return results, valid_scores
        seq = [seq for ti, seq in enumerate(
            data.sequence) if ti in split_indices]
        if type(all_gls_outputs)==list:
            results, valid_scores = get_recognition_results(all_gls_outputs, results, valid_scores)
            results['sequence'] = seq
        else:# dict
            assert not do_translation
            for k, ao in all_gls_outputs.items():
                results[k], valid_scores[k] = get_recognition_results(ao, 
                    {'sequence': seq}, {})
        results['gls_prob'] = all_gls_probs #N,T,C
            

    if do_translation:
        valid_scores["num_seq"] = torch.tensor(len(split_indices), device='cuda')
        all_txt_outputs = all_txt_outputs[:len(split_indices)]
        assert len(all_txt_outputs) == len(split_indices)
        if (
            translation_loss_function is not None
            and translation_loss_weight != 0
            and total_num_txt_tokens > 0
        ):
            # total validation translation loss
            valid_translation_loss = total_translation_loss
            # exponent of token-level negative log prob
            valid_ppl = np.exp(total_translation_loss / total_num_txt_tokens)
        else:
            valid_translation_loss = -1
            valid_ppl = -1
        # decode back to symbols
        if model_name.lower() in ['huggingface_transformer','transformer_spm']:
            decoded_txt = all_txt_outputs
        else:
            decoded_txt = model.module.txt_vocab.arrays_to_sentences(arrays=all_txt_outputs)
        # evaluate with metric on full data_splitset
        join_char = " " if level in ["word", "bpe"] else ""
        # Construct text sequences for metrics
        data_split_txt = [t for ti, t in enumerate(
            data.txt) if ti in split_indices]
        txt_ref = [join_char.join(t) for t in data_split_txt]
        if model_name.lower() in ['huggingface_transformer', 'transformer_spm']:
            txt_hyp = decoded_txt
        else:
            txt_hyp = [join_char.join(t) for t in decoded_txt]
        # post-process
        if level == "bpe":
            txt_ref = [bpe_postprocess(v) for v in txt_ref]
            txt_hyp = [bpe_postprocess(v) for v in txt_hyp]
        assert len(txt_ref) == len(txt_hyp)

        # TXT Metrics
        #print('ref ', txt_ref)
        #print('hyp ', txt_hyp)
        txt_bleu = bleu(references=txt_ref, hypotheses=txt_hyp)
        txt_chrf = chrf(references=txt_ref, hypotheses=txt_hyp)
        txt_rouge = rouge(references=txt_ref, hypotheses=txt_hyp)

        valid_scores["bleu"] = torch.tensor(txt_bleu["bleu4"], device='cuda')
        valid_scores["bleu_scores"] = {k: torch.tensor(
            s, device='cuda') for k, s in txt_bleu.items()}
        valid_scores["chrf"] = torch.tensor(txt_chrf, device='cuda')
        valid_scores["rouge"] = torch.tensor(txt_rouge, device='cuda')

        results["valid_translation_loss"] = valid_translation_loss
        results["valid_ppl"] = valid_ppl
        results["decoded_txt"] = decoded_txt
        results["txt_ref"] = txt_ref
        results["txt_hyp"] = txt_hyp
        results["sequence"] = [seq for ti, seq in enumerate(data.sequence) if ti in split_indices]
        results['all_attention_scores'] = all_attention_scores

    if output_attention:
        assert int(os.environ['WORLD_SIZE'])==1, os.environ['WORLD_SIZE']
        results['all_self_attention_scores'] = all_self_attention_scores
    if output_feature:
        assert int(os.environ['WORLD_SIZE']) == 1, os.environ['WORLD_SIZE']
        results['all_sgn_feature'] = all_sgn_feature
        results['all_encoder_outputs'] = all_encoder_outputs

    #all gather
    valid_scores_cuda_gather = [None for _ in range(int(os.environ['WORLD_SIZE']))]
    torch.distributed.all_gather_object(
        valid_scores_cuda_gather, valid_scores)
    #compute mean
    # not strictly corpus-level  -> strictly corpus-level wer (by yutong 9/28)
    estimated_mean_scores = {}

    if do_translation:
        # not strictly corpus-level bleu
        for k,v in valid_scores.items():
            if 'wer' in k:
                continue
            if k != 'num_seq':
                if type(v) == dict:
                    estimated_mean_scores[k] = {}
                    for k_, v_ in v.items():
                        estimated_mean_scores[k][k_] = 0
                else:
                    estimated_mean_scores[k] = 0

        total_num = 0
        for ri, scores_split in enumerate(valid_scores_cuda_gather):
            total_num += scores_split['num_seq'].item()

        for ri, scores_split in enumerate(valid_scores_cuda_gather):
            for k, s in scores_split.items():
                if 'wer' in k:
                    continue
                num_seq = scores_split['num_seq'].item()
                if k!='num_seq':
                    if type(s)==dict:
                        for k_, s_ in s.items():
                            s_ = s_.item()
                            estimated_mean_scores[k][k_] += num_seq*s_/total_num
                    else:
                        s = s.item()
                        estimated_mean_scores[k] += num_seq*s/total_num

    if do_recognition:
        def get_recognition_estimated_mean_scores(valid_scores_cuda_gather, estimated_mean_scores):
            total_ref_len, total_del, total_ins, total_sub = 0, 0, 0, 0
            total_error = 0
            for ri, scores_split in enumerate(valid_scores_cuda_gather):
                total_ref_len += scores_split['wer_scores']['ref_len'].item()
                total_del += scores_split['wer_scores']['del'].item()
                total_ins += scores_split['wer_scores']['ins'].item()
                total_sub += scores_split['wer_scores']['sub'].item()
                total_error += scores_split['wer_scores']['error'].item()
            estimated_mean_scores['wer'] = total_error/total_ref_len*100
            estimated_mean_scores['wer_scores'] = {
                'del_rate':total_del/total_ref_len*100,
                'sub_rate':total_sub/total_ref_len*100,
                'ins_rate':total_ins/total_ref_len*100}
            return estimated_mean_scores

        if type(all_gls_outputs)==list:
            estimated_mean_scores = get_recognition_estimated_mean_scores(valid_scores_cuda_gather, estimated_mean_scores)
        else: #dict
            assert not do_translation
            valid_scores_cuda_gather_ = defaultdict(list)
            for ri, rank_dict in enumerate(valid_scores_cuda_gather):
                for k, vg in rank_dict.items():
                    valid_scores_cuda_gather_[k].append(vg)
            for k, vg in valid_scores_cuda_gather_.items():
                estimated_mean_scores[k] = get_recognition_estimated_mean_scores(vg, {})

    #overwrite valid_scores
    if type(all_gls_outputs)==list or do_recognition==False:
        results['valid_scores_gathered'] = {}
        for k, s in estimated_mean_scores.items():
            results['valid_scores_gathered'][k] = s 
    else:
        for k, sc in estimated_mean_scores.items():
            results[k]['valid_scores_gathered'] = sc
    torch.cuda.empty_cache()
    return results


# pylint: disable-msg=logging-too-many-args
def test(
    cfg_file, ckpt: str, output_path: str = None, logger: logging.Logger = None,
    num: int=50, save_immediate_results=False
) -> None:
    """
    Main test function. Handles loading a model from checkpoint, generating
    translations and storing them and attention plots.

    :param cfg_file: path to configuration file
    :param ckpt: path to checkpoint to load
    :param output_path: path to output
    :param logger: log output to this logger (creates new logger if not set)
    """

    if logger is None:
        logger = logging.getLogger(__name__)
        if not logger.handlers:
            FORMAT = "%(asctime)-15s - %(message)s"
            logging.basicConfig(format=FORMAT)
            logger.setLevel(level=logging.DEBUG)

    cfg = load_config(cfg_file)

    if "test" not in cfg["data"].keys():
        raise ValueError("Test data must be specified in config.")

    # when checkpoint is not specified, take latest (best) from model dir
    if ckpt is None:
        model_dir = cfg["training"]["model_dir"]
        ckpt = get_latest_checkpoint(model_dir)
        if ckpt is None:
            raise FileNotFoundError(
                "No checkpoint found in directory {}.".format(model_dir)
            )
    batch_size = cfg["training"]["batch_size"]
    batch_type = cfg["training"].get("batch_type", "sentence")
    use_cuda = cfg["training"].get("use_cuda", False)
    level = cfg["data"]["level"]
    dataset_version = cfg["data"].get("version", "phoenix_2014_trans")
    translation_max_output_length = cfg["training"].get(
        "translation_max_output_length", None
    )
    # load vocab
    vocab_dir = os.path.dirname(ckpt) if ckpt else model_dir
    cfg['data']['gls_vocab'] = os.path.join(os.path.join(vocab_dir, 'gls.vocab'))
    cfg['data']['txt_vocab'] = os.path.join(os.path.join(vocab_dir, 'txt.vocab'))
    print('Load  vocab file from ', vocab_dir)
    # load the data
    _, dev_data, test_data, gls_vocab, txt_vocab, _, _ = load_data(data_cfg=cfg["data"])
    if save_immediate_results and num>0:
        set_seed(0)
        dev_data, _ = dev_data.split(split_ratio=num/len(dev_data))

    # load model state from disk
    model_checkpoint = load_checkpoint(ckpt)#, map_location='cuda:0') #-default to gpu

    # build model and load parameters into it
    do_recognition = cfg["training"].get("recognition_loss_weight", 1.0) > 0.0
    do_translation = cfg["training"].get("translation_loss_weight", 1.0) > 0.0
    input_data = cfg["data"].get("input_data", "feature")
    if input_data == 'feature':
        model = build_model(
            cfg=cfg["model"],
            gls_vocab=gls_vocab,
            txt_vocab=txt_vocab,
            sgn_dim=sum(cfg["data"]["feature_size"])
            if isinstance(cfg["data"]["feature_size"], list)
            else cfg["data"]["feature_size"],
            do_recognition=do_recognition,
            do_translation=do_translation,
        )
    elif input_data == 'image':
        if cfg["model"]["tokenizer"]["architecture"] == 'cnn':
            assert cfg["data"]["feature_size"] == 2048, 'feature_size={}? When input_data is img->cnn, only support resnet50 logits.'.format(
                cfg["data"]["feature_size"])

        model = build_model(
            cfg=cfg["model"],
            gls_vocab=gls_vocab,
            txt_vocab=txt_vocab,
            sgn_dim=sum(cfg["data"]["feature_size"])
            if isinstance(cfg["data"]["feature_size"], list)
            else cfg["data"]["feature_size"],
            do_recognition=do_recognition,
            do_translation=do_translation,
            input_data=input_data
        )
    elif input_data=='gloss':
        model = build_gloss2text_model(
            cfg=cfg["model"],
            gls_vocab=gls_vocab,
            txt_vocab=txt_vocab,
        )

    model.load_state_dict(model_checkpoint["model_state"])
    logger.info('Load Model state dict from {}'.format(ckpt))
    if use_cuda:
        model.cuda()
        assert 'LOCAL_RANK' in os.environ, 'Only support distributed training'
        local_rank = int(os.environ['LOCAL_RANK'])
        model = DDP(model, device_ids=[
            local_rank], output_device=local_rank)

    # Data Augmentation Parameters
    frame_subsampling_ratio = cfg["data"].get("frame_subsampling_ratio", None)
    # Note (Cihan): we are not using 'random_frame_subsampling' and
    #   'random_frame_masking_ratio' in testing as they are just for training.

    # whether to use beam search for decoding, 0: greedy decoding
    if "testing" in cfg.keys() and not save_immediate_results:
        recognition_beam_sizes = cfg["testing"].get("recognition_beam_sizes", [1])
        translation_beam_sizes = cfg["testing"].get("translation_beam_sizes", [1])
        translation_beam_alphas = cfg["testing"].get("translation_beam_alphas", [-1])
    else:
        recognition_beam_sizes = [1]
        translation_beam_sizes = [1]
        translation_beam_alphas = [-1]
        # cfg["testing"]['output_attention'] = True
        # cfg["testing"]['output_feature'] = True written in config
    if "testing" in cfg.keys():
        max_recognition_beam_size = cfg["testing"].get(
            "max_recognition_beam_size", None
        )
        if max_recognition_beam_size is not None:
            recognition_beam_sizes = list(range(1, max_recognition_beam_size + 1))

    if do_recognition:
        recognition_loss_function = torch.nn.CTCLoss(
            blank=model.module.gls_vocab.stoi[SIL_TOKEN], zero_infinity=True
        )
        if use_cuda:
            recognition_loss_function.cuda()
    if do_translation:
        translation_loss_function = XentLoss(
            pad_index=txt_vocab.stoi[PAD_TOKEN], smoothing=0.0
        )
        if use_cuda:
            translation_loss_function.cuda()

    # NOTE (Cihan): Currently Hardcoded to be 0 for TensorFlow decoding
    if do_recognition:
        assert model.module.gls_vocab.stoi[SIL_TOKEN] == 0

    if do_recognition:
        # Dev Recognition CTC Beam Search Results
        dev_recognition_results = {}
        dev_best_wer_score = float("inf")
        dev_best_recognition_beam_size = 1
        logger.info("-" * 60)
        valid_start_time = time.time()
        logger.info("[DEV] partition [RECOGNITION] experiment [BW]: {}".format(recognition_beam_sizes))
        #!!
        model.module.do_translation = False
        dev_recognition_results = validate_on_data(
            model=model,
            data=dev_data,
            split='dev',
            cfg=cfg,
            batch_size=batch_size,
            use_cuda=use_cuda,
            batch_type=batch_type,
            dataset_version=dataset_version,
            sgn_dim=sum(cfg["data"]["feature_size"])
            if isinstance(cfg["data"]["feature_size"], list)
            else cfg["data"]["feature_size"],
            txt_pad_index=txt_vocab.stoi[PAD_TOKEN],
            gls_pad_index=gls_vocab.stoi[PAD_TOKEN],
            # Recognition Parameters
            do_recognition=do_recognition,
            recognition_loss_function=recognition_loss_function if do_recognition else None,
            recognition_loss_weight=1,
            recognition_beam_size=recognition_beam_sizes,
            # Translation Parameters
            do_translation=False,
            translation_loss_function=None,
            translation_loss_weight=None,
            translation_max_output_length=None,
            level=None,
            translation_beam_size= None,
            translation_beam_alpha= None,
            frame_subsampling_ratio=frame_subsampling_ratio,
            use_amp = cfg["training"].get('use_amp',False),
            output_attention=cfg["testing"].get("output_attention",False),
            output_feature=cfg["testing"].get("output_feature", False)
        ) #return {'}
        model.module.do_translation = do_translation
        logger.info("finished in %.4fs ", time.time() - valid_start_time)

        if save_immediate_results:
            save_immediate_results_fun(
                data=dev_data,
                output_file_format=output_path+'.{}.dev',
                gls_prob=dev_recognition_results['gls_prob'] if 'gls_prob' in dev_recognition_results else None,
                encoder_attention=dev_recognition_results['all_self_attention_scores'] if 'all_self_attention_scores' in dev_recognition_results else None,
                sgn_feature=dev_recognition_results['all_sgn_feature'],
                encoder_outputs=dev_recognition_results['all_encoder_outputs'])

        for rbw in recognition_beam_sizes:
            if dev_recognition_results[rbw]["valid_scores_gathered"]["wer"] < dev_best_wer_score:
                dev_best_wer_score = dev_recognition_results[rbw]["valid_scores_gathered"]["wer"]
                dev_best_recognition_beam_size = rbw
                dev_best_recognition_result = dev_recognition_results[rbw]
                logger.info("*" * 60)
                logger.info(
                    "[DEV] partition [RECOGNITION] results:\n\t"
                    "New Best CTC Decode Beam Size: %d\n\t"
                    "WER %3.2f\t(DEL: %3.2f,\tINS: %3.2f,\tSUB: %3.2f)",
                    dev_best_recognition_beam_size,
                    dev_best_recognition_result["valid_scores_gathered"]["wer"],
                    dev_best_recognition_result["valid_scores_gathered"]["wer_scores"][
                        "del_rate"
                    ],
                    dev_best_recognition_result["valid_scores_gathered"]["wer_scores"][
                        "ins_rate"
                    ],
                    dev_best_recognition_result["valid_scores_gathered"]["wer_scores"][
                        "sub_rate"
                    ],
                )
                logger.info("*" * 60)
    if do_translation:
        logger.info("=" * 60)
        dev_translation_results = {}
        dev_best_bleu_score = float("-inf")
        dev_best_translation_beam_size = 1
        dev_best_translation_alpha = 1
        for tbw in translation_beam_sizes:
            dev_translation_results[tbw] = {}
            for ta in translation_beam_alphas:
                dev_translation_results[tbw][ta] = validate_on_data(
                    model=model,
                    data=dev_data,
                    split='dev',
                    cfg=cfg,
                    batch_size=batch_size,
                    use_cuda=use_cuda,
                    level=level,
                    sgn_dim=sum(cfg["data"]["feature_size"])
                    if isinstance(cfg["data"]["feature_size"], list)
                    else cfg["data"]["feature_size"],
                    batch_type=batch_type,
                    dataset_version=dataset_version,
                    do_recognition=do_recognition,
                    recognition_loss_function=recognition_loss_function
                    if do_recognition
                    else None,
                    recognition_loss_weight=1 if do_recognition else None,
                    recognition_beam_size=1 if do_recognition else None,
                    do_translation=do_translation,
                    translation_loss_function=translation_loss_function,
                    translation_loss_weight=1,
                    translation_max_output_length=translation_max_output_length,
                    txt_pad_index=txt_vocab.stoi[PAD_TOKEN],
                    gls_pad_index=gls_vocab.stoi[PAD_TOKEN],
                    translation_beam_size=tbw,
                    translation_beam_alpha=ta,
                    frame_subsampling_ratio=frame_subsampling_ratio,
                    output_feature=cfg["testing"].get("output_feature",False),
                    output_attention=cfg["testing"].get(
                        "output_attention", False)
                )
                if save_immediate_results:
                    save_immediate_results_fun(
                        data=dev_data,
                        output_file_format=output_path+'.{}.dev_'+str(tbw)+'_'+str(ta),
                        decoder_attention=dev_translation_results[tbw][ta]['all_attention_scores'])
                    if not do_recognition:
                        save_immediate_results_fun(
                            data=dev_data,
                            output_file_format=output_path+'.{}.dev',
                            encoder_attention=dev_translation_results[tbw][ta]['all_self_attention_scores'] if 'all_self_attention_scores' in dev_translation_results[tbw][ta] else None,
                            encoder_outputs=dev_translation_results[tbw][ta]['all_encoder_outputs'],
                            sgn_feature=dev_translation_results[tbw][ta]['all_sgn_feature'])

                if (
                    dev_translation_results[tbw][ta]["valid_scores_gathered"]["bleu"]
                    > dev_best_bleu_score
                ):
                    logger.info('New Best! beam_width={} alpha={}'.format(tbw, ta))
                    dev_best_bleu_score = dev_translation_results[tbw][ta][
                        "valid_scores_gathered"
                    ]["bleu"]
                    dev_best_translation_beam_size = tbw
                    dev_best_translation_alpha = ta
                    dev_best_translation_result = dev_translation_results[tbw][ta]
                logger.info(
                    "[DEV] partition [Translation] results:\n\t"
                    "Translation Beam Size: %d and Alpha: %.2f\n\t"
                    "BLEU-4 %.2f\t(BLEU-1: %.2f,\tBLEU-2: %.2f,\tBLEU-3: %.2f,\tBLEU-4: %.2f)\n\t"
                    "CHRF %.2f\t"
                    "ROUGE %.2f",
                    tbw,
                    ta,
                    dev_translation_results[tbw][ta]["valid_scores_gathered"]["bleu"],
                    dev_translation_results[tbw][ta]["valid_scores_gathered"]["bleu_scores"][
                        "bleu1"
                    ],
                    dev_translation_results[tbw][ta]["valid_scores_gathered"]["bleu_scores"][
                        "bleu2"
                    ],
                    dev_translation_results[tbw][ta]["valid_scores_gathered"]["bleu_scores"][
                        "bleu3"
                    ],
                    dev_translation_results[tbw][ta]["valid_scores_gathered"]["bleu_scores"][
                        "bleu4"
                    ],
                    dev_translation_results[tbw][ta]["valid_scores_gathered"]["chrf"],
                    dev_translation_results[tbw][ta]["valid_scores_gathered"]["rouge"],
                )
                logger.info("-" * 60)

    logger.info("*" * 60)
    logger.info(
        "[DEV] partition [Recognition & Translation] results:\n\t"
        "Best CTC Decode Beam Size: %d\n\t"
        "Best Translation Beam Size: %d and Alpha: %.2f\n\t"
        "WER %3.2f\t(DEL: %3.2f,\tINS: %3.2f,\tSUB: %3.2f)\n\t"
        "BLEU-4 %.2f\t(BLEU-1: %.2f,\tBLEU-2: %.2f,\tBLEU-3: %.2f,\tBLEU-4: %.2f)\n\t"
        "CHRF %.2f\t"
        "ROUGE %.2f",
        dev_best_recognition_beam_size if do_recognition else -1,
        dev_best_translation_beam_size if do_translation else -1,
        dev_best_translation_alpha if do_translation else -1,
        dev_best_recognition_result["valid_scores_gathered"]["wer"] if do_recognition else -1,
        dev_best_recognition_result["valid_scores_gathered"]["wer_scores"]["del_rate"]
        if do_recognition
        else -1,
        dev_best_recognition_result["valid_scores_gathered"]["wer_scores"]["ins_rate"]
        if do_recognition
        else -1,
        dev_best_recognition_result["valid_scores_gathered"]["wer_scores"]["sub_rate"]
        if do_recognition
        else -1,
        dev_best_translation_result["valid_scores_gathered"]["bleu"] if do_translation else -1,
        dev_best_translation_result["valid_scores_gathered"]["bleu_scores"]["bleu1"]
        if do_translation
        else -1,
        dev_best_translation_result["valid_scores_gathered"]["bleu_scores"]["bleu2"]
        if do_translation
        else -1,
        dev_best_translation_result["valid_scores_gathered"]["bleu_scores"]["bleu3"]
        if do_translation
        else -1,
        dev_best_translation_result["valid_scores_gathered"]["bleu_scores"]["bleu4"]
        if do_translation
        else -1,
        dev_best_translation_result["valid_scores_gathered"]["chrf"] if do_translation else -1,
        dev_best_translation_result["valid_scores_gathered"]["rouge"] if do_translation else -1,
    )
    logger.info("*" * 60)

    test_best_result = validate_on_data(
        model=model,
        data=test_data,
        split='test',
        cfg=cfg,
        batch_size=batch_size,
        use_cuda=use_cuda,
        batch_type=batch_type,
        dataset_version=dataset_version,
        sgn_dim=sum(cfg["data"]["feature_size"])
        if isinstance(cfg["data"]["feature_size"], list)
        else cfg["data"]["feature_size"],
        txt_pad_index=txt_vocab.stoi[PAD_TOKEN],
        gls_pad_index=gls_vocab.stoi[PAD_TOKEN],
        do_recognition=do_recognition,
        recognition_loss_function=recognition_loss_function if do_recognition else None,
        recognition_loss_weight=1 if do_recognition else None,
        recognition_beam_size=dev_best_recognition_beam_size
        if do_recognition
        else None,
        do_translation=do_translation,
        translation_loss_function=translation_loss_function if do_translation else None,
        translation_loss_weight=1 if do_translation else None,
        translation_max_output_length=translation_max_output_length
        if do_translation
        else None,
        level=level if do_translation else None,
        translation_beam_size=dev_best_translation_beam_size
        if do_translation
        else None,
        translation_beam_alpha=dev_best_translation_alpha if do_translation else None,
        frame_subsampling_ratio=frame_subsampling_ratio,
    )

    logger.info(
        "[TEST] partition [Recognition & Translation] results:\n\t"
        "Best CTC Decode Beam Size: %d\n\t"
        "Best Translation Beam Size: %d and Alpha: %.2f\n\t"
        "WER %3.2f\t(DEL: %3.2f,\tINS: %3.2f,\tSUB: %3.2f)\n\t"
        "BLEU-4 %.2f\t(BLEU-1: %.2f,\tBLEU-2: %.2f,\tBLEU-3: %.2f,\tBLEU-4: %.2f)\n\t"
        "CHRF %.2f\t"
        "ROUGE %.2f",
        dev_best_recognition_beam_size if do_recognition else -1,
        dev_best_translation_beam_size if do_translation else -1,
        dev_best_translation_alpha if do_translation else -1,
        test_best_result["valid_scores_gathered"]["wer"] if do_recognition else -1,
        test_best_result["valid_scores_gathered"]["wer_scores"]["del_rate"]
        if do_recognition
        else -1,
        test_best_result["valid_scores_gathered"]["wer_scores"]["ins_rate"]
        if do_recognition
        else -1,
        test_best_result["valid_scores_gathered"]["wer_scores"]["sub_rate"]
        if do_recognition
        else -1,
        test_best_result["valid_scores_gathered"]["bleu"] if do_translation else -1,
        test_best_result["valid_scores_gathered"]["bleu_scores"]["bleu1"]
        if do_translation
        else -1,
        test_best_result["valid_scores_gathered"]["bleu_scores"]["bleu2"]
        if do_translation
        else -1,
        test_best_result["valid_scores_gathered"]["bleu_scores"]["bleu3"]
        if do_translation
        else -1,
        test_best_result["valid_scores_gathered"]["bleu_scores"]["bleu4"]
        if do_translation
        else -1,
        test_best_result["valid_scores_gathered"]["chrf"] if do_translation else -1,
        test_best_result["valid_scores_gathered"]["rouge"] if do_translation else -1,
    )
    logger.info("*" * 60)

    def _write_to_file(file_path: str, sequence_ids: List[str], hypotheses: List[str]):
        with open(file_path, mode="w", encoding="utf-8") as out_file:
            for seq, hyp in zip(sequence_ids, hypotheses):
                out_file.write(seq + "|" + hyp + "\n")

    if output_path is not None:
        if do_recognition:
            dev_gls_output_path_set = "{}.BW_{:03d}.{}.gls.rank{}".format(
                output_path, dev_best_recognition_beam_size, "dev", os.environ['LOCAL_RANK']
            )
            _write_to_file(
                dev_gls_output_path_set,
                dev_best_recognition_result["sequence"],
                dev_best_recognition_result["gls_hyp"],
            )
            test_gls_output_path_set = "{}.BW_{:03d}.{}.gls.rank{}".format(
                output_path, dev_best_recognition_beam_size, "test", os.environ['LOCAL_RANK']
            )
            _write_to_file(
                test_gls_output_path_set,
                test_best_result["sequence"],
                test_best_result["gls_hyp"],
            )

        if do_translation:
            if dev_best_translation_beam_size > -1:
                dev_txt_output_path_set = "{}.BW_{:02d}.A_{:.1f}.{}.txt.rank{}".format(
                    output_path,
                    dev_best_translation_beam_size,
                    dev_best_translation_alpha,
                    "dev",
                    os.environ['LOCAL_RANK']
                )
                test_txt_output_path_set = "{}.BW_{:02d}.A_{:.1f}.{}.txt.rank{}".format(
                    output_path,
                    dev_best_translation_beam_size,
                    dev_best_translation_alpha,
                    "test",
                    os.environ['LOCAL_RANK']
                )
            else:
                dev_txt_output_path_set = "{}.BW_{:02d}.{}.txt.rank{}".format(
                    output_path, dev_best_translation_beam_size, "dev",
                    os.environ['LOCAL_RANK']
                )
                test_txt_output_path_set = "{}.BW_{:02d}.{}.txt.rank{}".format(
                    output_path, dev_best_translation_beam_size, "test",
                    os.environ['LOCAL_RANK']
                )

            _write_to_file(
                dev_txt_output_path_set,
                dev_best_translation_result["sequence"],
                dev_best_translation_result["txt_hyp"],
            )
            _write_to_file(
                test_txt_output_path_set,
                test_best_result["sequence"],
                test_best_result["txt_hyp"],
            )

        with open(output_path + ".dev_results.pkl.rank{}".format(os.environ['LOCAL_RANK']), "wb") as out:
            pickle.dump(
                {
                    "recognition_results": dev_recognition_results
                    if do_recognition
                    else None,
                    "translation_results": dev_translation_results
                    if do_translation
                    else None,
                },
                out,
            )
        with open(output_path + ".test_results.pkl.rank{}".format(os.environ['LOCAL_RANK']), "wb") as out:
            pickle.dump(test_best_result, out)



if __name__ == "__main__":
    parser = argparse.ArgumentParser("Joey-NMT")
    parser.add_argument(
        "--config",
        default="configs/default.yaml",
        type=str,
        help="Training configuration file (yaml).",
    )

    parser.add_argument(
        "--num",
        default=-1,
        type=int,
        help='effective when mode=visualize_attention or output_feature=True'
    )

    parser.add_argument(
        "--save_immediate_results",
        action='store_true'
    )

    parser.add_argument(
        "--save_subdir",
        default='immediate'
    )
    args = parser.parse_args()
    assert 'LOCAL_RANK' in os.environ, 'Only support distributed training/evaluation(gpu=1) now!'
    cfg = load_config(args.config)
    train_config = cfg["training"]
    model_dir = train_config["model_dir"]
    ckpt = "{}/{}.ckpt".format(model_dir, 'best')
    output_name = "best.IT_best"
    if args.save_immediate_results:
        output_name += '.immediate'
        model_dir = os.path.join(model_dir,args.save_subdir)
        make_model_dir(model_dir, overwrite=False)
    output_path = os.path.join(model_dir, output_name)
    logger = make_logger(model_dir=model_dir, log_file='test.rank{}.log'.format(os.environ['RANK']))

    distributed = 'WORLD_SIZE' in os.environ
    if distributed:
        local_rank = int(os.environ['LOCAL_RANK'])
        torch.cuda.set_device(local_rank)
        torch.distributed.init_process_group(backend="nccl", init_method="env://")
    test(cfg_file=args.config, ckpt=ckpt, output_path=output_path, logger=logger, num=args.num, 
        save_immediate_results=args.save_immediate_results)
