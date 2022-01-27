import math, pickle, os, numpy as np
import torch, torchvision
import torch.nn as nn
from torch import Tensor, dist
import torch.nn.functional as F
from signjoey import batch
from signjoey.vocabulary import (
    TextVocabulary,
    GlossVocabulary,
    SIL_TOKEN,
    UNK_TOKEN,
    PAD_TOKEN,
    EOS_TOKEN,
    BOS_TOKEN,
)
from transformers import MBartForConditionalGeneration, MBartTokenizer
from transformers import GPT2Tokenizer, GPT2LMHeadModel, AutoTokenizer
from signjoey.loss import XentLoss
from collections import defaultdict
from signjoey.helpers import freeze_params, get_spans, sparse_sample, shift_tokens_right, ctc_decode_func
from copy import deepcopy
from helpers import get_spans_sequence, batch_random_copy
import random
DEBUG = False
class PrecedingLayer_weighted_embed(nn.Module):
    def __init__(
        self, 
        index2str,
        init_weight,
        out_features=1024,
        freeze=True
    ):
        super().__init__()
        self.fc = nn.Linear(in_features=len(index2str),out_features=out_features, bias=False)
        with torch.no_grad():
            for i,s in enumerate(index2str):
                if s in [SIL_TOKEN, UNK_TOKEN, PAD_TOKEN, EOS_TOKEN,BOS_TOKEN]:
                    print('special token {}, set fc to zero'.format(s))
                    self.fc.weight[i,:] = 0
                elif s in init_weight:
                    self.fc.weight[:,i] = init_weight[s]
                    if i==100:
                        print(i, s, self.fc.weight[:,i])
                else:
                    print('{} not in gls2embed, set fc to zero'.format(s))
                    self.fc.weight[i,:] = 0

        if freeze:
            freeze_params(self.fc)

    def forward(self,x):
        return self.fc(x)


class PrecedingLayer(nn.Module):
    def __init__(
        self,
        in_features,
        out_features,
        hidden_size,
        num_layers=2):
        super().__init__()
        assert num_layers>=2, num_layers
        layers =  [nn.Linear(in_features=in_features, out_features=hidden_size)]
        for i in range(1, num_layers-1):
            layers.extend([nn.ReLU(),nn.Linear(in_features=hidden_size, out_features=hidden_size)])
        layers.extend([nn.ReLU(),nn.Linear(in_features=hidden_size, out_features=out_features)])
        self.layers = nn.Sequential(*layers)
    def forward(self, x):
        return self.layers(x)
    

class SignModel_PLM(nn.Module):
    def __init__(
        self,
        encoder: nn.Module, #here we refer to sign->gloss
        gloss_output_layer: nn.Module,
        plm_cfg: dict,
        sgn_embed: nn.Module,
        gls_vocab: GlossVocabulary,
        txt_vocab: TextVocabulary,
        txt_level: str='word',
        sample_strategy: str='all',
        do_recognition: bool=True,
        do_translation: bool=True,
        do_distillation: bool=False,
    ):  
        super().__init__()
        self.do_recognition = do_recognition
        self.do_translation = do_translation
        self.do_distillation = do_distillation
        self.gls_vocab = gls_vocab
        self.txt_vocab = txt_vocab
        self.txt_level = txt_level
        self.gls_pad_index = gls_vocab.stoi[PAD_TOKEN]
        self.txt_pad_index = txt_vocab.stoi[PAD_TOKEN]  # ???
        self.sgn_embed = sgn_embed
        self.encoder = encoder
        self.input_feature_after_sgnmbed = plm_cfg.get('input_feature_after_sgnmbed',False)
        if self.input_feature_after_sgnmbed:
            print('input feature after sgnembed to translation net')
        self.gls_target_embedding_layer = None
        self.sample_strategy = sample_strategy
        print('sample_strategy= ', self.sample_strategy)
        self.plm_cfg = plm_cfg
        self.src_lang = plm_cfg.get('src_lang','de_DGS')
        self.tgt_lang = plm_cfg.get('tgt_lang','de_DE')

        self.plm_type = plm_cfg.get('type','mbart').lower()
        self.use_logits = plm_cfg.get('use_logits', False)
        self.pipeline = plm_cfg.get('pipeline',False)
        self.freeze_ctc = plm_cfg.get('freeze_ctc', False)
        self.use_gt_gloss = plm_cfg.get('use_gt_gloss', False)
        if self.use_gt_gloss and 'copy_gloss' in plm_cfg:
            self.copy_gloss = True
            self.copy_gloss_min, self.copy_gloss_max = plm_cfg['copy_gloss']['min'], plm_cfg['copy_gloss']['max']
            print('Copy gloss to [{},{}]'.format(self.copy_gloss_min, self.copy_gloss_max))
            self.copy_gloss_unk_ratio = plm_cfg['copy_gloss'].get('unk_ratio', 0)
        else:
            self.copy_gloss = False

        if 'decoder_mask' in plm_cfg:
            self.decoder_mask_prob = plm_cfg['decoder_mask'].get('prob', 1)
            print('randomly mask decoder inputs prob=', self.decoder_mask_prob)
        else:
            self.decoder_mask_prob = 0
        if self.use_gt_gloss:
            print('use gt gloss, freeze_ctc=',self.freeze_ctc)
            assert not self.pipeline
            assert not do_distillation and not do_recognition
        if self.pipeline:
            assert not do_distillation and do_recognition
        
        if 'gloss_augmentation' in self.plm_cfg:
            self.gloss_augmentation = True
            with open(self.plm_cfg['gloss_augmentation']['bank_file'],'rb') as f:
                self.gloss_bank = pickle.load(f)
            self.gloss_bank = dict(self.gloss_bank) # not default_dict
            self.gloss_replace_prob = self.plm_cfg['gloss_augmentation']['prob']
        else:
            self.gloss_augmentation = False


        assert self.plm_type in ['mbart', 'gpt2']
        #gloss2embed
        assert os.path.isfile(plm_cfg['gloss_embedding_file'])
        self.gls2embed = torch.load(plm_cfg['gloss_embedding_file'])

        if 'fusion' in plm_cfg:
            assert self.do_distillation==False
            self.fusion = True
            self.fusion_cfg = plm_cfg['fusion']
            self.fusion_method = self.fusion_cfg['method']
            self.fusion_fc = self.fusion_cfg['fusion_fc']
            self.fusion_vis_weight = self.fusion_cfg.get('fusion_vis_weight', 1)
            print('Implementing fusion ', self.fusion_method) #after 'before_add_word_embedding'
            self.fusion_use_gloss, self.fusion_use_vis = self.fusion_cfg['use_gloss'], self.fusion_cfg['use_vis']
            assert self.fusion_use_vis, 'currently do not support fusion_use_vis=False'
            if self.fusion_method == 'prepend':
                assert self.fusion_use_gloss
            if self.fusion_fc:
                print('Linear transform after fusion')
                assert self.fusion_use_gloss and self.fusion_use_vis
                assert self.fusion_method in ['plus','cat']
                if self.fusion_method=='plus':
                    self.fusion_fc = torch.nn.Linear(1024, 1024)
                elif self.fusion_method=='cat':
                    self.fusion_fc = torch.nn.Linear(2048, 1024)
                else:
                    raise ValueError
            else:
                self.fusion_fc = torch.nn.Identity()
            self.gls_target_embedding_layer = torch.nn.Embedding(
                num_embeddings=len(self.gls_vocab),#len(self.gls2embed),
                embedding_dim=1024,
                padding_idx=self.gls_vocab.stoi[PAD_TOKEN]) # we would like a padding idx
            #initialize!
            with torch.no_grad():
                print('initialize gls_target_embedding_layer as gls2embed')
                for i,s in enumerate(self.gls_vocab.itos):
                    if s in self.gls2embed:
                        init_emb = self.gls2embed[s]
                        self.gls_target_embedding_layer.weight[i,:] = init_emb
                    else:
                        print(s,'not in gls2embed set as zero')
                        self.gls_target_embedding_layer.weight[i,:] = 0
                        
            if self.fusion_cfg['freeze_gloss_embedding']:
                freeze_params(self.gls_target_embedding_layer)
                print('gls_target_embedding_layer freeze=True')
            else:
                print('gls_target_embedding_layer freeze=False')
            print('Txt embedding layer in plm, freeze={}'.format(plm_cfg.get('freeze_embed',False)))
        else:
            self.fusion = False

        if self.plm_type == 'mbart':
            if 'overwrite_mbart_cfg' in plm_cfg:
                print('Overwrite mbart cfg')
                print(plm_cfg['overwrite_mbart_cfg'])
            else:
                plm_cfg['overwrite_mbart_cfg'] = {}
            #tokenizer is still needed to encode/decode txt
            #manually set src_lang_code (newly added) 
            self.tokenizer = MBartTokenizer.from_pretrained(
                plm_cfg['pretrained_dir'], tgt_lang=self.tgt_lang)
            if self.src_lang == 'de_DGS':
                assert 'de' in plm_cfg['pretrained_dir'].lower(), plm_cfg['pretrained_dir']
                self.tokenizer.lang_code_to_id['de_DGS'] = 30
            elif self.src_lang == 'zh_CSL':
                assert 'zh' in plm_cfg['pretrained_dir'].lower(), plm_cfg['pretrained_dir']
                self.tokenizer.lang_code_to_id['zh_CSL'] = -2   
            else:
                print('Warning! src_lang={}'.format(self.src_lang))            
            self.tokenizer.src_lang = self.src_lang
        if self.plm_type == 'gpt2':
            self.tokenizer = AutoTokenizer.from_pretrained(plm_cfg['pretrained_dir'])

        if do_recognition:
            self.gloss_output_layer = gloss_output_layer
        if (do_distillation or do_translation) and not self.pipeline and not self.use_gt_gloss:
            if self.encoder.output_size == 1024:
                if 'force_preceding_layer' in self.plm_cfg:
                    self.preceding_layer = PrecedingLayer(
                        in_features=1024,
                        out_features=1024,  # mBart
                        hidden_size=plm_cfg['force_preceding_layer'].get(
                            'hidden_size', 1024),
                        num_layers=plm_cfg['force_preceding_layer'].get('num_layers', 2))
                else:
                    self.preceding_layer = nn.Identity()
            elif self.input_feature_after_sgnmbed:
                print('input feature after sgnembed in_features=',
                      self.sgn_embed.embedding_dim)
                self.preceding_layer = PrecedingLayer(
                    in_features=self.sgn_embed.embedding_dim,
                    out_features=1024,  # mBart
                    hidden_size=plm_cfg['preceding_layer'].get(
                        'hidden_size', 1024),
                    num_layers=plm_cfg['preceding_layer'].get('num_layers', 2))
            elif self.use_logits:
                print('input feature as gloss logits')
                self.preceding_layer = PrecedingLayer(
                    in_features=len(self.gls_vocab),
                    out_features=1024,  # mBart
                    hidden_size=plm_cfg['preceding_layer'].get(
                        'hidden_size', 1024),
                    num_layers=plm_cfg['preceding_layer'].get('num_layers', 2))           
            elif 'preceding_layer_weighted_embed' in plm_cfg:
                freeze_preceding_layer = plm_cfg['preceding_layer_weighted_embed'].get('freeze', True)
                print('Use gls2embed to initialize preceding_Layer, freeze=', freeze_preceding_layer)
                #assert self.encoder.output_size==len(self.gls_vocab.itos),(self.encoder.output_size, self.gls_vocab.itos)
                self.preceding_layer = PrecedingLayer_weighted_embed(
                    index2str = self.gls_vocab.itos,
                    init_weight = self.gls2embed,
                    freeze = freeze_preceding_layer,
                    out_features=1024
                )
            else:
                out_features = 1024 if self.plm_type=='mbart' else 768
                self.preceding_layer = PrecedingLayer(
                    in_features=self.encoder.output_size,
                    out_features=out_features, #mBart
                    hidden_size=plm_cfg['preceding_layer'].get('hidden_size', out_features),
                    num_layers=plm_cfg['preceding_layer'].get('num_layers', 2))        
        if do_translation:
            if self.plm_type == 'mbart':
                self.plm_model = MBartForConditionalGeneration.from_pretrained(
                    plm_cfg['pretrained_dir'],
                    **plm_cfg['overwrite_mbart_cfg']
                    ) 
                self.plm_embed_scale = math.sqrt(self.plm_model.config.d_model)
                #OLD2NEW file
                old2new_file=os.path.join(plm_cfg['pretrained_dir'], 'old2new_vocab.pkl')
                assert os.path.isfile(old2new_file), old2new_file
                print('Map old id to new id use ',old2new_file)
                with open(old2new_file,'rb') as f:
                    self.old2new = pickle.load(f)
                # old (id output by tokenizer 0~250026) new(id in mBart_De with restricted vocab 0~4107/3046)
                # self.old2new = defaultdict(lambda: 3, old2new) #3 unk I remove this line as the newly updated mBart_gls_embedding.bin should cover all glosses in the corpus
                # However, it is still possible that the mBart outputs some word that is not existed in original mBart, e.g. de_DGS
                self.new2old = defaultdict(lambda: 3) #
                for o,n in self.old2new.items():
                    assert n>=0 and n<self.plm_model.config.vocab_size, (n, self.plm_model.config.vocab_size)
                    if type(o) != str:
                        self.new2old[n] = o

                if plm_cfg.get('src_lang_code_from_scratch',True):
                    src_lang_id = self.tokenizer.lang_code_to_id[self.tokenizer.src_lang]
                    src_lang_id_in_model = self.old2new[src_lang_id]
                    print('Reinitialize src_lang_code {} {}  {}'.format(
                        self.tokenizer.src_lang, 
                        src_lang_id, src_lang_id_in_model))
                    print('Before re-initialize')
                    print(self.tokenizer.src_lang)
                    print(self.plm_model.model.shared.weight[src_lang_id_in_model, :])
                    print('=corresponding ?tgt_lang ',self.tgt_lang)
                    tgt_lang_id_in_model = self.old2new[self.tokenizer.lang_code_to_id[self.tgt_lang]]
                    print(self.plm_model.model.shared.weight[tgt_lang_id_in_model, :])
                    torch.nn.init.normal_(self.plm_model.model.shared.weight[src_lang_id_in_model,:])
                    print('after re-initialize')
                    print(self.plm_model.model.shared.weight[src_lang_id_in_model, :])

            elif self.plm_type == 'gpt2':
                self.plm_model = GPT2LMHeadModel.from_pretrained(plm_cfg['pretrained_dir'])
                self.plm_embed_scale = 1
                vocab_size = self.plm_model.config.vocab_size
                self.old2new = {i:i for i in range(vocab_size)}
                self.new2old = {i:i for i in range(vocab_size)}



            if plm_cfg.get('from_scratch',False):
                print('reinitialize plm_model to train from scratch')
                self.plm_model.init_weights()

            if plm_cfg.get('freeze_embed',False):
                print('freeze plm embedding!')
                if self.plm_type=='mbart':
                    freeze_params(self.plm_model.model.shared)
                elif self.plm_type=='gpt2':
                    freeze_params(self.plm_model.transformer.wte)
            
        if 'gloss_embedding_layer' in plm_cfg:
            gls_emb_cfg = plm_cfg['gloss_embedding_layer']
            self.gls_emb_freeze = gls_emb_cfg.get('freeze',False)
            self.gls_emb_from_scratch = gls_emb_cfg.get('from_scratch',False)
            self.gls_target_embedding_layer = torch.nn.Embedding(
                num_embeddings=len(self.gls_vocab),
                embedding_dim=self.plm_model.config.d_model,
                padding_idx=self.gls_vocab.stoi[PAD_TOKEN]) # we would like a padding idx
            if not self.gls_emb_from_scratch:
                #initialize
                print('initialize gls_emb from gls2embed matrix')
                with torch.no_grad():
                    for i,s in enumerate(self.gls_vocab.itos):
                        if s in self.gls2embed:
                            init_emb = self.gls2embed[s]
                            self.gls_target_embedding_layer.weight[i,:] = init_emb
                            if i==0:
                                print('init example i{}:s{}'.format(i,s))
                                print(init_emb)
                        else:
                            print('in gls_emb {}:{} from scratch'.format(i,s))
            else:
                print('gls_emb from scratch, num_embeddings=', len(self.gls_vocab))
            if self.gls_emb_freeze:
                freeze_params(self.gls_target_embedding_layer)
            print('Implement regularization target as a layer, freeze={}'.format(self.gls_emb_freeze))
            print('Txt embedding layer in plm, freeze={}'.format(plm_cfg.get('freeze_embed',False)))
        else:
            self.gls_target_embedding_layer = None

        if do_distillation:
            assert 'distillation' in plm_cfg

            self.distillation_loss_type = plm_cfg['distillation'].get('loss_type','MSE')
            if self.distillation_loss_type == 'MSE':
                self.distillation_loss_fun = nn.MSELoss(reduction='none')
            elif self.distillation_loss_type == 'L1':
                self.distillation_loss_fun = nn.L1Loss(reduction='none')
            elif self.distillation_loss_type == 'SmoothL1':
                self.distillation_loss_fun = nn.SmoothL1Loss(reduction='none')
            else:
                raise ValueError
            
            if self.sample_strategy=='all':
                assert 'distill_sample_strategy' in plm_cfg
                self.distill_sample_strategy = plm_cfg['distill_sample_strategy']
            else:
                self.distill_sample_strategy = None

        if do_translation:
            self.loss_level = plm_cfg.get('loss_level', 'sentence')
            self.label_smoothing = plm_cfg.get('label_smoothing', 0.2)

            if self.plm_type == 'mbart':
                self.ignore_index = self.tokenizer.pad_token_id  # ???
                self.gls_lang_index = self.old2new[self.tokenizer.lang_code_to_id[self.tokenizer.src_lang]]
                self.gls_eos_index = self.old2new[self.tokenizer.eos_token_id]
                self.txt_bos_index = self.old2new[self.tokenizer.convert_tokens_to_ids(
                    self.tokenizer.tgt_lang)]
                self.txt_eos_index = self.old2new[self.tokenizer.eos_token_id]
                print('txt_eos_index', self.txt_eos_index)
                self.pad_token_id = self.tokenizer.pad_token_id
            elif self.plm_type == 'gpt2':
                self.tokenizer.pad_token = '<pad>'
                self.ignore_index = self.tokenizer.convert_tokens_to_ids('<pad>') 
                self.gls_eos_index = self.tokenizer.convert_tokens_to_ids('</s>') 
                self.txt_bos_index = self.tokenizer.convert_tokens_to_ids('<s>') 
                self.txt_eos_index = self.tokenizer.convert_tokens_to_ids('</s>')
                self.pad_token_id = self.tokenizer.convert_tokens_to_ids('<pad>')
            self.translation_loss_fun = XentLoss(
                pad_index=self.ignore_index,  # ignore
                smoothing=self.label_smoothing)

            




        if self.freeze_ctc:
            print('freeze_ctc ...')
            for m in ['sgn_embed','encoder','gloss_output_layer']:
                if hasattr(self, m):
                    print('freeze ', m)
                    freeze_params(getattr(self, m))

    def set_train(self, verbose=False):
        self.train()
        if self.freeze_ctc:
            for m in ['sgn_embed','encoder','gloss_output_layer']:
                if hasattr(self, m):
                    getattr(self, m).eval()
    
    def set_eval(self):
        self.eval()

    def map_old2new(self, batch_input_ids):
        if self.old2new == None:
            return batch_input_ids
        new_batch_input_ids = batch_input_ids.clone()
        for bi, input_ids in enumerate(batch_input_ids):
            for ii, id_ in enumerate(input_ids):
                if not id_.item() in self.old2new:
                    print(id_)
                    print(new_batch_input_ids)
                    new_batch_input_ids[bi, ii] = 3 # unk
                else:
                    new_batch_input_ids[bi, ii] = self.old2new[id_.item()]
        return new_batch_input_ids

    def map_new2old(self, batch_input_ids):
        if self.old2new==None:
            return batch_input_ids
        new_batch_input_ids = deepcopy(batch_input_ids)#.clone()
        for bi,input_ids in enumerate(batch_input_ids):
            for ii, id_ in enumerate(input_ids):
                if self.plm_type=='mbart' and id_.item() in [self.gls_lang_index]: #say 31
                    #is a special token but won't be ignored by batch_decode, so here we convert it to a special token
                    new_batch_input_ids[bi][ii] = 2 # </s>
                else:
                    new_batch_input_ids[bi][ii] = self.new2old[id_.item()]
        return new_batch_input_ids

    def prepare_txt_input(self, txt, txt_lengths):
        batch_size, padded_length = txt.shape
        device = txt.device
        batch_raw_txt = []
        for i in range(batch_size):
            raw_txt = [self.txt_vocab.itos[txt[i,j]] for j in range(1,txt_lengths[i]) \
                if self.txt_vocab.itos[txt[i,j]] != EOS_TOKEN] #the first one is [BOS] (prepended by torchtext field)
            #level!
            if self.txt_level=='word':
                batch_raw_txt.append(' '.join(raw_txt))
            elif self.txt_level=='char':
                batch_raw_txt.append(''.join(raw_txt))
            else:
                raise ValueError
        #print(batch_raw_txt) #debughere!
        if self.plm_type == 'mbart':
            with self.tokenizer.as_target_tokenizer():
                labels = self.tokenizer(
                    batch_raw_txt,            
                    padding='longest', #we've already control the maximum length of input seq in dataloader
                    return_attention_mask=True,
                    return_length=True,
                    return_tensors="pt")
            #print(labels['input_ids'])
            labels['input_ids'] = self.map_old2new(labels['input_ids'])
            decoder_input_ids, label_input_ids = shift_tokens_right(
                labels['input_ids'].clone(), 
                self.pad_token_id,
                ignore_index=self.ignore_index) #[lang, ]
            decoder_attention_mask = labels['attention_mask'] #attention mask keeps the same after shifting
            if self.decoder_mask_prob>0:
                random_mask = torch.rand(decoder_attention_mask.shape)>self.decoder_mask_prob
                decoder_attention_mask = decoder_attention_mask*random_mask
            return label_input_ids.to(device), decoder_input_ids.to(device), decoder_attention_mask.to(device)

        elif self.plm_type == 'gpt2':
            tokenizer_outputs = self.tokenizer(batch_raw_txt, return_length=True)
            input_ids, input_lengths = tokenizer_outputs['input_ids'], tokenizer_outputs['length']
            input_ids = [[self.txt_bos_index] + ids for ids in input_ids]
            input_lengths = [le+1 for le in input_lengths]
            return input_ids, input_lengths



    def prepare_plm_inputs(self,input_embed, input_mask, txt_input=None, txt_mask=None,
            fusion=False, batch_pred_ids=None):
        #here we need to append </s> <src_lang_code>  to input_embed
        #print('eos_index', self.gls_eos_index) #debughere!
        #print('gls_lang_index', self.gls_lang_index) #debughere!
        if self.plm_type=='mbart':
            eos_embedding = self.plm_model.model.shared.weight[self.gls_eos_index,:]
            src_lang_code_embedding = self.plm_model.model.shared.weight[self.gls_lang_index,:]
            suffix_emb = torch.stack([eos_embedding, src_lang_code_embedding], dim=0) # 2,1024
        elif self.plm_type=='gpt2':
            suffix_emb = None#self.plm_model.transformer.wte.weight[self.txt_bos_index,:].unsqueeze(0) #1, 768
        if fusion:
            assert batch_pred_ids!=None
            gloss_embeddings, gloss_mask = self.build_gloss_target_embedding(
                src_embeddings=input_embed,
                tgt_ids=batch_pred_ids,
                return_mask=True
            )
            if self.fusion_use_gloss and not self.fusion_use_vis:
                return self.prepare_plm_inputs(
                    input_embed = gloss_embeddings,
                    input_mask = gloss_mask,
                    txt_input = txt_input,
                    txt_mask = txt_mask, fusion=False)

        #input_embed_appended = torch.stacm
        input_mask = input_mask.squeeze(1) #B,L
        batch_size, max_length_wo_suffix, dim_ = input_embed.shape
        valid_lengths = torch.sum(input_mask, dim=-1) #B
        max_length = max_length_wo_suffix + 2
        input_embed_w_suffix, input_mask_w_suffix = [], torch.zeros([batch_size, max_length],dtype=torch.long, device=input_embed.device)
        for i in range(batch_size):
            valid_emb = input_embed[i,:valid_lengths[i]]
            if suffix_emb!=None:
                emb_w_suffix = torch.cat([valid_emb, suffix_emb],  dim=0) # VALID_LEN+2, 1024
            else:
                emb_w_suffix = valid_emb
            if self.plm_type == 'mbart': #pad encoder inputs for mbart
                pad_len = max_length-(valid_lengths[i]+2)
                if pad_len>0:
                    paddings = torch.zeros([pad_len,dim_],device=input_embed.device, dtype=input_embed.dtype)
                    padded_emb_w_suffix = torch.cat([emb_w_suffix, paddings], dim=0)
                else:
                    padded_emb_w_suffix = emb_w_suffix
                input_embed_w_suffix.append(padded_emb_w_suffix)
                input_mask_w_suffix[i,:valid_lengths[i]+2] = 1
            elif self.plm_type == 'gpt2':
                input_embed_w_suffix.append(emb_w_suffix)
        
        if self.plm_type == 'mbart':
            input_embed_w_suffix = torch.stack(input_embed_w_suffix, dim=0) #B,L,D
            if fusion:
                assert self.fusion_use_vis #currently, don't support only use gloss embeddings here?
                if self.fusion_method=='prepend' and self.fusion_use_gloss:
                    #we don't need any fc layer? yes the visual feature has already been transformed before (preceding layer)
                    fusion_input_embed_w_suffix = []
                    max_length = 2*max_length_wo_suffix + 2
                    new_input_mask_w_suffix = torch.zeros([batch_size, max_length],dtype=torch.long, device=gloss_embeddings.device)
                    for bi in range(batch_size):
                        gls_embed = gloss_embeddings[bi, :valid_lengths[bi]]
                        input_embed_prepended = torch.cat([gls_embed, input_embed_w_suffix[bi,:]],dim=0) 
                        pad_len = max_length-input_embed_prepended.shape[0]
                        if pad_len>=0:
                            paddings = torch.zeros([pad_len,dim_],
                                device=gloss_embeddings.device, 
                                dtype=gloss_embeddings.dtype)
                            padded_input_embed_prepended = torch.cat([input_embed_prepended,paddings], dim=0)
                        else:
                            padded_input_embed_prepended = input_embed_prepended
                        fusion_input_embed_w_suffix.append(padded_input_embed_prepended)
                        new_input_mask_w_suffix[bi, :input_embed_prepended.shape[0]] = 1
                    #input_mask_w_suffix = new_input_mask_w_suffix
                    fusion_input_embed_w_suffix = torch.stack(fusion_input_embed_w_suffix, dim=0)
                    input_mask_w_suffix = new_input_mask_w_suffix
                else:
                    #replace pad with zero
                    zeros = torch.zeros_like(gloss_embeddings)
                    gloss_embeddings = torch.where(
                        gloss_mask.transpose(1,2),#B,L,1
                        gloss_embeddings,
                        zeros) 
                    #append  another two zeros
                    zeros_suffix = torch.zeros([batch_size, 2, dim_], dtype=gloss_embeddings.dtype, device=gloss_embeddings.device)
                    gloss_embedding_w_suffix = torch.cat([gloss_embeddings, zeros_suffix], dim=1) #B,L,D B,2,D
                    if self.fusion_method=='plus':
                        fusion_input_embed_w_suffix = 0
                        if self.fusion_use_vis:
                            # print('input_embed_w_suffix')
                            # print(input_embed_w_suffix)
                            fusion_input_embed_w_suffix += input_embed_w_suffix*self.fusion_vis_weight
                        if self.fusion_use_gloss:
                            # print('gloss_embedding_w_suffix')
                            # print(gloss_embedding_w_suffix)
                            fusion_input_embed_w_suffix += gloss_embedding_w_suffix
                    elif self.fusion_method=='cat':
                        fusion_input_embed_w_suffix = []
                        if self.fusion_use_vis:
                            fusion_input_embed_w_suffix.append(input_embed_w_suffix)
                        if self.fusion_use_gloss:
                            fusion_input_embed_w_suffix.append(gloss_embedding_w_suffix)
                        fusion_input_embed_w_suffix = torch.cat(fusion_input_embed_w_suffix, dim=-1) #B,L,2D
                    elif self.fusion_method=='prepend':
                        raise ValueError #prepend
                    #fc layer!
                fusion_input_embed_w_suffix = self.fusion_fc(fusion_input_embed_w_suffix)
                input_embed_w_suffix = fusion_input_embed_w_suffix*self.plm_embed_scale
                if 0: #debug here!
                    print('with fusion input_embed')
                    print(self.plm_embed_scale)
                    print(input_embed_w_suffix)
                    #print(input_mask_w_suffix)
                    input()
            else:
                input_embed_w_suffix = input_embed_w_suffix*self.plm_embed_scale
            encoder_inputs = {
                'inputs_embeds': input_embed_w_suffix, #B,L,D
                'attention_mask': input_mask_w_suffix, #B,L
            }
            if txt_input != None and txt_mask != None: #copy from PLM 
                #txt_mask (B,1,L) is not causal yet, so we can get txt_lengths from it
                txt_lengths = torch.sum(
                    txt_mask, dim=-1).squeeze(1)  # B (including)
                txt_label, txt_input, txt_mask_transformer = self.prepare_txt_input(
                    txt_input, txt_lengths) #mask!
                decoder_inputs = {
                    'decoder_input_ids': txt_input,
                    'decoder_attention_mask': txt_mask_transformer,
                    'labels': txt_label
                }
                #print('decoder inputs')
                #print(decoder_inputs) #debughere!
            else:
                batch_start_ids = torch.ones([batch_size,1],dtype=torch.long, device=input_embed.device)*self.txt_bos_index
                decoder_inputs = {'decoder_input_ids':batch_start_ids} #for inference
            inputs = {**encoder_inputs, **decoder_inputs}
            return inputs
        elif self.plm_type == 'gpt2':
            #input_embed_w_suffix 
            #we need to convert txt to embeds 
            if txt_input != None and txt_mask != None: # training
                txt_lengths = torch.sum(
                    txt_mask, dim=-1).squeeze(1)  # B (including)
                txt_ids, txt_lengths = self.prepare_txt_input(txt_input, txt_lengths)
                max_length = max([input_embed_w_suffix[i].shape[0]+txt_lengths[i] for i in range(batch_size)])
                inputs_embeds, labels, token_type_ids = [], [], []
                for i in range(batch_size):
                    txt_emb = self.plm_model.transformer.wte(torch.tensor(txt_ids[i]).to('cuda')) #L, D
                    assert txt_emb.shape[0] == txt_lengths[i], (txt_emb.shape, txt_lengths[i])

                    gls_len, txt_len = input_embed_w_suffix[i].shape[0], txt_lengths[i]
                    pad_length = max_length - gls_len - txt_len
                    emb_list = [input_embed_w_suffix[i], txt_emb]+ \
                        [self.plm_model.transformer.wte.weight[self.pad_token_id,:].unsqueeze(0)]*pad_length
                    inputs_embeds.append(torch.cat(emb_list, dim=0)) #L,D
                    token_type_ids.append([0]*gls_len + [1]*txt_len + [1]*pad_length)
                    labels.append([self.ignore_index]*gls_len + txt_ids[i][1:] + [self.txt_eos_index] + [self.ignore_index]*pad_length)
                inputs_embeds = torch.stack(inputs_embeds, dim=0) # B,L,D
                token_type_ids = torch.tensor(token_type_ids, dtype=torch.long, device=inputs_embeds.device)
                labels = torch.tensor(labels, dtype=torch.long, device=inputs_embeds.device)
                #inputs = {'inputs_embeds':inputs_embeds, 'token_type_ids':token_type_ids, 'labels':labels}
                inputs = {'inputs_embeds':inputs_embeds, 'labels':labels}
                if DEBUG: #debug
                    print(inputs)
                    input()
                return inputs
            else: #testing  
                inputs_embeds, token_type_ids, attention_mask = [], [], []
                generate_attention_mask = []
                max_length = max([len(input_embed_w_suffix[i]) for i in range(batch_size)])
                for i in range(batch_size):
                    gls_len = input_embed_w_suffix[i].shape[0]
                    pad_length = max_length - gls_len
                    emb_list = [self.plm_model.transformer.wte.weight[self.pad_token_id,:].unsqueeze(0)]*pad_length + \
                        [input_embed_w_suffix[i]]
                    inputs_embeds.append(torch.cat(emb_list,dim=0)) #L,D
                    token_type_ids.append([0]*(max_length-1) + [1])
                    attention_mask.append([0]*(pad_length) + [1]*(max_length - pad_length))
                    generate_attention_mask.append([0]*(pad_length) + [1]*(max_length - pad_length) + [1])
                inputs_embeds = torch.stack(inputs_embeds, dim=0) # B,L,D
                #token_type_ids = torch.tensor(token_type_ids, dtype=torch.long, device=inputs_embeds.device)
                attention_mask = torch.tensor(attention_mask, dtype=torch.long, device=inputs_embeds.device)
                #inputs = {'inputs_embeds':inputs_embeds, 'token_type_ids':token_type_ids, 'attention_mask':attention_mask}
                forward_inputs = {'inputs_embeds':inputs_embeds, 'attention_mask':attention_mask}
                generate_input_ids = torch.ones([inputs_embeds.shape[0],1], 
                        device=inputs_embeds.device, dtype=torch.long)*self.txt_bos_index #B,1
                generate_attention_mask = torch.tensor(generate_attention_mask, dtype=torch.long, device=inputs_embeds.device)
                generate_inputs = {'input_ids':generate_input_ids,'attention_mask':generate_attention_mask}
                if DEBUG: #debug
                    print('forward')
                    print(forward_inputs)
                    print('generate')
                    print(generate_inputs)
                    input()

                return forward_inputs, generate_inputs
            

    def convert_gls_from_tensor_to_list(self, gls_tensor, gls_lengths):
        gls_list = []
        batch_size = gls_tensor.shape[0]
        gls_lengths = gls_lengths.long()
        for i in range(batch_size):
            gls_list.append([gls_tensor[i,j].item() for j in range(gls_lengths[i])])
        return gls_list

    def build_gloss_target_embedding(self, src_embeddings, tgt_ids, return_mask=False):
        batch_size, max_length, dim_ = src_embeddings.shape
        new_mask = torch.zeros([batch_size, max_length], device=src_embeddings.device, dtype=torch.bool)
        if self.gls_target_embedding_layer==None:
            with torch.no_grad():
                batch_target_embeddings = torch.zeros_like(src_embeddings) #B,T,D <pad>
                for bi in range(batch_size):
                    #print('sample ', bi)
                    for ii,gid in enumerate(tgt_ids[bi]):
                        g_str = self.gls_vocab.itos[gid].lower()
                        if not g_str in self.gls2embed:
                            if self.fusion and self.sample_strategy=='all':
                                #add zero
                                batch_target_embeddings[bi,ii,:] = 0
                            else:
                                print(gid, g_str, end=' ')
                                batch_target_embeddings[bi,ii,:] = src_embeddings[bi, ii, :]
                            #     input()
                        else:
                            g_tgt_emb = self.gls2embed[g_str]
                            batch_target_embeddings[bi,ii,:] = g_tgt_emb
                    #print()
                    for ii in range(len(tgt_ids[bi]), max_length): #pad
                        batch_target_embeddings[bi,ii,:] = src_embeddings[bi,ii,:]
                    new_mask[bi,:len(tgt_ids[bi])] = 1


        else:
            padded_tgt_ids = self.gls_vocab.stoi[PAD_TOKEN]*torch.ones(
                [batch_size, max_length],
                device=src_embeddings.device,
                dtype=torch.long)
            #B,L
            for bi in range(batch_size):
                ##print('sample ', bi) #debughere
                for ii, gid in enumerate(tgt_ids[bi]):
                    #print(self.gls_vocab.itos[gid], end=' ')#debughere
                    padded_tgt_ids[bi,ii] = gid
                new_mask[bi,:len(tgt_ids[bi])] = 1
                ##print()
            batch_target_embeddings = self.gls_target_embedding_layer(padded_tgt_ids)


        #print(new_mask) #debughere!
        if return_mask:
            return batch_target_embeddings, new_mask.unsqueeze(1)
        else:
            return batch_target_embeddings

    def compute_distillation_loss(self, src_embeddings, src_masks, tgt_ids):
        batch_size, max_length, dim_ = src_embeddings.shape
        assert src_masks.shape==torch.Size([batch_size,1,max_length])
        seq_lengths = torch.sum(src_masks, dim=-1).view(-1) #B
        tgt_ids_length = [len(t) for t in tgt_ids]
        valid_total_length = sum(tgt_ids_length)
        assert seq_lengths.detach().cpu().numpy().tolist()==tgt_ids_length, (seq_lengths,tgt_ids_length)
        #build target first
        batch_target_embeddings = self.build_gloss_target_embedding(
            src_embeddings=src_embeddings, 
            tgt_ids=tgt_ids)
        # print(src_embeddings)
        # print(batch_target_embeddings)
        loss = self.distillation_loss_fun(input=src_embeddings, target=batch_target_embeddings) #B,T,D
        #BTD reduce=None normalize by  valid_total_length
        eps = 1.0e-10

        loss = torch.sum(loss)/(valid_total_length+eps)/src_embeddings.shape[-1] # D
        return loss
 
    def encode(
        self, sgn: Tensor, sgn_mask: Tensor, sgn_length: Tensor, output_attention: bool = False
    ) -> (Tensor, Tensor):
        """
        Encodes the source sentence.

        :param sgn:
        :param sgn_mask:
        :param sgn_length:
        :return: encoder outputs (output, hidden_concat)
        """
        return self.encoder(
            embed_src=self.sgn_embed(x=sgn, mask=sgn_mask),
            src_length=sgn_length,
            mask=sgn_mask,
            output_attention=output_attention,
        )

    def forward(
        self,
        sgn: Tensor,
        sgn_mask: Tensor,
        sgn_lengths: Tensor,
        txt_input: Tensor,
        txt_mask: Tensor = None,
        output_attention: bool=False,
        batch=None
    ) -> (Tensor, Tensor, Tensor, Tensor):
        other_outputs = {}
        assert output_attention==False
        if not self.use_gt_gloss:
            encoder_output, encoder_hidden, attention, intermediate = self.encode(
                sgn=sgn, sgn_mask=sgn_mask, sgn_length=sgn_lengths, 
                output_attention=output_attention
            )
            other_outputs = {**intermediate}
            if self.do_recognition:
                # Gloss Recognition Part
                # N x T x C
                gloss_scores = self.gloss_output_layer(encoder_output)
                other_outputs['gloss_logits'] = gloss_scores
                # N x T x C
                gloss_probabilities = gloss_scores.log_softmax(2)
                other_outputs['gloss_distribution'] = gloss_probabilities
                # Turn it into T x N x C
                gloss_probabilities = gloss_probabilities.permute(1, 0, 2)
                if self.use_logits:
                    encoder_output = gloss_scores
                if self.preceding_layer.__class__.__name__ == 'PrecedingLayer_weighted_embed':
                    softmax = torch.nn.Softmax(dim=2)
                    encoder_output = softmax(gloss_scores)
                encoder_output, sgn_mask, batch_pred_gls = sparse_sample(
                    batch_enc_op=encoder_output, 
                    batch_gls_prob=gloss_probabilities.permute(1,0,2).detach(), # n,t,c
                    batch_mask = sgn_mask,  #B,1,L
                    select_strategy=self.sample_strategy,
                    return_pred_gls=True)

            else:
                gloss_probabilities = None
        else:
            assert batch!=None
            batch_size = batch.gls_lengths.shape[0]
            max_len = torch.max(batch.gls_lengths)
            #print(batch.sequence)
            tgt_ids = self.convert_gls_from_tensor_to_list(batch.gls, batch.gls_lengths)
            if self.copy_gloss:
                tgt_ids = batch_random_copy(tgt_ids, 
                    min_=self.copy_gloss_min, max_=self.copy_gloss_max,
                    unk_ratio=self.copy_gloss_unk_ratio,
                    unk_index=self.gls_vocab.stoi['<unk>'])
                max_len = max([len(ids) for ids in tgt_ids])
            encoder_output, sgn_mask = self.build_gloss_target_embedding(
                src_embeddings=torch.zeros([batch_size, max_len, 1024 if self.plm_type=='mbart' else 768], 
                    dtype=batch.sgn.dtype, 
                    device=batch.sgn.device),
                tgt_ids=tgt_ids,  #ground_truth gls
                return_mask=True
            )
            gloss_probabilities, attention = None, None
            
        if self.pipeline:
            #directly use gloss embedding (S->G->T) replace predicted encoder_output with gloss embedding
            batch_size, max_len, _ = encoder_output.shape
            encoder_output = self.build_gloss_target_embedding(
                src_embeddings=torch.zeros([batch_size, max_len, 1024], dtype=encoder_output.dtype, device=encoder_output.device),
                tgt_ids=batch_pred_gls
            )
            if 0:
                print('pipeline')
                print('batch_pred_gls')
                print(batch_pred_gls)
                print('encoder_output')
                print(encoder_output)
                input()
        elif (self.do_translation or self.do_distillation) and not self.use_gt_gloss:
            #intermediate layer 512->1024
            if self.gloss_augmentation and self.training:
                assert self.sample_strategy=='all'
                batch_size = encoder_output.shape[0]
                augmented_encoder_outputs, augmented_sgn_lengths = [], []
                for b in range(batch_size):
                    length0 = batch.sgn_lengths[b].long()
                    sort_idx = torch.argsort(-1*gloss_scores[b,:length0,:], dim=-1) #T,C
                    spans = get_spans_sequence(sort_idx) #[st, end, pred_id]
                    aug_feature_list = [] # [(t1,D),(t2,D)]
                    for (st, ed, gid) in spans:
                        if gid==0 or (self.gls_vocab.itos[gid] not in self.gloss_bank):
                            aug_feature_list.append(encoder_output[b,st:ed,:]) #T,D
                        elif np.random.rand(1)>self.gloss_replace_prob:
                            aug_feature_list.append(encoder_output[b,st:ed,:])
                        else:
                            #randomly choose one from memory bank
                            try:
                                feature_from_bank = random.choice(self.gloss_bank[self.gls_vocab.itos[gid]]) #T,D
                            except:
                                print(self.gloss_bank[self.gls_vocab.itos[gid]])
                            feature_from_bank = torch.tensor(feature_from_bank, 
                                dtype=encoder_output.dtype,
                                device=encoder_output.device)
                            aug_feature_list.append(feature_from_bank)
                    aug_feature = torch.cat(aug_feature_list, dim=0) #T,D
                    augmented_encoder_outputs.append(aug_feature)
                    augmented_sgn_lengths.append(aug_feature.shape[0])

                #padding and buid mask
                max_length = max(augmented_sgn_lengths)
                augmented_sgn_masks = torch.zeros(
                    [batch_size, 1, max_length],
                    dtype=sgn_mask.dtype,
                    device=sgn_mask.device)
                for b in range(batch_size):
                    cur_len = augmented_sgn_lengths[b]
                    augmented_sgn_masks[b,:,:cur_len] = 1
                    if cur_len<max_length:
                        augmented_encoder_outputs[b] = torch.cat(
                            [augmented_encoder_outputs[b], 
                            torch.zeros(
                                [max_length-cur_len,augmented_encoder_outputs[b].shape[1]],
                                device=encoder_output.device,
                                dtype=encoder_output.dtype
                            )],
                            dim=0) #T,D
                encoder_output = torch.stack(augmented_encoder_outputs, dim=0) #B,T,D
                sgn_mask = augmented_sgn_masks


            if self.input_feature_after_sgnmbed:
                assert not self.do_distillation
                encoder_output = self.preceding_layer(self.sgn_embed(x=sgn, mask=sgn_mask))
            else:
                encoder_output = self.preceding_layer(encoder_output)  # B,T,D'
            #note that after transmormation padded features are not zeros!
            #print('after transform ', encoder_output.shape)

        if self.do_distillation:
            assert self.do_recognition
            if self.sample_strategy=='all':
                assert self.distill_sample_strategy!=None
                encoder_output_for_distill, sgn_mask_for_distill, batch_pred_gls_for_distill = sparse_sample(
                    batch_enc_op=encoder_output, #after preceding layer?
                    batch_gls_prob=gloss_probabilities.permute(
                        1, 0, 2).detach(),  # n,t,c
                    batch_mask=sgn_mask,  # B,1,L
                    select_strategy=self.distill_sample_strategy,
                    return_pred_gls=True)
                distillation_loss = self.compute_distillation_loss(
                    src_embeddings=encoder_output_for_distill,
                    src_masks=sgn_mask_for_distill,
                    tgt_ids=batch_pred_gls_for_distill)
                
            else:
                distillation_loss = self.compute_distillation_loss(
                                        src_embeddings=encoder_output, 
                                        src_masks=sgn_mask,
                                        tgt_ids=batch_pred_gls)
        else:
            distillation_loss = None # we don't consider distillation when using dense feature

        if self.do_translation:
            #encoder_output -> translation loss
            if self.fusion: 
                #assert self.sample_strategy!='all'
                #input batch_pred_gls
                inputs = self.prepare_plm_inputs(
                    input_embed=encoder_output, input_mask=sgn_mask,
                    txt_input=txt_input, txt_mask=txt_mask,
                    fusion=True, 
                    batch_pred_ids=batch_pred_gls)  
                if 0:
                    print('fusion')
                    print(inputs)
                    input()
            else:
                inputs = self.prepare_plm_inputs(
                    input_embed=encoder_output, input_mask=sgn_mask,
                    txt_input=txt_input, txt_mask=txt_mask)      
            # 'input_ids', 'attention_mask', 'decoder_input_ids' 'decoder_attention_mask' 'labels'
            if DEBUG:
                print('forward in get_loss')
                print(inputs)
                input()
            output_dict = self.plm_model(
                **inputs, 
                return_dict=True,
                output_attentions=output_attention)
            assert self.loss_level == 'sentence', self.loss_level
            batch_size = output_dict['logits'].shape[0]
            log_prob = torch.nn.functional.log_softmax(
                output_dict['logits'], dim=-1)  # B, T, L
            if DEBUG:
                print('torch.argmax')
                print(torch.argmax(log_prob, dim=-1))
                input()
            batch_loss_sum = self.translation_loss_fun(
                log_probs=log_prob,
                targets=inputs['labels']
            )
            if 0:
                print('forward')
                print(inputs)
                print('pred in forward')
                print(self.map_new2old(torch.argmax(output_dict['logits'], dim=-1)))#B,L,V

            translation_loss = batch_loss_sum/batch_size
            other_outputs['translation_input'] = inputs['inputs_embeds']
        else:
            translation_loss = None

        return translation_loss, gloss_probabilities, attention, encoder_output, distillation_loss, other_outputs


    def run_batch(
        self,
        batch,
        recognition_beam_size: int = 1,
        translation_beam_size: int = 1,
        translation_beam_alpha: float = -1,
        translation_max_output_length: int = 100,
        output_gloss_prob: bool = False
    ) -> (np.array, np.array, np.array):
        if not self.use_gt_gloss:
            encoder_output, encoder_hidden, attention, intermediate = self.encode(
                sgn=batch.sgn, sgn_mask=batch.sgn_mask, sgn_length=batch.sgn_lengths
            )

            if self.do_recognition:
                gloss_scores = self.gloss_output_layer(encoder_output)
                gloss_probabilities_0 = gloss_scores.log_softmax(2)
                gloss_probabilities = gloss_probabilities_0.permute(1, 0, 2)
                gloss_probabilities = gloss_probabilities.cpu().detach().numpy()
                tf_gloss_probabilities = np.concatenate(
                    (gloss_probabilities[:, :, 1:],
                    gloss_probabilities[:, :, 0, None]),
                    axis=-1,
                )
                if type(recognition_beam_size) != list:
                    decoded_gloss_sequences = ctc_decode_func(
                        tf_gloss_probabilities, batch, recognition_beam_size, gloss_scores)
                else:
                    decoded_gloss_sequences = {}
                    for rbs in recognition_beam_size:
                        decoded_gloss_sequences[rbs] = ctc_decode_func(
                            tf_gloss_probabilities, batch, rbs, gloss_scores)

                # print('run batch')
                # print('sample_strategy', self.sample_strategy)
                # input()
                if self.use_logits:
                    encoder_output = gloss_scores
                if self.preceding_layer.__class__.__name__ == 'PrecedingLayer_weighted_embed':
                    softmax = torch.nn.Softmax(dim=2)
                    encoder_output = softmax(gloss_scores)
                encoder_output, new_sgn_mask, batch_pred_gls, has_empty = sparse_sample(
                    batch_enc_op=encoder_output,
                    batch_gls_prob=gloss_probabilities_0,  # n,t,c
                    batch_mask=batch.sgn_mask,  # B,1,L
                    select_strategy=self.sample_strategy if self.sample_strategy!='top1_random' else 'top1_mean',
                    return_pred_gls=True,
                    return_empty_flag=True)   


            else:
                new_sgn_mask = batch.sgn_mask
                gloss_probabilities = None
                decoded_gloss_sequences = None
        else:
            batch_size = batch.gls_lengths.shape[0]
            max_len = torch.max(batch.gls_lengths)
            tgt_ids = self.convert_gls_from_tensor_to_list(batch.gls, batch.gls_lengths)
            if self.copy_gloss:
                tgt_ids = batch_random_copy(tgt_ids, 
                    min_=self.copy_gloss_min, max_=self.copy_gloss_max,
                    unk_ratio=self.copy_gloss_unk_ratio,
                    unk_index=self.gls_vocab.stoi['<unk>'])
                max_len = max([len(ids) for ids in tgt_ids])
            encoder_output,  new_sgn_mask = self.build_gloss_target_embedding(
                src_embeddings=torch.zeros([batch_size, max_len, 1024 if self.plm_type=='mbart' else 768], 
                    dtype=batch.sgn.dtype, 
                    device=batch.sgn.device),
                tgt_ids=tgt_ids,  #ground_truth gls
                return_mask=True
            )
            decoded_gloss_sequences = None

        if self.pipeline:
            #directly use gloss embedding (S->G->T) replace predicted encoder_output with gloss embedding
            batch_size, max_len, _ = encoder_output.shape
            encoder_output = self.build_gloss_target_embedding(
                src_embeddings=torch.zeros([batch_size, max_len, 1024], dtype=encoder_output.dtype, device=encoder_output.device),
                tgt_ids=batch_pred_gls
            )
        elif (self.do_translation or self.do_distillation) and not self.use_gt_gloss:
            #intermediate layer 512->1024
            encoder_output = self.preceding_layer(encoder_output)  # B,T,D'
            #note that after transmormation padded features are not zeros!
            #print('after transform ', encoder_output.shape)
        
        if self.do_translation:
            if self.fusion: 
                #assert self.sample_strategy!='all'
                #input batch_pred_gls
                inputs = self.prepare_plm_inputs(
                    input_embed=encoder_output, input_mask=new_sgn_mask,
                    txt_input=None, txt_mask=None,
                    fusion=True, 
                    batch_pred_ids=batch_pred_gls)  
            else:
                inputs = self.prepare_plm_inputs(
                    input_embed=encoder_output, input_mask=new_sgn_mask,
                    txt_input=None, txt_mask=None) 

            if self.plm_type == 'mbart':
                output_dict = self.plm_model.generate(
                    **inputs,  #include decoder_input_ids
                    max_length=translation_max_output_length,
                    num_beams=translation_beam_size,
                    length_penalty=translation_beam_alpha,
                    return_dict_in_generate=True,
                    output_attentions=True)
            elif self.plm_type == 'gpt2':
                assert translation_beam_size==1, translation_beam_size
                forward_inputs, generate_inputs = inputs[0], inputs[1]
                forward_output = self.plm_model(**forward_inputs,  return_dict=True)
                #print('here!')
                #print(forward_output['past_key_values'][0][0][0].shape)
                output_dict = self.plm_model.generate(
                    **generate_inputs,
                    past=forward_output['past_key_values'],
                    max_length=translation_max_output_length,
                    num_beams=translation_beam_size,
                    length_penalty=translation_beam_alpha,
                    return_dict_in_generate=True,
                    pad_token_id = self.pad_token_id, # silence warning
                    eos_token_id=self.txt_eos_index,
                    output_attentions=True)                 

            
            output_dict['sequences'] = self.map_new2old(output_dict['sequences'])

            stacked_txt_output_decoded = self.tokenizer.batch_decode(output_dict['sequences'], 
                skip_special_tokens=True)
            if DEBUG:
                print('decoded sequences')
                print(output_dict['sequences'])

                print('batch_decode')
                print(stacked_txt_output_decoded)
                input()
            if self.tgt_lang=='de_DE':
                for di, d in enumerate(stacked_txt_output_decoded):
                    if len(d)>2 and d[-1]=='.' and d[-2]!=' ':
                        d = d[:-1]+ ' .'
                        stacked_txt_output_decoded[di] = d
            elif self.tgt_lang=='zh_CN':
                pass
                #perhaps some post process is needed here
            stacked_attention_scores = None
        else:
            stacked_txt_output_decoded = None
            stacked_attention_scores = None
        if output_gloss_prob:
            assert self.do_recognition
            return decoded_gloss_sequences, stacked_txt_output_decoded, stacked_attention_scores, gloss_probabilities_0.cpu().numpy()
        else:
            return decoded_gloss_sequences, stacked_txt_output_decoded, stacked_attention_scores, None

    def run_batch_ensemble_translation(
        self,
        batch,
        max_ensemble_n: int=10,
        num_return_sequences: int=1,
        ensemble_sample_random: str='uniform', #or weighted by probabilities
        translation_beam_size: int=4,
        translation_beam_alpha: float=1,
        translation_max_output_length: int = 100,
        **kwargs
    ):  
        #print('ensemble!')
        assert num_return_sequences <= translation_beam_size
        assert not self.sample_strategy=='all'
        assert self.do_recognition and self.do_translation and not self.pipeline and not self.use_gt_gloss
        encoder_outputs0, encoder_hidden = self.encode(
            sgn=batch.sgn, sgn_mask=batch.sgn_mask, sgn_length=batch.sgn_lengths
        ) #keep the same
        batch_size = encoder_outputs0.shape[0]        
        gloss_scores = self.gloss_output_layer(encoder_outputs0) 
        gloss_probabilities = gloss_scores.log_softmax(2) #B,D,L

        from helpers import get_all_combinations
        import random
        ensemble_predictions_decoded = [[] for i in range(batch_size)]
        device = encoder_outputs0.device
        batch_predictions = []
        batch_all_results = []
        for i in range(batch_size):
            #process one by one

            # step1. collect possible combinations
            length = torch.sum(batch.sgn_mask[i]) #B,1,L
            all_combinations_ids = get_all_combinations(gls_prob=gloss_probabilities[i, :length])  
            if len(all_combinations_ids)>max_ensemble_n:
                select_comb_ind = np.random.permutation(np.arange(len(all_combinations_ids)))[:max_ensemble_n]
                all_combinations_ids = [all_combinations_ids[ind] for ind in select_comb_ind]
                ensemble_n = max_ensemble_n
            else:
                ensemble_n = len(all_combinations_ids)

            # step2. prepare encoder_output_sample new_sgn_mask and generate
            sparse_samples = [{} for ii in range(ensemble_n)]
            for ci, (comb_ids, probs) in enumerate(all_combinations_ids):
                # a list
                encoder_output_sample = torch.stack(
                    [encoder_outputs0[i,id_,:] for id_ in comb_ids], dim=0).unsqueeze(0) #1,T,D
                new_sgn_mask = torch.ones(
                    [1,1,len(comb_ids)], 
                    dtype=batch.sgn_mask.dtype,
                    device=batch.sgn_mask.device)
                if hasattr(self, 'preceding_layer'):
                    encoder_output_sample = self.preceding_layer(encoder_output_sample)
                sparse_samples[ci] = {
                    'span_probs': probs,
                    'encoder_output_sample':encoder_output_sample,
                    'new_sgn_mask': new_sgn_mask}
                sparse_samples[ci]['encoder_inputs'] = self.prepare_plm_inputs(
                    input_embed=encoder_output_sample, input_mask=new_sgn_mask,
                    txt_input=None, txt_mask=None
                )              
                #translate
                sparse_samples[ci]['output_sequences'] = self.plm_model.generate(
                    **sparse_samples[ci]['encoder_inputs'],
                    max_length=translation_max_output_length,
                    num_beams=translation_beam_size,
                    num_return_sequences=num_return_sequences,
                    length_penalty=translation_beam_alpha,
                    #return_dict_in_generate=True,
                    output_attentions=False)  
                    # some redundant computation is made here, I'll improve it later     
                    #              
                if 0:
                    #print(num_return_sequences)
                    #print(sparse_samples[ci]['output_sequences'].shape)
                    #print debug
                    sparse_samples[ci]['output_sequences_mapped'] = self.map_new2old(sparse_samples[ci]['output_sequences'])
                    sparse_samples[ci]['stacked_txt_output_decoded'] = self.tokenizer.batch_decode(
                        sparse_samples[ci]['output_sequences_mapped'], 
                        skip_special_tokens=True)                  
                    #print(len(sparse_samples[ci]['stacked_txt_output_decoded']))
                    for b in range(1):
                        #print('example #',b)
                        for j in range(num_return_sequences):
                            ind = b*num_return_sequences+j
                            #print(sparse_samples[ci]['stacked_txt_output_decoded'][ind])
                        #print()    
                    #input()  
                    ensemble_predictions_decoded[i].append(sparse_samples[ci]['stacked_txt_output_decoded'])

            # step3. compute average score
            #  sparse_samples[ci]: 'encoder_inputs'
            #print('self.txt_bos_index', self.txt_bos_index)
            best_scores = None
            best_hyp = None
            all_results = []
            for ci in range(ensemble_n):
                sparse_samples[ci]['scores'] = []
                for si in range(num_return_sequences):
                    #for each hypothesis
                    hypothesis = sparse_samples[ci]['output_sequences'][si].detach().cpu().numpy() #new id
                    hyp_input, hyp_label = [] ,[]
                    for h in hypothesis:
                        if h == self.txt_bos_index:
                            hyp_input.append(h)
                        elif h == self.txt_eos_index:
                            hyp_label.append(h)
                            break
                        else:
                            hyp_input.append(h)
                            hyp_label.append(h)
                    # print(hypothesis)
                    # print(hyp_input)
                    # print(hyp_label)
                    # input()
                    # compute loss for each inputs_embeds
                    hyp_len = len(hyp_input)
                    hyp_input = torch.tensor(hyp_input, dtype=torch.long, device=device).unsqueeze(0)
                    hyp_label = torch.tensor(hyp_label, dtype=torch.long, device=device).unsqueeze(0)
                    scores = []
                    for cii in range(ensemble_n):
                        outputs_dict = self.plm_model(
                            inputs_embeds=sparse_samples[cii]['encoder_inputs']['inputs_embeds'],
                            attention_mask=sparse_samples[cii]['encoder_inputs']['attention_mask'],
                            decoder_input_ids=hyp_input,
                            decoder_attention_mask=torch.ones([1, hyp_len], device=device, dtype=torch.bool),
                            labels=hyp_label,
                            return_dict=True
                        )
                        XE_loss = torch.nn.CrossEntropyLoss(reduction='sum') # no length penalty, normalize by length[]
                        score = -1*XE_loss(outputs_dict['logits'].view(hyp_len,-1), hyp_label.view(-1)) #batch_size=1 -logprob
                        score = score/(hyp_len**translation_beam_alpha)
                        #log_probs = torch.nn.functional.log_softmax(outputs_dict['logits'], dim=-1)  # B(1), T, L
                        #score = self.translation_loss_fun(log_probs,targets=hyp_label).detach().cpu().item()
                        scores.append(score.item())
                        #print(ci,si, cii, score.item())
                    mean_scores = np.mean(scores)
                    sparse_samples[ci]['scores'].append(mean_scores)
                    all_results.append({
                        'hyp': sparse_samples[ci]['stacked_txt_output_decoded'][si],
                        'sample_ind': ci,
                        'beam_rank': si,
                        'scores': scores,
                        'probs': [ss['span_probs'] for ss in sparse_samples],
                        'mean_scores': mean_scores
                    })
                    # print(all_results[-1])
                    # input()
                #print(sparse_samples[ci]['scores'])
                for sii, s in enumerate(sparse_samples[ci]['scores']):
                    if best_scores==None or s > best_scores:
                        best_scores = s
                        best_hyp = sparse_samples[ci]['stacked_txt_output_decoded'][sii]

            batch_predictions.append(best_hyp.replace('.',' .'))
            batch_all_results.append(all_results)
        return batch_predictions, batch_all_results

