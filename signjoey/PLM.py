from json import decoder
from signjoey import attention, batch
import torch, os, pickle
import torch.nn as nn
import transformers
from transformers import MBartForConditionalGeneration, MBartTokenizer
from signjoey.model import SignModel
from signjoey.vocabulary import (
    TextVocabulary,
    GlossVocabulary,
    SIL_TOKEN,
    UNK_TOKEN,
    PAD_TOKEN,
    EOS_TOKEN,
    BOS_TOKEN,
)
from signjoey.helpers import (
    shift_tokens_right,
    freeze_params
)
from torch import Tensor
from collections import defaultdict
from signjoey.loss import XentLoss
class transformer_spm(nn.Module):
    def __init__(self,
        model,
        tokenizer,
        old2new,
        old_txt_vocab,
        old_gls_vocab
        ):
        super().__init__()
        self.base_model = model
        self.tokenizer = tokenizer
        self.old_txt_vocab = old_txt_vocab
        self.old_gls_vocab = old_gls_vocab

        self.old2new = defaultdict(lambda: 3, old2new) #3 unk
        self.new2old = defaultdict(lambda: 3) #
        for o,n in self.old2new.items():
            assert n >= 0 and n < len(
                self.base_model.txt_vocab), (n, len(self.base_model.txt_vocab))
            self.new2old[n] = o
        #self.txt_bos_index = self.base_model.txt_vocab.stoi[BOS_TOKEN]
        #We need to set bos_index (only exists in decoder input) to tgt_lang_code
        #to be used in decoding (run_batch)
        # bos for tgt_lang
        self.txt_bos_index = self.old2new[self.tokenizer.lang_code_to_id['de_DE']]
        self.base_model.txt_bos_index = self.txt_bos_index  # !!
        self.txt_pad_index = self.base_model.txt_vocab.stoi[PAD_TOKEN]
        self.txt_eos_index = self.base_model.txt_vocab.stoi[EOS_TOKEN]

        self.gls_pad_index = self.base_model.gls_vocab.stoi[PAD_TOKEN]
        print('bos', self.txt_bos_index)
        print('bos base_model', self.base_model.txt_bos_index)
        print('pad', self.txt_pad_index)
        print('pad base_model', self.base_model.txt_pad_index)
        print('eos', self.txt_eos_index)
        print('eos base_model', self.base_model.txt_eos_index)
        print('pad', self.gls_pad_index)
    def set_train(self, verbose=False):
        self.train()
    
    def set_eval(self):
        self.eval()

    def map_new2old(self, batch_input_ids):
        if self.old2new == None:
            return batch_input_ids
        new_batch_input_ids = batch_input_ids.clone()
        for bi, input_ids in enumerate(batch_input_ids):
            is_finished = False
            for ii, id_ in enumerate(input_ids):
                if is_finished:
                    new_batch_input_ids[bi, ii] = 2 # to skip
                    continue
                if not is_finished and id_.item()==2:
                    new_batch_input_ids[bi, ii] = 2
                    is_finished=True
                    
                if id_.item() == 30:
                    #is a special token but won't be ignored by batch_decode, so here we convert it to a special token
                    new_batch_input_ids[bi, ii] = 2  # </s>
                else:
                    new_batch_input_ids[bi, ii] = self.new2old[id_.item()]
        return new_batch_input_ids

    def map_old2new(self, batch_input_ids):
        if self.old2new == None:
            return batch_input_ids
        new_batch_input_ids = batch_input_ids.clone()
        for bi, input_ids in enumerate(batch_input_ids):
            for ii, id_ in enumerate(input_ids):
                new_batch_input_ids[bi, ii] = self.old2new[id_.item()]
        return new_batch_input_ids

    def prepare_gls_input(self, gls, gls_lengths):
        #!new_sgn_mask B,1,L
        batch_size, padded_length = gls.shape
        device = gls.device
        batch_raw_gls = []
        for i in range(batch_size):
            raw_gls = [self.old_gls_vocab.itos[gls[i, j]]
                       for j in range(gls_lengths[i])
                       if self.old_gls_vocab.itos[gls[i, j]] != EOS_TOKEN] #see batch.py we append <EOS> token when input_feature='gloss'
            batch_raw_gls.append(' '.join(raw_gls))
        inputs = self.tokenizer(  # already contain </s> lang_code at the end
            batch_raw_gls,
            padding='longest',  # we've already control the maximum length of input seq in dataloader
            return_attention_mask=True,
            return_length=True,
            return_tensors="pt")  # input_ids, length, attention_mask
        inputs['input_ids'] = self.map_old2new(inputs['input_ids'])
        return inputs['input_ids'].to(device), inputs['length'].to(device), inputs['attention_mask'].unsqueeze(1).bool().to(device)

    def prepare_txt_input(self, txt, txt_lengths):
        batch_size, padded_length = txt.shape
        device = txt.device
        batch_raw_txt = []
        for i in range(batch_size):
            raw_txt = [self.old_txt_vocab.itos[txt[i, j]] for j in range(1, txt_lengths[i])
                       if self.old_txt_vocab.itos[txt[i, j]] != EOS_TOKEN]  # the first one is [BOS] (prepended by torchtext field)
            batch_raw_txt.append(' '.join(raw_txt))
        with self.tokenizer.as_target_tokenizer():
            labels = self.tokenizer(
                batch_raw_txt,
                padding='longest',  # we've already control the maximum length of input seq in dataloader
                return_attention_mask=True,
                return_length=True,
                return_tensors="pt")
        labels['input_ids'] = self.map_old2new(labels['input_ids'])
        decoder_input_ids, label_input_ids = shift_tokens_right(
            labels['input_ids'].clone(), 
            self.tokenizer.pad_token_id, 
            ignore_index=self.base_model.txt_pad_index)  # [lang, ]
        # attention mask keeps the same after shifting
        # print('batch_raw_txt', batch_raw_txt)
        # print('decoder input ids')
        # print(decoder_input_ids)
        # print('label input ids')
        # print(label_input_ids)
        decoder_attention_mask = labels['attention_mask'].unsqueeze(1).bool() #b,1,L
        return label_input_ids.to(device), decoder_input_ids.to(device), decoder_attention_mask.to(device)

    def prepare_inputs(self, gls, gls_lengths, txt_input=None, txt_mask=None):
        gls, gls_lengths, gls_mask = self.prepare_gls_input(
            gls, gls_lengths)
        encoder_inputs = {
            'sgn': gls,
            'sgn_mask': gls_mask,
            'sgn_lengths': gls_lengths,
        }
        if txt_input != None and txt_mask != None:
            #txt_mask (B,1,L) is not causal yet, so we can get txt_lengths from it
            txt_lengths = torch.sum(
                txt_mask, dim=-1).squeeze(1)  # B (including)
            txt_label, txt_input, txt_mask = self.prepare_txt_input(
                txt_input, txt_lengths)
            decoder_inputs = {
                'txt_input': txt_input,
                'txt_mask': txt_mask,
            }
        else:
            txt_label = None
            decoder_inputs = {}
        inputs = {**encoder_inputs, **decoder_inputs}
        return inputs, txt_label

    def forward(
        self,
        sgn: Tensor,
        sgn_mask: Tensor,
        sgn_lengths: Tensor,
        txt_input: Tensor,
        txt_mask: Tensor = None,
        output_attention: bool = False,
        name: list=[] #only for debug
    ) -> (Tensor, Tensor, Tensor, Tensor):
        inputs, txt_label = self.prepare_inputs(gls=sgn, gls_lengths=sgn_lengths, 
            txt_input=txt_input, txt_mask=txt_mask)
        
        # print(inputs)
        # print(txt_label)
        # input()
        # if 'dev/28May_2010_Friday_tagesschau-7496'  in name:
        #     print('in get_loss_for_batch')
        #     print(inputs)
        model_output =  self.base_model(**inputs,output_attention=output_attention)
        #note that we need to return txt_label!
        return model_output, txt_label
    
    def run_batch(
        self,
        batch,
        **kwargs,
    ):
        inputs, _ = self.prepare_inputs(
            gls=batch.gls, 
            gls_lengths=batch.gls_lengths)
        batch.gls = inputs['sgn']
        batch.gls_mask = inputs['sgn_mask' ]
        batch.gls_lengths = inputs['sgn_lengths']
        outputs = self.base_model.run_batch(
            batch,
            **kwargs
        )
        _, stacked_txt_output, stacked_attention_scores, _ = outputs
        # if 'dev/28May_2010_Friday_tagesschau-7496' in batch.sequence:
        #     print('batch.gls')
        #     print(batch.gls)
        #     print('batch.gls_mask')
        #     print(batch.gls_mask)
        #     print('batch.gls_length')
        #     print(batch.gls_lengths)
        #     print(stacked_txt_output) 
        #     input()           
        #print(batch.sequence)
        # print(batch.txt)
        #print(stacked_txt_output)
        stacked_txt_output = self.map_new2old(torch.tensor(stacked_txt_output))
        #print('after mapping...')
        #print(stacked_txt_output)
        stacked_txt_output_decoded = self.tokenizer.batch_decode(
            stacked_txt_output,
            skip_special_tokens=True)
        #print(stacked_txt_output_decoded)  # list of string
        #input()
        for di, d in enumerate(stacked_txt_output_decoded):
            if len(d)>=2 and d[-1] == '.' and d[-2] != ' ':
                d = d[:-1] + ' .'
                stacked_txt_output_decoded[di] = d
        # input()
        # print(stacked_txt_output_decoded)
        #print('after detokenize')
        # print(stacked_txt_output_decoded)
        # input()
        return _, stacked_txt_output_decoded, stacked_attention_scores, _

class huggingface_transformer(nn.Module):
    def __init__(self, 
        plm_type, 
        plm, 
        tokenizer,
        gls_vocab, 
        txt_vocab,
        old2new_file: str=None,
        freeze_embed: bool=False,
        src_lang: str='de_DE',
        gloss_tokenizer: str='default',
        from_scratch: bool=False,
        loss_level: str='sentence',
        lower_case: bool=False, #convert to lower case for gloss
        label_smoothing: float=0,
        ):
        super().__init__()
        self.plm_type = plm_type #mBart
        self.plm = plm
        self.lower_case = lower_case
        if self.plm_type.lower()=='mbart':
            tokenizer.src_lang = src_lang 
        elif self.plm_type.lower()=='gpt2':
            tokenizer.pad_token = '<pad>'
            tokenizer.eos_token = '</s>'
            #we cannot use endoftext as padding token as it will be used as ignoreindex in labels later
        self.tokenizer = tokenizer
        assert gloss_tokenizer in ['default', 'custom']
        self.gloss_tokenizer = gloss_tokenizer 
        if gloss_tokenizer == 'custom':
            print('Use customized gloss tokenizer')
        self.loss_level = loss_level
        self.label_smoothing = label_smoothing
        self.ignore_index = self.tokenizer.pad_token_id
        self.translation_loss_fun = XentLoss(
            pad_index=self.ignore_index,  # ignore
            smoothing=self.label_smoothing)
        #input (log_probs, targets)

        self.gls_vocab = gls_vocab
        self.txt_vocab = txt_vocab #itos
        if os.path.isfile(old2new_file):
            print('Map old id to new id use ',old2new_file)
            with open(old2new_file,'rb') as f:
                old2new = pickle.load(f)
            self.old2new = defaultdict(lambda: 3, old2new) #3 unk
            self.new2old = defaultdict(lambda: 3) #
            for o,n in self.old2new.items():
                assert n>=0 and n<self.plm.config.vocab_size, (n, self.plm.config.vocab_size)
                if type(o) != str:
                    self.new2old[n] = o

        else:
            print(old2new_file,' is not a file. Use original ids')
            self.old2new = None
            self.new2old = None

        if self.plm_type.lower()=='mbart':        
            self.gls_pad_index = gls_vocab.stoi[PAD_TOKEN] 
            self.gls_lang_index = self.old2new[tokenizer.lang_code_to_id[src_lang]]
            print('self.gls_lang_index', self.gls_lang_index)
            self.txt_pad_index = txt_vocab.stoi[PAD_TOKEN]
            self.txt_bos_index = tokenizer.convert_tokens_to_ids(self.tokenizer.tgt_lang)
        elif self.plm_type.lower()=='gpt2':
            self.gls_token_type_id = 1
            self.txt_token_type_id = 0
            self.gls_pad_index = gls_vocab.stoi[PAD_TOKEN] 
            self.txt_pad_index = txt_vocab.stoi[PAD_TOKEN]
            self.txt_pad_index_gpt = tokenizer.pad_token_id
            self.txt_bos_index = txt_vocab.stoi[BOS_TOKEN]#! a special token
            self.txt_bos_index_gpt = tokenizer.convert_tokens_to_ids('<s>') #0
            self.txt_eos_index = txt_vocab.stoi[EOS_TOKEN]
            self.txt_eos_index_gpt = tokenizer.eos_token_id


        
        if from_scratch:
            print('Train from scratch, re-initialize_model')
            self.plm.init_weights()

        if freeze_embed:
            print('freeze plm embedding ...')
            if self.plm_type.lower()=='mbart': 
                freeze_params(self.plm.model.shared)
                #freeze_params(self.plm.lm_head) #already tied up
            elif self.plm_type.lower()=='gpt2':
                freeze_params(self.plm.transformer.wte)
        

    def prepare_gls_input(self, gls, gls_lengths):
        #gls B,L
        #gls_lengths B
        batch_size, padded_length = gls.shape
        device = gls.device
        batch_raw_gls = []
        for i in range(batch_size):
            raw_gls = [self.gls_vocab.itos[gls[i,j]] 
                for j in range(gls_lengths[i])
                if self.gls_vocab.itos[gls[i,j]]!=EOS_TOKEN]
            if self.lower_case:
                batch_raw_gls.append(' '.join(raw_gls.lower()))
            else:
                batch_raw_gls.append(' '.join(raw_gls))
        if self.gloss_tokenizer=='default':
            inputs = self.tokenizer( #already contain </s> lang_code at the end
                batch_raw_gls,  
                padding='longest', #we've already control the maximum length of input seq in dataloader
                return_attention_mask=True,
                return_length=True,
                return_tensors="pt") #input_ids, length, attention_mask
            inputs['input_ids'] = self.map_old2new(inputs['input_ids'])
        else:
            inputs = {'input_ids':[], 'length':[], 'attention_mask':None}
            eos_ids = [self.tokenizer.eos_token_id, self.gls_lang_index] # a bug here!!! self.gls_lang_index should be converted by old2new
            for raw_gls in batch_raw_gls: 
                input_id = [self.old2new[g] for g in raw_gls.split()] + eos_ids
                inputs['input_ids'].append(input_id)
                inputs['length'].append(len(input_id))
            max_length = max(inputs['length'])
            #pad and make attention_mask
            inputs['attention_mask'] = torch.zeros([batch_size, max_length], dtype=torch.long)
            for i in range(batch_size):
                length_ = inputs['length'][i]
                inputs['attention_mask'][i, :length_] = 1
                inputs['input_ids'][i] += [self.tokenizer.pad_token_id]*(max_length-length_)
            inputs['input_ids'] = torch.tensor(inputs['input_ids'], dtype=torch.long)
            inputs['length'] = torch.tensor(inputs['length'], dtype=torch.long)

        return inputs['input_ids'].to(device), inputs['length'].to(device), inputs['attention_mask'].to(device)  #B,L



    def prepare_txt_input(self, txt, txt_lengths):
        batch_size, padded_length = txt.shape
        device = txt.device
        batch_raw_txt = []
        for i in range(batch_size):
            raw_txt = [self.txt_vocab.itos[txt[i,j]] for j in range(1,txt_lengths[i]) \
                if self.txt_vocab.itos[txt[i,j]] != EOS_TOKEN] #the first one is [BOS] (prepended by torchtext field)
            batch_raw_txt.append(' '.join(raw_txt))
        with self.tokenizer.as_target_tokenizer():
            labels = self.tokenizer(
                batch_raw_txt,            
                padding='longest', #we've already control the maximum length of input seq in dataloader
                return_attention_mask=True,
                return_length=True,
                return_tensors="pt")
        labels['input_ids'] = self.map_old2new(labels['input_ids']) #tgt_lang_code are converted here (from 250003->6)
        decoder_input_ids, label_input_ids = shift_tokens_right(
            labels['input_ids'].clone(), 
            self.tokenizer.pad_token_id,
            ignore_index=self.ignore_index) #[lang, ]
        decoder_attention_mask = labels['attention_mask'] #attention mask keeps the same after shifting
        return label_input_ids.to(device), decoder_input_ids.to(device), decoder_attention_mask.to(device)

    def set_train(self, verbose=False):
        self.train()
    
    def set_eval(self):
        self.eval()
    
    def map_old2new(self, batch_input_ids):
        if self.old2new==None:
            return batch_input_ids
        new_batch_input_ids = batch_input_ids#.clone()
        for bi,input_ids in enumerate(batch_input_ids):
            for ii, id_ in enumerate(input_ids):
                if type(id_)==int:
                    new_batch_input_ids[bi][ii] = self.old2new[id_]
                else:
                    new_batch_input_ids[bi][ii] = self.old2new[id_.item()]
        return new_batch_input_ids

    def map_new2old(self, batch_input_ids):
        if self.old2new==None:
            return batch_input_ids
        new_batch_input_ids = batch_input_ids#.clone()
        for bi,input_ids in enumerate(batch_input_ids):
            for ii, id_ in enumerate(input_ids):
                if id_.item()==30:
                    #is a special token but won't be ignored by batch_decode, so here we convert it to a special token
                    new_batch_input_ids[bi][ii] = 2 # </s>
                else:
                    new_batch_input_ids[bi][ii] = self.new2old[id_.item()]
        return new_batch_input_ids

    def prepare_inputs(self, sgn, sgn_lengths, txt_input=None, txt_mask=None):
        if self.plm_type.lower()=='mbart':
            sgn, sgn_lengths, sgn_mask_transformer = self.prepare_gls_input(sgn, sgn_lengths)
            sgn_mask = sgn_mask_transformer.bool().unsqueeze(1) #B,1,L   
            encoder_inputs = {
                'input_ids':sgn,
                'attention_mask': sgn_mask_transformer,            
            }
            if txt_input!=None and txt_mask!=None:
                #txt_mask (B,1,L) is not causal yet, so we can get txt_lengths from it
                txt_lengths = torch.sum(txt_mask, dim=-1).squeeze(1) #B (including)
                txt_label, txt_input, txt_mask_transformer = self.prepare_txt_input(txt_input, txt_lengths)
                decoder_inputs = {
                    'decoder_input_ids': txt_input,
                    'decoder_attention_mask': txt_mask_transformer,
                    'labels': txt_label            
                }
            else:
                decoder_inputs = {}
            inputs = {**encoder_inputs, **decoder_inputs}
        elif self.plm_type.lower()=='gpt2':
            # step1 reverse gls
            gls, gls_lengths = sgn, sgn_lengths
            if txt_input!=None and txt_mask!=None:
                is_train = True
                txt, txt_lengths = txt_input, torch.sum(txt_mask, dim=-1).squeeze(1)
            else:
                is_train = False
            batch_size, padded_length = gls.shape
            device = gls.device
            inputs = {'input_ids':[],'token_type_ids':[]}
            if is_train:
                inputs['labels'] = []
            input_lengths = []
            # we don't need attention mask here now, causal mask is automatically used
            for i in range(batch_size):
                raw_gls = [self.gls_vocab.itos[gls[i,j]] 
                    for j in range(gls_lengths[i])
                    if self.gls_vocab.itos[gls[i,j]]!=EOS_TOKEN]
                raw_gls = ' '.join(raw_gls)
                if self.lower_case:
                    raw_gls = raw_gls.lower()
                if self.gloss_tokenizer=='default':
                    gls_ids = self.tokenizer(raw_gls)['input_ids']
                elif self.gloss_tokenizer=='custom':
                    gls_ids = []
                    for g in raw_gls.split():
                        if self.old2new[g]==3:
                            print(raw_gls.split(), g)
                        gls_ids.append(self.old2new[g])
                else:
                    raise ValueError

                if is_train:
                    raw_txt = [self.txt_vocab.itos[txt[i,j]] for j in range(1,txt_lengths[i]) \
                        if self.txt_vocab.itos[txt[i,j]] != EOS_TOKEN] 
                    raw_txt = ' '.join(raw_txt)
                    raw_ids = self.tokenizer(raw_txt)['input_ids']
                else:
                    raw_ids = []
                # print(raw_gls)
                # print(raw_txt)
                # print(gls_ids)
                # print(raw_ids)
                sequence = gls_ids+[self.txt_bos_index_gpt]+raw_ids
                sequence_token_type = [self.gls_token_type_id]*len(gls_ids) + [self.txt_token_type_id]*(len(raw_ids)+1)
                label = [self.ignore_index]*len(gls_ids) + raw_ids+[self.txt_eos_index_gpt]
                inputs['input_ids'].append(sequence)
                input_lengths.append(len(sequence))
                inputs['token_type_ids'].append(sequence_token_type)
                if is_train:
                    inputs['labels'].append(label)
            
            #useless for gpt2 in old2new for each gloss we've already got them to vocab_id
            #inputs['input_ids'] = self.map_old2new(inputs['input_ids'])
            #padding
            max_length = max(input_lengths)
            if is_train==False:
                inputs['attention_mask'] = torch.ones([batch_size, max_length],dtype=torch.long, device=device)
            for i in range(batch_size):
                pad_length = max_length-input_lengths[i]
                if is_train: #pad right
                    inputs['input_ids'][i] += [self.txt_pad_index_gpt]*pad_length
                    inputs['labels'][i] += [self.ignore_index]*pad_length
                    inputs['token_type_ids'][i] += [self.txt_token_type_id]*pad_length
                else: #pad left
                    inputs['input_ids'][i] = [self.txt_pad_index_gpt]*pad_length + inputs['input_ids'][i]
                    inputs['token_type_ids'][i] = [self.gls_token_type_id]*pad_length + inputs['token_type_ids'][i]
                    if 'attention_mask' in inputs and pad_length>0:
                        inputs['attention_mask'][i, :pad_length:] = 0
            inputs['input_ids'] = torch.tensor(inputs['input_ids'], dtype=torch.long, device=device)  
            inputs['token_type_ids'] = torch.tensor(inputs['token_type_ids'], dtype=torch.long, device=device)
            if is_train:
                inputs['labels'] = torch.tensor(
                    inputs['labels'], dtype=torch.long, device=device)
            # print(inputs)
            # input()
        else:
            raise ValueError
        return inputs

    def run_batch(
        self,
        batch,
        recognition_beam_size: int = 1,
        translation_beam_size: int = 1,
        translation_beam_alpha: float = 1,
        translation_max_output_length: int = 100,
        output_gloss_prob: bool = False
    ) :
        batch_size = batch.gls.shape[0]
        inputs = self.prepare_inputs(
            sgn=batch.gls,
            sgn_lengths=batch.gls_lengths)
        


        assert translation_beam_alpha>0, translation_beam_alpha
        if self.plm_type.lower()=='mbart':
            decoder_start_token_id=self.tokenizer.lang_code_to_id[self.tokenizer.tgt_lang]
            if self.old2new:
                decoder_start_token_id = self.old2new[decoder_start_token_id]
            output_dict = self.plm.generate(
                **inputs, 
                max_length=translation_max_output_length,
                num_beams=translation_beam_size,
                length_penalty=translation_beam_alpha,
                return_dict_in_generate=True,
                output_attentions=True,
                decoder_start_token_id=decoder_start_token_id)
        elif self.plm_type.lower()=='gpt2':
            # print('run batch')
            # print('input_ids')
            # print(inputs['input_ids'])
            # print('token type ids')
            # print(inputs['token_type_ids'])
            # print('attention mask')
            # print(inputs['attention_mask'])
            output_dict = self.plm.generate(
                input_ids=inputs['input_ids'],
                token_type_ids=inputs['token_type_ids'],
                attention_mask=inputs['attention_mask'],#,important here!
                max_length=translation_max_output_length,
                num_beams=translation_beam_size,
                length_penalty=translation_beam_alpha,
                return_dict_in_generate=True,
                pad_token_id = 1, # silence warning
                eos_token_id=self.txt_eos_index_gpt,
                output_attentions=True) 

            # note that we need to remove prefix (token_type_id=0) in output_dict['sequences']
            for i in range(batch_size):
                for j,tt in enumerate(inputs['token_type_ids'][i]):
                    if tt==self.gls_token_type_id:
                        output_dict['sequences'][i,j] = self.txt_pad_index_gpt# as special token, to be skipped by batch_decode
            # print('predict')
            # print(output_dict['sequences']) 
            # input()      
        # print(output_dict['sequences'])
        #print(output_dict.keys()) # sequences, encoder_attentions, decoder_attentions
        #return decoded_gloss_sequences, stacked_txt_output, stacked_attention_scores, gloss_probabilities_0.cpu().numpy()
        #print(output_dict['sequences'].shape) #Bs, Length
        #mbart_debug
        if 0:#'dev/14April_2010_Wednesday_heute-1879' in batch.sequence:
            print('predict')
            print(output_dict['sequences'][0,:])
            input()



        output_dict['sequences'] = self.map_new2old(output_dict['sequences'])
        stacked_txt_output_decoded = self.tokenizer.batch_decode(output_dict['sequences'], 
            skip_special_tokens=True)
        if 0:
            print(stacked_txt_output_decoded[0])
        #!! split end common and the last word!
        #print(stacked_txt_output_decoded) #list of string
        for di, d in enumerate(stacked_txt_output_decoded):
            if len(d)>2 and d[-1]=='.' and d[-2]!=' ':
                d = d[:-1]+ ' .'
                stacked_txt_output_decoded[di] = d
        #mbart_debug
        if 0:#'dev/30August_2011_Tuesday_heute-783' in batch.sequence:
            print('run_batch')
            print(translation_beam_size, translation_beam_alpha, translation_max_output_length)
            print(batch.sequence)
            print(batch.gls)
            batch_raw_gls = []
            for i in range(batch_size):
                raw_gls = [self.gls_vocab.itos[batch.gls[i,j]] 
                    for j in range(batch.gls_lengths[i])
                    if self.gls_vocab.itos[batch.gls[i,j]]!=EOS_TOKEN]
                if self.lower_case:
                    batch_raw_gls.append(' '.join(raw_gls.lower()))
                else:
                    batch_raw_gls.append(' '.join(raw_gls))
            print(batch_raw_gls)
            print(inputs)
            input_embeds = []
            for i in range(batch_size):
                ids = inputs['input_ids'][i]
                emb = torch.stack([self.plm.model.shared.weight[i,:] for i in ids], dim=0) #T,D
                input_embeds.append(emb)
            input_embeds = torch.stack(input_embeds, dim=0) #B,T,D
            print('input_embeds')
            #torch.set_printoptions(profile="full")
            input_embeds = self.plm.model.encoder.embed_tokens(inputs['input_ids']) * self.plm.model.encoder.embed_scale
            print('predict')
            print(inputs)
            print('decoder_start_token_id', decoder_start_token_id)
            print('emb', self.plm.model.shared.weight[decoder_start_token_id])
            print(output_dict['sequences'])
            print(stacked_txt_output_decoded)
            print('----use decoder_ids')
            batch_start_ids = torch.ones([batch_size,1],dtype=torch.long, device=inputs['input_ids'].device)*decoder_start_token_id
            print(batch_start_ids)
            output_dict2 = self.plm.generate(
                **inputs, 
                max_length=translation_max_output_length,
                num_beams=translation_beam_size,
                length_penalty=translation_beam_alpha,
                return_dict_in_generate=True,
                output_attentions=True,
                decoder_input_ids=batch_start_ids)
            print(self.map_new2old(output_dict2['sequences']))
            print('----use input_embs')
            print(input_embeds)
            output_dict3 = self.plm.generate(
                inputs_embeds = input_embeds,
                attention_mask = inputs['attention_mask'], 
                max_length=translation_max_output_length,
                num_beams=translation_beam_size,
                length_penalty=translation_beam_alpha,
                return_dict_in_generate=True,
                output_attentions=True,
                decoder_input_ids=batch_start_ids)
            print(self.map_new2old(output_dict3['sequences']))
            #input()


        # input()
        #print(stacked_txt_output_decoded)
        #stacked_attention_scores = output_dict['decoder_attentions'][0] #one for each layer
        return None, stacked_txt_output_decoded, None, None

        # batch_gls_predictions,
        # batch_txt_predictions,
        # batch_attention_scores,
        # batch_gls_prob #B, T, C return 


    def forward(
        self,
        sgn: Tensor,
        sgn_mask: Tensor,
        sgn_lengths: Tensor,
        txt_input: Tensor,
        txt_mask: Tensor = None,
        output_attention: bool=False,
        name: list=[]
    ) -> (Tensor, Tensor, Tensor, Tensor):
        '''
        note that here the gls sequence is already encoded as int index
        but for huggingface transformer, we expect raw string as tokenizer's input, 
        so we need to reverse them (even do some repadding, recreate attention mask)
        This seems a little complicated yet avoids modification on the code of dataloader part
        '''
        inputs = self.prepare_inputs(sgn, sgn_lengths, txt_input, txt_mask)
        if self.plm_type.lower()=='mbart':
            output_dict = self.plm(
                **inputs,
                return_dict=True,
                output_attentions=output_attention)
        elif self.plm_type.lower()=='gpt2':
            output_dict = self.plm(
                input_ids = inputs['input_ids'],
                token_type_ids= inputs['token_type_ids'],
                return_dict=True,
                output_attentions=output_attention)            
        assert self.loss_level=='sentence', self.loss_level
        batch_size = output_dict['logits'].shape[0]
        #B, T, L
        log_prob = torch.nn.functional.log_softmax(output_dict['logits'], dim=-1)

        #debug
        if 0:
            print('forward')
            print('input_ids')
            print(inputs)
            print(self.map_new2old(torch.argmax(log_prob, dim=-1)))
            input()
        batch_loss_sum = self.translation_loss_fun(
            log_probs=log_prob,
            targets=inputs['labels']
        ) 

        output_dict['loss'] = batch_loss_sum/batch_size
        #logits 1, t,v
        '''
        if 0:
            loss_fct = torch.nn.CrossEntropyLoss(reduction='sum')
            logits = output_dict['logits']
            logits0 = output_dict['logits'][:,0,:] #b,V
            labels0 = inputs['labels'][:,0]
            logits5 = output_dict['logits'][:,:2,:]
            labels5 = inputs['labels'][:,:2]
            masked_lm_loss0 = loss_fct(
                logits0.view(-1, self.plm.config.vocab_size), labels0.view(-1))
            print('t=0', torch.mean(masked_lm_loss0))
            masked_lm_loss5 = loss_fct(
                logits5.reshape(-1, self.plm.config.vocab_size), labels5.reshape(-1))
            masked_lm_loss = loss_fct(
                logits.view(-1, self.plm.config.vocab_size), inputs['labels'].view(-1))
            #print('t=1', torch.mean(masked_lm_loss1))
        if 0:#'dev/14April_2010_Wednesday_heute-1879' in name:
            print('forward')
            print(inputs)
            # input()
            logits = output_dict['logits']  #B,L,V
            prob = torch.nn.functional.softmax(logits, dim=-1)
            print('logits argmax ')
            print(torch.argmax(prob, dim=-1))
            print('L=0')
            print(torch.topk(prob[0,0,:], k=10))
            print(inputs['labels'][0,0], prob[0, 0, inputs['labels'][0,0]])
            print('L=1')
            print(torch.topk(prob[0,1,:], k=10))  
            print(inputs['labels'][0,1], prob[0, 1, inputs['labels'][0, 1]])
            # #loss
            loss_fct = torch.nn.CrossEntropyLoss(reduction='none')
            # #logits 1, t,v
            masked_lm_loss = loss_fct(
                logits.view(-1, self.plm.config.vocab_size), inputs['labels'].view(-1))
            # print('loss computed outside')
            # print(masked_lm_loss)
            print(torch.mean(masked_lm_loss[:-1]))
            print('loss computed in huggingface')
            print(output_dict['loss'])
            input()
        '''
        #Let's get sth crazy!
        #print(output_dict.keys()) loss, logits, past_key_values, encoder_last_hiddenstates
        return output_dict



