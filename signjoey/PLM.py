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
        from_scratch: bool=False,
        loss_level: str='token'
        ):
        super().__init__()
        self.plm_type = plm_type #mBart
        self.plm = plm
        tokenizer.src_lang = src_lang 
        tokenizer.lang_code_to_id['de_DGS'] = 30
        self.tokenizer = tokenizer
        self.loss_level = loss_level
        self.gls_vocab = gls_vocab
        self.gls_pad_index = gls_vocab.stoi[PAD_TOKEN] 
        self.txt_vocab = txt_vocab #itos
        self.txt_pad_index = txt_vocab.stoi[PAD_TOKEN]
        self.txt_bos_index = tokenizer.convert_tokens_to_ids(self.tokenizer.tgt_lang)
        print('bos_index', self.tokenizer.tgt_lang,self.txt_bos_index )

        if os.path.isfile(old2new_file):
            print('Map old id to new id use ',old2new_file)
            with open(old2new_file,'rb') as f:
                old2new = pickle.load(f)
            self.old2new = defaultdict(lambda: 3, old2new) #3 unk
            self.new2old = defaultdict(lambda: 3) #
            for o,n in self.old2new.items():
                assert n>=0 and n<self.plm.config.vocab_size, (n, self.plm.config.vocab_size)
                self.new2old[n] = o

        else:
            print(old2new_file,' is not a file. Use original ids')
            self.old2new = None
        
        if from_scratch:
            print('Train from scratch, re-initialize_model')
            self.plm.init_weights()

        if freeze_embed:
            print('freeze plm embedding ...')
            freeze_params(self.plm.model.shared)
            #freeze_params(self.plm.lm_head) #already tied up
        

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
            batch_raw_gls.append(' '.join(raw_gls))
        inputs = self.tokenizer( #already contain </s> lang_code at the end
            batch_raw_gls,  
            padding='longest', #we've already control the maximum length of input seq in dataloader
            return_attention_mask=True,
            return_length=True,
            return_tensors="pt") #input_ids, length, attention_mask
        inputs['input_ids'] = self.map_old2new(inputs['input_ids'])
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
        labels['input_ids'] = self.map_old2new(labels['input_ids'])
        decoder_input_ids, label_input_ids = shift_tokens_right(labels['input_ids'].clone(), self.tokenizer.pad_token_id) #[lang, ]
        decoder_attention_mask = labels['attention_mask'] #attention mask keeps the same after shifting
        return label_input_ids.to(device), decoder_input_ids.to(device), decoder_attention_mask.to(device)

    def set_train(self, verbose=False):
        self.train()
    
    def set_eval(self):
        self.eval()
    
    def map_old2new(self, batch_input_ids):
        if self.old2new==None:
            return batch_input_ids
        new_batch_input_ids = batch_input_ids.clone()
        for bi,input_ids in enumerate(batch_input_ids):
            for ii, id_ in enumerate(input_ids):
                new_batch_input_ids[bi,ii] = self.old2new[id_.item()]
        return new_batch_input_ids

    def map_new2old(self, batch_input_ids):
        if self.old2new==None:
            return batch_input_ids
        new_batch_input_ids = batch_input_ids.clone()
        for bi,input_ids in enumerate(batch_input_ids):
            for ii, id_ in enumerate(input_ids):
                if id_.item()==30:
                    #is a special token but won't be ignored by batch_decode, so here we convert it to a special token
                    new_batch_input_ids[bi,ii] = 2 # </s>
                else:
                    new_batch_input_ids[bi,ii] = self.new2old[id_.item()]
        return new_batch_input_ids

    def prepare_inputs(self, sgn, sgn_lengths, txt_input=None, txt_mask=None):
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
        # print(inputs)
        # input()
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
        inputs = self.prepare_inputs(
            sgn=batch.gls,
            sgn_lengths=batch.gls_lengths)
        
        #mbart_debug
        if  0:#'dev/30August_2011_Tuesday_heute-783' in batch.sequence:
            print('run_batch')
            print(inputs)

        assert translation_beam_alpha>0, translation_beam_alpha
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
        # print(output_dict['sequences'])
        #print(output_dict.keys()) # sequences, encoder_attentions, decoder_attentions
        #return decoded_gloss_sequences, stacked_txt_output, stacked_attention_scores, gloss_probabilities_0.cpu().numpy()
        #print(output_dict['sequences'].shape) #Bs, Length
        #mbart_debug
        if 0:#'dev/14April_2010_Wednesday_heute-1879' in batch.sequence:
            print('predict')
            print(output_dict['sequences'])
            input()

        output_dict['sequences'] = self.map_new2old(output_dict['sequences'])
        stacked_txt_output_decoded = self.tokenizer.batch_decode(output_dict['sequences'], 
            skip_special_tokens=True)
        #!! split end common and the last word!
        #print(stacked_txt_output_decoded) #list of string
        for di, d in enumerate(stacked_txt_output_decoded):
            if len(d)>2 and d[-1]=='.' and d[-2]!=' ':
                d = d[:-1]+ ' .'
                stacked_txt_output_decoded[di] = d
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
        output_dict = self.plm(
            **inputs,
            return_dict=True,
            output_attentions=output_attention)
        if self.loss_level=='sentence':
            batch_size = output_dict['logits'].shape[0]
            loss_fct = torch.nn.CrossEntropyLoss(reduction='sum')
            masked_lm_loss = loss_fct(
                output_dict['logits'].view(-1, self.plm.config.vocab_size), 
                inputs['labels'].view(-1))
            output_dict['loss'] = masked_lm_loss/batch_size
        #logits 1, t,v
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
        
        #Let's get sth crazy!
        #print(output_dict.keys()) loss, logits, past_key_values, encoder_last_hiddenstates
        return output_dict



