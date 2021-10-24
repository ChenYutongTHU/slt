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
        ):
        super().__init__()
        self.plm_type = plm_type #mBart
        self.plm = plm
        tokenizer.src_lang = src_lang 
        tokenizer.lang_code_to_id['de_DGS'] = 30
        self.tokenizer = tokenizer
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
            raw_gls = [self.gls_vocab.itos[gls[i,j]] for j in range(gls_lengths[i])]
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
        output_dict['sequences'] = self.map_new2old(output_dict['sequences'])
        stacked_txt_output_decoded = self.tokenizer.batch_decode(output_dict['sequences'], 
            skip_special_tokens=True)
        
        # print(stacked_txt_output_decoded) #list of string
        # input()
        stacked_attention_scores = output_dict['decoder_attentions']
        return None, stacked_txt_output_decoded, stacked_attention_scores, None

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
        output_attention: bool=False
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
        #print(output_dict.keys()) loss, logits, past_key_values, encoder_last_hiddenstates
        return output_dict



