import pickle, os, numpy as np
import torch, torchvision
import torch.nn as nn
from torch import Tensor, dist
import torch.nn.functional as F
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
from signjoey.loss import XentLoss
from collections import defaultdict
from signjoey.helpers import sparse_sample, shift_tokens_right, ctc_decode_func

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
        sample_strategy: str='all',
        do_recognition: bool=True,
        do_translation: bool=True,
        do_distillation: bool=False,
    ):  
        super().__init__()
        self.sgn_embed = sgn_embed
        self.encoder = encoder
        self.sample_strategy = sample_strategy
        print('sample_strategy= ', self.sample_strategy)
        self.plm_cfg = plm_cfg
        self.plm_type = plm_cfg.get('type','mbart').lower()
        assert self.plm_type in ['mbart']
        #gloss2embed
        assert os.path.isfile(plm_cfg['gloss_embedding_file'])
        self.gls2embed = torch.load(plm_cfg['gloss_embedding_file'])

        if self.plm_type == 'mbart':
            if 'overwrite_mbart_cfg' in plm_cfg:
                print('Overwrite mbart cfg')
                print(plm_cfg['overwrite_mbart_cfg'])
            else:
                plm_cfg['overwrite_mbart_cfg'] = {}
            #tokenizer is still needed to encode/decode txt
            self.tokenizer = MBartTokenizer.from_pretrained(
                plm_cfg['pretrained_dir'], tgt_lang='de_DE')
            self.tokenizer.lang_code_to_id['de_DGS'] = 30
            self.tokenizer.src_lang = plm_cfg.get('src_lang', 'de_DGS')
            self.gls_lang_index = self.tokenizer.lang_code_to_id[self.tokenizer.src_lang]
            self.txt_bos_index = self.tokenizer.convert_tokens_to_ids(
                self.tokenizer.tgt_lang)

        if do_recognition:
            self.gloss_output_layer = gloss_output_layer
        if do_distillation or do_translation:
            self.preceding_layer = PrecedingLayer(
                in_features=self.encoder.output_size,
                out_features=1024, #mBart
                hidden_size=plm_cfg['preceding_layer'].get('hidden_size', 1024),
                num_layers=plm_cfg['preceding_layer'].get('num_layers', 2))        
        if do_translation:
            self.plm_model = MBartForConditionalGeneration.from_pretrained(
                plm_cfg['pretrained_dir'],
                **plm_cfg['overwrite_mbart_cfg']
                ) 
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
                print('Reinitialize src_lang_code {} {}'.format(
                    self.tokenizer.src_lang, 
                    src_lang_id, src_lang_id_in_model))
                # print('Before re-initialize')
                # print(self.tokenizer.src_lang)
                # print(self.plm_model.model.shared.weight[src_lang_id_in_model, :])
                # print('=de_DE?')
                tgt_lang_id_in_model = self.old2new[self.tokenizer.lang_code_to_id['de_DE']]
                # print(self.plm_model.model.shared.weight[tgt_lang_id_in_model, :])
                torch.nn.init.normal_(self.plm_model.model.shared.weight[src_lang_id_in_model,:])
                # print('after re-initialize')
                # print(self.plm_model.model.shared.weight[src_lang_id_in_model, :])
                #to-do here!!

            if plm_cfg.get('from_scratch',False):
                print('reinitialize plm_model to train from scratch')
                self.plm_model.init_weights()
        

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
        if do_translation:
            self.loss_level = plm_cfg.get('loss_level', 'sentence')
            self.label_smoothing = plm_cfg.get('label_smoothing', 0.2)
            self.ignore_index = self.tokenizer.pad_token_id  # ???
            self.translation_loss_fun = XentLoss(
                pad_index=self.ignore_index,  # ignore
                smoothing=self.label_smoothing)
                
        self.do_recognition = do_recognition
        self.do_translation = do_translation
        self.do_distillation = do_distillation
        self.gls_vocab = gls_vocab
        self.txt_vocab = txt_vocab
        self.gls_pad_index = gls_vocab.stoi[PAD_TOKEN]
        self.txt_pad_index = txt_vocab.stoi[PAD_TOKEN]  # ???

    def set_train(self, verbose=False):
        self.train()
    
    def set_eval(self):
        self.eval()

    def map_old2new(self, batch_input_ids):
        if self.old2new == None:
            return batch_input_ids
        new_batch_input_ids = batch_input_ids.clone()
        for bi, input_ids in enumerate(batch_input_ids):
            for ii, id_ in enumerate(input_ids):
                new_batch_input_ids[bi, ii] = self.old2new[id_.item()]
        return new_batch_input_ids

    def prepare_txt_input(self, txt, txt_lengths):
        batch_size, padded_length = txt.shape
        device = txt.device
        batch_raw_txt = []
        for i in range(batch_size):
            raw_txt = [self.txt_vocab.itos[txt[i,j]] for j in range(1,txt_lengths[i]) \
                if self.txt_vocab.itos[txt[i,j]] != EOS_TOKEN] #the first one is [BOS] (prepended by torchtext field)
            batch_raw_txt.append(' '.join(raw_txt))
        #print(batch_raw_txt)
        with self.tokenizer.as_target_tokenizer():
            labels = self.tokenizer(
                batch_raw_txt,            
                padding='longest', #we've already control the maximum length of input seq in dataloader
                return_attention_mask=True,
                return_length=True,
                return_tensors="pt")
        labels['input_ids'] = self.map_old2new(labels['input_ids'])
        decoder_input_ids, label_input_ids = shift_tokens_right(
            labels['input_ids'].clone(), 
            self.tokenizer.pad_token_id,
            ignore_index=self.ignore_index) #[lang, ]
        decoder_attention_mask = labels['attention_mask'] #attention mask keeps the same after shifting
        return label_input_ids.to(device), decoder_input_ids.to(device), decoder_attention_mask.to(device)

    def prepare_plm_inputs(self,input_embed, input_mask, txt_input=None, txt_mask=None):
        encoder_inputs = {
            'inputs_embeds': input_embed, #B,L,D
            'attention_mask': input_mask.long().squeeze(1), #B,L
        }
        if txt_input != None and txt_mask != None: #copy from PLM 
            #txt_mask (B,1,L) is not causal yet, so we can get txt_lengths from it
            txt_lengths = torch.sum(
                txt_mask, dim=-1).squeeze(1)  # B (including)
            txt_label, txt_input, txt_mask_transformer = self.prepare_txt_input(
                txt_input, txt_lengths)
            decoder_inputs = {
                'decoder_input_ids': txt_input,
                'decoder_attention_mask': txt_mask_transformer,
                'labels': txt_label
            }
        else:
            decoder_inputs = {}
        decoder_inputs = {
            'decoder_input_ids': txt_input,
            'decoder_attention_mask': txt_mask_transformer,
            'labels': txt_label
        }
        inputs = {**encoder_inputs, **decoder_inputs}
        return inputs

    def compute_distillation_loss(self, src_embeddings, src_masks, tgt_ids):
        batch_size, max_length, dim_ = src_embeddings.shape
        assert src_masks.shape==torch.Size([batch_size,1,max_length])
        seq_lengths = torch.sum(src_masks, dim=-1).view(-1) #B
        tgt_ids_length = [len(t) for t in tgt_ids]
        valid_total_length = sum(tgt_ids_length)
        assert seq_lengths.detach().cpu().numpy().tolist()==tgt_ids_length, (seq_lengths,tgt_ids_length)
        #build target first
        with torch.no_grad():
            batch_target_embeddings = torch.zeros_like(src_embeddings) #B,T,D <pad>
            for bi in range(batch_size):
                # print('sample ', bi)
                for ii,gid in enumerate(tgt_ids[bi]):
                    g_str = self.gls_vocab.itos[gid]
                    #print(gid, g_str, end=' ')
                    g_tgt_emb = self.gls2embed[g_str]
                    batch_target_embeddings[bi,ii,:] = g_tgt_emb
                # print()
                for ii in range(len(tgt_ids[bi]), max_length): #pad
                    batch_target_embeddings[bi,ii,:] = src_embeddings[bi,ii,:]
        # print(src_embeddings)
        # print(batch_target_embeddings)
        loss = self.distillation_loss_fun(input=src_embeddings, target=batch_target_embeddings) #B,T,D
        #BTD reduce=None normalize by  valid_total_length
        eps = 1.0e-10
        loss = torch.sum(loss)/valid_total_length+eps
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
            output_attention=output_attention
        )

    def forward(
        self,
        sgn: Tensor,
        sgn_mask: Tensor,
        sgn_lengths: Tensor,
        txt_input: Tensor,
        txt_mask: Tensor = None,
        output_attention: bool=False
    ) -> (Tensor, Tensor, Tensor, Tensor):
        encoder_outputs = self.encode(
            sgn=sgn, sgn_mask=sgn_mask, sgn_length=sgn_lengths, output_attention=output_attention
        )
        if len(encoder_outputs) == 3:
            encoder_output, encoder_hidden, attention = encoder_outputs
        else:
            encoder_output, encoder_hidden = encoder_outputs
            attention=None
        if self.do_recognition:
            # Gloss Recognition Part
            # N x T x C
            gloss_scores = self.gloss_output_layer(encoder_output)
            # N x T x C
            gloss_probabilities = gloss_scores.log_softmax(2)
            # Turn it into T x N x C
            gloss_probabilities = gloss_probabilities.permute(1, 0, 2)
            encoder_output, sgn_mask, batch_pred_gls = sparse_sample(
                batch_enc_op=encoder_output, 
                batch_gls_prob=gloss_probabilities.permute(1,0,2).detach(), # n,t,c
                batch_mask = sgn_mask,  #B,1,L
                select_strategy=self.sample_strategy,
                return_pred_gls=True)

        else:
            gloss_probabilities = None

        if self.do_translation or self.do_distillation:
            #intermediate layer 512->1024
            encoder_output = self.preceding_layer(encoder_output)  # B,T,D'
            #note that after transmormation padded features are not zeros!
            #print('after transform ', encoder_output.shape)

        if self.do_distillation:
            assert self.sample_strategy != 'all' and self.do_recognition
            #print('after sample ', encoder_output.shape, sgn_mask.shape)
            # from here, code differs from sign_model.forward()
            # adapted from PLM.py
            # input: encoder_output [B,T,D] sgn_mask [B,1,T] true/false (masked) padded
            distillation_loss = self.compute_distillation_loss(
                                    src_embeddings=encoder_output, 
                                    src_masks=sgn_mask,
                                    tgt_ids=batch_pred_gls)
        else:
            distillation_loss = None # we don't consider distillation when using dense feature

        if self.do_translation:
            #encoder_output -> translation loss
            inputs = self.prepare_plm_inputs(
                input_embed=encoder_output, input_mask=sgn_mask,
                txt_input=txt_input, txt_mask=txt_mask)  
            # 'input_ids', 'attention_mask', 'decoder_input_ids' 'decoder_attention_mask' 'labels'
            #print(inputs)
            output_dict = self.plm_model(
                **inputs, 
                return_dict=True,
                output_attentions=output_attention)
            assert self.loss_level == 'sentence', self.loss_level
            batch_size = output_dict['logits'].shape[0]
            log_prob = torch.nn.functional.log_softmax(
                output_dict['logits'], dim=-1)  # B, T, L
            batch_loss_sum = self.translation_loss_fun(
                log_probs=log_prob,
                targets=inputs['labels']
            )
            translation_loss = batch_loss_sum/batch_size
        else:
            translation_loss = None

        return translation_loss, gloss_probabilities, attention, encoder_output, distillation_loss


    def run_batch(
        self,
        batch,
        recognition_beam_size: int = 1,
        translation_beam_size: int = 1,
        translation_beam_alpha: float = -1,
        translation_max_output_length: int = 100,
        output_gloss_prob: bool = False
    ) -> (np.array, np.array, np.array):
        encoder_outputs = self.encode(
            sgn=batch.sgn, sgn_mask=batch.sgn_mask, sgn_length=batch.sgn_lengths
        )
        if len(encoder_outputs) == 3:
            encoder_output, encoder_hidden, attention = encoder_outputs
        else:
            encoder_output, encoder_hidden = encoder_outputs
            attention = None

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

            encoder_output, new_sgn_mask, batch_pred_gls = sparse_sample(
                batch_enc_op=encoder_output,
                batch_gls_prob=gloss_probabilities_0,  # n,t,c
                batch_mask=batch.sgn_mask,  # B,1,L
                select_strategy=self.sample_strategy,
                return_pred_gls=True)

        else:
            gloss_probabilities = None
            decoded_gloss_sequences = None

        if self.do_translation or self.do_distillation:
            encoder_output = self.preceding_layer(encoder_output)  # B,T,D'
        
        if self.do_translation:
            #to do 
            stacked_txt_output = None
            stacked_attention_scores = None
        else:
            stacked_txt_output = None
            stacked_attention_scores = None
        if output_gloss_prob:
            assert self.do_recognition
            return decoded_gloss_sequences, stacked_txt_output, stacked_attention_scores, gloss_probabilities_0.cpu().numpy()
        else:
            return decoded_gloss_sequences, stacked_txt_output, stacked_attention_scores, None

