# coding: utf-8

from signjoey.gloss_cls import gloss_cls_head
from json import encoder
import os
from signjoey.model_3d import backbone_3D
from utils_3d import get_premodel_weight, pre_task
import tensorflow as tf

tf.config.set_visible_devices([], "GPU")

import numpy as np
import torch, torchvision
import torch.nn as nn
import torch.nn.functional as F

from itertools import groupby
from signjoey.initialization import initialize_model
from signjoey.embeddings import Embeddings, SpatialEmbeddings
from signjoey.encoders import CNNEncoder, Encoder, RecurrentEncoder, TransformerEncoder, NullEncoder
from signjoey.decoders import Decoder, RecurrentDecoder, TransformerDecoder
from signjoey.signmodel_plm import SignModel_PLM
from signjoey.search import beam_search, greedy
from signjoey.helpers import ctc_decode_func, sparse_sample
from signjoey.vocabulary import (
    TextVocabulary,
    GlossVocabulary,
    PAD_TOKEN,
    EOS_TOKEN,
    BOS_TOKEN,
)
from signjoey.batch import Batch
from signjoey.helpers import freeze_params
from torch import Tensor
from typing import Union
from signjoey.resnet import Resnet50
from signjoey.models_3d.CoCLR.utils.utils import neq_load_customized
from initialization import initialize_embed


def get_loss_for_batch(
    model,
    batch: Batch,
    recognition_loss_weight: float,
    translation_loss_weight: float,
    distillation_loss_weight: float,
    recognition_loss_function: nn.Module,
    translation_loss_function: nn.Module,
    input_data: str='feature',
    output_attention: bool=False
    ) -> (Tensor, Tensor):
    
    """
    Compute non-normalized loss and number of tokens for a batch

    :param batch: batch to compute loss for
    :param recognition_loss_function: Sign Language Recognition Loss Function (CTC)
    :param translation_loss_function: Sign Language Translation Loss Function (XEntropy)
    :param recognition_loss_weight: Weight for recognition loss
    :param translation_loss_weight: Weight for translation loss
    :param input_data: feature or images
    :return: recognition_loss: sum of losses over sequences in the batch
    :return: translation_loss: sum of losses over non-pad elements in the batch
    """
    # pylint: disable=unused-variable

    # Do a forward pass
    translation_loss, distillation_loss = None, None
    model_name = model.__class__.__name__
    if model_name == 'DistributedDataParallel':
        model_name = model.module.__class__.__name__
    if input_data=='feature':
        outputs = model(
            sgn=batch.sgn,
            sgn_mask=batch.sgn_mask,
            sgn_lengths=batch.sgn_lengths,
            txt_input=batch.txt_input,
            txt_mask=batch.txt_mask,
            output_attention=output_attention,
            batch=batch
            )
        if model_name=='SignModel':
            assert len(outputs) == 4, len(outputs)
            decoder_outputs, gloss_probabilities, attention, encoder_outputs = outputs
        elif model_name=='SignModel_PLM':
            translation_loss, gloss_probabilities, attention, encoder_outputs, distillation_loss = outputs
        else:
            raise ValueError

    elif input_data=='image':
        (decoder_outputs, gloss_probabilities, attention, encoder_outputs), batch.sgn, batch.sgn_mask, batch.sgn_lengths = model(
            sgn_img=batch.sgn_img,
            sgn_mask=batch.sgn_mask,
            sgn_lengths=batch.sgn_lengths,
            txt_input=batch.txt_input,
            txt_mask=batch.txt_mask,
            output_attention=output_attention
        )
    elif input_data=='gloss':
        if model_name == 'SignModel':
            decoder_outputs, _, attention, encoder_outputs = model(
                sgn=batch.gls,
                sgn_mask=batch.gls_mask,
                sgn_lengths=batch.gls_lengths,
                txt_input=batch.txt_input,
                txt_mask=batch.txt_mask,
                output_attention=output_attention
            )
        elif model_name.lower() == 'transformer_spm':
            #rewrite batch.txt! use new_id
            outputs, batch.txt = model(
                sgn=batch.gls,
                sgn_mask=batch.gls_mask,
                sgn_lengths=batch.gls_lengths,
                txt_input=batch.txt_input,
                txt_mask=batch.txt_mask,
                output_attention=output_attention,
                name = batch.sequence
            )
            decoder_outputs, _, attention, encoder_outputs = outputs

        elif model_name == 'huggingface_transformer':
            output_dict = model(
                sgn=batch.gls,
                sgn_mask=batch.gls_mask,
                sgn_lengths=batch.gls_lengths,
                txt_input=batch.txt_input,
                txt_mask=batch.txt_mask,
                output_attention=True,
                name=batch.sequence
            )
            translation_loss = output_dict['loss'] # default 'mean' token-level (-100 is ignored)
            encoder_outputs = output_dict.get('encoder_last_hidden_state',None) #B,L,H
            attention = output_dict.get('encoder_attentions',None) #B,H,L,L

    do_recognition = recognition_loss_function!=None
    do_translation = translation_loss_function!=None
    do_distillation = distillation_loss!=None

    if do_recognition:
            assert gloss_probabilities is not None
            # Calculate Recognition Loss
            recognition_loss = (
                recognition_loss_function(
                    gloss_probabilities,
                    batch.gls,
                    batch.sgn_lengths.long(),
                    batch.gls_lengths.long(),
                )
                * recognition_loss_weight
            )
    else:
        recognition_loss = None

    if do_translation:
        if translation_loss==None:
            assert decoder_outputs is not None
            word_outputs, _, _, _ = decoder_outputs
            # Calculate Translation Loss
            txt_log_probs = F.log_softmax(word_outputs, dim=-1)
            translation_loss = (
                translation_loss_function(txt_log_probs, batch.txt)
                * translation_loss_weight
            )
        else:
            translation_loss = translation_loss*translation_loss_weight
    else:
        translation_loss = None

    if do_distillation:
        assert distillation_loss_weight>0
        distillation_loss = distillation_loss*distillation_loss_weight
    else:
        assert distillation_loss_weight in [0,None], distillation_loss_weight
        distillation_loss = None

    return recognition_loss, translation_loss, distillation_loss, attention, encoder_outputs


class CNN(torch.nn.Module):
    def __init__(self, pretrained_ckpt, use_layer=4, freeze_layer=0):
        super().__init__()
        self.use_layer = int(use_layer)
        self.freeze_layer = int(freeze_layer)
        assert self.freeze_layer<=self.use_layer, (self.freeze_layer, self.use_layer)
        self.resnet = Resnet50(use_layer=use_layer)
        if os.path.isfile(pretrained_ckpt):
            print('CNN trained from {}'.format(pretrained_ckpt))
            try:
                self.resnet.load_state_dict(
                    torch.load(pretrained_ckpt), strict=False)
            except:
                neq_load_customized(model=self.resnet, 
                    pretrained_dict=torch.load(pretrained_ckpt), verbose=True)
        else:
            print('CNN from scratch, pretrained_ckpt {} is not a file'.format(
                pretrained_ckpt))
        
        self.frozen_modules = []
        if self.freeze_layer>0:
            self.frozen_modules = [self.resnet.conv1, self.resnet.bn1, self.resnet.relu, self.resnet.maxpool]
            for li in range(1, self.freeze_layer+1):
                self.frozen_modules.append(getattr(self.resnet, 'layer{}'.format(li)))
            
        self.set_frozen_layers()

    def set_train(self):
        self.train()
        for m in self.frozen_modules:
            m.eval()

    def set_frozen_layers(self):
        for m in self.frozen_modules:
            for param in m.parameters():
                #print(param)
                param.requires_grad = False
            m.eval()

    def forward(self, x):
        x = self.resnet(x)
        return x.squeeze(dim=-1).squeeze(dim=-1)


class SignModel(nn.Module):
    """
    Base Model class
    """

    def __init__(
        self,
        encoder: Encoder,
        gloss_output_layer: nn.Module,
        gloss_encoder: nn.Module,
        decoder: Decoder,
        sgn_embed: SpatialEmbeddings,
        txt_embed: Embeddings,
        gls_vocab: GlossVocabulary,
        txt_vocab: TextVocabulary,
        do_recognition: bool = True,
        do_translation: bool = True,
        sample_strategy: str='all',
        input_data: str='feature'
    ):
        """
        Create a new encoder-decoder model

        :param encoder: encoder
        :param gloss_encoder: gloss_encoder
        :param decoder: decoder
        :param sgn_embed: spatial feature frame embeddings
        :param txt_embed: spoken language word embedding
        :param gls_vocab: gls vocabulary
        :param txt_vocab: spoken language vocabulary
        :param do_recognition: flag to build the model with recognition output.
        :param do_translation: flag to build the model with translation decoder.
        """
        super().__init__()

        self.encoder = encoder
        self.sample_strategy = sample_strategy
        print('sample_strategy= ', self.sample_strategy)
        self.gloss_encoder = gloss_encoder
        self.decoder = decoder

        self.sgn_embed = sgn_embed
        self.txt_embed = txt_embed

        self.gls_vocab = gls_vocab
        self.txt_vocab = txt_vocab

        self.txt_bos_index = self.txt_vocab.stoi[BOS_TOKEN]
        self.txt_pad_index = self.txt_vocab.stoi[PAD_TOKEN]
        self.txt_eos_index = self.txt_vocab.stoi[EOS_TOKEN]

        self.gls_pad_index = self.gls_vocab.stoi[PAD_TOKEN]

        self.gloss_output_layer = gloss_output_layer
        self.do_recognition = do_recognition
        self.do_translation = do_translation
        self.input_data = input_data
    
    def set_train(self, verbose=False):
        self.train()
    
    def set_eval(self):
        self.eval()

    # pylint: disable=arguments-differ
    def forward(
        self,
        sgn: Tensor,
        sgn_mask: Tensor,
        sgn_lengths: Tensor,
        txt_input: Tensor,
        txt_mask: Tensor = None,
        output_attention: bool=False,
        **kwargs
    ) -> (Tensor, Tensor, Tensor, Tensor):
        """
        First encodes the source sentence.
        Then produces the target one word at a time.

        :param sgn: source input
        :param sgn_mask: source mask
        :param sgn_lengths: length of source inputs
        :param txt_input: target input
        :param txt_mask: target mask
        :return: decoder outputs
        """
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
        else:
            gloss_probabilities = None

        if self.do_translation:
            if self.do_recognition:
                #print('before sample ', encoder_output.shape, sgn_mask.shape, end=' --> ')
                encoder_output, sgn_mask = sparse_sample(
                    batch_enc_op=encoder_output, 
                    batch_gls_prob=gloss_probabilities.permute(1,0,2).detach(), # n,t,c
                    batch_mask = sgn_mask,  #B,1,L
                    select_strategy=self.sample_strategy)
                #print('after sample ', encoder_output.shape, sgn_mask.shape)
            if self.gloss_encoder:
                encoder_output, encoder_hidden = self.gloss_encoder(
                    embed_src = encoder_output,
                    src_length = sgn_lengths,
                    output_attention = False,
                    mask = sgn_mask
                )
            unroll_steps = txt_input.size(1)
            decoder_outputs = self.decode(
                encoder_output=encoder_output,
                encoder_hidden=encoder_hidden,
                sgn_mask=sgn_mask,
                txt_input=txt_input,
                unroll_steps=unroll_steps,
                txt_mask=txt_mask,
            )
        else:
            decoder_outputs = None

        return decoder_outputs, gloss_probabilities, attention, encoder_output

    def encode(
        self, sgn: Tensor, sgn_mask: Tensor, sgn_length: Tensor, output_attention: bool=False
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

    def decode(
        self,
        encoder_output: Tensor,
        encoder_hidden: Tensor,
        sgn_mask: Tensor,
        txt_input: Tensor,
        unroll_steps: int,
        decoder_hidden: Tensor = None,
        txt_mask: Tensor = None,
    ) -> (Tensor, Tensor, Tensor, Tensor):
        """
        Decode, given an encoded source sentence.

        :param encoder_output: encoder states for attention computation
        :param encoder_hidden: last encoder state for decoder initialization
        :param sgn_mask: sign sequence mask, 1 at valid tokens
        :param txt_input: spoken language sentence inputs
        :param unroll_steps: number of steps to unroll the decoder for
        :param decoder_hidden: decoder hidden state (optional)
        :param txt_mask: mask for spoken language words
        :return: decoder outputs (outputs, hidden, att_probs, att_vectors)
        """
        return self.decoder(
            encoder_output=encoder_output,
            encoder_hidden=encoder_hidden,
            src_mask=sgn_mask,
            trg_embed=self.txt_embed(x=txt_input, mask=txt_mask),
            trg_mask=txt_mask,
            unroll_steps=unroll_steps,
            hidden=decoder_hidden,
        )
                
    def run_batch(
        self,
        batch: Batch,
        recognition_beam_size: int = 1,
        translation_beam_size: int = 1,
        translation_beam_alpha: float = -1,
        translation_max_output_length: int = 100,
        output_gloss_prob: bool = False
    ) -> (np.array, np.array, np.array):
        """
        Get outputs and attentions scores for a given batch

        :param batch: batch to generate hypotheses for
        :param recognition_beam_size: size of the beam for CTC beam search
            if 1 use greedy
        :param translation_beam_size: size of the beam for translation beam search
            if 1 use greedy
        :param translation_beam_alpha: alpha value for beam search
        :param translation_max_output_length: maximum length of translation hypotheses
        :return: stacked_output: hypotheses for batch,
            stacked_attention_scores: attention scores for batch
        """
        if self.input_data=='gloss':
            encoder_outputs = self.encode(
                sgn=batch.gls, sgn_mask=batch.gls_mask, sgn_length=batch.gls_lengths
            )
        else:
            encoder_outputs = self.encode(
                sgn=batch.sgn, sgn_mask=batch.sgn_mask, sgn_length=batch.sgn_lengths
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
            gloss_probabilities_0 = gloss_scores.log_softmax(2)
            # Turn it into T x N x C
            gloss_probabilities = gloss_probabilities_0.permute(1, 0, 2) #
            gloss_probabilities = gloss_probabilities.cpu().detach().numpy()
            tf_gloss_probabilities = np.concatenate(
                (gloss_probabilities[:, :, 1:], gloss_probabilities[:, :, 0, None]),
                axis=-1,
            )
            if type(recognition_beam_size)!=list:
                decoded_gloss_sequences = ctc_decode_func(tf_gloss_probabilities, batch, recognition_beam_size, gloss_scores)
            else:
                decoded_gloss_sequences = {}
                for rbs in recognition_beam_size:
                    decoded_gloss_sequences[rbs] = ctc_decode_func(tf_gloss_probabilities, batch, rbs, gloss_scores)

        else:
            decoded_gloss_sequences = None

        if self.do_translation and translation_beam_size:
            if self.do_recognition:
                #print('before sample ', encoder_output.shape, sgn_mask.shape, end=' --> ')
                encoder_output, new_sgn_mask = sparse_sample(
                    batch_enc_op=encoder_output,
                    batch_gls_prob=gloss_probabilities_0,  # n,t,c
                    batch_mask=batch.sgn_mask,  # B,1,L
                    select_strategy='top1_maxprob' if self.sample_strategy == 'top1_random' else self.sample_strategy)
                #print('after sample ', encoder_output.shape, sgn_mask.shape)
            else:
                if self.input_data=='gloss':
                    new_sgn_mask = batch.gls_mask
                else:
                    new_sgn_mask = batch.sgn_mask
            if self.gloss_encoder:
                encoder_output, encoder_hidden = self.gloss_encoder(
                    embed_src=encoder_output,
                    src_length=None, #unused
                    output_attention=False,
                    mask=new_sgn_mask
                )

            # greedy decoding
            if translation_beam_size < 2:
                stacked_txt_output, stacked_attention_scores = greedy(
                    encoder_hidden=encoder_hidden,
                    encoder_output=encoder_output,
                    src_mask=new_sgn_mask, #batch.sgn_mask,
                    embed=self.txt_embed,
                    bos_index=self.txt_bos_index,
                    eos_index=self.txt_eos_index,
                    decoder=self.decoder,
                    max_output_length=translation_max_output_length,
                )
                # batch, time, max_sgn_length
            else:  # beam size
                stacked_txt_output, stacked_attention_scores = beam_search(
                    size=translation_beam_size,
                    encoder_hidden=encoder_hidden,
                    encoder_output=encoder_output,
                    src_mask=new_sgn_mask,
                    embed=self.txt_embed,
                    max_output_length=translation_max_output_length,
                    alpha=translation_beam_alpha,
                    eos_index=self.txt_eos_index,
                    pad_index=self.txt_pad_index,
                    bos_index=self.txt_bos_index,
                    decoder=self.decoder,
                )
        else:
            stacked_txt_output = stacked_attention_scores = None
        if output_gloss_prob:
            assert self.do_recognition
            return decoded_gloss_sequences, stacked_txt_output, stacked_attention_scores, gloss_probabilities_0.cpu().numpy()
        else:
            return decoded_gloss_sequences, stacked_txt_output, stacked_attention_scores, None

    def __repr__(self) -> str:
        """
        String representation: a description of encoder, decoder and embeddings

        :return: string representation
        """
        return (
            "%s(\n"
            "\tencoder=%s,\n"
            "\tdecoder=%s,\n"
            "\tsgn_embed=%s,\n"
            "\ttxt_embed=%s)"
            % (
                self.__class__.__name__,
                self.encoder,
                self.decoder,
                self.sgn_embed,
                self.txt_embed,
            )
        )


class Tokenizer_SignModel(nn.Module):
    def __init__(self, tokenizer_type, tokenizer, signmodel, 
        track_bn=True, bn_train_mode='train'):
        super().__init__()
        self.tokenizer_type = tokenizer_type
        self.tokenizer = tokenizer
        self.signmodel = signmodel  # already initialized
        self.txt_pad_index = self.signmodel.txt_pad_index
        self.txt_bos_index = self.signmodel.txt_bos_index
        self.txt_eos_index = self.signmodel.txt_eos_index
        self.gls_pad_index = self.signmodel.gls_pad_index
        self.gls_vocab = self.signmodel.gls_vocab
        self.txt_vocab = self.signmodel.txt_vocab
        self.do_recognition = self.signmodel.do_recognition
        self.do_translation = self.signmodel.do_translation
        self.track_bn = track_bn
        assert self.track_bn, 'No longer support track_bn=False now'
        self.bn_train_mode = bn_train_mode
        
        def set_track_running_stats(m):
            classname = m.__class__.__name__
            if classname.find('BatchNorm') != -1:
                m.track_running_stats = False
                m.register_buffer("running_mean", None)
                m.register_buffer("running_var", None)
                m.register_buffer("num_batches_tracked", None)
                # if int(os.environ['LOCAL_RANK'])==0:
                #     print('Set bn module {} to tracking running stats = {}'.format(
                #         m,  m.track_running_stats))

        if self.track_bn==False:
            print('Set batchnorm in Toeknizer track_running_stats to False')
            self.tokenizer.apply(set_track_running_stats)

    def set_bn_eval(self, verbose=False):
        def _set_bn_eval_(m):
            classname = m.__class__.__name__
            if classname.find('BatchNorm') != -1:
                m.eval()
        if self.bn_train_mode=='eval':
            if verbose:
                print('Set batchnorm in tokenizer  to eval mode')
            self.tokenizer.apply(_set_bn_eval_)
        elif verbose:
            print('Set batchnorm in tokenizer to Train mode')

    def set_train(self, verbose=False):
        self.tokenizer.set_train()
        self.signmodel.train()
        self.set_bn_eval(verbose)

    def set_eval(self):
        self.eval()

    def visual_tokenize(
        self,
        sgn_img: Tensor,  # B,C,H,W
        sgn_mask: Tensor,
        sgn_lengths: Tensor,
    ) -> (Tensor):
        sgn_feature = self.tokenizer(sgn_img, sgn_lengths)
        if self.tokenizer_type == 'cnn':
            #split and pad#
            assert torch.sum(
                sgn_lengths) == sgn_feature.shape[0], (sgn_feature.shape, torch.sum(sgn_lengths))
            sgn_seqs = torch.split(sgn_feature, sgn_lengths.tolist(), dim=0)
            sgn = []
            pad_length = sgn_mask.shape[-1]  # L
            for seq in sgn_seqs:
                #seq L,D
                if seq.shape[0] >= pad_length:
                    padded_seq = seq[:pad_length, :]  # pl,d
                else:
                    padding_len = pad_length-seq.shape[0]
                    paddings = torch.zeros(size=[padding_len, seq.shape[1]],
                                        dtype=seq.dtype, device=seq.device)  # L,d
                    padded_seq = torch.cat([seq, paddings], dim=0)  # pl,d
                sgn.append(padded_seq)
            sgn = torch.stack(sgn, dim=0)
            assert sgn.shape[0] == sgn_mask.shape[0], (
                sgn.shape, sgn_mask.shape)
            assert sgn.shape[1] == sgn_mask.shape[2], (sgn.shape, sgn_mask.shape)
            return sgn, sgn_mask, sgn_lengths
        elif self.tokenizer_type in ['s3d','s3ds','i3d','bntin']:
            #Spatial average pooling and MASKING
            B, _, T_in, _, _ = sgn_img.shape
            if self.tokenizer_type=='bntin':
                B, _, T_out = sgn_feature.shape
                pooled_sgn_feature = sgn_feature #already pooled within bntin B, D, T
            else:
                B, _, T_out, _, _ = sgn_feature.shape
                pooled_sgn_feature = torch.mean(sgn_feature, dim=[3,4]) #B, D, T_out
            sgn = torch.transpose(pooled_sgn_feature, 1, 2) #b, t_OUT, d
            sgn_mask = torch.zeros([B,1,T_out], dtype=torch.bool, device=sgn.device)
            valid_len_out = torch.floor(sgn_lengths*T_out/T_in).long() #B,
            for bi in range(B):
                sgn_mask[bi, :, :valid_len_out[bi]] = True

            # #debug
            # print('input shape ', sgn_img.shape)
            # print('output shape ', sgn_feature.shape)
            # print('valid_len_in ', sgn_lengths)
            # print('valid_len_out ', valid_len_out)
            # print('sgn_mask ',sgn_mask)
            return sgn, sgn_mask, valid_len_out
        else:
            raise NotImplementedError

    def forward(
        self,
        sgn_img: Tensor,  # B,C,H,W
        sgn_mask: Tensor,
        sgn_lengths: Tensor,
        **kwargs,
    ) -> (Tensor, Tensor, Tensor, Tensor):

        if self.tokenizer_type in ['s3d','s3ds','s3dt','i3d']:
            assert sgn_img.dim() == 5, sgn_img.shape #B,C,T,H,W
            assert sgn_mask==None
        sgn, sgn_mask, sgn_lengths = self.visual_tokenize(
            sgn_img=sgn_img,
            sgn_mask=sgn_mask,
            sgn_lengths=sgn_lengths)

        outputs = self.signmodel(
            sgn=sgn,
            sgn_mask=sgn_mask,
            sgn_lengths=sgn_lengths,
            **kwargs,
        )
        return outputs, sgn, sgn_mask, sgn_lengths



    def run_batch(
        self,
        batch: Batch,
        **kwargs,
    ) -> (np.array, np.array, np.array):
        # assert batch.sgn == None
        # batch.sgn, batch.sgn_mask, batch.sgn_lengths = self.visual_tokenize(
        #     sgn_img=batch.sgn_img,
        #     sgn_mask=batch.sgn_mask,
        #     sgn_lengths=batch.sgn_lengths)
        assert batch.sgn!=None, 'Please call forward before run_batch'
        outputs = self.signmodel.run_batch(
            batch=batch,
            **kwargs
        )
        return outputs

def build_model(
    cfg: dict,
    sgn_dim: int,
    gls_vocab: GlossVocabulary,
    txt_vocab: TextVocabulary,
    do_recognition: bool = True,
    do_translation: bool = True,
    do_distillation: bool = False,
    input_data: str='feature',
) -> SignModel:
    """
    Build and initialize the model according to the configuration.

    :param cfg: dictionary configuration containing model specifications
    :param sgn_dim: feature dimension of the sign frame representation, i.e. 2560 for EfficientNet-7.
    :param gls_vocab: sign gloss vocabulary
    :param txt_vocab: spoken language word vocabulary
    :return: built and initialized model
    :param do_recognition: flag to build the model with recognition output.
    :param do_translation: flag to build the model with translation decoder.
    :param input_data: feature or image.
    """

    txt_padding_idx = txt_vocab.stoi[PAD_TOKEN]

    sgn_embed: SpatialEmbeddings = SpatialEmbeddings(
        **cfg["encoder"]["embeddings"],
        num_heads=cfg["encoder"].get('num_heads', 8),  #used for groupnorm
        input_size=sgn_dim,
    )

    # build encoder
    enc_dropout = cfg["encoder"].get("dropout", 0.0)
    enc_emb_dropout = cfg["encoder"]["embeddings"].get("dropout", 0.1)
    if 'gls_embed' in cfg["encoder"]["embeddings"]:
        assert cfg["encoder"].get("type", "recurrent") == "transformer"
        encoder_output_size = cfg["encoder"]["embeddings"]['gls_embed']["embedding_dim"]
    else:
        encoder_output_size = cfg["encoder"].get("hidden_size", 512)
    if cfg["encoder"].get("type", "recurrent") == "transformer":
        assert (
            cfg["encoder"]["embeddings"]["embedding_dim"]
            == cfg["encoder"]["hidden_size"]
        ), "for transformer, emb_size must be hidden_size"
        encoder = TransformerEncoder(
            **cfg["encoder"], #default pe=True, fc_type='linear', kernel_size=1
            emb_size=sgn_embed.embedding_dim,
            emb_dropout=enc_emb_dropout,
            output_size=encoder_output_size
        )
    elif cfg["encoder"].get("type", "recurrent") == "empty":
        encoder = NullEncoder( 
            emb_size=sgn_embed.embedding_dim, #default pe=Fals
            pe=cfg["encoder"].get("pe",False),
        )
    elif cfg["encoder"].get("type", "recurrent") == 'cnn':
        encoder = CNNEncoder(
            emb_size=sgn_embed.embedding_dim,
            pe=cfg["encoder"].get("pe",False),
            hidden_size=cfg["encoder"].get("hidden_size", 512),
            num_layers=cfg["encoder"].get("num_layers", 1),
            masking_before=cfg["encoder"].get("masking_before",'zero'),
            LN=cfg["encoder"].get("LN", False),
            **cfg["encoder"]["cnn"] #dropout
        )
    else:
        encoder = RecurrentEncoder(
            **cfg["encoder"],
            emb_size=sgn_embed.embedding_dim,
            emb_dropout=enc_emb_dropout,
        )

    if do_recognition:
        bias = True
        freeze_special, freeze_normal = False, False
        init_normal_file = None
        if 'gls_embed' in cfg['encoder'].get('embeddings', {}):
            bias = cfg['encoder']['embeddings']['gls_embed'].get('bias',True)
            if bias==False:
                print('Turn off bias')
            freeze_mode = cfg['encoder']['embeddings']['gls_embed'].get(
                "freeze_mode", "all_tune")

            print('freeze_mode = ', freeze_mode)
            if freeze_mode=='all_freeze':
                raise ValueError #we should freeze special (they are trained from scratch)
                freeze_special, freeze_normal = True, True
            elif freeze_mode=='special_tune':
                freeze_special, freeze_normal = False, True
            
            init_normal_file = cfg['encoder']['embeddings']['gls_embed'].get('init_file',None)

        gloss_output_layer_version = cfg.get('gloss_output_layer_version',1)
        print('gloss output layer version = ', gloss_output_layer_version)
        if gloss_output_layer_version==1:
            gloss_output_layer = nn.Linear(encoder.output_size, 
                len(gls_vocab), bias=bias
                )
        else:
            gloss_output_layer = gloss_cls_head(
                in_features=encoder.output_size,
                special_vocab_size= len(gls_vocab.specials),
                vocab_size = len(gls_vocab),
                bias = bias,
                gls_vocab = gls_vocab,
                freeze_normal=freeze_normal, freeze_special=freeze_special,
                init_normal_file=init_normal_file
            )

    else:
        gloss_output_layer = None

    signmodel_type = cfg.get('signmodel_type','vanilla') #standard, signmodel_plm
    assert signmodel_type in ['signmodel_plm', 'vanilla']


    if signmodel_type=='vanilla':
        # build decoder and word embeddings
        if do_translation:
            txt_embed: Union[Embeddings, None] = Embeddings(
                **cfg["decoder"]["embeddings"],
                num_heads=cfg["decoder"]["num_heads"],
                vocab_size=len(txt_vocab),# 
                padding_idx=txt_padding_idx,
                output_dim=cfg["decoder"]["hidden_size"]
            )
            dec_dropout = cfg["decoder"].get("dropout", 0.0)
            dec_emb_dropout = cfg["decoder"]["embeddings"].get("dropout", dec_dropout)
            if "gloss_encoder" in cfg:
                if cfg["gloss_encoder"].get("type", "transformer"):
                    gloss_encoder = TransformerEncoder(
                        **cfg["gloss_encoder"],  # default pe=True, fc_type='linear', kernel_size=1
                    )
            else:
                gloss_encoder = None
            if cfg["decoder"].get("type", "recurrent") == "transformer":
                decoder = TransformerDecoder(
                    **cfg["decoder"],
                    encoder=encoder,
                    vocab_size=len(txt_vocab),
                    emb_size=txt_embed.embedding_dim,
                    emb_dropout=dec_emb_dropout,
                    tied_softmax=cfg.get("tied_softmax", False),
                    output_layer_size=txt_embed.embedding_dim if cfg.get("tied_softmax", False) else -1
                )
            else:
                decoder = RecurrentDecoder(
                    **cfg["decoder"],
                    encoder=encoder,
                    vocab_size=len(txt_vocab),
                    emb_size=txt_embed.embedding_dim,
                    emb_dropout=dec_emb_dropout,
                )
        else:
            txt_embed = None
            decoder = None
            gloss_encoder = None
        sign_model: SignModel = SignModel(
            encoder=encoder,
            gloss_output_layer=gloss_output_layer,
            sample_strategy=cfg.get('sample_strategy','all'),
            gloss_encoder = gloss_encoder,
            decoder=decoder,
            sgn_embed=sgn_embed,
            txt_embed=txt_embed,
            gls_vocab=gls_vocab,
            txt_vocab=txt_vocab,
            do_recognition=do_recognition,
            do_translation=do_translation,
        )
        # custom initialization of sign_model parameters
        if cfg.get("initialize_model", True)==True:
            initialize_model(sign_model, cfg, txt_padding_idx)
        else:
            print('Turn off initialize')    # custom initialization of sign_model parameters
    else:
        #signmodel_plm
        assert 'plm' in cfg
        sign_model: SignModel_PLM = SignModel_PLM(
            encoder=encoder, #here we refer to sign->gloss
            gloss_output_layer=gloss_output_layer,
            sample_strategy=cfg.get('sample_strategy','all'),
            plm_cfg=cfg['plm'],
            sgn_embed = sgn_embed,
            txt_vocab = txt_vocab,
            gls_vocab = gls_vocab, #gls_vocab is needed to convert CTC prediction back to gls str
            do_recognition=do_recognition,
            do_translation=do_translation, 
            do_distillation=do_distillation       
        )
        # custom initialization of sign_model parameters
        if cfg.get("initialize_model", True)==True:
            initialize_model(sign_model.encoder, cfg)
            initialize_model(sign_model.sgn_embed, cfg)
            initialize_model(sign_model.gloss_output_layer, cfg)
        else:
            print('Turn off initialize')    # custom initialization of sign_model parameters
    
    if do_recognition and gloss_output_layer_version==2:
        sign_model.gloss_output_layer.initialize_weights()

    if do_translation:
        if signmodel_type == 'vanilla':
            if 'txt_embed' in cfg['decoder']['embeddings']:
                initialize_embed(sign_model.txt_embed, vocab=txt_vocab,
                                cfg=cfg['decoder']['embeddings']['txt_embed'], verbose='txt')
            # tie softmax layer with txt embeddings
            if cfg.get("tied_softmax", False):
                print('Tied softmax')
                # noinspection PyUnresolvedReferences
                if txt_embed.lut.weight.shape == sign_model.decoder.output_layer.weight.shape:
                    # (also) share txt embeddings and softmax layer:
                    # noinspection PyUnresolvedReferences
                    sign_model.decoder.output_layer.weight = txt_embed.lut.weight
                else:
                    raise ValueError(
                        "For tied_softmax, the decoder embedding_dim and decoder "
                        "hidden_size must be the same."
                        "The decoder must be a Transformer."
                    )


    if input_data == 'feature':
        return sign_model
    else:
        if cfg["tokenizer"]["architecture"] == 'cnn':
            tokenizer = CNN(pretrained_ckpt=cfg["cnn"].get('pretrained_ckpt', None), 
                        use_layer=cfg["cnn"].get('use_layer',4),
                        freeze_layer=cfg["cnn"].get("freeze_layer",0))


        elif cfg["tokenizer"]["architecture"] in ['s3d','s3ds','i3d','bntin']:
            tokenizer = backbone_3D(
                    network=cfg["tokenizer"]["architecture"], 
                    ckpt_dir=cfg["tokenizer"]["pretrained_ckpt"],
                    use_block=cfg["tokenizer"].get('use_block', 5),
                    freeze_block=cfg['tokenizer'].get('freeze_block', 0),
                    stride=cfg['tokenizer'].get('block5_stride', 2))
            
            network = cfg["tokenizer"]["architecture"]
            if cfg["tokenizer"].get('pretask','default')=='default':
                pretask = pre_task[network]
            elif cfg["tokenizer"].get('pretask', 'default') == 'scratch':
                print('Train 3D backbone from scratch ...')
                pass 
            else:
                pretask = cfg["tokenizer"].get('pretask')
                ckpt_filename = os.path.join(
                    cfg["tokenizer"]["pretrained_ckpt"], 
                    '%s_%s_ckpt' % (network, pretask))
                success = get_premodel_weight(
                    network=network,
                    pretask=pretask,
                    model_without_dp=tokenizer, 
                    model_path=ckpt_filename)
                if success:
                    print('Load model {} from {} ... success {}'.format(
                        network, ckpt_filename, success))
                else:
                    raise NotImplementedError
            
        else:
            
            raise ValueError

        tokenizer_signmodel = Tokenizer_SignModel(
            tokenizer_type=cfg["tokenizer"]["architecture"],
            tokenizer=tokenizer,
            signmodel=sign_model,
            track_bn=cfg.get("track_bn", True),
            bn_train_mode=cfg.get("bn_train_mode", 'train'))
        return tokenizer_signmodel
