from json import encoder
from signjoey import decoders
from signjoey import embeddings
from transformers import MBartForConditionalGeneration, MBartTokenizer
import torch
import torch.nn as nn
import os, numpy as np
from signjoey.vocabulary import (
    TextVocabulary,
    GlossVocabulary,
    SIL_TOKEN,
    UNK_TOKEN,
    PAD_TOKEN,
    EOS_TOKEN,
    BOS_TOKEN,
)
from signjoey.model import SignModel
from signjoey.encoders import TransformerEncoder
from signjoey.decoders import TransformerDecoder
from signjoey.embeddings import Embeddings
from signjoey.initialization import initialize_model, initialize_embed
from signjoey.helpers import freeze_params




def build_gloss2text_model(
    cfg: dict,
    gls_vocab: GlossVocabulary,
    txt_vocab: TextVocabulary,
    pretrained_dir: str=None,
    gls_embed_cfg: dict={},
    txt_embed_cfg: dict={},
    **kwargs):

    if cfg.get('type','mBART')=='mBART': 
        tokenizer = MBartTokenizer.from_pretrained(
            pretrained_dir, src_lang='de_DE', trg_lang='de_DE')
        model = MBartForConditionalGeneration.from_pretrained(pretrained_dir)
    elif cfg.get('type', 'mBART') == 'Transformer':
        # three layer transformer
        txt_padding_idx = txt_vocab.stoi[PAD_TOKEN]
        gls_padding_idx = gls_vocab.stoi[PAD_TOKEN]
        enc_emb_dropout = cfg["encoder"]["embeddings"].get("dropout", 0.1)
        encoder = TransformerEncoder(
            **cfg["encoder"],  # default pe=True, fc_type='linear', kernel_size=1
            emb_size=cfg["encoder"]["embeddings"].get("embedding_dim", 512), #unused
            emb_dropout=enc_emb_dropout,
            output_size=cfg["encoder"]["hidden_size"]
        )
        gls_embed = Embeddings(
            **cfg["encoder"]["embeddings"],
            num_heads=cfg["encoder"]["num_heads"],
            vocab_size=len(gls_vocab),
            padding_idx=gls_padding_idx,
            output_dim=cfg["encoder"]["hidden_size"]
        )#
        txt_embed = Embeddings(
            **cfg["decoder"]["embeddings"],
            num_heads=cfg["decoder"]["num_heads"],
            vocab_size=len(txt_vocab),
            padding_idx=txt_padding_idx,
            output_dim=cfg["decoder"]["hidden_size"]
        )
        dec_dropout = cfg["decoder"].get("dropout", 0.0)
        dec_emb_dropout = cfg["decoder"]["embeddings"].get(
            "dropout", dec_dropout)
        decoder = TransformerDecoder(
            **cfg["decoder"],
            encoder=encoder,
            vocab_size=len(txt_vocab),
            emb_size=txt_embed.embedding_dim,
            emb_dropout=dec_emb_dropout,
        )
        model = SignModel(
            encoder=encoder,
            gloss_output_layer=None,
            gloss_encoder=None,
            decoder=decoder,
            sgn_embed=gls_embed,
            txt_embed=txt_embed,
            gls_vocab=gls_vocab,
            txt_vocab=txt_vocab,
            do_recognition=False,
            do_translation=True,
            input_data='gloss'
        )
        if cfg.get('initialize_model',False):
            initialize_model(model, cfg, txt_padding_idx)
        else:
            print('turn off initialize')
        initialize_embed(model.sgn_embed, vocab=gls_vocab, cfg=gls_embed_cfg, verbose='gls')
        initialize_embed(model.txt_embed, vocab=txt_vocab, cfg=txt_embed_cfg, verbose='txt')
        #initialize gloss embedding and txt embedding
        if cfg.get("tied_softmax", False):
            # noinspection PyUnresolvedReferences
            if txt_embed.lut.weight.shape == model.decoder.output_layer.weight.shape:
                # (also) share txt embeddings and softmax layer:
                # noinspection PyUnresolvedReferences
                model.decoder.output_layer.weight = txt_embed.lut.weight
            else:
                raise ValueError(
                    "For tied_softmax, the decoder embedding_dim and decoder "
                    "hidden_size must be the same."
                    "The decoder must be a Transformer."
                )    
    return model



