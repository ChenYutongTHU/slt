from collections import defaultdict
from json import encoder
from signjoey import decoders
from signjoey import embeddings
from transformers import MBartForConditionalGeneration, MBartTokenizer, MBartConfig
from transformers import GPT2Tokenizer, GPT2LMHeadModel, AutoTokenizer
import torch, pickle
import torch.nn as nn
import os, numpy as np
from signjoey.vocabulary import (
    TextVocabulary,
    GlossVocabulary,
    Vocabulary,
    SIL_TOKEN,
    UNK_TOKEN,
    PAD_TOKEN,
    EOS_TOKEN,
    BOS_TOKEN,
)
from signjoey.model import SignModel
from signjoey.PLM import huggingface_transformer, transformer_spm
from signjoey.encoders import TransformerEncoder, NullEncoder
from signjoey.decoders import TransformerDecoder
from signjoey.embeddings import Embeddings
from signjoey.initialization import initialize_model, initialize_embed
from signjoey.helpers import freeze_params


def rebuild_vocabulary(id2str, vocab_size):
    new_vocab = Vocabulary()
    new_vocab.itos = []
    new_vocab.stoi = {}
    for i in range(vocab_size):
        new_vocab.itos.append(id2str[i])
        new_vocab.stoi[id2str[i]] = i
    return new_vocab

def build_gloss2text_model(
    cfg: dict,
    gls_vocab: GlossVocabulary,
    txt_vocab: TextVocabulary,
    gls_embed_cfg: dict={},
    txt_embed_cfg: dict={},
    **kwargs):

    if cfg.get('type','mBart')=='mBart': 
        tokenizer = MBartTokenizer.from_pretrained(
            cfg['pretrained_dir'], tgt_lang='de_DE')
        if 'overwrite_mbart_cfg' in cfg:
            print('Overwrite mbart cfg')
            print(cfg['overwrite_mbart_cfg'])
        else:
            cfg['overwrite_mbart_cfg'] = {}
        plm_model = MBartForConditionalGeneration.from_pretrained(
            cfg['pretrained_dir'],
            **cfg['overwrite_mbart_cfg']
            )
        tokenizer.lang_code_to_id['de_DGS'] = 30
        model = huggingface_transformer(
            plm_type='mBart', 
            plm=plm_model, 
            tokenizer=tokenizer, 
            gls_vocab=gls_vocab, txt_vocab=txt_vocab,
            old2new_file=os.path.join(cfg['pretrained_dir'], 'old2new_vocab.pkl'),
            **cfg.get('mbart_config',{})) #old2new_file, freeze_embed, src_lang
    elif cfg.get('type', 'mBart') == 'gpt2':
        print(cfg['pretrained_dir'])
        tokenizer = AutoTokenizer.from_pretrained(cfg['pretrained_dir'])
        plm_model = GPT2LMHeadModel.from_pretrained(cfg['pretrained_dir'])
        model = huggingface_transformer(
            plm_type='gpt2',
            plm=plm_model,
            tokenizer=tokenizer,
            gls_vocab=gls_vocab, txt_vocab=txt_vocab,
            old2new_file=os.path.join(cfg['pretrained_dir'], 'old2new_vocab.pkl'),
            **cfg.get('gpt2_config',{})) 
    elif cfg.get('type', 'mBART') in ['Transformer', 'Transformer_spm']:
        if cfg.get('type', 'mBART') == 'Transformer_spm':
            #!! re-organize txt_vocab and gls_vocab according to tokenizer.json
            vocab_dir = cfg['tokenizer']['vocab_dir']
            tokenizer = MBartTokenizer.from_pretrained(
                vocab_dir,
                tgt_lang='de_DE')
            tokenizer.lang_code_to_id['de_DGS'] = 30
            tokenizer.src_lang = 'de_DGS'
            old2new_file = os.path.join(vocab_dir, 'old2new_vocab.pkl')
            with open(old2new_file, 'rb') as f:
                old2new = pickle.load(f)
            vocab_size = 0
            id2str = defaultdict(lambda: '?')
            for o,n in old2new.items():
                vocab_size = max(vocab_size, n+1)
                id2str[n] = tokenizer.convert_ids_to_tokens(o)
            print('Rebuild txt_vocab gls_vocab size=', vocab_size)
            print('')
            old_txt_vocab, old_gls_vocab = txt_vocab, gls_vocab
            txt_vocab = rebuild_vocabulary(id2str, vocab_size)
            gls_vocab = rebuild_vocabulary(id2str, vocab_size)


        # transformer
        txt_padding_idx = txt_vocab.stoi[PAD_TOKEN]
        gls_padding_idx = gls_vocab.stoi[PAD_TOKEN]
        enc_emb_dropout = cfg["encoder"]["embeddings"].get("dropout", 0.1)
        if cfg["encoder"].get("type",'transformer')=='empty':
            encoder = NullEncoder(
                emb_size=cfg["encoder"]["embeddings"].get(
                    "embedding_dim", 512),  # default pe=Fals
                pe=cfg["encoder"].get("pe", False),
            )
        else:
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
        
        sgn_model = SignModel(
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
        if cfg.get('type', 'mBART') == 'Transformer':
            model = sgn_model
            initialize_embed(model.sgn_embed, vocab=gls_vocab,
                            cfg=gls_embed_cfg, verbose='gls')
            initialize_embed(model.txt_embed, vocab=txt_vocab,
                            cfg=txt_embed_cfg, verbose='txt')
            #only support

        else:#'Transformer spm (use mBart-like spm)
            model = transformer_spm(
                model = sgn_model,
                tokenizer = tokenizer,
                old2new = old2new,
                old_txt_vocab=old_txt_vocab,
                old_gls_vocab=old_gls_vocab
                )
        if cfg.get('initialize_model',False):
            initialize_model(model, cfg, txt_padding_idx)
        else:
            print('turn off initialize')
        

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



