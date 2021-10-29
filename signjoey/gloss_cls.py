import os
import torch
import torch.nn as nn
from torch import Tensor
from signjoey.helpers import freeze_params

class gloss_cls_head(torch.nn.Module):
    def __init__(self, 
                in_features, 
                special_vocab_size, vocab_size, 
                gls_vocab,
                bias,
                freeze_normal=False, freeze_special=False,
                init_normal_file=None):
        super().__init__()
        self.in_features = in_features
        self.special_vocab_size = special_vocab_size
        self.vocab_size = vocab_size
        self.bias = bias
        self.special_head = nn.Linear(
            self.in_features, special_vocab_size, bias=bias)
        self.normal_head = nn.Linear(
            self.in_features, vocab_size-special_vocab_size, bias=bias)
        self.gls_vocab = gls_vocab
        self.init_normal_file = init_normal_file
        if freeze_special:
            raise ValueError
            #we only freeze weight here
            # freeze_params(self.special_head)
        if freeze_normal:
            #we only freeze weight here
            for name, p in self.normal_head.named_parameters():
                if 'weight' in name:
                    p.requires_grad = False


    def initialize_weights(self):
        if self.init_normal_file:
            assert os.path.isfile(self.init_normal_file)
            print('initialize gloss output layer from ', self.init_normal_file)
            init_weights = torch.load(self.init_normal_file)
            for i in range(len(self.gls_vocab)):
                if i<self.special_vocab_size:
                    assert self.gls_vocab.itos[i] in self.gls_vocab.specials
                else:
                    gls_str = self.gls_vocab.itos[i]
                    if gls_str in init_weights:
                        self.normal_head.weight.data[i-self.special_vocab_size,
                                                     :] = init_weights[gls_str]
                    else:
                        print('Unknown gls {} train from scratch'.format(gls_str))
                        print('Partly tune parameters are not supported now, please set freeze_mode to all_tune')


    def forward(self, x): 
        self.special_logits = self.special_head(x) #b, v1
        self.normal_logits = self.normal_head(x)#b,v2
        self.logits = torch.cat([self.special_logits, self.normal_logits], dim=-1) # b, V
        return self.logits


