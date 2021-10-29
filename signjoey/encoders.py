# coding: utf-8

import torch
import torch.nn as nn
from torch import Tensor
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from signjoey.helpers import freeze_params
from signjoey.transformer_layers import TransformerEncoderLayer, PositionalEncoding
from signjoey.embeddings import MaskedNorm

# pylint: disable=abstract-method
class Encoder(nn.Module):
    """
    Base encoder class
    """

    @property
    def output_size(self):
        """
        Return the output size

        :return:
        """
        return self._output_size

class NullEncoder(Encoder):
    def __init__(self,
        emb_size, pe=False):
        super(NullEncoder, self).__init__()
        self._output_size = emb_size
        if pe:
            print('Turn on positional encoding')
            self.pe = PositionalEncoding(emb_size)
        else:
            self.pe = None
    def forward(self, embed_src, src_length=None, mask=None,**kwargs):
        if self.pe:
            x = self.pe(embed_src)
            return x, x
        else:
            return embed_src, embed_src
class CNNEncoderLayer(nn.Module):
    def __init__(self,
                 input_size, output_size, kernel_size=5, stride=1, padding=0,
                 norm_type='sync_batch',
                 dropout=0):
        assert stride==1, 'only support stride=1 now otherwise mask should be adjusted'
        super(CNNEncoderLayer, self).__init__()
        self.conv_t = nn.Conv1d(input_size, output_size, kernel_size=kernel_size, 
            stride=stride, padding='same', bias=False)
        self.norm_t = MaskedNorm(norm_type=norm_type, 
                               num_groups=None, num_features=output_size)
        self.relu_t = nn.ReLU()
        print('set dropout rate to {}'.format(dropout))
        self.dropout = nn.Dropout(dropout)
    def forward(self, x, mask):
        #x B,T,D -> B, D, T
        x = self.conv_t(x.transpose(1,2)).transpose(1,2) #B, D', T -> B, T, D'
        x = self.norm_t(x, mask=mask)
        x = self.relu_t(x)
        x = self.dropout(x)
        return x, x


class CNNEncoder(Encoder):
    def __init__(self,
        emb_size, 
        hidden_size=512, 
        pe=False,
        num_layers=1,
        masking_before='zero',
        LN=False,
        **kwargs):
        super(CNNEncoder, self).__init__()
        self._output_size = hidden_size
        self.masking_before  = masking_before
        if pe:
            print('Turn on positional encoding')
            self.pe = PositionalEncoding(emb_size)
        else:
            print('Turn off positional encoding')
            self.pe = None
        self.layers = nn.ModuleList(
            [
                CNNEncoderLayer(
                    input_size=emb_size if i==0 else hidden_size,
                    output_size=hidden_size,
                    **kwargs
                )
                for i in range(num_layers)
            ]
        )
        self.LN = LN
        if self.LN:
            print('Turn on FC + LN at the end of CNN encoder')
            self.head_fc = nn.Linear(hidden_size, hidden_size)
            print('dropout rate betwen FC and LN = {}'.format(kwargs.get('dropout',0)))
            self.dropout = nn.Dropout(kwargs.get('dropout',0))
            self.head_ln = nn.LayerNorm(hidden_size)
            self.head = nn.Sequential(
                self.head_fc,
                self.dropout,
                self.head_ln
            )
        else:
            self.head = nn.Identity()

    def masking_before_cnn(self, embed_src, mask):
        assert self.masking_before=='zero', 'only support zero masking now'
        reshaped_mask = mask.transpose(1,2) # B, T, D
        if self.masking_before=='zero':
            masked = torch.zeros_like(embed_src)
        else:
            raise ValueError
        masked_embed = torch.where(reshaped_mask, embed_src, masked)
        return masked_embed

    def forward(self, embed_src,
                src_length,
                mask):
        embed_src = self.masking_before_cnn(embed_src, mask)
        if self.pe:
            x = self.pe(embed_src)
        else:
            x = embed_src
        for layer in self.layers:
            x,_ = layer(x, mask)
        x = self.head(x)
        return x,x

class RecurrentEncoder(Encoder):
    """Encodes a sequence of word embeddings"""

    # pylint: disable=unused-argument
    def __init__(
        self,
        rnn_type: str = "gru",
        hidden_size: int = 1,
        emb_size: int = 1,
        num_layers: int = 1,
        dropout: float = 0.0,
        emb_dropout: float = 0.0,
        bidirectional: bool = True,
        freeze: bool = False,
        **kwargs
    ) -> None:
        """
        Create a new recurrent encoder.

        :param rnn_type: RNN type: `gru` or `lstm`.
        :param hidden_size: Size of each RNN.
        :param emb_size: Size of the word embeddings.
        :param num_layers: Number of encoder RNN layers.
        :param dropout:  Is applied between RNN layers.
        :param emb_dropout: Is applied to the RNN input (word embeddings).
        :param bidirectional: Use a bi-directional RNN.
        :param freeze: freeze the parameters of the encoder during training
        :param kwargs:
        """

        super(RecurrentEncoder, self).__init__()

        self.emb_dropout = torch.nn.Dropout(p=emb_dropout, inplace=False)
        self.type = rnn_type
        self.emb_size = emb_size

        rnn = nn.GRU if rnn_type == "gru" else nn.LSTM

        self.rnn = rnn(
            emb_size,
            hidden_size,
            num_layers,
            batch_first=True,
            bidirectional=bidirectional,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        self._output_size = 2 * hidden_size if bidirectional else hidden_size

        if freeze:
            freeze_params(self)

    # pylint: disable=invalid-name, unused-argument
    def _check_shapes_input_forward(
        self, embed_src: Tensor, src_length: Tensor, mask: Tensor
    ) -> None:
        """
        Make sure the shape of the inputs to `self.forward` are correct.
        Same input semantics as `self.forward`.

        :param embed_src: embedded source tokens
        :param src_length: source length
        :param mask: source mask
        """
        assert embed_src.shape[0] == src_length.shape[0]
        assert embed_src.shape[2] == self.emb_size
        # assert mask.shape == embed_src.shape
        assert len(src_length.shape) == 1

    # pylint: disable=arguments-differ
    def forward(
        self, embed_src: Tensor, src_length: Tensor, mask: Tensor
    ) -> (Tensor, Tensor):
        """
        Applies a bidirectional RNN to sequence of embeddings x.
        The input mini-batch x needs to be sorted by src length.
        x and mask should have the same dimensions [batch, time, dim].

        :param embed_src: embedded src inputs,
            shape (batch_size, src_len, embed_size)
        :param src_length: length of src inputs
            (counting tokens before padding), shape (batch_size)
        :param mask: indicates padding areas (zeros where padding), shape
            (batch_size, src_len, embed_size)
        :return:
            - output: hidden states with
                shape (batch_size, max_length, directions*hidden),
            - hidden_concat: last hidden state with
                shape (batch_size, directions*hidden)
        """
        self._check_shapes_input_forward(
            embed_src=embed_src, src_length=src_length, mask=mask
        )

        # apply dropout to the rnn input
        embed_src = self.emb_dropout(embed_src)

        packed = pack_padded_sequence(embed_src, src_length.cpu(), batch_first=True)
        output, hidden = self.rnn(packed)

        # pylint: disable=unused-variable
        if isinstance(hidden, tuple):
            hidden, memory_cell = hidden

        output, _ = pad_packed_sequence(output, batch_first=True)
        # hidden: dir*layers x batch x hidden
        # output: batch x max_length x directions*hidden
        batch_size = hidden.size()[1]
        # separate final hidden states by layer and direction
        hidden_layerwise = hidden.view(
            self.rnn.num_layers,
            2 if self.rnn.bidirectional else 1,
            batch_size,
            self.rnn.hidden_size,
        )
        # final_layers: layers x directions x batch x hidden

        # concatenate the final states of the last layer for each directions
        # thanks to pack_padded_sequence final states don't include padding
        fwd_hidden_last = hidden_layerwise[-1:, 0]
        bwd_hidden_last = hidden_layerwise[-1:, 1]

        # only feed the final state of the top-most layer to the decoder
        # pylint: disable=no-member
        hidden_concat = torch.cat([fwd_hidden_last, bwd_hidden_last], dim=2).squeeze(0)
        # final: batch x directions*hidden
        return output, hidden_concat

    def __repr__(self):
        return "%s(%r)" % (self.__class__.__name__, self.rnn)


class TransformerEncoder(Encoder):
    """
    Transformer Encoder
    """

    # pylint: disable=unused-argument
    def __init__(
        self,
        hidden_size: int = 512,
        ff_size: int = 2048,
        num_layers: int = 8,
        num_heads: int = 4,
        dropout: float = 0.1,
        emb_dropout: float = 0.1,
        freeze: bool = False,
        pe: bool = True,
        LN: bool = True,
        skip_connection: bool=True,
        output_size: int=512,
        **kwargs
    ):
        """
        Initializes the Transformer.
        :param hidden_size: hidden size and size of embeddings
        :param ff_size: position-wise feed-forward layer size.
          (Typically this is 2*hidden_size.)
        :param num_layers: number of layers
        :param num_heads: number of heads for multi-headed attention
        :param dropout: dropout probability for Transformer layers
        :param emb_dropout: Is applied to the input (word embeddings).
        :param freeze: freeze the parameters of the encoder during training
        :param kwargs:
        """
        super(TransformerEncoder, self).__init__()
        # build all (num_layers) layers
        self.layers = nn.ModuleList(
            [
                TransformerEncoderLayer(
                    size=hidden_size,
                    ff_size=ff_size,
                    num_heads=num_heads,
                    dropout=dropout,
                    fc_type=kwargs.get('fc_type', 'linear'),
                    kernel_size=kwargs.get('kernel_size', 1),
                    skip_connection=skip_connection
                )
                for _ in range(num_layers)
            ]
        )
        if LN:
            self.layer_norm = nn.LayerNorm(output_size, eps=1e-6)
        else:
            print('Turn off layer norm at the last of encoder')
            self.layer_norm = nn.Identity()
        if pe:
            self.pe = PositionalEncoding(hidden_size)
        else:
            print('Turn off positional encoding')
            self.pe = None
        self.emb_dropout = nn.Dropout(p=emb_dropout)

        self._output_size = output_size
        if self._output_size != hidden_size:
            print('transformer outputsize {} != hidden size {}'.format(self._output_size, hidden_size))
            print('Create a mapping layer')
            self.map2gloss_embed = nn.Sequential(
                nn.Linear(hidden_size, self._output_size),
                nn.Dropout(dropout),
            )
        else:
            self.map2gloss_embed = nn.Identity()
        if freeze:
            freeze_params(self)

    # pylint: disable=arguments-differ
    def forward(
        self, embed_src: Tensor, src_length: Tensor, mask: Tensor,
        output_attention: bool=False
    ) -> (Tensor, Tensor):
        """
        Pass the input (and mask) through each layer in turn.
        Applies a Transformer encoder to sequence of embeddings x.
        The input mini-batch x needs to be sorted by src length.
        x and mask should have the same dimensions [batch, time, dim].

        :param embed_src: embedded src inputs,
            shape (batch_size, src_len, embed_size)
        :param src_length: length of src inputs
            (counting tokens before padding), shape (batch_size)
        :param mask: indicates padding areas (zeros where padding), shape
            (batch_size, src_len, embed_size)
        :return:
            - output: hidden states with
                shape (batch_size, max_length, directions*hidden),
            - hidden_concat: last hidden state with
                shape (batch_size, directions*hidden)
        """
        if self.pe:
            x = self.pe(embed_src)  # add position encoding to word embeddings
        else:
            x = embed_src
        x = self.emb_dropout(x)
        if output_attention:
            attentions = []
        for layer in self.layers:
            if output_attention:
                x, attention = layer(x, mask, output_attention=True)
                attentions.append(attention)
            else:
                x = layer(x, mask)

        x = self.map2gloss_embed(x)
        if output_attention:
            attentions = torch.stack(attentions, dim=1) #B, L, H, T,T
            return self.layer_norm(x), None, attentions # None -> encoder hidden(unused)
        else:
            return self.layer_norm(x), None

    def __repr__(self):
        return "%s(num_layers=%r, num_heads=%r)" % (
            self.__class__.__name__,
            len(self.layers),
            self.layers[0].src_src_att.num_heads,
        )
