"""
.. module:: basic
    :synopsis: basic rnn
 
.. moduleauthor:: Liyuan Liu, Jingbo Shang
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class BasicRNN(nn.Module):
    """
    The multi-layer recurrent networks for the vanilla stacked RNNs.

    Parameters
    ----------
    layer_num: ``int``, required.
        The number of layers. 
    unit : ``torch.nn.Module``, required.
        The type of rnn unit.
    input_dim : ``int``, required.
        The input dimension fo the unit.
    hid_dim : ``int``, required.
        The hidden dimension fo the unit.
    droprate : ``float``, required.
        The dropout ratrio.
    batch_norm: ``bool``, required.
        Incorporate batch norm or not. 
    """
    def __init__(self, layer_num, emb_dim, hid_dim, droprate, layer_norm):
        super(BasicRNN, self).__init__()

        self.lstm = nn.LSTM(input_size=emb_dim, hidden_size=hid_dim//2, num_layers=layer_num, batch_first=True, 
            bidirectional=True)
        self.output_dim = hid_dim
        self.layer_norm = layer_norm
        self.droprate = droprate
        if self.layer_norm:
            self.LayerNorm = nn.LayerNorm(hid_dim)

    def forward(self, x):
        """
        Calculate the output.

        Parameters
        ----------
        x : ``torch.LongTensor``, required.
            the input tensor, of shape (seq_len, batch_size, input_dim).

        Returns
        ----------
        output: ``torch.FloatTensor``.
            The output of RNNs.
        """
        out, _ = self.lstm(x)
        if self.layer_norm:
            out = self.LayerNorm(out)
        
        if self.droprate > 0:
            out = F.dropout(out, p=self.droprate, training=self.training)        
        return out