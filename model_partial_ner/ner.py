""" input example
<s> O None S
Effects I None S
of I None S
uninephrectomy I None S
and I None S
high I None S
protein I None S
feeding I None S
on I None S
lithium I Chemical S
-induced I None S
chronic I Disease S
renal O Disease S
failure O Disease S
in I None S
rats I None S
. I None S
<eof> I None S

<s> O None S
Fusidic O None D
acid O None D
was O None D
administered I None S
orally I None S
in I None S
a I None S
dose I None S
of I None S
500 I None S
mg O None D
t.d.s O None D
. I None S
<eof> I None S

"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import model_partial_ner.utils as utils
from model_partial_ner.highway import Highway
from utilities.common_utils import get_logger
import logging


logger = get_logger(name=__name__, log_file=None, log_level=logging.DEBUG, log_level_name='')


class NER(nn.Module):
    """
    Sequence Labeling model augumented with language model.

    Parameters
    ----------
    rnn : ``torch.nn.Module``, required.
        The RNN unit..
    w_num : ``int`` , required.
        The number of words.
    w_dim : ``int`` , required.
        The dimension of word embedding.
    c_num : ``int`` , required.
        The number of characters.
    c_dim : ``int`` , required.
        The dimension of character embedding.
    y_dim : ``int`` , required.
        The dimension of tags types.
    y_num : ``int`` , required.
        The number of tags types.
    droprate : ``float`` , required
        The dropout ratio.
    """
    def __init__(self, rnn, 
                w_num: int, 
                w_dim: int, 
                c_num: int, 
                c_dim: int, 
                y_dim: int, 
                y_num: int, 
                droprate: float):

        super(NER, self).__init__()

        self.rnn = rnn
        self.rnn_outdim = self.rnn.output_dim
        self.one_direction_dim = self.rnn_outdim // 2
        self.word_embed = nn.Embedding(w_num, w_dim)
        self.char_embed = nn.Embedding(c_num, c_dim)
        self.drop = nn.Dropout(p=droprate)
        self.add_proj = y_dim > 0
        self.to_chunk = Highway(self.rnn_outdim)
        self.to_type = Highway(self.rnn_outdim)

        if self.add_proj:
            self.to_chunk_proj = nn.Linear(self.rnn_outdim, y_dim)
            self.to_type_proj = nn.Linear(self.rnn_outdim, y_dim)
            self.chunk_weight = nn.Linear(y_dim, 1)
            self.type_weight = nn.Linear(y_dim, y_num)
            self.chunk_layer = nn.Sequential(self.to_chunk, self.drop, self.to_chunk_proj, self.drop, self.chunk_weight)
            self.type_layer = nn.Sequential(self.to_type, self.drop, self.to_type_proj, self.drop, self.type_weight)
        else:
            self.chunk_weight = nn.Linear(self.rnn_outdim, 1)
            self.type_weight = nn.Linear(self.rnn_outdim, y_num)
            self.chunk_layer = nn.Sequential(self.to_chunk, self.drop, self.chunk_weight)
            self.type_layer = nn.Sequential(self.to_type, self.drop, self.type_weight)

    def to_params(self):
        """
        To parameters.
        """
        return {
            "model_type": "char-lstm-two-level",
            # "rnn_params": self.rnn.to_params(),
            "word_embed_num": self.word_embed.num_embeddings,
            "word_embed_dim": self.word_embed.embedding_dim,
            "char_embed_num": self.char_embed.num_embeddings,
            "char_embed_dim": self.char_embed.embedding_dim,
            "type_dim": self.type_weight.in_features if self.add_proj else -1,
            "type_num": self.type_weight.out_features,
            "droprate": self.drop.p,
            "label_schema": "tie-or-break"
        }

    def load_pretrained_word_embedding(self, pre_word_embeddings):
        """
        Load pre-trained word embedding.

        Parameters
        ----------
        pre_word_embeddings : ``torch.FloatTensor``, required.
            pre-trained word embedding
        """
        self.word_embed.weight = nn.Parameter(pre_word_embeddings)

    def rand_ini(self):
        """
        Random initialization.
        """
        # self.rnn.rand_ini()  # simplify the rnn code
        self.to_chunk.rand_ini()
        self.to_type.rand_ini()
        utils.init_embedding(self.char_embed.weight)
        utils.init_linear(self.chunk_weight)
        utils.init_linear(self.type_weight)
        if self.add_proj:
            utils.init_linear(self.to_chunk_proj)
            utils.init_linear(self.to_type_proj)

    def forward(self, w_in, c_in, char_mask):
        """
        Sequence labeling model.

        Parameters
        ----------
        w_in : ``torch.LongTensor``, required.
            The RNN unit.
        c_in : ``torch.LongTensor`` , required.
            The number of characters.
        char_mask : ``torch.ByteTensor`` , required.
            The mask for character-level input.
        """
        w_emb = self.word_embed(w_in)
        c_emb = self.char_embed(c_in)
        emb = self.drop(torch.cat([w_emb, c_emb], 2))

        # batch size auto changes, the seq length is char length!
        # out torch.Size([115, 27, 300]), mask torch.Size([115, 27])
        # out torch.Size([88, 35, 300]), mask torch.Size([88, 35])
        out = self.rnn(emb)

        # out shape is 2d, the first size is always changed because the mask is always changed.
        # out first size: batch_size *seq minus "0 mask char token" number
        # out.shape torch.Size([756, 300])
        # out.shape torch.Size([419, 300])
        char_mask = char_mask.unsqueeze(2).expand_as(out)        
        out = out.masked_select(char_mask).view(-1, self.rnn_outdim)
        return out

    def chunking(self, z_in):
        """
        no mask, return 1d tensor of chunk break/tie label ids

        Parameters
        ----------
        z_in : ``torch.LongTensor``, required.
           The output of the character-level lstms.
        """
        z_in = self.drop(z_in)

        out = self.chunk_layer(z_in).squeeze(1)
        return out

    def typing(self, z_in, word_mask):
        """
        Typing 
        TODO use mask to filter relative meaning will ignore useful meanings of the following words. I will try to 
        TODO midify to add the following word info

        Parameters
        ----------
        z_in : ``torch.LongTensor``, required.
           The output of the character-level lstms.
        word_mask : ``torch.bool`` , required.
            The mask for word-level input.
        """
        
        word_mask = word_mask.unsqueeze(1).expand_as(z_in)
        z_in = z_in.masked_select(word_mask).view(-1, 2, self.one_direction_dim)

        # the seq length becomes len-1, the seq length is the same as the word number
        z_in = torch.cat([z_in[:-1, 1, :].squeeze(1), z_in[1:, 0, :].squeeze(1)], dim = 1)
        z_in = self.drop(z_in)

        out = self.type_layer(z_in)
        return out
        
    def to_span(self, chunk_label, type_ids, none_idx):
        """
        Convert word-level labels to entity spans.

        Parameters
        ----------
        chunk_label : ``torch.LongTensor``, required.
            The chunk label for one sequence.
        type_ids : ``torch.LongTensor`` , required.
            The type label for one sequence.
        none_idx: ``int``, required.
            Label index fot the not-target-type entity.
        """
        span_list = list()

        pre_idx = -1
        cur_idx = 0
        type_idx = 0
        while cur_idx < len(chunk_label):
            if chunk_label[cur_idx].data[0] == 1:
                if pre_idx >= 0:
                    cur_type = type_ids[type_idx].data[0]
                    if cur_type != none_idx:
                        span_list.append('('+str(pre_idx)+','+str(cur_idx)+')')
                    type_idx += 1
                pre_idx = cur_idx
            cur_idx += 1
            
        assert type_idx == len(type_ids)

        return set(span_list)


    def to_typed_span(self, chunk_label, type_ids, none_idx, id2label):
        """ not batch level, but sentence level
        TODO the author actually put word_mask as chunk_label, and it is right here!
        TODO merge chunk_label and word_mask
        Convert word-level labels to typed entity spans.

        Parameters
        ----------
        chunk_label : ``torch.LongTensor``, required.
           The output of the character-level lstms.
        mask : ``torch.ByteTensor`` , required.
            The mask for word-level input.
        none_idx: ``int``, required.
            Label index fot the not-target-type entity.
        """
        span_list = list()

        pre_idx = -1
        cur_idx = 0
        type_idx = 0
        while cur_idx < len(chunk_label):
            if chunk_label[cur_idx].item() == 1: 
                if pre_idx >=0:
                    cur_type_idx = type_ids[type_idx].item()
                    if cur_type_idx != none_idx:
                        span_list.append(id2label[cur_type_idx]+'@('+str(pre_idx)+','+str(cur_idx)+')')
                    type_idx += 1
                pre_idx = cur_idx
            cur_idx += 1

        assert type_idx == len(type_ids)

        return set(span_list)