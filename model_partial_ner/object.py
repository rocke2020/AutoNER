"""
.. module:: Objective function
    :synopsis: fuzzy objective function
    
.. moduleauthor:: Liyuan Liu, Jingbo Shang
"""

import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.optim as optim
import itertools
import numpy as np
import random

class softCE(nn.Module):
    """
    The objective function for the distant supervised typing.

    Parameters
    ----------
    if_average : ``bool``, optional, (default = True).
        Whether to average over batches or not.
    """
    def __init__(self, if_average = True):
        super(softCE, self).__init__()
        self.logSoftmax = nn.LogSoftmax(dim = 1)
        self.if_average = if_average

    @staticmethod
    def soft_max(scores, target):
        """
        This is softmax, not soft cross entropy loss 
        Calculate the softmax for the input with regard to a target which can be treat as a mask, as target contains 
        only 0 or 1.

        Parameters
        ----------
        scores : ``torch.FloatTensor``, required. shape, (seq_len, class_num)
            The input of the softmax.
        target : ``torch.ByteTensor`` , required, shape, (seq_len, class_num)
            The target as the mask for the softmax input.
        """
        max_score, idx = torch.max(scores, 1, keepdim=True)
        exp_score = torch.exp(scores - max_score.expand_as(scores))
        # exp_score = exp_score.masked_fill_(mask, 0)
        exp_score = exp_score * target
        batch_size = scores.size(0)        
        exp_score_sum = torch.sum(exp_score, 1).view(batch_size, 1).expand_as(exp_score)
        prob_score = exp_score / exp_score_sum
        return prob_score

    def forward(self, scores, target):
        """
        Calculate the cross entropy loss for distant supervision. 

        Parameters
        ----------
        scores : ``torch.FloatTensor``, required. shape, (seq_len, class_num)
            The input of the softmax.
        target : ``torch.ByteTensor`` , required, shape, (seq_len, class_num)
            The target as the mask for the softmax input.
        """
        # supervision_p shape (seq_len, class_num), 
        supervision_p = softCE.soft_max(scores, target)
        scores_logp = self.logSoftmax(scores)
        CE = (-supervision_p * scores_logp).sum()
        if self.if_average:
            CE = CE / scores.size(0)
        return CE


def hinge_loss(score, label):
    """
    Hinge loss for distant supervision.
    """
    ins_num = label.size(0)
    score = 1 - score * label
    return score.masked_select(score > 0).sum() / ins_num