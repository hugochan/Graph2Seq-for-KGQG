'''
Created on Nov, 2018

@author: hugo

'''
import yaml
import re
import string
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


re_art = re.compile(r'\b(a|an|the)\b')
re_punc = re.compile(r'[%s]' % re.escape(string.punctuation))
def normalize_answer(s):
    """Lower text and remove extra whitespace."""
    def remove_articles(text):
        return re_art.sub(' ', text)

    def remove_punc(text):
        return re_punc.sub(' ', text)  # convert punctuation to spaces

    def white_space_fix(text):
        return ' '.join(text.split())

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))

def tile(x, count, dim=0):
    """
    Tiles x on dimension dim count times.
    """
    perm = list(range(len(x.size())))
    if dim != 0:
        perm[0], perm[dim] = perm[dim], perm[0]
        x = x.permute(perm).contiguous()
    out_size = list(x.size())
    out_size[0] *= count
    batch = x.size(0)
    x = x.view(batch, -1) \
         .transpose(0, 1) \
         .repeat(count, 1) \
         .transpose(0, 1) \
         .contiguous() \
         .view(*out_size)
    if dim != 0:
        x = x.permute(perm).contiguous()
    return x

def to_cuda(x, device=None):
    if device:
        x = x.to(device)
    return x

def create_mask(x, N, device=None):
    x = x.data
    mask = np.zeros((x.size(0), N))
    for i in range(x.size(0)):
        mask[i, :x[i]] = 1
    return to_cuda(torch.Tensor(mask), device)

def get_config(config_path="config.yml"):
    with open(config_path, "r") as setting:
        config = yaml.load(setting)
    return config
