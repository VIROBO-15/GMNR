import torch as pt
import torch.nn.functional as F
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision.utils import save_image, make_grid
from torch.utils.tensorboard import SummaryWriter

import os, sys, json
import numpy as np
from skimage import io, transform
from datetime import datetime
from .mlp import ReluMLP



class Basis(nn.Module):
  def __init__(self, shape, out_view):
    super().__init__()
    #choosing illumination model
    self.order = 3

    # network for learn basis
    self.seq_basis = ReluMLP(
        1, #basis_mlp
        64, #basis_hidden
        self.order * 4 + 512,
        0.01,
        out_node = 4, #basis_out
      )
    
    print('Basis Network:',self.seq_basis)

    # positional encoding pre compute
    self.pos_freq_viewing = pt.Tensor([(2 ** i) for i in range(self.order)]).view(1, 1, 1, 1, -1)

  def forward(self, vi, ws, coeff = None):
    # vi, xy = get_viewing_angle(sfm, feature, ref_coords, planes)
    n, sel = vi.shape[:2]

    # positional encoding for learn basis
    hinv_xy = vi[...,  :2, None] * self.pos_freq_viewing.cuda()
    big = pt.reshape(hinv_xy, [n, sel, 1, hinv_xy.shape[-2] * hinv_xy.shape[-1]])
    vi = pt.cat([pt.sin(0.5*np.pi*big), pt.cos(0.5*np.pi*big)], -1)

    vi = pt.cat((vi, ws.unsqueeze(1).unsqueeze(1)), dim=-1)

    out2 = self.seq_basis(vi)
    out2 = pt.tanh(out2)

    vi = out2.view(n, sel,1, -1)

    coeff = coeff.view(coeff.shape[0], coeff.shape[1], 3,  -1)
    coeff = pt.tanh(coeff)

    illumination = pt.sum(coeff * vi,-1).permute([0, 2, 1])

    return illumination