import torch
import torch.nn as nn
import numpy as np
from torch.nn import functional as F

from models.SubLayers import MultiHeadAttention, PositionwiseFeedForward


class FFTBlock(torch.nn.Module):
    """FFT Block"""

    def __init__(self, d_model, n_head, d_k, d_v, d_inner, kernel_size, dropout=0.1):
        super(FFTBlock, self).__init__()
        self.slf_attn = MultiHeadAttention(n_head, d_model, d_k, d_v, dropout=dropout)  #d_model是fc输入层神经元数， n_head*d_v输出层神经元数

        self.pos_ffn = PositionwiseFeedForward(
            d_model, d_inner, kernel_size, dropout=dropout    #这部分是两层卷积层，通道数256>>1024>>256,第一个卷积核为9第二个卷积核为1
        )

    def forward(self, enc_input, mask=None,slf_attn_mask=None):


        enc_output, enc_slf_attn = self.slf_attn(
            enc_input, enc_input, enc_input, mask=slf_attn_mask
        )
        enc_output = enc_output.masked_fill(mask.unsqueeze(-1), 0)

        enc_output = self.pos_ffn(enc_output)
        enc_output = enc_output.masked_fill(mask.unsqueeze(-1), 0)

        return enc_output, enc_slf_attn