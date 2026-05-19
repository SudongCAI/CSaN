# --------------------------------------------------------
# Swin-CSaN with two optional CSaN-v1 variants
# Written by Sudong Cai
# --------------------------------------------------------
from __future__ import absolute_import
import logging
import math
from typing import Callable, List, Optional, Tuple, Union
#from typing import Optional

import torch
import torch.nn as nn

from torch.autograd import Variable 
from torch.nn.parameter import Parameter
import torch.nn.functional as F

from functools import partial
from itertools import chain
from torch.utils.checkpoint import checkpoint
from torch.hub import load_state_dict_from_url

from einops import rearrange
from einops.layers.torch import Rearrange

import torch.linalg as linalg

from torch.autograd import Function
from torch.nn.init import calculate_gain

import numpy as np

from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.layers import PatchEmbed, DropPath, ClassifierHead, to_2tuple, to_ntuple, trunc_normal_, \
    _assert, use_fused_attn#, Mlp
from timm.models._builder import build_model_with_cfg
from timm.models._features_fx import register_notrace_function
from timm.models._manipulate import checkpoint_seq, named_apply
from timm.models._registry import generate_default_cfgs, register_model, register_model_deprecations
from timm.models.vision_transformer import get_init_weights_vit


from torch.nn.functional import softplus as pesudo_linear2
from torch.nn.functional import softplus
import torch.fft as fft


from torch.cuda.amp import autocast

from torch.nn.functional import softplus
from torch import sigmoid
from torch import exp
from torch import asinh
from torch import log
from torch import tanh


#import torch._dynamo 
#from timm.layers import drop_path as drop_path_core

#@torch._dynamo.disable
#def safe_drop_path(x, drop_prob, training):
#    return drop_path_core(x, drop_prob, training)








__all__ = ['SwinTransformer']  # model_registry will add each entrypoint fn to this

_logger = logging.getLogger(__name__)

_int_or_tuple_2_t = Union[int, Tuple[int, int]]




#################################





### mlp
class Mlp(nn.Module):

    def __init__(
            self,
            in_features,
            hidden_features=None,
            out_features=None,
            act_layer=nn.GELU,
            norm_layer=None,
            bias=True,
            drop=0.,
            use_conv=False,
            drop_path=None,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        bias = to_2tuple(bias)
        drop_probs = to_2tuple(drop)
        linear_layer = partial(nn.Conv2d, kernel_size=1) if use_conv else nn.Linear

        self.fc1 = linear_layer(in_features, hidden_features, bias=bias[0])
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.act = act_layer()
        self.drop1 = nn.Dropout(drop_probs[0])
        self.norm = norm_layer(hidden_features) if norm_layer is not None else nn.Identity()
        self.fc2 = linear_layer(hidden_features, out_features, bias=bias[1]) # original
        self.drop2 = nn.Dropout(drop_probs[1]) 
        head_dropout = 0. #0. for tiny; 0.4 for small
        self.head_dropout = nn.Dropout(head_dropout) if head_dropout > 0. else nn.Identity() # 0 for tiny; 0.4 for small
        
        self.dim = in_features
        
        
    
    def forward(self, x):
        
        x = self.fc1(x)
        x = self.act(x)
        
        #x = self.drop1(x)
        #x = self.norm(x)
        
        x = self.head_dropout(x)
        
        x = self.fc2(x)
        
        #x = self.drop2(x)
        
        return x











def window_partition(x, window_size: int):
    """
    Args:
        x: (B, H, W, C)
        window_size (int): window size

    Returns:
        windows: (num_windows*B, window_size, window_size, C)
    """
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    return windows


@register_notrace_function  # reason: int argument is a Proxy
def window_reverse(windows, window_size: int, H: int, W: int):
    """
    Args:
        windows: (num_windows*B, window_size, window_size, C)
        window_size (int): Window size
        H (int): Height of image
        W (int): Width of image

    Returns:
        x: (B, H, W, C)
    """
    C = windows.shape[-1]
    x = windows.view(-1, H // window_size, W // window_size, window_size, window_size, C)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, H, W, C)
    return x


def get_relative_position_index(win_h: int, win_w: int):
    # get pair-wise relative position index for each token inside the window
    coords = torch.stack(torch.meshgrid([torch.arange(win_h), torch.arange(win_w)]))  # 2, Wh, Ww
    coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
    relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
    relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
    relative_coords[:, :, 0] += win_h - 1  # shift to start from 0
    relative_coords[:, :, 1] += win_w - 1
    relative_coords[:, :, 0] *= 2 * win_w - 1
    return relative_coords.sum(-1)  # Wh*Ww, Wh*Ww









# csan v1 (on swin window attention, version1: bounded kappas)
class WindowAttention(nn.Module):
    """ Window based multi-head self attention (W-MSA) module with relative position bias.
    It supports shifted and non-shifted windows.
    """
    fused_attn: torch.jit.Final[bool]

    def __init__(
            self,
            dim: int,
            num_heads: int,
            head_dim: Optional[int] = None,
            window_size: _int_or_tuple_2_t = 7,
            qkv_bias: bool = True,
            attn_drop: float = 0.,
            proj_drop: float = 0., 
            drop_path = 0., 
    ):
        """
        Args:
            dim: Number of input channels.
            num_heads: Number of attention heads.
            head_dim: Number of channels per head (dim // num_heads if not set)
            window_size: The height and width of the window.
            qkv_bias:  If True, add a learnable bias to query, key, value.
            attn_drop: Dropout ratio of attention weight.
            proj_drop: Dropout ratio of output.
        """
        super().__init__()
        self.dim = dim
        self.window_size = to_2tuple(window_size)  # Wh, Ww
        win_h, win_w = self.window_size 
        self.win_h = win_h 
        self.win_w = win_w 
        self.window_area = win_h * win_w
        self.num_heads = num_heads
        head_dim = head_dim or dim // num_heads 
        self.head_dim = head_dim 
        attn_dim = head_dim * num_heads
        self.scale = head_dim ** -0.5
        self.fused_attn = use_fused_attn(experimental=True)  # NOTE not tested for prime-time yet
        
        # define a parameter table of relative position bias, shape: 2*Wh-1 * 2*Ww-1, nH
        self.relative_position_bias_table = nn.Parameter(torch.zeros((2 * win_h - 1) * (2 * win_w - 1), num_heads))

        # get pair-wise relative position index for each token inside the window
        self.register_buffer("relative_position_index", get_relative_position_index(win_h, win_w))
        
        self.qkv = nn.Linear(dim, attn_dim * 3, bias=qkv_bias)
        
        ratio = 8 # or 8 12
        threshold = 24 
        hidden_dim = dim//ratio if dim//ratio > threshold else threshold # num_heads=(3, 6, 12, 24) 
        
        self.u = nn.Linear(dim, dim//4) 
        
        self.reduce = nn.Linear(dim//4, hidden_dim) 
        self.act = nn.GELU() 
        
        self.ka = nn.Linear(hidden_dim, dim) 
        self.kv = nn.Linear(hidden_dim, dim) 
        self.magk = nn.Linear(hidden_dim, self.num_heads) 
        self.temp = nn.Linear(hidden_dim, self.num_heads) 
        
        #attn_drop = 0.05 # swin-small 0.05 (test 0.1 once?)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(attn_dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        
        trunc_normal_(self.relative_position_bias_table, std=.02)
        self.softmax = nn.Softmax(dim=-1)
        
        # init conv1d as conv2d does
        self.apply(self._init_weights)

    
    def _get_rel_pos_bias(self) -> torch.Tensor:
        relative_position_bias = self.relative_position_bias_table[
            self.relative_position_index.view(-1)].view(self.window_area, self.window_area, -1)  # Wh*Ww,Wh*Ww,nH
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
        return relative_position_bias.unsqueeze(0)


    #@torch.jit.script
    def quasi_linear(self, x, eta):
        return torch.where(x > eta, x, eta * torch.exp(x.clamp(max=eta) / eta - 1))    

        
    def forward(self, x, mask: Optional[torch.Tensor] = None):
        """
        Args:
            x: input features with shape of (num_windows*B, N, C)
            mask: (0/-inf) mask with shape of (num_windows, Wh*Ww, Wh*Ww) or None
        """
        B_, N, C = x.shape
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        
        u = self.u(x).view(B_, N, self.num_heads, -1).transpose(1, 2) # b N g d -> b g N h 
        
        if self.fused_attn:
            attn_mask = self._get_rel_pos_bias()
            if mask is not None:
                num_win = mask.shape[0]
                mask = mask.view(1, num_win, 1, N, N).expand(B_ // num_win, -1, self.num_heads, -1, -1)
                attn_mask = attn_mask + mask.reshape(-1, self.num_heads, N, N)
            x = torch.nn.functional.scaled_dot_product_attention(
                q, k, v,
                attn_mask=attn_mask,
                dropout_p=self.attn_drop.p,
            )
        else: 
            q = q * self.scale 
            attn = q @ k.transpose(-2, -1)
            attn = attn + self._get_rel_pos_bias()
            if mask is not None:
                num_win = mask.shape[0]
                attn = attn.view(-1, num_win, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
                attn = attn.view(-1, self.num_heads, N, N)
            # scales compute 
            qscore = self.quasi_linear(attn, 0.25).mean(dim=-1, keepdim=True) # b g N 1 
            
            ind = qscore * u.sigmoid() # b g N d 
            ind = ind.transpose(1, 2).reshape(B_, N, -1) # b g N h -> b N g h -> b N H 
            
            indm = self.reduce(ind) 
            indm = self.act(indm) 
            
            ka = tanh(self.ka(indm))
            ka = ka.view(B_, N, self.num_heads, self.head_dim).transpose(1, 2) + 1. 
            kv = tanh(self.kv(indm))
            kv = kv.view(B_, N, self.num_heads, self.head_dim).transpose(1, 2)
            
            temp = tanh(self.temp(indm))
            temp = temp.transpose(1, 2).unsqueeze(-1) + 1. # b N g -> b g N 1
            magk = tanh(self.magk(indm))
            magk = magk.transpose(1, 2).unsqueeze(-2) + 1. # b N g -> b g 1 N 
            
            attn = self.softmax(attn * temp) * magk 
            attn = self.attn_drop(attn)
            
            x = attn @ v * ka + v * kv 
            
        x = x.transpose(1, 2).reshape(B_, N, -1)
        
        #x = self.actraw(x) # additive
        
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


    def _init_weights(self, m):
        if isinstance(m, (nn.Linear, nn.Conv2d, nn.Conv1d)):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)






# csan v1 (on swin window attention, version2: unbounded kappas)
class SoftGShift(nn.Module):
    def __init__(self, dim):
        super().__init__()
        
        self.proj = nn.Linear(dim, dim) 
        
        self.apply(self._init_weights)
        
    def forward(self, x):
        return x * torch.sigmoid(self.proj(x)) 
    
    def _init_weights(self, m):
        if isinstance(m, (nn.Linear, nn.Conv2d, nn.Conv1d)):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0) 


class WindowAttention(nn.Module):
    """ Window based multi-head self attention (W-MSA) module with relative position bias.
    It supports shifted and non-shifted windows.
    """
    fused_attn: torch.jit.Final[bool]

    def __init__(
            self,
            dim: int,
            num_heads: int,
            head_dim: Optional[int] = None,
            window_size: _int_or_tuple_2_t = 7,
            qkv_bias: bool = True,
            attn_drop: float = 0.,
            proj_drop: float = 0., 
            drop_path = 0., 
    ):
        """
        Args:
            dim: Number of input channels.
            num_heads: Number of attention heads.
            head_dim: Number of channels per head (dim // num_heads if not set)
            window_size: The height and width of the window.
            qkv_bias:  If True, add a learnable bias to query, key, value.
            attn_drop: Dropout ratio of attention weight.
            proj_drop: Dropout ratio of output.
        """
        super().__init__()
        self.dim = dim
        self.window_size = to_2tuple(window_size)  # Wh, Ww
        win_h, win_w = self.window_size 
        self.win_h = win_h 
        self.win_w = win_w 
        self.window_area = win_h * win_w
        self.num_heads = num_heads
        head_dim = head_dim or dim // num_heads 
        self.head_dim = head_dim 
        attn_dim = head_dim * num_heads
        self.scale = head_dim ** -0.5
        self.fused_attn = use_fused_attn(experimental=True)  # NOTE not tested for prime-time yet
        
        # define a parameter table of relative position bias, shape: 2*Wh-1 * 2*Ww-1, nH
        self.relative_position_bias_table = nn.Parameter(torch.zeros((2 * win_h - 1) * (2 * win_w - 1), num_heads))

        # get pair-wise relative position index for each token inside the window
        self.register_buffer("relative_position_index", get_relative_position_index(win_h, win_w))
        
        self.qkv = nn.Linear(dim, attn_dim * 3, bias=qkv_bias)
        
        ratio = 8 # or 8 12
        threshold = 24 
        hidden_dim = dim//ratio if dim//ratio > threshold else threshold # num_heads=(3, 6, 12, 24) 
        
        self.u = nn.Linear(dim, dim//4) 
        
        self.reduce = nn.Linear(dim//4, hidden_dim) 
        self.act = SoftGShift(dim=hidden_dim) 
        
        self.ka = nn.Linear(hidden_dim, dim) 
        self.kv = nn.Linear(hidden_dim, dim) 
        self.magk = nn.Linear(hidden_dim, self.num_heads) 
        self.temp = nn.Linear(hidden_dim, self.num_heads) 
        
        #attn_drop = 0.05 # swin-small 0.05 (test 0.1 once?)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(attn_dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        
        trunc_normal_(self.relative_position_bias_table, std=.02)
        self.softmax = nn.Softmax(dim=-1)
        
        # init conv1d as conv2d does
        self.apply(self._init_weights)

    
    def _get_rel_pos_bias(self) -> torch.Tensor:
        relative_position_bias = self.relative_position_bias_table[
            self.relative_position_index.view(-1)].view(self.window_area, self.window_area, -1)  # Wh*Ww,Wh*Ww,nH
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
        return relative_position_bias.unsqueeze(0)


    #@torch.jit.script
    def quasi_linear(self, x, eta):
        return torch.where(x > eta, x, eta * torch.exp(x.clamp(max=eta) / eta - 1))    

    
    #@torch.jit.script
    def comb_ind(self, ind, u): 
        return ind * softplus(u) # gating on the indicator 
    
    
    #@torch.jit.script
    def comb_scaled(self, attn, v, ka, kv): 
        return attn @ v * ka + v * kv 
    
    
    def forward(self, x, mask: Optional[torch.Tensor] = None):
        """
        Args:
            x: input features with shape of (num_windows*B, N, C)
            mask: (0/-inf) mask with shape of (num_windows, Wh*Ww, Wh*Ww) or None
        """
        B_, N, C = x.shape
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        
        u = self.u(x).view(B_, N, self.num_heads, -1).transpose(1, 2) # b N g d -> b g N h 
        
        if self.fused_attn:
            attn_mask = self._get_rel_pos_bias()
            if mask is not None:
                num_win = mask.shape[0]
                mask = mask.view(1, num_win, 1, N, N).expand(B_ // num_win, -1, self.num_heads, -1, -1)
                attn_mask = attn_mask + mask.reshape(-1, self.num_heads, N, N)
            x = torch.nn.functional.scaled_dot_product_attention(
                q, k, v,
                attn_mask=attn_mask,
                dropout_p=self.attn_drop.p,
            )
        else: 
            q = q * self.scale 
            attn = q @ k.transpose(-2, -1)
            attn = attn + self._get_rel_pos_bias()
            if mask is not None:
                num_win = mask.shape[0]
                attn = attn.view(-1, num_win, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
                attn = attn.view(-1, self.num_heads, N, N)
            # scales compute 
            qscore = self.quasi_linear(attn, 0.25).mean(dim=-1, keepdim=True) # b g N 1 
            
            ind = qscore * u.sigmoid() # b g N d 
            ind = ind.transpose(1, 2).reshape(B_, N, -1) # b g N h -> b N g h -> b N H 
            
            indm = self.reduce(ind) 
            indm = self.act(indm) 
            
            ka = self.ka(indm).view(B_, N, self.num_heads, self.head_dim).transpose(1, 2) + 1. 
            kv = self.kv(indm).view(B_, N, self.num_heads, self.head_dim).transpose(1, 2)
            
            temp = self.temp(indm).transpose(1, 2).unsqueeze(-1) + 1. # b N g -> b g N 1
            magk = self.magk(indm).transpose(1, 2).unsqueeze(-2) + 1. # b N g -> b g 1 N 
            
            attn = self.softmax(attn * temp) * magk 
            attn = self.attn_drop(attn)
            
            x = attn @ v * ka + v * kv 
            
        x = x.transpose(1, 2).reshape(B_, N, -1)
        
        #x = self.actraw(x) # additive
        
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


    def _init_weights(self, m):
        if isinstance(m, (nn.Linear, nn.Conv2d, nn.Conv1d)):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)







class SwinTransformerBlock(nn.Module):
    """ Swin Transformer Block.
    """

    def __init__(
            self,
            dim: int,
            input_resolution: _int_or_tuple_2_t,
            num_heads: int = 4,
            head_dim: Optional[int] = None,
            window_size: _int_or_tuple_2_t = 7,
            shift_size: int = 0,
            mlp_ratio: float = 4.,
            qkv_bias: bool = True,
            proj_drop: float = 0.,
            attn_drop: float = 0.,
            drop_path: float = 0.,
            act_layer: Callable = nn.GELU,
            norm_layer: Callable = nn.LayerNorm,
    ):
        """
        Args:
            dim: Number of input channels.
            input_resolution: Input resolution.
            window_size: Window size.
            num_heads: Number of attention heads.
            head_dim: Enforce the number of channels per head
            shift_size: Shift size for SW-MSA.
            mlp_ratio: Ratio of mlp hidden dim to embedding dim.
            qkv_bias: If True, add a learnable bias to query, key, value.
            proj_drop: Dropout rate.
            attn_drop: Attention dropout rate.
            drop_path: Stochastic depth rate.
            act_layer: Activation layer.
            norm_layer: Normalization layer.
        """
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        if min(self.input_resolution) <= self.window_size:
            # if window size is larger than input resolution, we don't partition windows
            self.shift_size = 0
            self.window_size = min(self.input_resolution)
        assert 0 <= self.shift_size < self.window_size, "shift_size must in 0-window_size"

        self.norm1 = norm_layer(dim)
        self.attn = WindowAttention(
            dim,
            num_heads=num_heads,
            head_dim=head_dim,
            window_size=to_2tuple(self.window_size),
            qkv_bias=qkv_bias,
            attn_drop=attn_drop,
            proj_drop=proj_drop, 
            drop_path=drop_path, 
        )
        
        
        #self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()         
        
        # keep dp_attn = 0.1 ./3 for swin-small and ./2 for -tiny 
        # ./3 swin-small:  83.50x
        if drop_path < 0.25:
            self.dp_attn = DropPath(drop_path) if drop_path > 0. else nn.Identity() 
        else:
            self.dp_attn = DropPath(drop_path / 3.) 
        
        self.dp_ffn = DropPath(drop_path) if drop_path > 0. else nn.Identity() 
        
        
        
        
        
        self.norm2 = norm_layer(dim)
        self.mlp = Mlp(
            in_features=dim,
            hidden_features=int(dim * mlp_ratio),
            act_layer=act_layer,
            drop=proj_drop,
            drop_path = drop_path,
        )
        
        
        
        if self.shift_size > 0:
            # calculate attention mask for SW-MSA
            H, W = self.input_resolution
            img_mask = torch.zeros((1, H, W, 1))  # 1 H W 1
            cnt = 0
            for h in (
                    slice(0, -self.window_size),
                    slice(-self.window_size, -self.shift_size),
                    slice(-self.shift_size, None)):
                for w in (
                        slice(0, -self.window_size),
                        slice(-self.window_size, -self.shift_size),
                        slice(-self.shift_size, None)):
                    img_mask[:, h, w, :] = cnt
                    cnt += 1
            mask_windows = window_partition(img_mask, self.window_size)  # num_win, window_size, window_size, 1
            mask_windows = mask_windows.view(-1, self.window_size * self.window_size)
            attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
            attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))
        else:
            attn_mask = None
        self.register_buffer("attn_mask", attn_mask)

    
    def no_weight_decay(self):
        return {'gamma_attn', 'gamma_ffn'}
        

    
    def forward(self, x):
        B, H, W, C = x.shape
        _assert(H == self.input_resolution[0], "input feature has wrong size")
        _assert(W == self.input_resolution[1], "input feature has wrong size")

        shortcut = x
        x = self.norm1(x)
        
        # cyclic shift
        if self.shift_size > 0:
            shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
        else:
            shifted_x = x

        # partition windows
        x_windows = window_partition(shifted_x, self.window_size)  # num_win*B, window_size, window_size, C
        x_windows = x_windows.view(-1, self.window_size * self.window_size, C)  # num_win*B, window_size*window_size, C

        # W-MSA/SW-MSA
        attn_windows = self.attn(x_windows, mask=self.attn_mask)  # num_win*B, window_size*window_size, C

        # merge windows
        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C)
        shifted_x = window_reverse(attn_windows, self.window_size, H, W)  # B H' W' C
        
        # reverse cyclic shift
        if self.shift_size > 0:
            x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        else:
            x = shifted_x
            
        # FFN
        x = shortcut + self.dp_attn(x)

        #x = x.reshape(B, -1, C)
        x = x + self.dp_ffn(self.mlp(self.norm2(x)))
        #x = x.reshape(B, H, W, C)
        return x















class PatchMerging(nn.Module):
    """ Patch Merging Layer.
    """

    def __init__(
            self,
            dim: int,
            out_dim: Optional[int] = None,
            norm_layer: Callable = nn.LayerNorm,
    ):
        """
        Args:
            dim: Number of input channels.
            out_dim: Number of output channels (or 2 * dim if None)
            norm_layer: Normalization layer.
        """
        super().__init__()
        self.dim = dim
        self.out_dim = out_dim or 2 * dim
        self.norm = norm_layer(4 * dim)
        self.reduction = nn.Linear(4 * dim, self.out_dim, bias=False)

    def forward(self, x):
        B, H, W, C = x.shape
        _assert(H % 2 == 0, f"x height ({H}) is not even.")
        _assert(W % 2 == 0, f"x width ({W}) is not even.")
        x = x.reshape(B, H // 2, 2, W // 2, 2, C).permute(0, 1, 3, 4, 2, 5).flatten(3)
        x = self.norm(x)
        x = self.reduction(x)
        return x









class SwinTransformerStage(nn.Module):
    """ A basic Swin Transformer layer for one stage.
    """

    def __init__(
            self,
            dim: int,
            out_dim: int,
            input_resolution: Tuple[int, int],
            depth: int,
            downsample: bool = True,
            num_heads: int = 4,
            head_dim: Optional[int] = None,
            window_size: _int_or_tuple_2_t = 7,
            mlp_ratio: float = 4.,
            qkv_bias: bool = True,
            proj_drop: float = 0.,
            attn_drop: float = 0.,
            drop_path: Union[List[float], float] = 0.,
            norm_layer: Callable = nn.LayerNorm,
    ):
        """
        Args:
            dim: Number of input channels.
            input_resolution: Input resolution.
            depth: Number of blocks.
            downsample: Downsample layer at the end of the layer.
            num_heads: Number of attention heads.
            head_dim: Channels per head (dim // num_heads if not set)
            window_size: Local window size.
            mlp_ratio: Ratio of mlp hidden dim to embedding dim.
            qkv_bias: If True, add a learnable bias to query, key, value.
            proj_drop: Projection dropout rate.
            attn_drop: Attention dropout rate.
            drop_path: Stochastic depth rate.
            norm_layer: Normalization layer.
        """
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.output_resolution = tuple(i // 2 for i in input_resolution) if downsample else input_resolution
        self.depth = depth
        self.grad_checkpointing = False

        # patch merging layer
        if downsample:
            self.downsample = PatchMerging(
                dim=dim,
                out_dim=out_dim,
                norm_layer=norm_layer,
            )
        else:
            assert dim == out_dim
            self.downsample = nn.Identity()

        # build blocks
        self.blocks = nn.Sequential(*[
            SwinTransformerBlock(
                dim=out_dim,
                input_resolution=self.output_resolution,
                num_heads=num_heads,
                head_dim=head_dim,
                window_size=window_size,
                shift_size=0 if (i % 2 == 0) else window_size // 2,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                proj_drop=proj_drop,
                attn_drop=attn_drop,
                drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                norm_layer=norm_layer,
            )
            for i in range(depth)])

    def forward(self, x):
        x = self.downsample(x)

        if self.grad_checkpointing and not torch.jit.is_scripting():
            x = checkpoint_seq(self.blocks, x)
        else:
            x = self.blocks(x)
        return x


class SwinTransformer(nn.Module):
    """ Swin Transformer

    A PyTorch impl of : `Swin Transformer: Hierarchical Vision Transformer using Shifted Windows`  -
          https://arxiv.org/pdf/2103.14030
    """

    def __init__(
            self,
            img_size: _int_or_tuple_2_t = 224,
            patch_size: int = 4,
            in_chans: int = 3,
            num_classes: int = 1000,
            global_pool: str = 'avg',
            embed_dim: int = 96,
            depths: Tuple[int, ...] = (2, 2, 6, 2),
            num_heads: Tuple[int, ...] = (3, 6, 12, 24),
            head_dim: Optional[int] = None,
            window_size: _int_or_tuple_2_t = 7,
            mlp_ratio: float = 4.,
            qkv_bias: bool = True,
            drop_rate: float = 0.,
            proj_drop_rate: float = 0.,
            attn_drop_rate: float = 0.,
            drop_path_rate: float = 0.1,
            norm_layer: Union[str, Callable] = nn.LayerNorm,
            weight_init: str = '',
            **kwargs,
    ):
        """
        Args:
            img_size: Input image size.
            patch_size: Patch size.
            in_chans: Number of input image channels.
            num_classes: Number of classes for classification head.
            embed_dim: Patch embedding dimension.
            depths: Depth of each Swin Transformer layer.
            num_heads: Number of attention heads in different layers.
            head_dim: Dimension of self-attention heads.
            window_size: Window size.
            mlp_ratio: Ratio of mlp hidden dim to embedding dim.
            qkv_bias: If True, add a learnable bias to query, key, value.
            drop_rate: Dropout rate.
            attn_drop_rate (float): Attention dropout rate.
            drop_path_rate (float): Stochastic depth rate.
            norm_layer (nn.Module): Normalization layer.
        """
        super().__init__()
        assert global_pool in ('', 'avg')
        self.num_classes = num_classes
        self.global_pool = global_pool
        self.output_fmt = 'NHWC'

        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        self.num_features = int(embed_dim * 2 ** (self.num_layers - 1))
        self.feature_info = []

        if not isinstance(embed_dim, (tuple, list)):
            embed_dim = [int(embed_dim * 2 ** i) for i in range(self.num_layers)]

        # split image into non-overlapping patches
        self.patch_embed = PatchEmbed(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim[0],
            norm_layer=norm_layer,
            output_fmt='NHWC',
        )
        self.patch_grid = self.patch_embed.grid_size

        # build layers
        head_dim = to_ntuple(self.num_layers)(head_dim)
        window_size = to_ntuple(self.num_layers)(window_size)
        mlp_ratio = to_ntuple(self.num_layers)(mlp_ratio)
        dpr = [x.tolist() for x in torch.linspace(0, drop_path_rate, sum(depths)).split(depths)]
        layers = []
        in_dim = embed_dim[0]
        scale = 1
        for i in range(self.num_layers):
            out_dim = embed_dim[i]
            layers += [SwinTransformerStage(
                dim=in_dim,
                out_dim=out_dim,
                input_resolution=(
                    self.patch_grid[0] // scale,
                    self.patch_grid[1] // scale
                ),
                depth=depths[i],
                downsample=i > 0,
                num_heads=num_heads[i],
                head_dim=head_dim[i],
                window_size=window_size[i],
                mlp_ratio=mlp_ratio[i],
                qkv_bias=qkv_bias,
                proj_drop=proj_drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=dpr[i],
                norm_layer=norm_layer,
            )]
            in_dim = out_dim
            if i > 0:
                scale *= 2
            self.feature_info += [dict(num_chs=out_dim, reduction=4 * scale, module=f'layers.{i}')]
        self.layers = nn.Sequential(*layers)

        self.norm = norm_layer(self.num_features)
        self.head = ClassifierHead(
            self.num_features,
            num_classes,
            pool_type=global_pool,
            drop_rate=drop_rate,
            input_fmt=self.output_fmt,
        )
        if weight_init != 'skip':
            self.init_weights(weight_init)

    @torch.jit.ignore
    def init_weights(self, mode=''):
        assert mode in ('jax', 'jax_nlhb', 'moco', '')
        head_bias = -math.log(self.num_classes) if 'nlhb' in mode else 0.
        named_apply(get_init_weights_vit(mode, head_bias=head_bias), self)
    
    
    @torch.jit.ignore
    def no_weight_decay(self):
        nwd = set()
        for n, _ in self.named_parameters():
            if 'relative_position_bias_table' in n:
                nwd.add(n)
        return nwd
    
    
    """
    @torch.jit.ignore
    def no_weight_decay(self):
        nwd = set()
        for n, _ in self.named_parameters():
            if (
                'relative_position_bias_table' in n
                or n.endswith('.ka.weight')
                or n.endswith('.kv.weight')
                or n.endswith('.temp.weight')
                or n.endswith('.magk.weight')
            ):
                nwd.add(n)
        return nwd
    """
    
    
    @torch.jit.ignore
    def group_matcher(self, coarse=False):
        return dict(
            stem=r'^patch_embed',  # stem and embed
            blocks=r'^layers\.(\d+)' if coarse else [
                (r'^layers\.(\d+).downsample', (0,)),
                (r'^layers\.(\d+)\.\w+\.(\d+)', None),
                (r'^norm', (99999,)),
            ]
        )

    @torch.jit.ignore
    def set_grad_checkpointing(self, enable=True):
        for l in self.layers:
            l.grad_checkpointing = enable

    @torch.jit.ignore
    def get_classifier(self):
        return self.head.fc

    def reset_classifier(self, num_classes, global_pool=None):
        self.num_classes = num_classes
        self.head.reset(num_classes, pool_type=global_pool)

    def forward_features(self, x):
        x = self.patch_embed(x)
        x = self.layers(x)
        x = self.norm(x)
        return x

    def forward_head(self, x, pre_logits: bool = False):
        return self.head(x, pre_logits=True) if pre_logits else self.head(x)

    def forward(self, x):
        x = self.forward_features(x)
        x = self.forward_head(x)
        return x


def checkpoint_filter_fn(state_dict, model):
    """ convert patch embedding weight from manual patchify + linear proj to conv"""
    if 'head.fc.weight' in state_dict:
        return state_dict
    import re
    out_dict = {}
    state_dict = state_dict.get('model', state_dict)
    state_dict = state_dict.get('state_dict', state_dict)
    for k, v in state_dict.items():
        k = re.sub(r'layers.(\d+).downsample', lambda x: f'layers.{int(x.group(1)) + 1}.downsample', k)
        k = k.replace('head.', 'head.fc.')
        out_dict[k] = v
    return out_dict


def _create_swin_transformer(variant, pretrained=False, **kwargs):
    default_out_indices = tuple(i for i, _ in enumerate(kwargs.get('depths', (1, 1, 3, 1))))
    out_indices = kwargs.pop('out_indices', default_out_indices)

    model = build_model_with_cfg(
        SwinTransformer, variant, pretrained,
        pretrained_filter_fn=checkpoint_filter_fn,
        feature_cfg=dict(flatten_sequential=True, out_indices=out_indices),
        **kwargs)

    return model


def _cfg(url='', **kwargs):
    return {
        'url': url,
        'num_classes': 1000, 'input_size': (3, 224, 224), 'pool_size': (7, 7),
        'crop_pct': .9, 'interpolation': 'bicubic', 'fixed_input_size': True,
        'mean': IMAGENET_DEFAULT_MEAN, 'std': IMAGENET_DEFAULT_STD,
        'first_conv': 'patch_embed.proj', 'classifier': 'head.fc',
        'license': 'mit', **kwargs
    }


default_cfgs = generate_default_cfgs({
    'swin_small_patch4_window7_224.ms_in22k_ft_in1k': _cfg(
        hf_hub_id='timm/',
        url='https://github.com/SwinTransformer/storage/releases/download/v1.0.8/swin_small_patch4_window7_224_22kto1k_finetune.pth', ),
    'swin_base_patch4_window7_224.ms_in22k_ft_in1k': _cfg(
        hf_hub_id='timm/',
        url='https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_base_patch4_window7_224_22kto1k.pth',),
    'swin_base_patch4_window12_384.ms_in22k_ft_in1k': _cfg(
        hf_hub_id='timm/',
        url='https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_base_patch4_window12_384_22kto1k.pth',
        input_size=(3, 384, 384), pool_size=(12, 12), crop_pct=1.0),
    'swin_large_patch4_window7_224.ms_in22k_ft_in1k': _cfg(
        hf_hub_id='timm/',
        url='https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_large_patch4_window7_224_22kto1k.pth',),
    'swin_large_patch4_window12_384.ms_in22k_ft_in1k': _cfg(
        hf_hub_id='timm/',
        url='https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_large_patch4_window12_384_22kto1k.pth',
        input_size=(3, 384, 384), pool_size=(12, 12), crop_pct=1.0),

    'swin_tiny_patch4_window7_224.ms_in1k': _cfg(
        hf_hub_id='timm/',
        url='https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_tiny_patch4_window7_224.pth',),
    'swin_small_patch4_window7_224.ms_in1k': _cfg(
        hf_hub_id='timm/',
        url='https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_small_patch4_window7_224.pth',),
    'swin_base_patch4_window7_224.ms_in1k': _cfg(
        hf_hub_id='timm/',
        url='https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_base_patch4_window7_224.pth',),
    'swin_base_patch4_window12_384.ms_in1k': _cfg(
        hf_hub_id='timm/',
        url='https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_base_patch4_window12_384.pth',
        input_size=(3, 384, 384), pool_size=(12, 12), crop_pct=1.0),

    # tiny 22k pretrain is worse than 1k, so moved after (untagged priority is based on order)
    'swin_tiny_patch4_window7_224.ms_in22k_ft_in1k': _cfg(
        hf_hub_id='timm/',
        url='https://github.com/SwinTransformer/storage/releases/download/v1.0.8/swin_tiny_patch4_window7_224_22kto1k_finetune.pth',),

    'swin_tiny_patch4_window7_224.ms_in22k': _cfg(
        hf_hub_id='timm/',
        url='https://github.com/SwinTransformer/storage/releases/download/v1.0.8/swin_tiny_patch4_window7_224_22k.pth',
        num_classes=21841),
    'swin_small_patch4_window7_224.ms_in22k': _cfg(
        hf_hub_id='timm/',
        url='https://github.com/SwinTransformer/storage/releases/download/v1.0.8/swin_small_patch4_window7_224_22k.pth',
        num_classes=21841),
    'swin_base_patch4_window7_224.ms_in22k': _cfg(
        hf_hub_id='timm/',
        url='https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_base_patch4_window7_224_22k.pth',
        num_classes=21841),
    'swin_base_patch4_window12_384.ms_in22k': _cfg(
        hf_hub_id='timm/',
        url='https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_base_patch4_window12_384_22k.pth',
        input_size=(3, 384, 384), pool_size=(12, 12), crop_pct=1.0, num_classes=21841),
    'swin_large_patch4_window7_224.ms_in22k': _cfg(
        hf_hub_id='timm/',
        url='https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_large_patch4_window7_224_22k.pth',
        num_classes=21841),
    'swin_large_patch4_window12_384.ms_in22k': _cfg(
        hf_hub_id='timm/',
        url='https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_large_patch4_window12_384_22k.pth',
        input_size=(3, 384, 384), pool_size=(12, 12), crop_pct=1.0, num_classes=21841),

    'swin_s3_tiny_224.ms_in1k': _cfg(
        hf_hub_id='timm/',
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/s3_t-1d53f6a8.pth'),
    'swin_s3_small_224.ms_in1k': _cfg(
        hf_hub_id='timm/',
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/s3_s-3bb4c69d.pth'),
    'swin_s3_base_224.ms_in1k': _cfg(
        hf_hub_id='timm/',
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/s3_b-a1e95db4.pth'),
})



@register_model
def swin_min_csan(pretrained=False, **kwargs) -> SwinTransformer:
    """ @ 224x224, trained ImageNet-1k
    """
    model_args = dict(patch_size=4, window_size=7, embed_dim=96, depths=(1, 1, 1, 1), num_heads=(3, 6, 12, 24))
    return _create_swin_transformer(
        'swin_tiny_patch4_window7_224', pretrained=pretrained, **dict(model_args, **kwargs))



@register_model
def swin_micro_csan(pretrained=False, **kwargs) -> SwinTransformer:
    """ @ 224x224, trained ImageNet-1k
    """
    model_args = dict(patch_size=4, window_size=7, embed_dim=96, depths=(1, 2, 2, 2), num_heads=(3, 6, 12, 24))
    return _create_swin_transformer(
        'swin_tiny_patch4_window7_224', pretrained=pretrained, **dict(model_args, **kwargs))



@register_model
def swin_tiny_slim_csan(pretrained=False, **kwargs) -> SwinTransformer:
    """ @ 224x224, trained ImageNet-1k
    """
    model_args = dict(patch_size=4, window_size=7, embed_dim=64, depths=(2, 2, 6, 2), num_heads=(2, 4, 8, 16))
    return _create_swin_transformer(
        'swin_tiny_patch4_window7_224', pretrained=pretrained, **dict(model_args, **kwargs))



@register_model
def swin_tiny_csan(pretrained=False, **kwargs) -> SwinTransformer:
    """ @ 224x224, trained ImageNet-1k
    """
    model_args = dict(patch_size=4, window_size=7, embed_dim=96, depths=(2, 2, 6, 2), num_heads=(3, 6, 12, 24))
    return _create_swin_transformer(
        'swin_tiny_patch4_window7_224', pretrained=pretrained, **dict(model_args, **kwargs))



@register_model
def swin_small_csan(pretrained=False, **kwargs) -> SwinTransformer:
    """ @ 224x224
    """
    model_args = dict(patch_size=4, window_size=7, embed_dim=96, depths=(2, 2, 18, 2), num_heads=(3, 6, 12, 24))
    return _create_swin_transformer(
        'swin_small_patch4_window7_224', pretrained=pretrained, **dict(model_args, **kwargs))



@register_model
def swin_base_csan(pretrained=False, **kwargs) -> SwinTransformer:
    """ @ 224x224
    """
    model_args = dict(patch_size=4, window_size=7, embed_dim=128, depths=(2, 2, 18, 2), num_heads=(4, 8, 16, 32))
    return _create_swin_transformer(
        'swin_base_patch4_window7_224', pretrained=pretrained, **dict(model_args, **kwargs))

