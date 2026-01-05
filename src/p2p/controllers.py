import abc
import torch
import math
from typing import Union, Tuple, Optional, Dict
from tqdm import tqdm

import src.utils.seq_aligner as seq_aligner
from src.p2p.step_callback import StepCallbackBase

class AttentionControl(abc.ABC):
    def step_callback(self, x_t):
        return x_t
    
    def between_steps(self):
        return
    
    @abc.abstractmethod
    def forward (self, attn_weight):
        raise NotImplementedError

    def __call__(self, attn_weight):
        attn_weight = self.forward(attn_weight)
        return attn_weight
    
    def set_diffusion_step(self, step):
        self.between_steps()
        self._diffusion_step = step
    
    def __init__(
            self,
            num_diffusion_steps: int
    ):
        self._diffusion_step = 0  # must be set externally at each step through set_diffusion_step
        self.num_diffusion_steps = num_diffusion_steps
    
class AttentionStore(AttentionControl):
    def forward(self, attn_weight):
        self.step_store.append(attn_weight)
        return attn_weight

    def between_steps(self):
        if len(self.attention_store) == 0:
            self.attention_store = self.step_store
        else:
            for i in range(len(self.attention_store)):
                self.attention_store[i] = \
                    (1 - self.running_mean_coef) * self.attention_store[i] \
                    + self.running_mean_coef * self.step_store[i]
    
        self.step_store = []

    def reset(self):
        self.step_store = []
        self.attention_store = []

    def step_callback(self, x_t):
        if self.step_callback_ is not None:
            x_t = self.step_callback_(x_t, self.attention_store, self._diffusion_step)
        return x_t

    def __init__(
            self,
            num_diffusion_steps: int,
            step_callback: Optional[StepCallbackBase] = None,
            running_mean_coef: float = 0.8
    ):
        super().__init__(num_diffusion_steps)
        self.step_store = []
        self.attention_store = []
        self.running_mean_coef = running_mean_coef
        self.step_callback_ = step_callback

class AttentionControlEdit(AttentionStore, abc.ABC):
    def __init__(
        self,
        prompts,
        num_diffusion_steps: int,
        step_callback: Optional[StepCallbackBase] = None
    ):
        super().__init__(num_diffusion_steps, step_callback)
        # batch_size = number of prompts (src + targets)
        self.batch_size = len(prompts)

    @abc.abstractmethod
    def replace_cross_attention(self, attn_weight_base, attn_weight_replace):
        pass

    def forward(self, attn_weight):
        super().forward(attn_weight)
        base = attn_weight[0]
        start_replaces_idx = (attn_weight.shape[0] - 1) // 2 + 1
        replaces = attn_weight[start_replaces_idx:]

        new_replaces = self.replace_cross_attention(base, replaces)
        attn_weight[start_replaces_idx:] = new_replaces

        return attn_weight
    
class AttentionControlReplace(AttentionControlEdit, abc.ABC):
    def __init__(
        self,
        prompts,
        tokenizer,
        num_diffusion_steps: int,
        step_callback: Optional[StepCallbackBase] = None,
        eta_min: float = 0.0,
        eta_max: float = 1.0,
        diffusion_step_start: Optional[int] = None,
        diffusion_step_end: Optional[int] = None,
    ):
        super().__init__(prompts, num_diffusion_steps, step_callback)
        self.mapper, self.edit_mask = self._get_replacement_mapper(prompts, tokenizer)

        self.eta_min = eta_min
        self.eta_max = eta_max
        self.diffusion_step_start = 0 if diffusion_step_start is None else diffusion_step_start
        self.diffusion_step_end = num_diffusion_steps if diffusion_step_end is None else diffusion_step_end

    def _get_alpha_from_cosine_schedule(self):
        step = min(max(self._diffusion_step, self.diffusion_step_start), self.diffusion_step_end)
        denom = self.diffusion_step_end - self.diffusion_step_start
        if denom <= 0:
            raise ValueError("Someting strange: denom <= 0")
        scale = 0.5 * (1 + math.cos(math.pi * (step - self.diffusion_step_start) / denom))
        return self.eta_min + (self.eta_max - self.eta_min) * scale
    
    def _replace_cross_attention_impl(
        self,
        attn_weight_base,
        attn_weight_replace,
        mapper_pos_src,
        mapper_pos_tgt
    ):
        
        len_src_seq = attn_weight_base.shape[-1]
        len_tgt_seq = attn_weight_replace.shape[-1] 

        M_full = attn_weight_replace.new_zeros((self.batch_size - 1, len_src_seq, len_tgt_seq))
        idx = torch.arange(min(len_src_seq, len_tgt_seq), device=M_full.device)
        M_full[:, idx, idx] = 1.0
        M_full[:, 
               mapper_pos_src : mapper_pos_src + self.mapper.shape[-2],
               mapper_pos_tgt : mapper_pos_tgt + self.mapper.shape[-1]] = self.mapper
        
        edit_mask_full = torch.zeros(M_full.shape[0], M_full.shape[-1], device=M_full.device)
        edit_mask_full[:,mapper_pos_tgt:mapper_pos_tgt + self.edit_mask.shape[-1]] = self.edit_mask

        attn_weight_base_mapped = torch.einsum('hpw,bwn->bhpn', attn_weight_base, M_full)
        alpha = self._get_alpha_from_cosine_schedule()

        interpolation = alpha * attn_weight_replace + (1 - alpha) * attn_weight_base_mapped
        return interpolation * edit_mask_full + attn_weight_base_mapped * (1 - edit_mask_full)
    
    @abc.abstractmethod
    def _get_replacement_mapper(self, prompts, tokenizer):
        raise NotImplementedError

class AttentionReplaceLyrics(AttentionControlReplace):
    def _get_replacement_mapper(self, prompts, tokenizer):
        return seq_aligner.get_lyrics_replacement_mapper(prompts)

    def replace_cross_attention(self, attn_weight_base, attn_weight_replace):
        mapper_pos_src = attn_weight_base.shape[-1] - self.mapper.shape[-2]
        mapper_pos_tgt = attn_weight_replace.shape[-1] - self.mapper.shape[-2]
        return self._replace_cross_attention_impl(
            attn_weight_base,
            attn_weight_replace,
            mapper_pos_src,
            mapper_pos_tgt
        )

class AttentionReplaceTags(AttentionControlReplace):    
    def _get_replacement_mapper(self, prompts, tokenizer):
        return seq_aligner.get_tags_replacement_mapper(prompts, tokenizer)

    def replace_cross_attention(self, attn_weight_base, attn_weight_replace):
        return self._replace_cross_attention_impl(
            attn_weight_base,
            attn_weight_replace,
            1, 1
        )
        