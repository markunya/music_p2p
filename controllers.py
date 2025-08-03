import abc
import torch
import p2p_utils
import seq_aligner
from typing import Union, Tuple, Optional, Dict
import math

class LocalBlend:
    pass

class AttentionControl(abc.ABC):
    
    def step_callback(self, x_t):
        return x_t
    
    def between_steps(self):
        return
    
    @abc.abstractmethod
    def forward (self, attn_weight):
        raise NotImplementedError

    def __call__(self, attn_weight):
        h = attn_weight.shape[0]
        attn_weight = self.forward(attn_weight)
        self.cur_att_layer += 1
        if self.cur_att_layer == self.num_att_layers:
            self.cur_att_layer = 0
            self.cur_step += 1
            self.between_steps()
        return attn_weight
    
    def reset(self):
        self.cur_step = 0
        self.cur_att_layer = 0

    def __init__(self):
        self.cur_step = 0
        self.num_att_layers = -1
        self.cur_att_layer = 0

class EmptyControl(AttentionControl):
    
    def forward (self, attn_weight):
        return attn_weight
    
class AttentionStore(AttentionControl):
    def forward(self, attn_weight):
        self.step_store.append(attn_weight)
        return attn_weight

    def between_steps(self):
        if len(self.attention_store) == 0:
            self.attention_store = self.step_store
        else:
            for i in range(len(self.attention_store)):
                self.attention_store[i] += self.step_store[i]
        self.step_store = []

    def get_average_attention(self):
        average_attention = [item / self.cur_step for item in self.attention_store]
        return average_attention

    def reset(self):
        super(AttentionStore, self).reset()
        self.step_store = []
        self.attention_store = []

    def __init__(self):
        super(AttentionStore, self).__init__()
        self.step_store = []
        self.attention_store = []


# class AttentionControlEdit(AttentionStore, abc.ABC):
    
#     def step_callback(self, x_t):
#         if self.local_blend is not None:
#             x_t = self.local_blend(x_t, self.attention_store, self.cur_step)
#         return x_t
    
#     @abc.abstractmethod
#     def replace_cross_attention(self, attn_weight_base, attn_weight_replace):
#         raise NotImplementedError
    
#     def forward(self, attn_weight):
#         super(AttentionControlEdit, self).forward(attn_weight)
#         h = attn_weight.shape[0] // (self.batch_size)
#         attn_weight = attn_weight.reshape(self.batch_size, h, *attn_weight.shape[1:])
#         attn_weight_base, attn_weight_repalce = attn_weight[0], attn_weight[1:] # explain what is attn_weight[0] and attn_weight[1:] please

#         alpha_words = self.cross_replace_alpha[self.cur_step]
#         attn_weight_repalce_new = self.replace_cross_attention(attn_weight_base, attn_weight_repalce) * alpha_words + (1 - alpha_words) * attn_weight_repalce
#         attn_weight[1:] = attn_weight_repalce_new
#         attn_weight = attn_weight.reshape(self.batch_size * h, *attn_weight.shape[2:])

#         return attn_weight
    
#     def __init__(
#             self,
#             prompts,
#             tokenizer,
#             num_steps: int,
#             cross_replace_steps: Union[float, Tuple[float, float],Dict[str, Tuple[float, float]]],
#             self_replace_steps: Union[float, Tuple[float, float]],
#             local_blend: Optional[LocalBlend]
#         ):
#         super(AttentionControlEdit, self).__init__()
#         self.batch_size = len(prompts)
#         self.cross_replace_alpha = p2p_utils.get_time_words_attention_alpha(prompts, num_steps, cross_replace_steps, tokenizer)
#         if type(self_replace_steps) is float:
#             self_replace_steps = 0, self_replace_steps
#         self.num_self_replace = int(num_steps * self_replace_steps[0]), int(num_steps * self_replace_steps[1])
#         self.local_blend = local_blend

class AttentionControlEdit(AttentionStore, abc.ABC):
    """
    Simplified AttentionControl for P2P edit without temporal alpha blending.
    """
    def __init__(
        self,
        prompts,
        tokenizer,
        local_blend: Optional[LocalBlend] = None
    ):
        super().__init__()
        # batch_size = number of prompts (src + targets)
        self.batch_size = len(prompts)
        self.local_blend = local_blend

    def step_callback(self, x_t):
        if self.local_blend is not None:
            x_t = self.local_blend(x_t, self.attention_store, self.cur_step)
        return x_t

    @abc.abstractmethod
    def replace_cross_attention(self, attn_weight_base, attn_weight_replace):
        """
        Implement how to replace cross-attention weights.
        - attn_weight_base: [H, P, W_src]
        - attn_weight_replace: [B-1, H, P, W_src]
        Return: [B-1, H, P, W_tgt]
        """
        pass

    def forward(self, attn_weight):
        # store incoming attention
        super().forward(attn_weight)
        # split src vs targets
        base = attn_weight[0]           # [H, P, W_src]
        replaces = attn_weight[1:]      # [B-1, H, P, W_src]

        # replace all targets' attention with control
        new_replaces = self.replace_cross_attention(base, replaces)
        attn_weight[1:] = new_replaces

        return attn_weight

class AttentionReplaceLyrics(AttentionControlEdit):
    def __init__(
        self,
        prompts,
        tokenizer,
        num_steps: int,
        local_blend: Optional[LocalBlend] = None,
        eta_min: float = 0.0,
        eta_max: float = 1.0,
        step_start: Optional[int] = None,
        step_end: Optional[int] = None,
    ):
        super(AttentionReplaceLyrics, self).__init__(prompts, tokenizer, local_blend)
        self.step = None  # must be set externally at each step
        self.num_steps = num_steps
        self.mapper = seq_aligner.get_replacement_mapper(prompts, tokenizer)

        self.eta_min = eta_min
        self.eta_max = eta_max
        self.step_start = 0 if step_start is None else step_start
        self.step_end = num_steps if step_end is None else step_end

    def Sca(self):
        step = min(max(self.step, self.step_start), self.step_end)
        scale = 0.5 * (1 + math.cos(math.pi * (step - self.step_start) / (self.step_end - self.step_start)))
        return self.eta_min + (self.eta_max - self.eta_min) * scale

    def replace_cross_attention(self, attn_weight_base, attn_weight_replace):
        B, _, _, W_src = attn_weight_replace.shape
        _, L_src, L_tgt = self.mapper.shape
        T_text = W_src - L_src - 1
        W_tgt = 1 + T_text + L_tgt

        M_full = attn_weight_replace.new_zeros((B, W_src, W_tgt))
        idx = torch.arange(1 + T_text, device=M_full.device)
        M_full[:, idx, idx] = 1.0

        start_src = 1 + T_text
        start_tgt = 1 + T_text
        M_full[:, 
               start_src : start_src + L_src,
               start_tgt : start_tgt + L_tgt] = self.mapper

        attn_weight_replaced = torch.einsum('hpw,bwn->bhpn', attn_weight_base, M_full)

        alpha = self.Sca()
        return alpha * attn_weight_replaced + (1 - alpha) * attn_weight_base

        