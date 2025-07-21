import torch
import torch.nn.functional as F
from diffusers.models.attention_processor import Attention
from acestep.models.customer_attention_processor import CustomLiteLAProcessor2_0
from typing import Optional, Union, Tuple

class CustomLiteLAProcessorWithP2PController2_0(CustomLiteLAProcessor2_0):
    def __init__(self, controller):
        super().__init__()
        self.controller = controller
    
    def __call__(
        self,
        attn: Attention,
        hidden_states: torch.FloatTensor,
        encoder_hidden_states: torch.FloatTensor = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        rotary_freqs_cis: Union[torch.Tensor, Tuple[torch.Tensor]] = None,
        rotary_freqs_cis_cross: Union[torch.Tensor, Tuple[torch.Tensor]] = None,
        *args,
        **kwargs,
    ) -> torch.FloatTensor:
        hidden_states_len = hidden_states.shape[1]

        input_ndim = hidden_states.ndim
        if input_ndim == 4:
            batch_size, channel, height, width = hidden_states.shape
            hidden_states = hidden_states.view(
                batch_size, channel, height * width
            ).transpose(1, 2)
        if encoder_hidden_states is not None:
            context_input_ndim = encoder_hidden_states.ndim
            if context_input_ndim == 4:
                batch_size, channel, height, width = encoder_hidden_states.shape
                encoder_hidden_states = encoder_hidden_states.view(
                    batch_size, channel, height * width
                ).transpose(1, 2)

        batch_size = hidden_states.shape[0]

        # `sample` projections.
        dtype = hidden_states.dtype
        query = attn.to_q(hidden_states)
        key = attn.to_k(hidden_states)
        value = attn.to_v(hidden_states)

        # `context` projections.
        has_encoder_hidden_state_proj = (
            hasattr(attn, "add_q_proj")
            and hasattr(attn, "add_k_proj")
            and hasattr(attn, "add_v_proj")
        )
        if encoder_hidden_states is not None and has_encoder_hidden_state_proj:
            encoder_hidden_states_query_proj = attn.add_q_proj(encoder_hidden_states)
            encoder_hidden_states_key_proj = attn.add_k_proj(encoder_hidden_states)
            encoder_hidden_states_value_proj = attn.add_v_proj(encoder_hidden_states)

            # attention
            if not attn.is_cross_attention:
                query = torch.cat([query, encoder_hidden_states_query_proj], dim=1)
                key = torch.cat([key, encoder_hidden_states_key_proj], dim=1)
                value = torch.cat([value, encoder_hidden_states_value_proj], dim=1)
            else:
                query = hidden_states
                key = encoder_hidden_states
                value = encoder_hidden_states

        inner_dim = key.shape[-1]
        head_dim = inner_dim // attn.heads

        query = query.transpose(-1, -2).reshape(batch_size, attn.heads, head_dim, -1)
        key = (
            key.transpose(-1, -2)
            .reshape(batch_size, attn.heads, head_dim, -1)
            .transpose(-1, -2)
        )
        value = value.transpose(-1, -2).reshape(batch_size, attn.heads, head_dim, -1)

        # RoPE需要 [B, H, S, D] 输入
        # 此时 query是 [B, H, D, S], 需要转成 [B, H, S, D] 才能应用RoPE
        query = query.permute(0, 1, 3, 2)  # [B, H, S, D]  (从 [B, H, D, S])

        # Apply query and key normalization if needed
        if attn.norm_q is not None:
            query = attn.norm_q(query)
        if attn.norm_k is not None:
            key = attn.norm_k(key)

        # Apply RoPE if needed
        if rotary_freqs_cis is not None:
            query = self.apply_rotary_emb(query, rotary_freqs_cis)
            if not attn.is_cross_attention:
                key = self.apply_rotary_emb(key, rotary_freqs_cis)
            elif rotary_freqs_cis_cross is not None and has_encoder_hidden_state_proj:
                key = self.apply_rotary_emb(key, rotary_freqs_cis_cross)

        # 此时 query是 [B, H, S, D]，需要还原成 [B, H, D, S]
        query = query.permute(0, 1, 3, 2)  # [B, H, D, S]

        if attention_mask is not None:
            # attention_mask: [B, S] -> [B, 1, S, 1]
            attention_mask = attention_mask[:, None, :, None].to(
                key.dtype
            )  # [B, 1, S, 1]
            query = query * attention_mask.permute(
                0, 1, 3, 2
            )  # [B, H, S, D] * [B, 1, S, 1]
            if not attn.is_cross_attention:
                key = (
                    key * attention_mask
                )  # key: [B, h, S, D] 与 mask [B, 1, S, 1] 相乘
                value = value * attention_mask.permute(
                    0, 1, 3, 2
                )  # 如果 value 是 [B, h, D, S]，那么需调整mask以匹配S维度

        if (
            attn.is_cross_attention
            and encoder_attention_mask is not None
            and has_encoder_hidden_state_proj
        ):
            encoder_attention_mask = encoder_attention_mask[:, None, :, None].to(
                key.dtype
            )  # [B, 1, S_enc, 1]
            # 此时 key: [B, h, S_enc, D], value: [B, h, D, S_enc]
            key = key * encoder_attention_mask  # [B, h, S_enc, D] * [B, 1, S_enc, 1]
            value = value * encoder_attention_mask.permute(
                0, 1, 3, 2
            )  # [B, h, D, S_enc] * [B, 1, 1, S_enc]

        query = self.kernel_func(query)
        key = self.kernel_func(key)

        query, key, value = query.float(), key.float(), value.float()

        value = F.pad(value, (0, 0, 0, 1), mode="constant", value=self.pad_val)

        if self.controller is not None: # the only difference with base class here
            hidden_states = self.controller(query, key, value) 
        else:
            vk = torch.matmul(value, key)

            hidden_states = torch.matmul(vk, query)

        if hidden_states.dtype in [torch.float16, torch.bfloat16]:
            hidden_states = hidden_states.float()

        hidden_states = hidden_states[:, :, :-1] / (hidden_states[:, :, -1:] + self.eps)

        hidden_states = hidden_states.view(
            batch_size, attn.heads * head_dim, -1
        ).permute(0, 2, 1)

        hidden_states = hidden_states.to(dtype)
        if encoder_hidden_states is not None:
            encoder_hidden_states = encoder_hidden_states.to(dtype)

        # Split the attention outputs.
        if (
            encoder_hidden_states is not None
            and not attn.is_cross_attention
            and has_encoder_hidden_state_proj
        ):
            hidden_states, encoder_hidden_states = (
                hidden_states[:, :hidden_states_len],
                hidden_states[:, hidden_states_len:],
            )

        # linear proj
        hidden_states = attn.to_out[0](hidden_states)
        # dropout
        hidden_states = attn.to_out[1](hidden_states)
        if (
            encoder_hidden_states is not None
            and not attn.context_pre_only
            and not attn.is_cross_attention
            and hasattr(attn, "to_add_out")
        ):
            encoder_hidden_states = attn.to_add_out(encoder_hidden_states)

        if input_ndim == 4:
            hidden_states = hidden_states.transpose(-1, -2).reshape(
                batch_size, channel, height, width
            )
        if encoder_hidden_states is not None and context_input_ndim == 4:
            encoder_hidden_states = encoder_hidden_states.transpose(-1, -2).reshape(
                batch_size, channel, height, width
            )

        if torch.get_autocast_gpu_dtype() == torch.float16:
            hidden_states = hidden_states.clip(-65504, 65504)
            if encoder_hidden_states is not None:
                encoder_hidden_states = encoder_hidden_states.clip(-65504, 65504)

        return hidden_states, encoder_hidden_states
