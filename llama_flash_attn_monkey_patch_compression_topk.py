from typing import List, Optional, Tuple

import torch
from torch import nn
import math
import numpy as np

import transformers
from transformers.models.llama.modeling_llama import apply_rotary_pos_emb

from einops import rearrange

from flash_attn.flash_attn_interface import flash_attn_varlen_qkvpacked_func
from flash_attn.bert_padding import unpad_input, pad_input

global_static_ratio = []
record_static_ratio = []
global_dynamic_ratio = []

def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        # attn_dynamic_mask = None,  # 619
        layer_idx = None,
        # not_all_zeros_cols_idx = None,  # 812
) -> Tuple[torch.Tensor, Optional[torch.Tensor],
           Optional[Tuple[torch.Tensor]]]:
    """Input shape: Batch x Time x Channel

    attention_mask: [bsz, q_len]
    """
    
    bsz, q_len, _ = hidden_states.size()
    not_all_zeros_cols_idx, attn_dynamic_mask = None, None

    query_states = self.q_proj(hidden_states).view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
    key_states = self.k_proj(hidden_states).view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
    value_states = self.v_proj(hidden_states).view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
    # [bsz, q_len, nh, hd]
    # [bsz, nh, q_len, hd]

    kv_seq_len = key_states.shape[-2]
    # assert past_key_value is None, "past_key_value is not supported"
    if past_key_value is not None:
        kv_seq_len += past_key_value[0].shape[-2]

    cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)
    query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, position_ids)
    # [bsz, nh, t, hd]
    # assert not output_attentions, "output_attentions is not supported"
    # assert not use_cache, "use_cache is not supported"
    if past_key_value is not None:
        # reuse k, v, self_attention
        key_states = torch.cat([past_key_value[0], key_states], dim=2)
        value_states = torch.cat([past_key_value[1], value_states], dim=2)

    past_key_value = (key_states, value_states) if use_cache else None

    # Flash attention codes from
    # https://github.com/HazyResearch/flash-attention/blob/main/flash_attn/flash_attention.py

    # transform the data into the format required by flash attention
    
    # print(f'q: {query_states.shape}, k: {key_states.shape}, v: {value_states.shape}')

    # start decoding
    if query_states.shape[-2] == 1 or query_states.shape[-2] != key_states.shape[-2]:
        # decode token-by-token, do not use flash attention
        # in incremental state, do not use flash attention
        
        # 812
        if layer_idx != 0 and layer_idx != 1:
            if not_all_zeros_cols_idx is None:
                # print(self.not_all_zeros_cols_idx.shape)
                # print('it is None:', self.not_all_zeros_cols_idx.shape)
                generated_token_idx = torch.arange(self.attn_static_mask.shape[-1], key_states.shape[2]).to(self.not_all_zeros_cols_idx)
                generated_token_idx = generated_token_idx.unsqueeze(0).unsqueeze(0).unsqueeze(-1).expand(-1, query_states.shape[1], -1, -1)
                # print(generated_token_idx.shape)
                remain_token = torch.cat((self.not_all_zeros_cols_idx, generated_token_idx), dim=2)
                remain_token = remain_token.expand(-1, -1, -1, query_states.shape[-1])
                key_states_evict_static = torch.gather(key_states, 2, remain_token)
                value_states_evict_static = torch.gather(value_states, 2, remain_token)
                # print(key_states_evict_static.shape)
            elif not_all_zeros_cols_idx is not None:
                generated_token_idx = torch.arange(self.attn_static_mask.shape[-1], key_states.shape[2]).to(not_all_zeros_cols_idx)
                generated_token_idx = generated_token_idx.unsqueeze(0).unsqueeze(0).unsqueeze(-1).expand(-1, query_states.shape[1], -1, -1)
                remain_token = torch.cat((not_all_zeros_cols_idx, generated_token_idx), dim=2)
                remain_token = remain_token.expand(-1, -1, -1, query_states.shape[-1])
                key_states_evict_static = torch.gather(key_states, 2, remain_token)
                value_states_evict_static = torch.gather(value_states, 2, remain_token)
        else:
            key_states_evict_static = key_states
            value_states_evict_static = value_states

        # print(f'Verfify static >>> {key_states.shape}, {key_states_evict_static.shape}')
        
        if layer_idx != 0 and layer_idx != 1:
            # zwx: hierarchical grouping
            alpha = 0.6
            ratio1 = 0.6
            ratio2 = 0.12

            # 1st level: gs=32
            group_size1 = 32
            group_size2 = 16
            b_max, b_min, num_padding_token = group_key_min_max(key_states_evict_static, group_size=group_size1)
            sim = torch.sum((alpha * query_states * b_max + (1 - alpha) * query_states * b_min), dim=-1).unsqueeze(2)
            # sim = torch.sum(torch.max(query_states * b_max, query_states * b_min), dim=-1).unsqueeze(2)
            # key_states_evict_static_avg, num_padding_token = group_token_key(key_states_evict_static, group_size1, padding_value=0)
            # sim = torch.matmul(query_states.to(dtype=torch.float16), key_states_evict_static_avg.to(dtype=torch.float16).transpose(2, 3))
            
            new_mask = torch.full((1, query_states.shape[1], 1, 1), fill_value=torch.finfo(query_states.dtype).min).to(query_states.device)  # (1, H, 1, 1)
            self.base_dynamic_mask = torch.cat((self.base_dynamic_mask, new_mask), dim=-1)
            self.attn_dynamic_mask = self.base_dynamic_mask.clone()
            k = int(sim.shape[-1] * ratio1)
            topk = torch.topk(sim, k=k, dim=-1).indices
            # enforce to keep the first group
            sink_group_idx = torch.full((topk.shape[0], topk.shape[1], 1, 1), fill_value=0).to(topk)
            topk = torch.concat([topk, sink_group_idx], dim=-1)
            # enforce to keep the last group
            if key_states_evict_static.shape[-2] % group_size1 != 0:
                last_group_idx = torch.full((topk.shape[0], topk.shape[1], 1, 1), fill_value=b_max.shape[2]-1).to(topk)
                topk = torch.concat([topk, last_group_idx], dim=-1)
                
            # 2nd level: gs=16
            topk_gs16 = groupidx_to_groupidx(topk, gap=2)
            b_max, b_min, num_padding_token = group_key_min_max(key_states_evict_static, group_size=group_size2)
            # b_max, b_min = fake_quant(b_max, 3), fake_quant(b_min, 3)  # quant
            group_mask = torch.full((1, b_max.shape[1], 1, b_max.shape[2]), fill_value=torch.finfo(query_states.dtype).min).to(query_states.device)
            topk_gs16[topk_gs16 > group_mask.shape[-1] - 1] = group_mask.shape[-1] - 1
            group_mask.scatter_(-1, topk_gs16, 0)
            sim = torch.sum((alpha * query_states * b_max + (1 - alpha) * query_states * b_min), dim=-1).unsqueeze(2)
            # sim = torch.sum(torch.max(query_states * b_max, query_states * b_min), dim=-1).unsqueeze(2)
            # key_states_evict_static_avg, num_padding_token = group_token_key(key_states_evict_static, group_size2, padding_value=0)
            # sim = torch.matmul(query_states.to(dtype=torch.float16), key_states_evict_static_avg.to(dtype=torch.float16).transpose(2, 3))
            
            sim += group_mask
            topk = torch.topk(sim, k=int(sim.shape[-1] * ratio2), dim=-1).indices
            # enforce to keep the first group
            sink_group_idx = torch.full((topk.shape[0], topk.shape[1], 1, 1), fill_value=0).to(topk)
            topk = torch.concat([topk, sink_group_idx], dim=-1)
            # enforce to keep the last group
            if key_states_evict_static.shape[-2] % group_size2 != 0:
                last_group_idx = torch.full((topk.shape[0], topk.shape[1], 1, 1), fill_value=b_max.shape[2]-1).to(topk)
                topk = torch.concat([topk, last_group_idx], dim=-1)
            # set the dynamic mask
            topk = groupidx_to_tokenidx(topk, b_max.shape[2], group_size=group_size2)
            topk[topk > self.attn_dynamic_mask.shape[-1] - 1] -= num_padding_token
            self.attn_dynamic_mask.scatter_(-1, topk.unsqueeze(0).unsqueeze(2), 0)
            # print(self.attn_dynamic_mask.shape, topk.shape)
        
        # attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self.head_dim)
        attn_weights = torch.matmul(query_states, key_states_evict_static.transpose(2, 3)) / math.sqrt(self.head_dim)  # zwx

        # if attn_weights.size() != (bsz, self.num_heads, q_len, kv_seq_len):
        #     raise ValueError(
        #         f"Attention weights should be of size {(bsz * self.num_heads, q_len, kv_seq_len)}, but is"
        #         f" {attn_weights.size()}"
        #     )

        if attention_mask is not None:
            if attention_mask.size() != (bsz, 1, q_len, kv_seq_len):
                raise ValueError(
                    f"Attention mask should be of size {(bsz, 1, q_len, kv_seq_len)}, but is {attention_mask.size()}"
                )
            # zwx add
            if attn_weights.shape[-1] != attention_mask.shape[-1]:
                attention_mask = attention_mask[..., :attn_weights.shape[-1]]
            attn_weights = attn_weights + attention_mask
            attn_weights = torch.max(attn_weights, torch.tensor(torch.finfo(attn_weights.dtype).min))
        
        # 619
        if layer_idx != 0 and layer_idx != 1:
            if attn_dynamic_mask is None:
                attn_weights += self.attn_dynamic_mask
                # print(self.attn_dynamic_mask[0, 0, 0, :50].tolist())
            elif attn_dynamic_mask is not None:
                # print(attn_dynamic_mask[0, 0, 0, :50].tolist())
                # if attn_weights.shape[-1] > attn_dynamic_mask.shape[-1]:
                    # pad = torch.zeros(attn_weights.shape[0], attn_weights.shape[1], 1, attn_weights.shape[-1]-attn_dynamic_mask.shape[-1]).to(attn_weights.device)
                    # attn_dynamic_mask = torch.cat([attn_dynamic_mask, pad], dim=-1)
                # elif attn_weights.shape[-1] < attn_dynamic_mask.shape[-1]:
                    # attn_dynamic_mask = attn_dynamic_mask[..., :attn_weights.shape[-1]]
                attn_weights += attn_dynamic_mask
            

        # upcast attention to fp32
        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        # print(attn_weights.shape, attn_weights[0, 0, 0].tolist().count(0))
        # print(attn_weights[0, 1, 0, :100].tolist())
        # print(f'dynamic select: {(1 - attn_weights[0, 0, 0].tolist().count(0) / attn_weights.shape[-1]) * 100}%')

        # global_dynamic_ratio.append((1 - attn_weights[0, 0, 0].tolist().count(0) / attn_weights.shape[-1]) * 100)
        # if len(global_dynamic_ratio) == 32:
        #     print(f'Avg dynamic selected ratio across layers: {sum(global_dynamic_ratio) / 32}%')
        #     global_dynamic_ratio.clear()
        
        # attn_output = torch.matmul(attn_weights, value_states)
        attn_output = torch.matmul(attn_weights, value_states_evict_static)

        if attn_output.size() != (bsz, self.num_heads, q_len, self.head_dim):
            raise ValueError(
                f"`attn_output` should be of size {(bsz, self.num_heads, q_len, self.head_dim)}, but is"
                f" {attn_output.size()}"
            )

        attn_output = attn_output.transpose(1, 2)
        attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)

        attn_output = self.o_proj(attn_output)

        if not output_attentions:
            attn_weights = None

        return attn_output, attn_weights, past_key_value
        # return attn_output, attn_weights, past_key_value, self.attn_dynamic_mask, self.not_all_zeros_cols_idx  # 619
    # end decoding
    
    
    # start prefill
    qkv = torch.stack([query_states, key_states, value_states], dim=2)  # [bsz, nh, 3, q_len, hd]
    qkv = qkv.transpose(1, 3)  # [bsz, q_len, 3, nh, hd]
    # We have disabled _prepare_decoder_attention_mask in LlamaModel
    # the attention_mask should be the same as the key_padding_mask
    key_padding_mask = attention_mask
    
    # zwx: during prefill, only compute the last attention weights
    # if layer_idx != 0 or layer_idx != 1:
    if key_states.shape[2] > 24000:
        last_attn_weights = torch.matmul(query_states[:, :, -int(key_states.shape[2]*0.1):, :], key_states.transpose(-1, -2))
    else:
        last_attn_weights = torch.matmul(query_states[:, :, -int(key_states.shape[2]*0.2):, :], key_states.transpose(-1, -2))
    last_attn_weights = nn.functional.softmax(last_attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
    self.attn_static_mask = torch.zeros(1, 1, 1, last_attn_weights.shape[-1]).to(last_attn_weights.device)  # (1, 1, 1, T)
    
    # fix the static ratio
    last_attn_weights_sum = torch.sum(last_attn_weights, dim=2, keepdim=True)  # (1, H, 1, T)
    static_ratio = 0.5
    _, idx = torch.topk(last_attn_weights_sum, k=int(last_attn_weights_sum.shape[-1]*static_ratio), dim=-1)
    idx = idx.transpose(-1, -2)
    self.not_all_zeros_cols_idx = idx
    
    # 812
    if not_all_zeros_cols_idx is None:
        self.base_dynamic_mask = torch.full((1, last_attn_weights.shape[1], 1, self.not_all_zeros_cols_idx.shape[2]), fill_value=torch.finfo(last_attn_weights.dtype).min).to(last_attn_weights.device)
    else:
        self.base_dynamic_mask = torch.full((1, last_attn_weights.shape[1], 1, not_all_zeros_cols_idx.shape[2]), fill_value=torch.finfo(last_attn_weights.dtype).min).to(last_attn_weights.device)

    if not_all_zeros_cols_idx is None:
        global_static_ratio.append(100 * self.not_all_zeros_cols_idx.shape[2] / q_len)
    else:
        global_static_ratio.append(100 * not_all_zeros_cols_idx.shape[2] / q_len)
    
    if len(global_static_ratio) == 32:
        print(f'Avg static remained ratio across layers: {sum(global_static_ratio) / 32}%')
        record_static_ratio.append(sum(global_static_ratio) / 32)
        global_static_ratio.clear()
    # if 0 < len(record_static_ratio) <= 100 and len(record_static_ratio) % 5 == 0:
    #     print(f'Global avg static remained ratio across: {sum(record_static_ratio) / len(record_static_ratio)}%')

    if key_padding_mask is None:
        qkv = rearrange(qkv, 'b s ... -> (b s) ...')
        max_s = q_len
        cu_q_lens = torch.arange(0, (bsz + 1) * q_len, step=q_len, dtype=torch.int32,
                                 device=qkv.device)
        output = flash_attn_varlen_qkvpacked_func(
            qkv, cu_q_lens, max_s, 0.0,
            softmax_scale=None, causal=True
        )
        output = rearrange(output, '(b s) ... -> b s ...', b=bsz)
    else:
        nheads = qkv.shape[-2]
        x = rearrange(qkv, 'b s three h d -> b s (three h d)')
        x_unpad, indices, cu_q_lens, max_s = unpad_input(x, key_padding_mask)
        x_unpad = rearrange(x_unpad, 'nnz (three h d) -> nnz three h d', three=3, h=nheads)
        output_unpad = flash_attn_varlen_qkvpacked_func(
            x_unpad, cu_q_lens, max_s, 0.0,  # zwx
            softmax_scale=None, causal=True,
            # return_attn_probs=True  # zwx
        )
        # print(f'attn_probs: {output_unpad[2]}')
        # output_unpad = output_unpad[0]  # zwx
        output = rearrange(pad_input(rearrange(output_unpad, 'nnz h d -> nnz (h d)'),
                                     indices, bsz, q_len),
                           'b s (h d) -> b s h d', h=nheads)
    attn_output = self.o_proj(rearrange(output, 'b s h d -> b s (h d)'))

    return attn_output, None, past_key_value
    # return attn_output, None, past_key_value, None, self.not_all_zeros_cols_idx  # 619, 812
    # end prefill


def group_key_min_max(key_states, group_size):
    # if #tokens cannot be evenly divided by group_size, there should be padded with padding_value
    num_padding_token = (key_states.shape[2] + group_size - 1) // group_size * group_size - key_states.shape[2]
    padding_token = key_states[:, :, -group_size:-group_size+num_padding_token, :]
    key_states_padded = torch.cat((key_states, padding_token), dim=2)
    
    key_states_reshaped = key_states_padded.reshape(key_states.shape[0], key_states.shape[1], -1, group_size, key_states.shape[-1])  # (1, num_head, num_group, group_size, dim) # group_size must be put behind before mean

    b_max = torch.max(key_states_reshaped, dim=-2).values
    b_min = torch.min(key_states_reshaped, dim=-2).values

    return b_max, b_min, num_padding_token


def groupidx_to_tokenidx(idx, num_group, group_size):
    # idx: (1, num_head, 1, num_select_group)
    num_select_group = idx.shape[-1]
    num_select_token = num_select_group * group_size
    num_full_token = num_group * group_size
    full_idx = torch.repeat_interleave(torch.arange(num_full_token).reshape(num_group, group_size).unsqueeze(0), idx.shape[1], dim=0).to(idx)  # (num_head, num_group, group_size)
    groupidx = torch.repeat_interleave(idx.squeeze(0).transpose(-1, -2), group_size, dim=-1)  # (num_head, num_select_group, group_size)
    tokenidx = torch.gather(full_idx, 1, groupidx).reshape(idx.shape[1], -1)  # (num_head, num_select_group, group_size)

    return tokenidx


def groupidx_to_groupidx(idx, gap=2):
    if gap == 2:
        # idx: (1, H, 1, k)
        idx_expand = idx.unsqueeze(-1)  # (1, H, 1, k, 1)
        first = idx_expand * 2
        second = first + 1
        idx_new = torch.cat([first, second], dim=-1)  # (1, H, 1, k, 2)
        idx_new = idx_new.reshape(1, idx.shape[1], 1, -1)  # (1, H, 1, 2k)
    if gap == 4:
        # idx: (1, H, 1, k)
        idx_expand = idx.unsqueeze(-1)  # (1, H, 1, k, 1)
        first = idx_expand * 4
        second = first + 1
        third = first + 2
        fourth = first + 3
        idx_new = torch.cat([first, second], dim=-1)  # (1, H, 1, k, 2)
        idx_new = torch.cat([idx_new, third], dim=-1)  # (1, H, 1, k, 3)
        idx_new = torch.cat([idx_new, fourth], dim=-1)  # (1, H, 1, k, 4)
        idx_new = idx_new.reshape(1, idx.shape[1], 1, -1)  # (1, H, 1, 4k)
    
    return idx_new


def group_token_key(key_states, group_size, padding_value=0):
    # if #tokens cannot be evenly divided by group_size, there should be padded with padding_value
    num_padding_token = (key_states.shape[2] + group_size - 1) // group_size * group_size - key_states.shape[2]
    padding_token = torch.full((key_states.shape[0], key_states.shape[1], num_padding_token, key_states.shape[-1]), fill_value=padding_value).to(key_states.device)
    # padding_token = key_states[:, :, -group_size:-group_size+num_padding_token, :]
    key_states_padded = torch.cat((key_states, padding_token), dim=2)
    
    # compute the mean of each group
    key_states_reshaped = key_states_padded.reshape(key_states.shape[0], key_states.shape[1], -1, group_size, key_states.shape[-1])  # (1, num_head, num_group, group_size, dim) # group_size must be put behind before mean
    key_states_mean = torch.mean(key_states_reshaped, dim=-2, keepdim=False)

    return key_states_mean.half(), num_padding_token


def fake_quant(x, bit=8):
    min_val, max_val = x.min(), x.max()
    max_range = max(-min_val, max_val)
    qmin = - 2 ** (bit - 1)
    qmax = 2 ** (bit - 1) - 1
    
    scale = max_range / ((qmax - qmin) / 2)
    scale = max(scale, 1e-8)
    output = torch.clamp(torch.round(x / scale), qmin, qmax) * scale
    
    return output
    

# Disable the transformation of the attention mask in LlamaModel as the flash attention
# requires the attention mask to be the same as the key_padding_mask
def _prepare_decoder_attention_mask(self, attention_mask, input_shape,
                                    inputs_embeds, past_key_values_length):
    # [bsz, seq_len]
    if input_shape[-1] > 1 and past_key_values_length == 0:  # encode
        return attention_mask
    return transformers.models.bart.modeling_bart.BartDecoder._prepare_decoder_attention_mask(self, attention_mask,
                                                                                              input_shape,
                                                                                              inputs_embeds,
                                                                                              past_key_values_length)


def replace_llama_attn_with_flash_attn():
    print('use FlashAttention')
    transformers.models.llama.modeling_llama.LlamaModel._prepare_decoder_attention_mask = _prepare_decoder_attention_mask
    transformers.models.llama.modeling_llama.LlamaAttention.forward = forward
