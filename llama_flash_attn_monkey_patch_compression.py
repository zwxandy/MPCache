from typing import List, Optional, Tuple

import torch
from torch import nn
import math
import transformers
from transformers.models.llama.modeling_llama import apply_rotary_pos_emb
from einops import rearrange
from utils import group_key_min_max, groupidx_to_tokenidx, groupidx_to_groupidx

from flash_attn.flash_attn_interface import flash_attn_varlen_qkvpacked_func
from flash_attn.bert_padding import unpad_input, pad_input

static_ratio = []
record_static_ratio = []
dynamic_ratio = []
dataset = None
is_print_static, is_print_dynamic = True, True

LARGE_TOKEN_NUM = 24000
REPORT_STEP = 30
skip_first2layers = False
if skip_first2layers:
    print('⚠️ Skip the first 2 layers.')
    skip_layer_idx = [0, 1]  # skip the first 2 layers
    layer_num = 30
else:
    print('⚠️ Do not skip any layer.')
    skip_layer_idx = []  # do not skip any layer
    layer_num = 32

def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        layer_idx = None,
) -> Tuple[torch.Tensor, Optional[torch.Tensor],
           Optional[Tuple[torch.Tensor]]]:
    """Input shape: Batch x Time x Channel

    attention_mask: [bsz, q_len]
    """
    global dataset, is_print_static, is_print_dynamic
    ic_token_idx, attn_dynamic_mask = None, None
    
    bsz, q_len, _ = hidden_states.size()

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
        
        if layer_idx not in skip_layer_idx:
            generated_token_idx = torch.arange(self.attn_static_mask.shape[-1], key_states.shape[2]).to(self.ic_token_idx)
            remain_token = torch.cat((self.ic_token_idx.squeeze(), generated_token_idx), dim=0)
            key_states_evict_static = key_states[:, :, remain_token, :]
            value_states_evict_static = value_states[:, :, remain_token, :]
        else:
            key_states_evict_static = key_states
            value_states_evict_static = value_states
        
        if layer_idx not in skip_layer_idx:
            # hierarchical clustering
            alpha = 0.6
            ratio1 = 0.5  # 1st level selection ratio
            ratio2 = 0.2  # overall dynamic selection ratio
            cluster_size1 = 32  # 1st level: s=32
            cluster_size2 = 16  # 2st level: s=16
            if is_print_dynamic:
                print(f'⚙️ Selection ratio at 1st level: {ratio1 * 100:.1f}%')
                print(f'⚙️ Overall dynamic selection ratio: {ratio2 * 100:.1f}%')
                is_print_dynamic = False
            b_max, b_min, num_padding_token = group_key_min_max(key_states_evict_static, group_size=cluster_size1)
            sim = torch.sum((alpha * query_states * b_max + (1 - alpha) * query_states * b_min), dim=-1).unsqueeze(2)
            
            new_mask = torch.full((1, query_states.shape[1], 1, 1), fill_value=torch.finfo(query_states.dtype).min).to(query_states.device)  # (1, H, 1, 1)
            self.base_dynamic_mask = torch.cat((self.base_dynamic_mask, new_mask), dim=-1)
            self.attn_dynamic_mask = self.base_dynamic_mask.clone()
            k = int(sim.shape[-1] * ratio1)
            topk = torch.topk(sim, k=k, dim=-1).indices
            # keep the first group
            sink_group_idx = torch.full((topk.shape[0], topk.shape[1], 1, 1), fill_value=0).to(topk)
            topk = torch.concat([topk, sink_group_idx], dim=-1)
            #  keep the last group
            if key_states_evict_static.shape[-2] % cluster_size1 != 0:
                last_group_idx = torch.full((topk.shape[0], topk.shape[1], 1, 1), fill_value=b_max.shape[2]-1).to(topk)
                topk = torch.concat([topk, last_group_idx], dim=-1)

            # 2nd level: s=16
            topk_gs16 = groupidx_to_groupidx(topk, gap=2)
            b_max, b_min, num_padding_token = group_key_min_max(key_states_evict_static, group_size=cluster_size2)
            group_mask = torch.full((1, b_max.shape[1], 1, b_max.shape[2]), fill_value=torch.finfo(query_states.dtype).min).to(query_states.device)
            topk_gs16[topk_gs16 > group_mask.shape[-1] - 1] = group_mask.shape[-1] - 1
            group_mask.scatter_(-1, topk_gs16, 0)
            sim = torch.sum((alpha * query_states * b_max + (1 - alpha) * query_states * b_min), dim=-1).unsqueeze(2)
            
            sim += group_mask
            topk = torch.topk(sim, k=int(sim.shape[-1] * ratio2), dim=-1).indices
            # keep the first group
            sink_group_idx = torch.full((topk.shape[0], topk.shape[1], 1, 1), fill_value=0).to(topk)
            topk = torch.concat([topk, sink_group_idx], dim=-1)
            # keep the last group
            if key_states_evict_static.shape[-2] % cluster_size1 != 0:
                last_group_idx = torch.full((topk.shape[0], topk.shape[1], 1, 1), fill_value=b_max.shape[2]-1).to(topk)
                topk = torch.concat([topk, last_group_idx], dim=-1)
            # set the dynamic mask
            topk = groupidx_to_tokenidx(topk, b_max.shape[2], group_size=cluster_size2)
            topk[topk > self.attn_dynamic_mask.shape[-1] - 1] -= num_padding_token
            self.attn_dynamic_mask.scatter_(-1, topk.unsqueeze(0).unsqueeze(2), 0)
        
        attn_weights = torch.matmul(query_states, key_states_evict_static.transpose(2, 3)) / math.sqrt(self.head_dim)

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
            if attn_weights.shape[-1] != attention_mask.shape[-1]:
                attention_mask = attention_mask[..., :attn_weights.shape[-1]]
            attn_weights = attn_weights + attention_mask
            attn_weights = torch.max(attn_weights, torch.tensor(torch.finfo(attn_weights.dtype).min))
        
        if layer_idx not in skip_layer_idx:
            attn_weights += self.attn_dynamic_mask

        # upcast attention to fp32
        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)

        if layer_idx not in skip_layer_idx:
            dynamic_ratio.append((1 - attn_weights[0, 0, 0].tolist().count(0) / attn_weights.shape[-1]) * 100)
        # if len(dynamic_ratio) == layer_num:
        #     print(f'Dynamic selection ratio: {sum(dynamic_ratio) / layer_num}%')
        #     dynamic_ratio.clear()
        
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
    # end decoding
    
    
    # start prefill
    qkv = torch.stack([query_states, key_states, value_states], dim=2)  # [bsz, nh, 3, q_len, hd]
    qkv = qkv.transpose(1, 3)  # [bsz, q_len, 3, nh, hd]
    # We have disabled _prepare_decoder_attention_mask in LlamaModel
    # the attention_mask should be the same as the key_padding_mask
    key_padding_mask = attention_mask
    
    static_threshold_list = {
        'hotpotqa': (1e-8, 0.3),
        'narrativeqa': (1e-3, 0.15),
        'triviaqa': (1e-4, 0.21),
        'qasper': (1e-8, 0.4),
        'gov_report': (1e-20, 0.66),
    }
    if key_states.shape[2] > LARGE_TOKEN_NUM:
        last_attn_weights = torch.matmul(query_states[:, :, -int(key_states.shape[2]*0.1):, :], key_states.transpose(-1, -2))
    else:
        last_attn_weights = torch.matmul(query_states[:, :, -int(key_states.shape[2]*0.2):, :], key_states.transpose(-1, -2))
    last_attn_weights = nn.functional.softmax(last_attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
    self.attn_static_mask = torch.zeros(1, 1, 1, last_attn_weights.shape[-1]).to(last_attn_weights.device)
    static_threshold = static_threshold_list[dataset][0]
    last_attn_weights_head_mean = torch.mean(last_attn_weights, dim=1, keepdim=False)
    is_important = torch.zeros_like(last_attn_weights_head_mean)
    is_important[last_attn_weights_head_mean >= static_threshold] = 1
    accum_sum = torch.sum(is_important[0, :, :], dim=0, keepdim=False)
    self.ic_token_idx = torch.argwhere(accum_sum > 0)

    self.base_dynamic_mask = torch.full((1, last_attn_weights.shape[1], 1, self.ic_token_idx.shape[0]), fill_value=torch.finfo(last_attn_weights.dtype).min).to(last_attn_weights.device)

    if layer_idx not in skip_layer_idx:
        if ic_token_idx is None:
            static_ratio.append(100 * self.ic_token_idx.shape[0] / q_len)
        else:
            static_ratio.append(100 * ic_token_idx.shape[0] / q_len)
    if len(static_ratio) == layer_num:
        record_static_ratio.append(sum(static_ratio) / layer_num)
        static_ratio.clear()
    # if 0 < len(record_static_ratio) <= 100 and len(record_static_ratio) % 10 == 0:
    # if len(record_static_ratio) == REPORT_STEP:
        # print(f'Static remained ratio: {sum(record_static_ratio) / len(record_static_ratio):.1f}%')
    if is_print_static:
        print(f'⚙️ Remained ratio after static eviction: {static_threshold_list[dataset][1] * 100:.1f}%')
        is_print_static = False

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
            x_unpad, cu_q_lens, max_s, 0.0,
            softmax_scale=None, causal=True,
        )
        output = rearrange(pad_input(rearrange(output_unpad, 'nnz h d -> nnz (h d)'),
                                     indices, bsz, q_len),
                           'b s (h d) -> b s h d', h=nheads)
    attn_output = self.o_proj(rearrange(output, 'b s h d -> b s (h d)'))

    return attn_output, None, past_key_value
    # end prefill


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


def replace_llama_attn_with_flash_attn(ds=None):
    print('✨🚀 FlashAttention is enabled')
    global dataset
    if ds is not None:
        dataset = ds
    transformers.models.llama.modeling_llama.LlamaModel._prepare_decoder_attention_mask = _prepare_decoder_attention_mask
    transformers.models.llama.modeling_llama.LlamaAttention.forward = forward
