import torch

def group_key_min_max(key_states, group_size):
    num_padding_token = (key_states.shape[2] + group_size - 1) // group_size * group_size - key_states.shape[2]
    padding_token = key_states[:, :, -group_size:-group_size+num_padding_token, :]
    key_states_padded = torch.cat((key_states, padding_token), dim=2)
    
    key_states_reshaped = key_states_padded.reshape(key_states.shape[0], key_states.shape[1], -1, group_size, key_states.shape[-1])

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