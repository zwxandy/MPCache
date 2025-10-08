import argparse
import json

import flax.linen as nn
import jax.numpy as jnp
import numpy as np
from jax import lax

import spu.utils.distributed as ppd
import spu.spu_pb2 as spu_pb2


parser = argparse.ArgumentParser(description='distributed driver.')
parser.add_argument("-c", "--config", default="/home/admin/dev/examples/python/conf/3pc.json")
args = parser.parse_args()

with open(args.config, 'r') as file:
    conf = json.load(file)

ppd.init(conf["nodes"], conf["devices"])

copts = spu_pb2.CompilerOptions()
copts.enable_pretty_print = False
copts.xla_pp_kind = 2
# enable x / broadcast(y) -> x * broadcast(1/y)
copts.enable_optimize_denominator_with_broadcast = True


def topk(x):
    """
    Returns the top-k largest values from the input array.
    x: (H, T) -> y: (H, k)
    """
    topk_values, _ = lax.top_k(x, k=50)
    return topk_values

def compute_sim(q, k):
    """
    Computes the similarity matrix between query and key.
    q: (H, Tq, D), k: (H, Tk, D) -> y: (H, Tq, Tk)
    """
    return q @ jnp.swapaxes(k, -1, -2)

def idx2onehot(x):
    """
    Converts indices to onehot vectors with 512 classes.
    x: (H, k) -> y: (H, k, T)
    """
    return nn.one_hot(x, num_classes=512, axis=-1)

def max_in_tensor(x):
    """
    Returns the maximum values along the second-to-last axis.
    x: (H, 1, S, D) -> y: (H, 1, D)
    """
    res = jnp.max(x, axis=-2)
    return res

def max_two_tensor(x, y):
    """
    Element-wise maximum between two tensors.
    x: (H, G, D), y: (H, G, D) -> y: (H, G, D)
    """
    max_val = jnp.maximum(x, y)
    return max_val

def qb_ewmm(q, b):
    """
    Element-wise multiplication of q and b.
    q: (H, Tq, D), b: (H, G, D) -> y: (H, G, D)
    """
    return q * b

def onehot_k_gemv(onehot, K):
    """
    Matrix multiplication between one-hot vectors and k.
    onehot: (H, k, T), K: (H, T, D) -> y: (H, k, D)
    """
    return onehot @ K

def token_gather(idx, K):
    """
    Token gathering from k based on index.
    idx: (H, k), K: (H, T, D) -> y: (H, k, D)
    """
    one_hot = nn.one_hot(idx, num_classes=512, axis=-1)
    return one_hot @ K


def run_on_spu_topk(sim):
    sim_enc = ppd.device("P2")(lambda x: x)(sim)
    topk_values = ppd.device("SPU")(topk)(sim_enc)
    topk_values_pt = ppd.get(topk_values)
    return topk_values_pt   

def run_on_spu_compute_sim(q, k):
    input_enc1 = ppd.device("P2")(lambda x: x)(q)
    input_enc2 = ppd.device("P2")(lambda x: x)(k)
    y_enc = ppd.device("SPU")(compute_sim)(input_enc1, input_enc2)
    y_pt = ppd.get(y_enc)
    return y_pt

def run_on_spu_idx2onehot(index):
    input_enc1 = ppd.device("P2")(lambda x: x)(index)
    y_enc = ppd.device("SPU")(idx2onehot, copts=copts)(input_enc1)
    y_pt = ppd.get(y_enc)
    return y_pt

def run_on_spu_max_in_tensor(input):
    input_enc1 = ppd.device("P2")(lambda x: x)(input)
    y_enc = ppd.device("SPU")(max_in_tensor)(input_enc1)
    y_pt = ppd.get(y_enc)
    return y_pt

def run_on_spu_max_two_tensor(a, b):
    input_enc1 = ppd.device("P2")(lambda x: x)(a)
    input_enc2 = ppd.device("P2")(lambda x: x)(b)
    y_enc = ppd.device("SPU")(max_two_tensor)(input_enc1, input_enc2)
    y_pt = ppd.get(y_enc)
    return y_pt

def run_on_spu_qb_ewmm(q, b):
    input_enc1 = ppd.device("P2")(lambda x: x)(q)
    input_enc2 = ppd.device("P2")(lambda x: x)(b)
    y_enc = ppd.device("SPU")(qb_ewmm)(input_enc1, input_enc2)
    y_pt = ppd.get(y_enc)
    return y_pt

def run_on_spu_onehot_k_gemv(onehot, K):
    input_enc1 = ppd.device("P2")(lambda x: x)(onehot)
    input_enc2 = ppd.device("P2")(lambda x: x)(K)
    y_enc = ppd.device("SPU")(onehot_k_gemv)(input_enc1, input_enc2)
    y_pt = ppd.get(y_enc)
    return y_pt

def run_on_spu_token_gather(idx, K):
    input_enc1 = ppd.device("P2")(lambda x: x)(idx)
    input_enc2 = ppd.device("P2")(lambda x: x)(K)
    y_enc = ppd.device("SPU")(token_gather)(input_enc1, input_enc2)
    y_pt = ppd.get(y_enc)
    return y_pt
    

if __name__ == '__main__':
    # model config
    batch_size = 1
    num_head = 32  # H
    dim_per_head = 32  # D
    num_token = 512  # T
    k = 50  # k
    group_size = 16  # S
    num_group = num_token // group_size  # G=32

    eval_mode = 'topk'

    if eval_mode == 'topk':
        sim = np.random.randn(num_head, num_token)
        p = run_on_spu_topk(sim)
    
    elif eval_mode == 'compute_sim':
        q = np.random.randn(batch_size, num_head, 1, dim_per_head)
        K = np.random.randn(batch_size, num_head, num_token, dim_per_head)
        p = run_on_spu_compute_sim(q, K)
    
    elif eval_mode == 'idx2onehot':
        index = np.random.randint(low=0, high=127, size=(batch_size, num_head, k))
        p = run_on_spu_idx2onehot(index)
    
    elif eval_mode == 'max_in_tensor':
        input = np.random.randn(batch_size, num_head, 1, group_size, dim_per_head)  # only need to compute the latest group
        p = run_on_spu_max_in_tensor(input)
    
    elif eval_mode == 'max_two_tensor':
        a = np.random.randn(batch_size, num_head, num_group, dim_per_head)
        b = np.random.randn(batch_size, num_head, num_group, dim_per_head)
        p = run_on_spu_max_two_tensor(a, b)
    
    elif eval_mode == 'qb_ewmm':
        q = np.random.randn(batch_size, num_head, 1, dim_per_head)
        b = np.random.randn(batch_size, num_head, num_group, dim_per_head)
        p = run_on_spu_qb_ewmm(q, b)
    
    elif eval_mode == 'onehot_k_gemv':
        onehot = np.random.randint(low=0, high=1, size=(batch_size, num_head, k, num_token))
        K = np.random.randn(batch_size, num_head, num_token, dim_per_head)
        p = run_on_spu_onehot_k_gemv(onehot, K)
    
    elif eval_mode == 'run_on_spu_token_gather':
        index = np.random.randint(low=0, high=127, size=(batch_size, num_head, k))
        K = np.random.randn(batch_size, num_head, num_token, dim_per_head)
        p = run_on_spu_token_gather(index, K)