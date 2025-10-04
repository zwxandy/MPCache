import argparse
import json
from collections import OrderedDict
from contextlib import contextmanager
from typing import Any, Optional, Tuple, Union

import torch
from jax.tree_util import tree_map
from transformers import (
    AutoTokenizer,
    GPT2LMHeadModel,
    AutoConfig,
    activations,
)

from transformers.activations import ACT2FN
import transformers

import spu.utils.distributed as ppd
import spu.spu_pb2 as spu_pb2

import time
import numpy as np
from transformers import GPT2Config, LlamaForCausalLM, AutoModelForCausalLM

# This is an experimental example to show legacy pytorch program could be run
# by SPU. Currently we rely on torch-xla to convert torch code into MLIR
# (specifically StableHLO) which is then consumed by SPU. To run this example,
# torch-xla python package should be installed.

# Start nodes.
# > bazel run -c opt //examples/python/utils:nodectl -- up

# Run this example script.
# > bazel run -c opt //examples/python/ml/torch_resnet_experiment:torch_resnet_experiment


parser = argparse.ArgumentParser(description='distributed driver.')
parser.add_argument("-c", "--config", default="examples/python/conf/3pc.json")
args = parser.parse_args()

with open(args.config, 'r') as file:
    conf = json.load(file)

ppd.init(conf["nodes"], conf["devices"], framework=ppd.Framework.EXP_TORCH)

copts = spu_pb2.CompilerOptions()
copts.enable_pretty_print = False
copts.xla_pp_kind = 2
copts.enable_optimize_denominator_with_broadcast = True
print(f'copts: {copts}')


def hack_softmax(x, dim=-1):
    print('hack softmax')
    x_max = torch.max(x, dim=dim, keepdims=True).values
    x = x - x_max
    # exp on large negative is clipped to zero
    b = x > -14
    nexp = torch.exp(x) * b
    divisor = torch.sum(nexp, dim=dim, keepdims=True)
    divisor_recip = 1 / divisor
    # return nexp / divisor
    return nexp * divisor_recip


@contextmanager
def hack_softmax_context(msg: str, enabled: bool = True):
    if not enabled:
        yield
        return
    # hijack some target functions
    raw_softmax = torch.nn.functional.softmax
    torch.nn.functional.softmax = hack_softmax
    yield
    # recover back
    torch.nn.functional.softmax = raw_softmax


def hack_silu(x, inplace=True):
    print('hack silu')
    b0 = x < -8.0
    b1 = x < -4.0
    b2 = x > 4.0
    
    b3 = b1 ^ b2 ^ True  # x in [-4.0, 4.0)
    b4 = b0 ^ b1  # x in [-8.0, -4.0)
    
    x2 = torch.square(x)
    x4 = torch.square(x2)
    x6 = x2 * x4
    
    seg1 = -0.0055465625580307 * x2 - 0.0819767021525476 * x - 0.3067541139982155
    seg2 = 0.0002743776353465 * x6 - 0.011113046708173 * x4 + 0.2281430841728270 * x2 + 0.5 * x + 0.0085064025895951
    ret = b2 * x + b4 * seg1 + b3 * seg2
    
    return ret


@contextmanager
def hack_silu_context(msg: str, enabled: bool = True):
    if not enabled:
        yield
        return
    # hijack some target functions
    raw_silu = torch.nn.functional.silu
    torch.nn.functional.silu = hack_silu
    yield
    # recover back
    torch.nn.functional.silu = raw_silu


class TextGenerationModel(torch.nn.Module):
    def __init__(self, model):
        super(TextGenerationModel, self).__init__()
        self.model = model

    def forward(self, past_key_values, pred_token_idx):
        # note: model.generate() is not supported in SPU!
        
        enable_kv_cache = True

        if enable_kv_cache:
            print('>>> Use `for loop` to generate tokens w/ KV cache (decoding stage)')
            generated_ids = pred_token_idx
            print('Decoding stage...')
            for _ in range(max_new_tokens - 1):  # note: prefill stage has generated 1 token already
                outputs = self.model(input_ids=pred_token_idx, past_key_values=past_key_values, use_cache=True)
                past_key_values = outputs.past_key_values  # if use_cache=False, past_key_values=None
                pred_token_idx = outputs.logits[:, -1, :].argmax(dim=-1).unsqueeze(1)
                generated_ids = torch.cat([generated_ids, pred_token_idx], dim=-1)
            return generated_ids
        else:
            print('>>> Use `for loop` to generate tokens w/o KV cache')
            for _ in range(max_new_tokens):
                outputs = self.model(input_ids=input_ids)
                logits = outputs.logits
                next_token = torch.argmax(logits[:, -1, :], dim=-1).unsqueeze(-1)
                input_ids = torch.cat([input_ids, next_token], dim=-1)
            return input_ids


def run_inference_on_cpu(input_ids):
    print('Run on CPU\n==========\n')
    # generate_ids = model.generate(input_ids, max_new_tokens=max_new_tokens)  # not supported in SPU!
    # return generate_ids
    
    outputs = model(input_ids=input_ids, past_key_values=None, use_cache=True)
    past_key_values = outputs.past_key_values
    pred_token_idx = outputs.logits[:, -1, :].argmax(dim=-1).unsqueeze(1)
    
    text_gen = TextGenerationModel(model)
    text_gen.eval()
    generate_ids = text_gen(past_key_values, pred_token_idx)
    return generate_ids


def run_inference_on_spu(input_ids):
    print('Run on SPU\n==========\n')
    
    with hack_softmax_context("hack exp of softmax", enabled=True), hack_silu_context("hack silu", enabled=True):
        text_gen = TextGenerationModel(model)
        text_gen.eval()
        
        outputs = model(input_ids=input_ids, past_key_values=None, use_cache=True)
        past_key_values = outputs.past_key_values
        pred_token_idx = outputs.logits[:, -1, :].argmax(dim=-1).unsqueeze(1)
        print('Finish generating the 1st token during prefill stage.')

        params_buffers = OrderedDict()
        for k, v in text_gen.named_parameters():
            params_buffers[k] = v
        for k, v in text_gen.named_buffers():
            params_buffers[k] = v
        for k, v in text_gen.state_dict().items():
            params_buffers[k] = v
            # print(f'{k} {v}')
        params = ppd.device("P1")(
            lambda input: tree_map(lambda x: x.detach().numpy(), input)
        )(params_buffers)
        # params = ppd.device("P1")(lambda x: x)(text_gen.state_dict())  # error: data format

        # input_ids_hat = ppd.device("P2")(lambda x: x.detach().numpy())(input_ids)
        pred_token_idx_hat = ppd.device("P2")(lambda x: x.detach().numpy())(pred_token_idx)
        past_key_values_hat = ppd.device("P2")(
            lambda input: tree_map(lambda x: x.detach().numpy(), input)
        )(past_key_values)
        print('Finish loading params and input_ids to devices.')

        generate_ids = ppd.device("SPU")(text_gen, copts=copts)(
            params, past_key_values_hat, pred_token_idx_hat
        )  # torch._dynamo.exc.Unsupported: call_function BuiltinVariable(str) [UserFunctionVariable()] {}
        
        # generate_ids = ppd.device("SPU")(model, copts=copts)(params, input_ids_hat)  # RuntimeError: Currently only torch models with named parameters and buffers are supported
        # generate_ids = ppd.device("SPU")(text_generation, copts=copts)(params, input_ids_hat)  # AssertionError: currently only torch.nn.Module is supported
    return generate_ids


if __name__ == '__main__':
    torch.manual_seed(0)
    
    model_config = GPT2Config.from_pretrained("gpt2", cache_dir='path/ckpt_torch_gpt2')

    # modify the model config to simulate different scales
    model_config.num_attention_heads = 32
    model_config.hidden_size = 1024
    model_config.n_layer = 1
    model_config.activation_function = 'silu'
    print(model_config)

    tokenizer = AutoTokenizer.from_pretrained("gpt2", cache_dir='path/ckpt_torch_gpt2')
    model = GPT2LMHeadModel.from_pretrained("gpt2", cache_dir='path/ckpt_torch_gpt2', config=model_config, ignore_mismatched_sizes=True)
    
    print('Finish loading tokenizer and model.')
    tokenizer.pad_token_id = tokenizer.eos_token_id
    max_new_tokens = 2  # decoding_len=max_new_tokens-1

    # input_ids = tokenizer.encode('I love the way you', return_tensors='pt')
    input_ids = torch.randint(0, 1000, (1, 512))  # set a input length
    print(
        f'Length of prompt: {input_ids.shape[-1]}, Length of generation: {max_new_tokens}'
    )

    stime = time.time()
    ret = run_inference_on_spu(input_ids)
    print(ret.shape)
    print(tokenizer.decode(ppd.get(ret)[0], skip_special_tokens=True))
    print(f'spu execution runtime: {time.time() - stime} s')