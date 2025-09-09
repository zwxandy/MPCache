# MPCache: MPC-Friendly KV Cache Eviction for Efficient Private LLM Inference

In this work, we follow the framework of LongBench to build MPCache.

## Abstract
Private LLM inference based on multi-party computation (MPC) offers cryptographically-secure protection for both user prompt and proprietary model weights. However, it suffers from large latency overhead for long input sequences. While key-value (KV) cache eviction algorithms have been proposed to reduce the computation and memory cost for plaintext inference, they are not designed for MPC and may even introduce more overhead. In this paper, we propose an accurate and MPC-friendly KV cache eviction framework, dubbed MPCache. MPCache is built on the observation that historical tokens in a long sequence may have different effects on the downstream decoding. Hence, MPCache combines a look-once static eviction algorithm to discard unimportant tokens and a query-aware dynamic selection algorithm to further choose a small subset of tokens for attention computation. As existing dynamic selection algorithms incur too much latency, we propose a series of optimizations to drastically reduce the KV cache selection overhead, including MPC-friendly similarity approximation, hierarchical KV cache clustering, and layer-wise index sharing strategy. With extensive experiments, we demonstrate that MPCache consistently outperforms prior-art KV cache eviction baselines across different LLM generation tasks and achieves 1.8 ∼ 2.01× and 3.39 ∼ 8.37× decoding latency and communication reduction on different sequence lengths, respectively.

## Dataset Preparation
You can download and load the LongBench dataset through the Huggingface datasets ([HF Repo](https://huggingface.co/datasets/THUDM/LongBench)):
```python
from datasets import load_dataset

datasets = ["narrativeqa", "qasper", "multifieldqa_en", "multifieldqa_zh", "hotpotqa", "2wikimqa", "musique", \
            "dureader", "gov_report", "qmsum", "multi_news", "vcsum", "trec", "triviaqa", "samsum", "lsht", \
            "passage_count", "passage_retrieval_en", "passage_retrieval_zh", "lcc", "repobench-p"]

for dataset in datasets:
    data = load_dataset('THUDM/LongBench', dataset, split='test')
```
You can also download the datasets from the website [this link](https://huggingface.co/datasets/THUDM/LongBench/resolve/main/data.zip).


## Evaluation

**Packages and environment:**

Install the required packages with pip tool: 
```bash
pip install -r requirements.txt
```
For LLaMA-2-based model inference on long sequences, we follow the optimization of FlashAttention during the prefill stage for saving GPU memory.
The relevant dependencies can be installed according to the codebase of [FlashAttention](https://github.com/Dao-AILab/flash-attention).

**Dataset choice:**

To evaluate a specific dataset, we can modify the following code in `pred_mine.py` (we choose the hotpotqa dataset as an example):
```python
datasets = ["narrativeqa", "qasper", "multifieldqa_en", "multifieldqa_zh", "hotpotqa", "2wikimqa", "musique", \
            "dureader", "gov_report", "qmsum", "multi_news", "vcsum", "trec", "triviaqa", "samsum", "lsht", \
            "passage_count", "passage_retrieval_en", "passage_retrieval_zh", "lcc", "repobench-p"]

datasets = ["hotpotqa"]  # define the chosen dataset
```

**Model file and configuration:**

The main codes of KV cache eviction algorithm, including MPC-friendly similarity approximation and hierarchical KV cache clustering are implemented in `llama_flash_attn_monkey_patch_compression.py`.

For hierarchical clustering, the variable `alpha` controls the ratio between $\mathbf r^{\min}$ and $\mathbf r^{\max}$ (set $\alpha=0.6$ by default).
`cluster_size1` and `cluster_size2` control the granularities of two hierarchical levels (set `cluster_size1=32` and `cluster_size2=16` by default).
`ratio1` and `ratio2` control the dynamic selection ratio at the 1st hierarchical level and the final dynamic selection ratio, respectively.

Layer-wise index sharing is implemented in `modeling_llama.py`, and we provide the modified file in this repo. To enable layer-wise index sharing, you can simply replace the original `modeling_llama.py` with ours.

We give an example on the hotpotqa dataset here, 70\% tokens are pruned (i.e., 30\% preserved) on average after static eviction based on the accumulated attention sum (refer to H2O and SnapKV); set `ratio2=0.1` for a final KV cache budget of 3\%, `ratio2=0.2` for a final KV cache budget of 6\%, `ratio2=0.4` for a final KV cache budget of 12\%.

**Model inference and evaluation:**

First, run `pred_mine.py` to perform the model inference on longchat-v1.5-7b-32k:
```bash
CUDA_VISIBLE_DEVICES=0 python pred_mine.py --model longchat-v1.5-7b-32k
```
You can also run inference on multi-gpus in parallel (one model per gpu):
```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 python pred_mine.py --model longchat-v1.5-7b-32k
```
Then, you can obtain the inference output of the model on the dataset under the `pred_mine/` folder corresponding to the model name.

After the inference, run `eval_mine.py` to evaluate the model performance (no need of GPU):
```bash
python eval_mine.py --model longchat-v1.5-7b-32k
```
We can get the results on the datasets in `result.json`.
