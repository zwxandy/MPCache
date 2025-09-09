from datasets import load_dataset

datasets = ["narrativeqa", "qasper", "multifieldqa_en", "multifieldqa_zh", "hotpotqa", "2wikimqa", "musique", \
            "dureader", "gov_report", "qmsum", "multi_news", "vcsum", "trec", "triviaqa", "samsum", "lsht", \
            "passage_count", "passage_retrieval_en", "passage_retrieval_zh", "lcc", "repobench-p"]

for dataset in datasets:
    print(f'dataset: {dataset}')
    data = load_dataset('THUDM/LongBench', dataset, split='test', cache_dir='./lb_datasets')
    print(data)

# data = load_dataset('THUDM/LongBench', "samsum", split='test', cache_dir='./lb_datasets')
# print(data)