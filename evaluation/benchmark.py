import os
import evaluate
import json

bleu = evaluate.load("sacrebleu")


def foo():
    references = []
    dir = "/mnt/cephfs2/xiangyang/Code/LLM/ColossalAI/applications/Chat/examples/BENCHMARK"
    deepl = open("benchmark_deepl.txt").read().splitlines()
    google = open("benchmark_google.txt").read().splitlines()
    en = open("benchmark_en.txt").read().splitlines()
    for d, g, e in zip(deepl, google, en):
        references.append([d, g, e])
    # for e in zip(en):
    #     references.append(e)

    files = [
        "benchmark_1b7_1w.txt",
        "benchmark_1b7_yiran_1w.txt",
        "benchmark_1b7_yiran_v2_1w.txt",
        "benchmark_3b_1w.txt",
        "benchmark_baichuan_7b.txt",
        "benchmark_3b-a6000_t6000_d-1000.txt",
        "benchmark_3b-17120.txt",
        "benchmark_3b-17210-v2.txt",
        "benchmark_3b-22210.txt",
        "benchmark_3b-8624-lima.txt",
        "benchmark_3b-8624.txt",
        "benchmark_3b-17210-sub-8624.txt",
        "benchmark_3b-4457-lima.txt",
        "benchmark_3b-4457.txt",
        "benchmark_3b-17210-sub-4457.txt",
    ]
    for f in files:
        predictions = open(os.path.join(dir, f)).read().splitlines()
        predictions = [i[1:-1] if i[0] == '"' and i[-1] == '"' else i for i in predictions]
        results = bleu.compute(predictions=predictions, references=references)
        print(f)
        print(results)


def calc_benchmark_bleu(predictions_file):
    with open(
        "/mnt/cephfs2/xiangyang/Code/LLM/ColossalAI/applications/Chat/code/benchmarkData/benchmark_deepl.txt"
    ) as f:
        deepl_refs = f.read().splitlines()

    with open(
        "/mnt/cephfs2/xiangyang/Code/LLM/ColossalAI/applications/Chat/code/benchmarkData/benchmark_google.txt"
    ) as f:
        google_refs = f.read().splitlines()

    with open(predictions_file) as f:
        predictions_result = json.load(f)
        predictions = [pred["predict"] for pred in predictions_result]
        predictions = [i[1:-1] if i[0] == '"' and i[-1] == '"' else i for i in predictions]

    print(predictions_file)
    print("--- google refs bleu ---")
    results = bleu.compute(predictions=predictions, references=google_refs)
    print(results)

    print("--- deepl refs bleu ---")
    results = bleu.compute(predictions=predictions, references=deepl_refs)
    print(results)


if __name__ == "__main__":
    calc_benchmark_bleu(
        "output/bloom3b_sft_lora/2023-08-01_103831/checkpoint-1000/benchmark_eval_result/generated_predictions.json"
    )
