import re
import os
import json
import glob
import random
from pathlib import Path
from tqdm import tqdm
import multiprocessing
from datasketch import MinHash, MinHashLSH, MinHashLSHForest


CONTENT_FIELD = "text"


class MinHashDeDup:
    def __init__(self, num_workers=16, similarity_threshold=0.9, num_perm=128):
        self.num_workers = num_workers
        self.similarity_threshold = similarity_threshold
        self.num_perm = num_perm

    def calculate_minhash(self, text):
        tokens = text.split()
        minhash = MinHash()

        # Generate hash values for each token
        for token in tokens:
            minhash.update(token.encode("utf-8"))

        return minhash

    def deduplicate_chunk(self, chunk, lsh):
        deduplicated_records = []

        for record in tqdm(chunk):
            text = record[CONTENT_FIELD]
            minhash = self.calculate_minhash(text)
            similar_hashes = lsh.query(minhash)

            if not similar_hashes:
                lsh.insert(text, minhash)
                deduplicated_records.append(record)

        return deduplicated_records

    def deduplicate_file_lsh(self, input_data):
        lsh = MinHashLSH(threshold=self.similarity_threshold, num_perm=self.num_perm)

        random.shuffle(input_data)
        chunk_size = len(input_data) // self.num_workers
        if chunk_size == 0:
            chunk_size = 1
        chunks = [input_data[i : i + chunk_size] for i in range(0, len(input_data), chunk_size)]

        pool = multiprocessing.Pool(processes=self.num_workers)
        results = pool.starmap(self.deduplicate_chunk, [(chunk, lsh) for chunk in chunks])
        deduplicated_records = [record for sublist in results for record in sublist]
        pool.close()
        pool.join()

        return deduplicated_records


def load_dataset_from_texts(dir_path):
    txt_files = glob.glob(os.path.join(dir_path, "*.txt"))

    def load_txt_file(file_path):
        result = []
        print(f"=> load txt file: {file_path}")
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f.readlines():
                line = line.strip()
                if line:
                    result.append({CONTENT_FIELD: line})
        return result

    data = []
    for txt_file in txt_files:
        data.extend(load_txt_file(txt_file))

    return data


def dedup_pt_data():
    dataset_dir = "/mnt/cephfs2/peichao/NLP/translate/LLM/datasets/chinese_llama2/pt_training/"
    data = load_dataset_from_texts(dataset_dir)

    min_hash = MinHashDeDup(num_workers=16)
    dedup_data = min_hash.deduplicate_file_lsh(data)
    print(
        f"input data size: {len(data)}\ndedup data size: {len(dedup_data)}\ndup ratio: {1- len(dedup_data) / len(data)}"
    )

    dedup_data = [data[CONTENT_FIELD] for data in dedup_data]
    line_count = 0
    max_line_per_file = 100_0000
    while line_count < len(dedup_data):
        with open(Path(dataset_dir) / f"dedup_{line_count // max_line_per_file}.txt", "w", encoding="utf-8") as f:
            f.write("\n".join(dedup_data[line_count : line_count + max_line_per_file]))
        line_count += max_line_per_file


if __name__ == "__main__":
    dedup_pt_data()
