from zhconv import convert
from tqdm import tqdm


def zh_cn_tw_conv(input_file):
    """Convert Traditional Chinese to Simplified Chinese."""
    output_file = input_file.replace(".txt", "_conv.txt")
    with open(input_file, "r") as f:
        lines = f.readlines()
    with open(output_file, "w") as f:
        result = []
        for line in tqdm(lines):
            line = convert(line, "zh-cn")
            result.append(line)
        f.write("\n".join(result))


if __name__ == "__main__":
    zh_cn_tw_conv(
        "/mnt/cephfs2/peichao/NLP/translate/LLM/datasets/chinese_llama2/pt_training/zetavg_coct_en_zh_tw_translations_twp_300k.txt"
    )
