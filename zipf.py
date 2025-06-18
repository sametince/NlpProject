# 1_zipf_analysis.py

import os
import yaml
from collections import Counter
import matplotlib.pyplot as plt
from tqdm import tqdm


DATA_DIR = "data/APIs"

def get_path_words_from_yaml_files(data_dir):
    word_list = []

    for root, _, files in os.walk(data_dir):
        for file in tqdm(files, desc="YAML dosyaları taranıyor"):
            if file.endswith(".yaml") or file.endswith(".yml"):
                file_path = os.path.join(root, file)
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content = yaml.safe_load(f)
                        if not isinstance(content, dict):
                            continue
                        paths = content.get("paths")
                        if isinstance(paths, dict):
                            for path in paths.keys():
                                clean_path = path.replace('{', '').replace('}', '')
                                parts = clean_path.strip("/").split("/")
                                word_list.extend(parts)
                except Exception as e:
                    print(f"HATA ({file_path}): {e}")
    return word_list

def draw_zipf_graph(word_freq: Counter, output_path: str):
    sorted_freqs = sorted(word_freq.values(), reverse=True)
    ranks = range(1, len(sorted_freqs) + 1)

    plt.figure(figsize=(10, 6))
    plt.loglog(ranks, sorted_freqs, marker='.')
    plt.title("Zipf Grafiği - Endpoint Kelimeleri (YAML)")
    plt.xlabel("Sıra (Rank)")
    plt.ylabel("Frekans (Frequency)")
    plt.grid(True)
    plt.savefig(output_path)
    plt.close()

if __name__ == "__main__":
    os.makedirs("data/zipf", exist_ok=True)
    print("Endpoint kelimeleri YAML'den çıkarılıyor...")
    words = get_path_words_from_yaml_files(DATA_DIR)
    word_freq = Counter(words)

    print("\n📊 En sık geçen 10 kelime:")
    for word, freq in word_freq.most_common(10):
        print(f"{word}: {freq}")

    draw_zipf_graph(word_freq, "data/zipf/zipf_raw.png")
    print("\n✅ Zipf grafiği oluşturuldu → data/zipf/zipf_raw.png")
