import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from gensim.models import Word2Vec
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from numpy.linalg import norm

# === Ayarlar ===
giris_cumlesi = "Performs introspection of the provided Bearer JWT token"
model_klasoru = "modeller"
veri_dosyalar = {
    "lemmatized": "lemmatizasyon_kelimeler.csv",
    "stemmed": "stemming_kelimeler.csv"
}

# === TF-IDF Benzerlik ===
def tfidf_benzerlik(isim, veri_yolu):
    veri_df = pd.read_csv(veri_yolu, header=None)
    cumleler = [" ".join(satir.dropna()) for _, satir in veri_df.iterrows()]

    vectorizer = TfidfVectorizer()
    vectorizer.fit(cumleler)
    giris_vec = vectorizer.transform([giris_cumlesi])
    tfidf_vec = vectorizer.transform(cumleler)

    skorlar = cosine_similarity(giris_vec, tfidf_vec)[0]
    top5 = np.argsort(skorlar)[-5:][::-1]

    print(f"\n[TF-IDF] {isim}")
    for i in top5:
        print(f"{i}. cümle (skor={skorlar[i]:.4f}):", cumleler[i])

    return top5.tolist(), skorlar[top5].tolist()

# === Word2Vec Benzerlik ===
def ortalama_vektor(kelimeler, model):
    vektorler = [model.wv[k] for k in kelimeler if k in model.wv]
    return np.mean(vektorler, axis=0) if vektorler else None

def word2vec_benzerlik(model_yolu, veri_yolu):
    model = Word2Vec.load(model_yolu)
    veri_df = pd.read_csv(veri_yolu, header=None)
    cumleler = [satir.dropna().tolist() for _, satir in veri_df.iterrows()]
    giris_vec = ortalama_vektor(giris_cumlesi.split(), model)

    if giris_vec is None:
        return [], []

    skorlar = []
    for cumle in cumleler:
        vec = ortalama_vektor(cumle, model)
        if vec is None:
            skorlar.append(0)
        else:
            skorlar.append(np.dot(giris_vec, vec) / (norm(giris_vec) * norm(vec)))

    skorlar = np.array(skorlar)
    top5 = np.argsort(skorlar)[-5:][::-1]

    print(f"\n[Word2Vec] {os.path.basename(model_yolu)}")
    for i in top5:
        print(f"{i}. cümle (skor={skorlar[i]:.4f}):", " ".join(cumleler[i]))

    return top5.tolist(), skorlar[top5].tolist()

# === Anlamsal Değerlendirme (Manuel Puanlama) ===
personal_scores = {
    "tfidf_lemmatized": [4, 4, 3, 5, 4],
    "tfidf_stemmed": [3, 2, 2, 3, 1],
    # "word2vec_...": [5, 4, 3, 4, 5], # manuel eklenecek
}

# === Ana Fonksiyon ===
def main():
    jaccard_dict = {}

    # TF-IDF Modelleri
    for isim, veri_yolu in veri_dosyalar.items():
        top5, skorlar = tfidf_benzerlik(isim, veri_yolu)
        jaccard_dict[f"tfidf_{isim}"] = set(top5)

    # Word2Vec Modelleri
    for veri_tipi, veri_yolu in veri_dosyalar.items():
        for dosya in os.listdir(model_klasoru):
            if dosya.endswith(".model") and dosya.startswith(f"word2vec_{veri_tipi}"):
                model_yolu = os.path.join(model_klasoru, dosya)
                top5, skorlar = word2vec_benzerlik(model_yolu, veri_yolu)
                jaccard_dict[dosya] = set(top5)
                # personal_scores[dosya] = [..] # elle ekleyebilirsin

    # Ortalama skor yazdır
    print("\n--- Anlamsal Değerlendirme ---")
    for model_adi, skorlar in personal_scores.items():
        ort = np.mean(skorlar)
        print(f"{model_adi}: Ortalama Anlamsal Puan = {ort:.2f}")

    # Jaccard Benzerlik Matrisi
    print("\n--- Jaccard Benzerlik Matrisi ---")
    model_adlari = list(jaccard_dict.keys())
    n = len(model_adlari)
    jaccard_matrisi = np.zeros((n, n))

    for i in range(n):
        for j in range(n):
            if i == j:
                jaccard_matrisi[i, j] = 1.0
            else:
                kesisim = len(jaccard_dict[model_adlari[i]] & jaccard_dict[model_adlari[j]])
                birlesim = len(jaccard_dict[model_adlari[i]] | jaccard_dict[model_adlari[j]])
                jaccard_matrisi[i, j] = kesisim / birlesim

    plt.figure(figsize=(12, 10))
    sns.heatmap(jaccard_matrisi, annot=True, xticklabels=model_adlari, yticklabels=model_adlari, cmap="coolwarm")
    plt.title("Jaccard Benzerlik Matrisi")
    plt.tight_layout()
    plt.savefig("jaccard_matrisi.png")
    plt.show()

if __name__ == "__main__":
    main()
