import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer, PorterStemmer

# Gerekli NLTK veri setlerini indir (bir kez yeterli)
nltk.download("stopwords")
nltk.download("wordnet")
nltk.download("omw-1.4")

# 1. Adım: Temizlenmiş kelime dosyasını oku
veri = pd.read_csv("data/clean/temiz_kelimeler.csv")
kelimeler = veri["kelime"].tolist()

# 2. Adım: Küçük harf yap, noktalama işaretlerini temizle, sadece harf içerenleri al
duzenli_kelimeler = [
    kelime.lower().strip(string.punctuation)
    for kelime in kelimeler
    if kelime and kelime.isalpha()
]

# 3. Adım: Stopword'leri çıkar
durak_kelimeler = set(stopwords.words("english"))
anlamli_kelimeler = [
    kelime for kelime in duzenli_kelimeler if kelime not in durak_kelimeler
]

# 4. Adım: Lemmatization ve Stemming işlemleri
kok_bulucu = WordNetLemmatizer()
kok_alici = PorterStemmer()

lemmatize_edilmis = [kok_bulucu.lemmatize(kelime) for kelime in anlamli_kelimeler]
stem_edilmis = [kok_alici.stem(kelime) for kelime in anlamli_kelimeler]

# 5. Adım: Frekansları hesapla ve CSV olarak kaydet
def frekans_kaydet(kelime_listesi, dosya_adi):
    sayim = Counter(kelime_listesi)
    df = pd.DataFrame(sayim.items(), columns=["kelime", "frekans"])
    df = df.sort_values(by="frekans", ascending=False)
    df.to_csv(dosya_adi, index=False)
    return df

df_lemmatize = frekans_kaydet(lemmatize_edilmis, "lemmatizasyon_kelimeler.csv")
df_stem = frekans_kaydet(stem_edilmis, "stemming_kelimeler.csv")

# 6. Adım: Zipf grafiği çiz
def zipf_grafigi(df, baslik, dosya_adi):
    plt.figure(figsize=(10, 6))
    plt.loglog(range(1, len(df) + 1), df["frekans"], marker=".")
    plt.title(f"Zipf Grafiği - {baslik}")
    plt.xlabel("Kelime Sırası (log)")
    plt.ylabel("Frekans (log)")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(dosya_adi)
    plt.close()

zipf_grafigi(df_lemmatize, "Lemmatizasyon", "zipf_lemmatizasyon.png")
zipf_grafigi(df_stem, "Stemming", "zipf_stemming.png")

# 7. Adım: Özet Bilgi Yazdır
print("\nÖzet Bilgi:")
print(f"Toplam orijinal kelime sayısı: {len(kelimeler)}")
print(f"Anlamlı (temiz) kelime sayısı: {len(anlamli_kelimeler)}")
print(f"Lemmatizasyon sonrası toplam kelime sayısı: {len(lemmatize_edilmis)}")
print(f"Stemming sonrası toplam kelime sayısı: {len(stem_edilmis)}")
print(f"Lemmatizasyon sonrası eşsiz kelime sayısı: {df_lemmatize.shape[0]}")
print(f"Stemming sonrası eşsiz kelime sayısı: {df_stem.shape[0]}")
print(f"Çıkarılan durak kelime sayısı: {len(duzenli_kelimeler) - len(anlamli_kelimeler)}")
