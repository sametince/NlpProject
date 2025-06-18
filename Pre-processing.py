# 2_on_isleme.py

import os
import yaml
import pandas as pd
import string
from tqdm import tqdm
from collections import Counter
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Dizin yolu
veri_klasoru = "data/APIs"
cikti_dosyasi = "data/clean/temiz_kelimeler.csv"

# Hazırlıklar
durak_kelimeler = set(stopwords.words("english"))
kok_bulucu = WordNetLemmatizer()

def kelime_cek(kok_dizin):
    kelimeler = []

    for kok, _, dosyalar in os.walk(kok_dizin):
        for dosya in tqdm(dosyalar, desc="Dosyalar taranıyor"):
            if dosya.endswith(".yaml") or dosya.endswith(".yml"):
                tam_yol = os.path.join(kok, dosya)
                try:
                    with open(tam_yol, "r", encoding="utf-8") as f:
                        icerik = yaml.safe_load(f)
                        if not isinstance(icerik, dict):
                            continue
                        yollar = icerik.get("paths")
                        if isinstance(yollar, dict):
                            for yol in yollar.keys():
                                temiz_yol = yol.replace("{", "").replace("}", "")
                                parcala = temiz_yol.strip("/").split("/")
                                kelimeler.extend(parcala)
                except:
                    continue
    return kelimeler

def on_isle(kelime_listesi):
    sonuc = []

    for kelime in kelime_listesi:
        kelime = kelime.lower().strip()
        kelime = kelime.translate(str.maketrans('', '', string.punctuation))
        if kelime and kelime not in durak_kelimeler and kelime.isalpha():
            kok = kok_bulucu.lemmatize(kelime)
            sonuc.append(kok)

    return sonuc

if __name__ == "__main__":
    os.makedirs("data/clean", exist_ok=True)

    print("Endpoint kelimeleri okunuyor...")
    ham_kelimeler = kelime_cek(veri_klasoru)

    print(f"Toplam kelime (ham): {len(ham_kelimeler)}")
    temiz_kelimeler = on_isle(ham_kelimeler)
    print(f"Temizlenmiş kelime sayısı: {len(temiz_kelimeler)}")

    frekans = Counter(temiz_kelimeler)
    tablo = pd.DataFrame(frekans.items(), columns=["kelime", "frekans"])
    tablo = tablo.sort_values(by="frekans", ascending=False)
    tablo.to_csv(cikti_dosyasi, index=False)

    print(f"Temiz veri kaydedildi: {cikti_dosyasi}")
    print("En sık geçen 10 kelime:")
    print(tablo.head(10))
