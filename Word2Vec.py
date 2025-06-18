import pandas as pd
import os
import time
from gensim.models import Word2Vec
import logging

# Konsola bilgi yazdırmak için ayar
logging.basicConfig(format="%(asctime)s : %(levelname)s : %(message)s", level=logging.INFO)

# Kullanılacak parametreler (toplam 8 kombinasyon)
parametreler = [
    {'model_tipi': 'cbow', 'pencere': 2, 'vektor_boyutu': 100},
    {'model_tipi': 'skipgram', 'pencere': 2, 'vektor_boyutu': 100},
    {'model_tipi': 'cbow', 'pencere': 4, 'vektor_boyutu': 100},
    {'model_tipi': 'skipgram', 'pencere': 4, 'vektor_boyutu': 100},
    {'model_tipi': 'cbow', 'pencere': 2, 'vektor_boyutu': 300},
    {'model_tipi': 'skipgram', 'pencere': 2, 'vektor_boyutu': 300},
    {'model_tipi': 'cbow', 'pencere': 4, 'vektor_boyutu': 300},
    {'model_tipi': 'skipgram', 'pencere': 4, 'vektor_boyutu': 300}
]

# Kullanılacak veri dosyaları
veri_setleri = {
    "lemmatized": "lemmatizasyon_kelimeler.csv",
    "stemmed": "stemming_kelimeler.csv"
}

# Modellerin kaydedileceği klasör
os.makedirs("modeller", exist_ok=True)

# Örnek olarak benzer kelimeleri alacağımız kelime
ornek_kelime = "count"

# Raporu tutacak satırlar
rapor_satirlari = []

# Tüm veri setleri için döngü
for veri_adi, dosya_yolu in veri_setleri.items():
    print(f"\n>>> {veri_adi} veri seti işleniyor...")

    # CSV dosyasını oku
    veri = pd.read_csv(dosya_yolu)
    cumleler = veri.values.tolist()  # [['kelime1', 'kelime2', ...], ...]

    # Her parametre seti için model eğit
    for ayar in parametreler:
        sg_degeri = 1 if ayar["model_tipi"] == "skipgram" else 0
        pencere = ayar["pencere"]
        boyut = ayar["vektor_boyutu"]

        model_adi = f"word2vec_{veri_adi}_{ayar['model_tipi']}_pencere{pencere}_boyut{boyut}.model"
        model_yolu = os.path.join("modeller", model_adi)

        print(f"\nModel eğitiliyor: {model_adi}")
        baslangic = time.time()

        # Word2Vec modeli oluştur
        model = Word2Vec(
            sentences=cumleler,
            vector_size=boyut,
            window=pencere,
            sg=sg_degeri,
            min_count=1,
            workers=4,
            epochs=10
        )

        egitim_suresi = time.time() - baslangic
        model.save(model_yolu)

        # Model boyutunu MB cinsinden al
        boyut_mb = os.path.getsize(model_yolu) / (1024 * 1024)

        # "count" kelimesine en yakın 5 kelime
        try:
            benzerler = model.wv.most_similar(ornek_kelime, topn=5)
        except KeyError:
            benzerler = [("Bulunamadı", 0.0)]

        # Rapor için bilgileri ekle
        rapor_satirlari.append(f"Model: {model_adi}")
        rapor_satirlari.append(f"Veri seti: {veri_adi}")
        rapor_satirlari.append(f"Pencere: {pencere}, Vektör boyutu: {boyut}, Model tipi: {ayar['model_tipi']}")
        rapor_satirlari.append(f"Eğitim süresi: {egitim_suresi:.2f} saniye")
        rapor_satirlari.append(f"Model boyutu: {boyut_mb:.2f} MB")
        rapor_satirlari.append("Benzer kelimeler:")
        for kelime, benzerlik in benzerler:
            rapor_satirlari.append(f"  {kelime} -> {benzerlik:.4f}")
        rapor_satirlari.append("-" * 40)

# Raporu dosyaya yaz
with open("word2vec_raporu.txt", "w", encoding="utf-8") as dosya:
    dosya.write("\n".join(rapor_satirlari))

print("\nTüm modeller eğitildi ve 'word2vec_raporu.txt' dosyasına rapor yazıldı.")
