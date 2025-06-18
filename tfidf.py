import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

# Dosya adlarını belirleyelim
veri_dosyalari = {
    "lemmatized": "lemmatizasyon_kelimeler.csv",
    "stemmed": "stemming_kelimeler.csv"
}

# Her veri seti için işlem yapalım
for isim, yol in veri_dosyalari.items():
    print(f"{isim} veri seti işleniyor...")

    # Veri dosyasını okuyalım
    veri = pd.read_csv(yol, header=None)

    # Her satırdaki kelimeleri cümleye çevirelim
    cumleler = [" ".join(satir.dropna()) for _, satir in veri.iterrows()]

    # TF-IDF modeli oluşturalım
    tfidf_modeli = TfidfVectorizer()
    tfidf_sonuclari = tfidf_modeli.fit_transform(cumleler)

    # Sonuçları tabloya dönüştürelim
    kelimeler = tfidf_modeli.get_feature_names_out()
    tfidf_tablosu = pd.DataFrame(tfidf_sonuclari.toarray(), columns=kelimeler)

    # CSV olarak kaydedelim
    tfidf_tablosu.to_csv(f"tfidf_{isim}.csv", index=False)
    print(f"'{isim}' için TF-IDF dosyası kaydedildi.")
