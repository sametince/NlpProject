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

    # Veri dosyasını okuyalım (başlıksız olduğundan header=None)
    veri = pd.read_csv(yol, header=None)

    # Her satırdaki kelimeleri birer cümle gibi birleştir
    cumleler = [" ".join(satir.dropna().astype(str)) for _, satir in veri.iterrows()]

    # TF-IDF modelini oluşturalım
    tfidf_modeli = TfidfVectorizer()
    tfidf_sonuclari = tfidf_modeli.fit_transform(cumleler)
    # TF-IDF modelini oluşturalım - sütun sayısını sınırla
    tfidf_modeli = TfidfVectorizer(max_features=10000)
    tfidf_sonuclari = tfidf_modeli.fit_transform(cumleler)
    # Sonuçları pandas DataFrame'e dönüştürelim
    kelimeler = tfidf_modeli.get_feature_names_out()
    tfidf_tablosu = pd.DataFrame(tfidf_sonuclari.toarray(), columns=kelimeler)

    # Excel olarak kaydedelim
    dosya_adi = f"tfidf_{isim}.xlsx"
    tfidf_tablosu.to_excel(dosya_adi, index=False)
    print(f"'{isim}' için TF-IDF dosyası Excel olarak kaydedildi: {dosya_adi}")
