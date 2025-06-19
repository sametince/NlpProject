Bu çalışmada, GitHub üzerinden temin ettiğim openapi-directory-main veri seti kullanılarak yazılım dökümantasyonları arasındaki benzerliklerin ölçülmesine yönelik bir analiz gerçekleştirdim. Amacım, farklı API servislerinin tanımlarını karşılaştırarak içeriksel benzerliklerini ortaya koymak ve bu benzerlikler üzerinden gruplayarak anlamlı sonuçlara ulaşmaktı.
Veri seti, çok sayıda farklı servise ait OpenAPI tanımlarını YAML formatlarında içermekteydi. İlk olarak bu dökümanları uygun biçimde ayrıştırdım ve her bir API’nin açıklama, başlık, endpoint ve parametre gibi bileşenlerini analiz edilebilir metinlere dönüştürdüm. Daha sonra bu metinleri ön işleme tabi tutarak (büyük-küçük harf duyarlılığı giderme, noktalama temizleme, durak kelime çıkarımı vb.), metinsel temsil açısından daha sağlıklı hale getirdim.
Benzerliği ölçmek için metinleri vektörleştirdim ve TF-IDF temsilleri üzerinden API’ler arasındaki içeriksel yakınlıkları hesapladım. Bu analiz sonucunda, benzer işlevleri yerine getiren API'lerin dökümantasyonlarının genellikle birbirine yakın açıklamalara sahip olduğu gözlemlendi.

Kullanılan kütüphaneler:
•	os, yaml: Dosya gezintisi ve YAML dosyalarını okumak için
•	pandas: Veri çerçevesi (DataFrame) oluşturmak ve CSV kaydı yapmak için
•	string: Noktalama işaretlerini temizlemek için
•	tqdm: Dosya tarama sırasında görsel ilerleme çubuğu sağlamak için
•	collections.Counter: Kelime frekanslarını hesaplamak için
•	nltk.corpus.stopwords: İngilizce durak (stop) kelimeleri filtrelemek için
•	nltk.stem.WordNetLemmatizer: Kelimeleri kök (lemma) forma indirmek için (örneğin “users” → “user”)

Model Nasıl Oluşturulur?

1. TF-IDF Modeli Oluşturma
Lemmatize ve stem edilmiş veri setlerinden TF-IDF matrislerini çıkartmak için `tfidf.py` gibi bir script kullanılır:
```python
from sklearn.feature_extraction.text import TfidfVectorizer
# Veri yüklenir ve cümle listesine dönüştürülür
# TF-IDF matrisleri CSV olarak kaydedilir
```

2. Word2Vec Modeli Eğitimi
`Word2Vec.py` adlı script her iki veri seti için 8 farklı parametre kombinasyonu ile (CBOW/Skipgram, window size, vector size) toplam 16 model üretir ve `modeller/` klasörüne kaydeder:
```python
model = Word2Vec(sentences, vector_size=100, window=4, sg=1)
model.save("modeller/word2vec_stemmed_skipgram_pencere4_boyut100.model")
```

Veri Seti Amacı
Veri seti her satırında işlenmiş (temizlenmiş, lemmatize veya stem uygulanmış) bir cümle içerir. Bu cümleler aşağıdaki amaçlar için kullanılır:
- TF-IDF ile her cümleye vektör temsili kazandırmak
- Word2Vec ile her kelimeye vektör kazandırarak cümleleri temsil etmek
- Giriş cümlesine en benzer diğer cümleleri bulmak

Kullanım

Tüm analizleri çalıştırmak için:
```bash
python benzerlik.py
```
Bu komut ile:
- TF-IDF ve Word2Vec modelleri yüklenir
- Giriş cümlesine en yakın 5 cümle her model için yazdırılır
- Anlamsal değerlendirme puanları hesaplanır
- Jaccard benzerlik matrisi grafik olarak gösterilir ve `jaccard_matrisi.png` dosyasına kaydedilir

---
