# Northwind DBSCAN Kümeleme ve API Projesi

Bu proje, Northwind veritabanındaki müşteri, ürün, tedarikçi ve ülke verilerini kullanarak DBSCAN kümeleme analizleri yapar ve bu modelleri FastAPI aracılığıyla sunar.

## İçerik

1.  **Analiz Scriptleri (`sample*.py`)**: Veritabanından veri çeker, özellikleri hesaplar, veriyi ölçekler, DBSCAN için `eps` ve `min_samples` parametrelerini optimize eder, modeli eğitir, sonuçları görselleştirir ve eğitilmiş modeli (`scaler` ve `dbscan`) `.pkl` dosyası olarak kaydeder.
    *   `sample1.py`: Müşteri Segmentasyonu
    *   `sample2.py`: Ürün Kümeleme
    *   `sample3.py`: Tedarikçi Segmentasyonu
    *   `sample4.py`: Ülkelere Göre Satış Deseni Analizi
2.  **API Scriptleri (`api*.py`)**: Kaydedilmiş `.pkl` modellerini yükler ve FastAPI kullanarak ilgili kümeleme modelini sunan bir API endpoint'i sağlar.
    *   `api1.py`: Müşteri Segmentasyonu Modeli (Port 8001)
    *   `api2.py`: Ürün Kümeleme Modeli (Port 8002)
    *   `api3.py`: Tedarikçi Segmentasyonu Modeli (Port 8003)
    *   `api4.py`: Ülke Satış Deseni Modeli (Port 8004)
3.  **Model Dosyaları (`model*.pkl`)**: `joblib` ile kaydedilmiş, ilgili `StandardScaler` ve eğitilmiş `DBSCAN` modelini içeren dosyalar.
4.  **Gereksinimler (`requirements.txt`)**: Projenin çalışması için gerekli Python kütüphaneleri.

## Kurulum

### Ön Gereksinimler

*   Python 3.7+
*   PostgreSQL veritabanı sunucusu
*   Northwind veritabanı (PostgreSQL'e yüklenmiş olmalı)

### Veritabanı Bağlantısı

`sample*.py` dosyalarının başındaki veritabanı bağlantı bilgilerini (`user`, `password`, `host`, `port`, `database`) kendi PostgreSQL kurulumunuza göre güncelleyin:

```python
# --- Database Connection ---
user = "sizin_kullanici_adiniz"
password = "sizin_sifreniz"
host = "localhost" # veya sunucu adresiniz
port = "5432" # veya port numaranız
database = "Northwind"
```

### Bağımlılıkların Yüklenmesi

Proje dizininde bir terminal veya komut istemcisi açın ve aşağıdaki komutu çalıştırın:

```bash
pip install -r requirements.txt
```

## Kullanım

### 1. Analizlerin Çalıştırılması ve Modellerin Eğitilmesi

Her bir analiz scriptini ayrı ayrı çalıştırarak ilgili modeli eğitebilir ve `.pkl` dosyasını oluşturabilirsiniz:

```bash
python sample1.py
python sample2.py
python sample3.py
python sample4.py
```

Bu scriptler çalışırken:
*   Veritabanına bağlanıp veriyi çekecek.
*   Optimum `eps` ve `min_samples` parametrelerini bulmak için k-mesafe grafikleri ve siluet skorlarını kullanacak (grafikler gösterilebilir).
*   DBSCAN kümelemesini yapacak.
*   Sonuçları `matplotlib` ile görselleştirecek (grafikler gösterilebilir).
*   Bulunan kümeler ve aykırı değerler hakkında bilgi yazdıracak.
*   `model*.pkl` dosyalarını kaydedecek.

### 2. API Sunucularının Başlatılması

Her bir model için API sunucusunu ayrı bir terminalde başlatabilirsiniz:

```bash
# Terminal 1
uvicorn api1:app --reload --host 127.0.0.1 --port 8001

# Terminal 2
uvicorn api2:app --reload --host 127.0.0.1 --port 8002

# Terminal 3
uvicorn api3:app --reload --host 127.0.0.1 --port 8003

# Terminal 4
uvicorn api4:app --reload --host 127.0.0.1 --port 8004
```

`--reload` parametresi, kodda değişiklik yaptığınızda sunucunun otomatik olarak yeniden başlamasını sağlar.

### 3. API Endpoint'lerinin Kullanımı

API'ler çalışır durumdayken, `/predict` endpoint'ine POST isteği göndererek yeni veriler için küme tahminleri alabilirsiniz. Her API, kendi modeline uygun yapıda bir JSON listesi bekler.

**Örnek (API 1 - Müşteri Segmentasyonu - Port 8001):**

```bash
curl -X 'POST' \
  'http://127.0.0.1:8001/predict' \
  -H 'accept: application/json' \
  -H 'Content-Type: application/json' \
  -d '[
  {
    "total_orders": 10,
    "total_spent": 1250.75,
    "avg_order_value": 125.08
  },
  {
    "total_orders": 2,
    "total_spent": 300.50,
    "avg_order_value": 150.25
  }
]'
```

**Beklenen Yanıt:**

```json
{
  "clusters": [
    -1,  // İlk müşteri için küme etiketi (-1 = aykırı)
    0    // İkinci müşteri için küme etiketi (0 = küme 0)
  ]
}
```

Diğer API'ler için de benzer şekilde, ilgili `api*.py` dosyasındaki Pydantic modeline uygun veri yapısını kullanarak istek gönderebilirsiniz.

Ayrıca, tarayıcınızdan API'lerin `/docs` adresine giderek (örn. `http://127.0.0.1:8001/docs`) otomatik oluşturulan Swagger UI arayüzünü kullanabilir, endpoint'leri inceleyebilir ve doğrudan test edebilirsiniz. Buradaki örnekler de size yardımcı olacaktır.

## Notlar

*   `min_samples` ve `eps` optimizasyonu, veri setinin özelliklerine göre farklı sonuçlar verebilir.
*   DBSCAN'ın `-1` olarak etiketlediği noktalar, herhangi bir kümeye ait olmayan aykırı değerlerdir. 
