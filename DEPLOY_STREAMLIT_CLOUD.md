# Streamlit Community Cloud Deploy Rehberi

Bu proje, Streamlit Community Cloud uzerinde ucretsiz olarak yayinlanabilir.

## 1. Repo Hazirligi

- Kodlar GitHub reposunda bulunmali
- `requirements.txt` dosyasi guncel olmali
- giris sistemi icin `secrets` tanimlanmali

Repo:

- `https://github.com/furkann44/finDL`

## 2. Streamlit Cloud Uzerinde Uygulama Olusturma

1. `https://share.streamlit.io/` adresine girin
2. GitHub hesabinizla baglanin
3. `New app` secin
4. Repository olarak `furkann44/finDL` secin
5. Branch: `master`
6. Main file path: `streamlit_app.py`

## 3. Secrets Ayarlari

Streamlit Cloud panelindeki `Secrets` alanina su tipte bir yapi ekleyin:

```toml
TWELVE_DATA_API_KEY = "your_api_key_here"

[auth]
username = "admin"
password = "change-me"
```

Isterseniz duz sifre yerine hash de kullanabilirsiniz:

```toml
[auth]
username = "admin"
password_hash_sha256 = "<sha256-hash>"
```

## 4. Ilk Acilista Beklenen Davranis

Bu repoda uretilmis artifact dosyalari git'e konulmuyor.
Bu nedenle deploy sonrasi ilk acilista dashboard acilir ama bazi sekmelerde veri bulunmayabilir.

Bu normaldir.

Ilk kurulum sonrasi:

1. login olun
2. `Management` sekmesine gidin
3. once `Guncel Veri + Ozetleri Yenile` butonunu calistirin
4. gerekiyorsa `Secili Varlik Icin Tam Pipeline` veya ozel pipeline calistirin

Bu adimlardan sonra:

- holdout ozetleri
- no-trade tablolari
- backtest sonuc dosyalari
- rolling retraining ciktilari

olusturulacak ve dashboard dolacaktir.

## 5. Notlar

- Ucretsiz Streamlit yayini `*.streamlit.app` alan adiyla gelir
- Tam ozel domain ayari genelde ekstra yonlendirme veya baska servis gerektirir
- Bu proje icin en pratik ucretsiz yayin modeli Streamlit'in verdigi alt alan adidir

## 6. Onerilen Ilk Kullanim

Deploy sonrasi ilk asamada tum pipeline'i calistirmak yerine:

- once `BTC/USD` ve `NVDA` ile baslayin
- sonra tum varliklara genisletin

Bu yaklasim hem sureyi kisaltir hem de hata ayiklamayi kolaylastirir.
