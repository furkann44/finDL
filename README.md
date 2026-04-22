# Asset Direction Project

Bu proje, Twelve Data API uzerinden alinan gunluk finansal zaman serileriyle bir sonraki donemde fiyat yonunu tahmin etmek icin kurulan moduler bir deney hattidir.

Ilk fazda odak noktasi `BTC/USD` icin su akisin uctan uca calismasidir:

1. Ham veriyi cekmek
2. Teknik ozellikleri uretmek
3. Sızıntısiz zaman bazli split ile Logistic Regression egitmek
4. Sequence veri seti ile LSTM egitmek
5. Ayni veri hattinda GRU ile ek sequence baseline kurmak

## Kurulum

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

`.env.example` dosyasini `.env` olarak kopyalayip API anahtarini ekleyin:

```env
TWELVE_DATA_API_KEY=your_api_key_here
```

Dashboard login ayari icin `.streamlit/secrets.toml.example` dosyasini referans alin. Yerel ornek:

```toml
[auth]
username = "admin"
password = "change-me"
```

Alternatif olarak environment variable da kullanabilirsiniz:

```bash
set STREAMLIT_AUTH_USERNAME=admin
set STREAMLIT_AUTH_PASSWORD=change-me
```

## Ilk Faz Komutlari

Sadece `BTC/USD` icin:

```bash
python src/build_raw_data.py
python src/build_processed_data.py
python src/train_baseline.py
python src/train_mlp.py
python src/train_lstm.py
python src/train_gru.py
python src/summarize_results.py
python src/report_results.py
python src/generate_holdout_diagnostics.py
python src/backtest_predictions.py
python src/run_threshold_experiment.py
python src/run_threshold_tuning.py
python src/run_no_trade_tuning.py
python src/run_4h_experiment.py
python src/walk_forward_baseline.py
python src/walk_forward_sequence.py
python src/report_walk_forward.py
```

Tum varliklar icin:

```bash
python src/build_raw_data.py --all
python src/build_processed_data.py --all
python src/train_baseline.py --all
python src/train_mlp.py --all
python src/train_lstm.py --all
python src/train_gru.py --all
python src/summarize_results.py
python src/report_results.py
python src/generate_holdout_diagnostics.py
python src/backtest_predictions.py
python src/run_threshold_experiment.py --all
python src/run_threshold_tuning.py --all
python src/run_no_trade_tuning.py --all
python src/run_4h_experiment.py --all
python src/walk_forward_baseline.py --all
python src/walk_forward_sequence.py --symbols BTC/USD ETH/USD --models lstm gru
python src/rolling_retrain_backtest.py --symbols BTC/USD NVDA --models baseline mlp --optimize-for total_return
python src/report_walk_forward.py
```

Sinif agirliklandirma denemesi icin:

```bash
python src/train_lstm.py --symbols AAPL NVDA --class-weight balanced
python src/train_gru.py --symbols AAPL NVDA --class-weight balanced
```

Dashboard calistirmak icin:

```bash
streamlit run streamlit_app.py
```

Streamlit Community Cloud deploy adimlari icin:

- `DEPLOY_STREAMLIT_CLOUD.md`

Dashboard icinde:

- `Recommended Config`: varlik bazli önerilen urun konfigürasyonu
- `Recommended Config` tablosu artik holdout, no-trade ve rolling retraining sonuclarini birlikte gosterir
- `Asset Detail`: secilen varlik icin signal, equity curve ve ROC/confusion gorselleri
- `Management`: ham veri cekme, feature uretme, egitim ve ozet yenileme adimlarini butonlarla calistirma
- `Rolling Retrain`: gecmise kadar egitip ileri dogru sinyal ureten rolling retraining sonuclari
- `Holdout / No-Trade / Backtest / Threshold Tuning / Walk-Forward`: arastirma ve urun karar katmanlarini birlikte inceleme

Dashboard tablolarinda artik test verisinin geldigi aralik da gosterilir:

- `test_start`
- `test_end`
- `test_rows`
- `active_trade_days`

## Uretilen Ciktilar

- `data/raw/`: ham OHLCV parquet dosyalari
- `data/processed/`: feature eklenmis parquet dosyalari
- `artifacts/metrics/`: baseline metrik ciktilari
- `artifacts/metrics/model_summary.csv`: varlik-model karsilastirma tablosu
- `artifacts/metrics/walk_forward_summary.csv`: walk-forward baseline ozet tablosu
- `artifacts/models/`: kaydedilen sequence model agirliklari
- `artifacts/predictions/`: validation ve test tahmin dosyalari
- `artifacts/backtests/`: basit yon stratejisi backtest ciktilari
- `artifacts/metrics/threshold_experiment_summary.csv`: threshold-based etiketleme deney ozeti
- `artifacts/metrics/intraday_4h_summary.csv`: 4 saatlik baseline deney ozeti
- `artifacts/metrics/no_trade_summary_total_return.csv`: no-trade toplam getiri odakli ozet
- `artifacts/reports/phase6_report.md`: Faz 6 otomatik ozet raporu
- `artifacts/reports/walk_forward_report.md`: walk-forward otomatik raporu
- `artifacts/reports/threshold_experiment_report.md`: threshold etiketleme raporu
- `artifacts/reports/intraday_4h_report.md`: 4 saatlik veri raporu
- `artifacts/figures/`: performans grafik ciktilari

## Bu Fazda Eklenen Moduller

- `src/config.py`: merkezi proje ayarlari ve yol yardimcilari
- `src/twelvedata_client.py`: Twelve Data istemcisi
- `src/build_raw_data.py`: ham veri cekme ve kaydetme
- `src/features.py`: teknik gosterge ve hedef uretimi
- `src/build_processed_data.py`: islenmis veri seti olusturma
- `src/train_baseline.py`: Logistic Regression baseline egitimi ve degerlendirme
- `src/train_mlp.py`: MLP egitimi, dogrulama ve kaydetme
- `src/dataset.py`: sequence veri hazirlama yardimcilari
- `src/models.py`: PyTorch sequence modelleri
- `src/evaluate.py`: ortak metrik yardimcilari
- `src/train_lstm.py`: LSTM egitimi, dogrulama ve kaydetme
- `src/train_gru.py`: GRU egitimi, dogrulama ve kaydetme
- `src/sequence_training.py`: ortak sequence egitim yardimcilari
- `src/summarize_results.py`: metrik JSON dosyalarindan ozet tablo olusturma
- `src/report_results.py`: Faz 6 karsilastirma raporu ve grafik uretimi
- `src/generate_holdout_diagnostics.py`: ROC ve confusion matrix gorselleri
- `src/backtest_predictions.py`: test tahminleri uzerinde basit long-short backtest
- `src/run_threshold_experiment.py`: threshold-based etiketleme deney hattisi
- `src/run_4h_experiment.py`: 4 saatlik veri icin ek baseline deney hattisi
- `src/walk_forward_baseline.py`: expanding-window walk-forward baseline degerlendirmesi
- `src/report_walk_forward.py`: walk-forward raporu ve grafik uretimi
- `src/dashboard_data.py`: dashboard veri toplama ve onerilen konfigürasyon secimi
- `streamlit_app.py`: etkileşimli web arayuzu

## Sonraki Mantikli Adimlar

1. Dashboard icinden deney tetikleme ve refresh aksiyonu eklemek
2. Canli/veriye en yakin inference katmanini eklemek
3. Streamlit yerine servis/API katmanina gecmek
