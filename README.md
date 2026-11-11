# ISG_VISION — Yapay Zekâ Destekli Çoklu Çalışan Tanıma ve İSG Denetim Sistemi

> **Mevcut durum: Aşama 3 — Canlı kamera altyapısı `IN_PROGRESS`.** Kamera/video okuma, FPS ve canlı pencere kodu yazıldı ve otomatik testlerden geçti;
> gerçek kamera donanım doğrulaması kullanıcı tarafından (macOS kamera izniyle) yapılacaktır (`READY_FOR_MANUAL_VERIFICATION`). Ayrıntı: [docs/camera_module.md](docs/camera_module.md).
> Aşama 1 (geliştirme ortamı) ve Aşama 2 (veri seti analizi, `COMPLETED`: `reports/dataset_audit/STAGE_2_CLOSURE_REPORT.md`) tamamlanmıştır.
> Tespit, takip, yüz tanıma ve KKD denetimi özellikleri, veri birleştirme ve model eğitimi **henüz geliştirilmemiştir**; Aşama 4 başlatılmamıştır.

## 1. Proje adı

**Yapay Zekâ Destekli Çoklu Çalışan Tanıma ve İSG Denetim Sistemi** (teknik kök klasör: `ISG_VISION`)

## 2. Akademik amaç

Yazılım Mühendisliği lisans bitirme projesidir. Bağımsız bir üniversite çalışmasıdır; herhangi bir kurumun
altyapısına veya onayına bağlı değildir. Geliştirme; modüler mimari, karşılaştırmalı deneyler, model performans
analizi (precision, recall, mAP, kimlik tanıma ölçütleri), gerçek zamanlı performans testleri (FPS, gecikme) ve
teknik dokümantasyonla akademik olarak değerlendirilebilir olacak şekilde yürütülür.

## 3. Çözülmek istenen problem

Fabrika, üretim tesisi ve şantiye gibi ortamlarda kişisel koruyucu donanım (KKD) kullanımının elle denetlenmesi
zor ve hataya açıktır. Hedef; tek bir kameradan aynı anda birden fazla çalışanı tespit edip takip eden, kayıtlı
çalışanları yüzlerinden tanıyan ve her çalışanın baret ile güvenlik yeleği kullanımını ayrı ayrı denetleyen bir
sistem geliştirmektir.

## 4. Birinci etap hedefleri (uzun vadeli, henüz uygulanmadı)

- Canlı kamera görüntüsü, çoklu insan tespiti ve çoklu insan takibi
- Çoklu yüz tespiti ve yüzlerin doğru insan takipleriyle ilişkilendirilmesi
- En fazla 10 kayıtlı çalışanın tanınması; güvenilir eşleşmeyen kişiler "Bilinmeyen kişi / Kimlik belirlenemedi"
- Baret ve yelek durumunun çalışan bazında değerlendirilmesi (kişi–ekipman eşleştirme, "Belirsiz" kararı desteği)
- Canlı sonuç ekranı ve temel olay kayıtları

İkinci etap (sensörler, MQTT, edge AI vb.) şu an kapsam dışıdır.

## 5. Kullanılacak teknolojiler

| Alan | Teknoloji | Aşama 1'de kurulu mu? |
|---|---|---|
| Dil | Python 3.11 | Evet |
| Derin öğrenme | PyTorch (+ torchvision), MPS hızlandırma | Evet |
| Görüntü işleme | OpenCV | Evet |
| Nesne tespiti | Ultralytics YOLO | Evet (model indirilmedi) |
| Yüz tespiti | MediaPipe (veya SCRFD) | MediaPipe kurulu, kullanılmadı |
| Çoklu kişi takibi | ByteTrack / BoT-SORT | Hayır (Aşama 4) |
| Yüz tanıma | ArcFace tabanlı önceden eğitilmiş model | Hayır (Aşama 7) |
| Sürüm kontrolü | Git | Evet |

## 6. Proje klasör mimarisi

```
ISG_VISION/
├── data/
│   ├── raw/ (construction_ppe, mendeley_ppe, site_safety)
│   ├── processed/  merged_ppe/  employees/
├── models/ (human_detection, tracking, face_detection, face_recognition, ppe_detection)
├── src/isg_vision/
│   ├── camera/  human_detection/  tracking/  face_detection/
│   ├── face_recognition/  ppe_detection/  integration/  storage/  utils/
├── configs/   scripts/   tests/   results/   reports/   docs/
├── scripts/git-hooks/pre-commit        (biyometrik veri koruması)
├── .vscode/settings.json               (.venv yorumlayıcısı, unittest ayarı)
├── requirements.txt   requirements-lock.txt   pyproject.toml
├── .gitignore   .env.example   README.md
```

Modül ilişkileri için bkz. [docs/architecture.md](docs/architecture.md).

## 7. macOS üzerinde kurulum

Python 3.11 gereklidir (Homebrew ile):

```bash
brew install python@3.11
```

## 8. Python sanal ortamı

```bash
cd /proje/yolu            # projenin bulunduğu klasör
/opt/homebrew/bin/python3.11 -m venv .venv
source .venv/bin/activate
python --version          # Python 3.11.x görmelisiniz
```

Kütüphaneleri asla sistem Python'una yüklemeyin; her zaman `.venv` etkinken çalışın.

VS Code: `.vscode/settings.json` yorumlayıcıyı otomatik olarak `.venv/bin/python` yapar ve `unittest` keşfini `tests/`
klasörüne ayarlar. Yorumlayıcı otomatik algılanmazsa: Cmd+Shift+P → "Python: Select Interpreter" → `.venv`.

## 9. Bağımlılık kurulumu

```bash
pip install --upgrade pip
pip install -r requirements.txt     # doğrudan bağımlılıklar
pip install -e . --no-deps          # isg_vision paketini geliştirme modunda kur
```

Birebir aynı ortam için tam sürüm kilidi: `pip install -r requirements-lock.txt`.

> Not: `mediapipe`, `opencv-contrib-python` gerektirir; Ultralytics ise `opencv-python` ister. İki paket aynı
> sürümde (5.0.0.93) sabitlenmiştir ve `pip check` çakışma bildirmemektedir.

## 10. MPS desteğinin kontrolü

```bash
python scripts/check_mps.py
```

MPS kullanılamazsa kod `cpu` cihazına geri düşebilir (CUDA gerekmez).

## 11. Ortam doğrulama komutları

```bash
python scripts/check_environment.py          # kritik hata varsa çıkış kodu 1
python -m unittest discover -s tests -v      # testler
```

Bu komutlar veri indirmez, kamera açmaz ve model yüklemez.

## 12. Veri seti klasörleri

| Klasör | Planlanan veri seti |
|---|---|
| `data/raw/construction_ppe/` | Ultralytics Construction-PPE |
| `data/raw/mendeley_ppe/` | Mendeley PPE v6 |
| `data/raw/site_safety/` | Roboflow Construction Site Safety v27 |
| `data/processed/`, `data/merged_ppe/` | İşlenmiş (Aşama 8 eğitim hazırlığı) / birleştirilmiş veriler (Aşama 9) |
| `data/employees/` | Çalışan referans fotoğrafları (yalnızca yerel, Git dışı) |

Aşama 2'de üç ZIP `~/Downloads` içinden `data/raw/` altına güvenli biçimde çıkarılmıştır (orijinal ZIP'ler yerinde duruyor); veriler
birleştirilmemiş, değiştirilmemiş ve yeniden bölünmemiştir.

## 13. Veri güvenliği yaklaşımı

- Ham/işlenmiş veri setleri, çalışan fotoğrafları, yüz embedding'leri, biyometrik kayıtlar, model ağırlıkları ve
  `.env` dosyaları `.gitignore` ile Git dışında tutulur (`data/` altında yalnızca klasörler ve `.gitkeep` takip edilir).
- Çalışan fotoğrafları yalnızca hukuki gerekliliklere (KVKK, açık rıza) uygun olarak toplanmalıdır.
- Kimlik tanıma (biyometrik) modülü ayrı bir pakette (`face_recognition`) tutulur.
- Uzak (GitHub) depo oluşturulmamıştır; otomatik commit/push yoktur.
- İkinci koruma katmanı olarak **pre-commit hook** kullanılır (aşağıda).

### Pre-commit hook (biyometrik veri koruması)

`scripts/git-hooks/pre-commit`, commit'e eklenen dosyaların yolunu ve uzantısını kontrol eder ve şunlar
varsa commit'i **reddeder**: `data/` altındaki her şey (`.gitkeep` hariç; çalışan fotoğrafları dahil),
`employees/`, `faces/`, `face_db/`, `biometric/`, `embeddings/` klasörlerindeki görüntüler, embedding dosyaları
(`.npy`, `.npz`, `.pkl`, `.faiss`, `.embedding` …), model ağırlıkları (`.pt`, `.pth`, `.onnx` …), arşivler (`.zip` …),
veritabanları, anahtar dosyaları ve `.env`. (`.env.example` serbesttir.)

Kurulum (depoyu klonladıktan/oluşturduktan sonra bir kez):

```bash
git config core.hooksPath scripts/git-hooks
chmod +x scripts/git-hooks/pre-commit
```

Bu depoda kurulum zaten yapılmıştır; `python scripts/check_environment.py` hook'un etkin olup olmadığını uyarı olarak bildirir.

**Sınırlar (bilmeniz gerekenler):**
- Hook yalnızca yerel makinede çalışır; `git commit --no-verify` ile atlanabilir.
- `core.hooksPath` ayarı depoyla birlikte taşınmaz; her klonda yukarıdaki komut yeniden çalıştırılmalıdır.
- Kontrol **dosya yoluna ve uzantısına** bakar, içeriğe bakmaz. Yasaklı olmayan bir ad/uzantıyla (ör. `ali.dat`,
  yeniden adlandırılmış bir fotoğraf) kaydedilen biyometrik veri tespit edilmeyebilir. Bu nedenle çalışan verileri yalnızca
  `data/employees/` altında tutulmalıdır.
- Hook yanlışlıkla ekleme (kaza) önlemek içindir; kasıtlı aşmaya veya zaten Git geçmişine girmiş verilere karşı koruma değildir.
- `.gitignore` korumasıyla birlikte çalışır; ikisi birbirinin yerine geçmez.

## 13.1 Aşama 2 — Veri seti denetimi (komutlar)

Ham veri setleri `data/raw/` altındadır (salt-okunur, Git dışı). Denetim araçları yalnızca OKUR; birleştirme, yeniden bölme,
etiket düzeltme, silme yapmaz. Betikler sırayla çalıştırılabilir (hepsi tekrar çalıştırılabilir):

```bash
python scripts/extract_datasets.py --dry-run    # ZIP'leri bul, ne yapılacağını göster
python scripts/extract_datasets.py              # ~/Downloads içindeki ZIP'leri güvenli çıkar (üzerine yazmaz)
python scripts/inspect_datasets.py              # klasör yapısı, YAML, envanter
python scripts/validate_labels.py               # etiket doğrulama -> annotation_issues.csv
python scripts/analyze_class_distribution.py    # sınıf dağılımı + grafikler
python scripts/detect_duplicates.py             # SHA-256 + dHash yakın kopya analizi
python scripts/visualize_annotations.py         # görsel doğrulama örnekleri (Git'e eklenmez)
python scripts/generate_dataset_report.py       # Türkçe rapor + eşleme CSV + manuel kontrol listesi
```

Çıktılar `reports/dataset_audit/` altındadır; ana rapor `dataset_audit_report.md`, manuel onay listesi `manual_review_checklist.md`.
Veri seti kayıtları (kaynak, lisans, yayımlanan sayılar): `configs/datasets.yaml`; aday sınıf eşlemesi (kesin DEĞİL):
`configs/candidate_class_mapping.yaml`.

### Aşama 2 — eksik giderme (ek analizler)

```bash
python scripts/analyze_box_semantics.py --render      # kutu semantiği ölçüleri + karşılaştırma/çelişki sayfaları
python scripts/review_roboflow_labels.py --render     # Roboflow şüpheli etiket sayfaları; kararlar configs/review/label_review_verdicts.yaml
python scripts/analyze_augmentation_artifacts.py      # siyah kare (cutout) sayımı
python scripts/verify_duplicates.py                   # pHash + ORB-LSH + ORB/RANSAC yakın kopya doğrulaması (~4-5 dk)
python scripts/generate_followup_reports.py           # semantic/duplicate/license/roboflow raporları + 59 maddelik kontrol listesi
python scripts/generate_dataset_report.py             # ana rapor (önce yukarıdakiler çalışmalı)
```

Yeni raporlar (`reports/dataset_audit/`): `semantic_validation_report.md`, `roboflow_label_review_report.md`, `roboflow_cleanup_suggestions.md`,
`duplicate_verification_report.md`, `license_verification_report.md`, `manual_review_checklist.md`, `remaining_manual_reviews.md`.
Görsel inceleme kayıtları ve lisans kanıtı `configs/review/` altındadır. Yeni bağımlılık eklenmemiştir (ORB/RANSAC `opencv-contrib-python`
içindeki OpenCV ile yapılır). Karşılaştırma görüntüleri (`semantic_review/`, `roboflow_review/`, `duplicate_verification/`) Git'e eklenmez.

### Aşama 2B — veri kullanım politikası ve kapanış

```bash
python scripts/build_source_groups.py        # kaynak grupları / bölme birimleri (sızıntı grafiği); mevcut bölünmeleri DEĞİŞTİRMEZ
python scripts/apply_usage_policy.py         # politikayı değerlendirir: image_usage_decisions.csv, quarantine_candidates.csv, needs_reannotation_list.csv
python scripts/generate_stage2b_reports.py   # final_dataset_decisions.md, manual_review_checklist.md (nihai), remaining_manual_reviews.md
```

Politika: `configs/dataset_usage_policy.yaml` (onaylı kapsam yalnızca Construction-PPE; Mendeley/Roboflow `pending_license_verification`).
Kararlar ve kanıtlar: `configs/review/stage2b_decisions.yaml`, `reports/dataset_audit/final_dataset_decisions.md`. Atıf kayıtları: `docs/dataset_attributions.md`.
Politika ham veriyi değiştirmeden manifeste uygulanmıştır: 990 kullanılabilir Construction-PPE görüntüsü (794 train, 196 validation).
Kullanıcı görsel kararları: 13 `approved`, 4 `needs_reannotation`, 5 `quarantine`. Mendeley ve Roboflow beklemededir.

## 13.2 Aşama 3 — Canlı kamera altyapısı (komutlar)

Durum: `IN_PROGRESS`. Amaç: MacBook kamerasından veya video dosyasından kare okuyan, FPS ölçen ve durum bilgisini OpenCV penceresinde gösteren,
sonraki aşamalardaki modüllerin (insan tespiti, takip, yüz, KKD) aynı kare akışını kullanabileceği modüler altyapı. Bu aşamada **hiçbir yapay zekâ
modeli çalıştırılmaz**, görüntü kaydedilmez. Tam teknik doküman: [docs/camera_module.md](docs/camera_module.md).

Kod: `src/isg_vision/camera/` (`config.py`, `frame.py`, `capture.py`, `fps.py`, `display.py`, `preview.py`, `cli.py`, `errors.py`), başlatıcı: `scripts/run_camera.py`.

```bash
.venv/bin/python scripts/run_camera.py --source camera                                   # varsayılan kamera, 1280x720 istenir
.venv/bin/python scripts/run_camera.py --source camera --camera-index 1                  # kamera indeksi seç
.venv/bin/python scripts/run_camera.py --source camera --width 640 --height 480          # çözünürlük (ikisi birlikte)
.venv/bin/python scripts/run_camera.py --source video --video-path /yol/video.mp4        # MP4 / MOV / AVI
.venv/bin/python scripts/run_camera.py --source camera --no-display --max-frames 100     # penceresiz kısa deneme
.venv/bin/python scripts/run_camera.py --help
```

* Pencerede: kaynak, **ölçülen FPS**, kaynağın bildirdiği FPS, gerçek çözünürlük, durum. Çıkış: `Q` / `Esc` / pencereyi kapatma / terminalde `Ctrl+C`.
* **macOS kamera izni:** programı çalıştıran uygulamaya (Terminal, VS Code ...) Sistem Ayarları → Gizlilik ve Güvenlik → Kamera bölümünden izin verilmelidir;
  `not authorized to capture video` hatası izin eksikliğini gösterir. Bu proje izinleri değiştirmez veya aşmaz.
* Kamera/video hataları (izin yok, kamera yok, kare okunamadı, bağlantı koptu, dosya yok/bozuk, pencere açılamadı) anlaşılır mesajla ve çıkış koduyla bildirilir; kaynaklar her koşulda serbest bırakılır.
* Testler: `.venv/bin/python -m unittest discover -s tests` (gerçek kamera açmaz, izin istemez, ağ kullanmaz).

## 14. Geliştirme aşamaları

Proje 12 aşamada geliştirilir. Bu sıra proje boyunca **tek referanstır**: [docs/project_plan.md](docs/project_plan.md)
(makine tarafından okunur sürüm: `configs/project_stages.yaml`). Her aşamanın ayrıntılı kapsamı o aşamanın başında belirlenir.

1. **Proje ortamının hazırlanması** — tamamlandı
2. **Veri setlerinin analizi ve doğrulanması** — `COMPLETED` (22/22 kullanıcı kararı uygulandı; 990 görüntü = 794 train + 196 validation)
3. **Canlı kamera altyapısı** — `IN_PROGRESS`: OpenCV kamera erişimi, video akışı, FPS hesaplama, kamera hata yönetimi, çoklu kişi analizi için akış altyapısı (YOLO eğitimi ve veri birleştirme YOK)
4. **Çoklu insan tespiti ve takibi**
5. **Çoklu yüz tespiti**
6. **Çalışan kayıt sistemi**
7. **Çoklu çalışan kimlik tanıma**
8. **İlk YOLO İSG modelinin eğitilmesi** — türetilmiş eğitim verisi ve manifest tabanlı sınıf dönüşümü bu aşamanın hazırlığında
9. **Veri setlerinin birleştirilmesi**
10. **Geliştirilmiş YOLO İSG modelinin eğitilmesi**
11. **Kimlik tanıma ve İSG kontrolünün birleştirilmesi**
12. **Canlı arayüz, sistem testleri ve raporlama**

Aşama kapanış kararları (Aşama 3 dahil) ve sonraki aşamaya geçiş kullanıcıya aittir.

## 15. Mevcut geliştirme durumu

**Aşama 3 — Canlı kamera altyapısı (`IN_PROGRESS`).** Eklenenler: `FrameSource` (kamera/video ortak kare okuma), `FramePacket` (kare + metadata), `FpsMeter`, OpenCV
canlı pencere (overlay), `scripts/run_camera.py`, hata yönetimi ve `tests/test_camera_*.py` otomatik testleri. Gerçek kamera donanım doğrulaması bekleniyor
(`READY_FOR_MANUAL_VERIFICATION`). Aşama 4 başlatılmamıştır.

**Aşama 2 — Veri seti analizi (`COMPLETED`).** Aşama 1'de tamamlananlar: klasör iskeleti, `.venv` (Python 3.11), temel bağımlılıklar,
`scripts/check_mps.py`, `scripts/check_environment.py`, `unittest` tabanlı temel testler, `.gitignore`, pre-commit hook,
VS Code ayarları, bu dokümantasyon. Aşama 2'de eklenenler: güvenli ZIP çıkarma, veri seti envanteri, etiket doğrulama, sınıf dağılımı,
tekrar/yakın kopya analizi, görsel doğrulama örnekleri, aday sınıf eşleme taslağı, Türkçe denetim raporu ve ilgili testler.
22/22 nihai kullanıcı kararı uygulanmış; onaylı Construction-PPE kapsamı 990 görüntüye indirilmiştir.
**Geliştirilmeyenler:** insan tespiti/takibi, yüz tespiti/tanıma, KKD tespiti, model eğitimi,
veri seti birleştirme/yeniden bölme/etiket dönüştürme, arayüz, olay kaydı.
