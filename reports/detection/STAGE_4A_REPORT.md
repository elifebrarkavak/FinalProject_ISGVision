# Aşama 4A — Çoklu İnsan Tespiti: Sonuç Raporu

**Tarih:** 2026-11-12 · **Durum: `COMPLETED`** (yalnızca kapsam içi prototip geliştirme ve doğrulama; resmî kapanış 2026-11-12, kullanıcının gerçek MacBook doğrulamalarıyla)
**Aşama 4 genel durumu: `IN_PROGRESS`** (4B takip `PLANNED`, başlatılmadı) · **Teknik belge:** [docs/person_detection.md](../../docs/person_detection.md)
**Durum geçmişi:** ilk teslim `READY_FOR_MANUAL_VERIFICATION` (Claude ortamında kamera izni olmadığından gerçek kamera testleri yapılamamıştı) → kullanıcı bu testlerin büyük bölümünü kendi MacBook'unda yaptı (§19) → `COMPLETED`.

> **Özet.** Hazır `yolo11n.pt` ile kamera paketinin kare akışındaki her karede aynı anda birden fazla insan tespit edilir (yalnızca `person`); kutu, güven skoru ve kişi sayısı canlı gösterilir; FPS ve çıkarım gecikmesi ölçülür.
> Gerçek model çıkarımı CPU ve MPS'te çalıştırılarak doğrulandı; kullanıcı canlı MacBook kamerasında üç kişiyi aynı anda tespit ettiğini (`PERSONS: 3`), penceresiz canlı hızın CPU ve MPS'te 30,0 FPS olduğunu ve Q/Esc/kırmızı düğme/Ctrl+C ile çıkışın çalıştığını doğruladı.
>
> **Kapanışın sınırı.** `COMPLETED`, yalnızca bu aşamanın kapsam içi prototip geliştirme ve doğrulamasını kapsar. **Kapatılmış sayılmayanlar** (§20–§21, §23): bağımsız doğruluk değerlendirmesi (OT-4A-002) —
> **Precision/Recall/F1/mAP ölçülmedi ve %85 hedefinin karşılandığı iddia edilmez**; gerçek IP kamera doğrulaması (OT-4A-003, `DEFERRED_TO_SYSTEM_INTEGRATION`); ilk oturumdaki en yüksek 7 tespit gözlemi (OBS-4A-001 / OT-4A-004); yorgan yanlış pozitifi (FP-4A-001, bilinen sınırlama; kapanıştan sonra yapılan `conf=0.35` testinde de kısa süreliğine görüldü, **çözülmedi** — §23).
>
> **Kapanış sonrası güncelleme (§23):** OT-4A-001 (`conf=0.35`) kullanıcı tarafından gerçekleştirildi (`TEST_PERFORMED`); yorgan 0,35'te de kısa süreliğine insan sanıldı (süre gözle daha kısaydı, ölçülmedi), gerçek insan tespitinin başarısı ölçülmedi; varsayılan `conf` 0,25 değişmedi. 4A `COMPLETED`, 4B `PLANNED`, Aşama 4 `IN_PROGRESS` kalır.

**Kanıt türleri bu raporda ayrı tutulmuştur:** *[Ölçüldü]* = Claude'un çalıştırıp sonucunu gördüğü; *[Mock]* = sahte YOLO ile test; *[Gerçek model]* = gerçek `yolo11n.pt` çıkarımı (Claude); *[Kullanıcı]* = kullanıcının kendi MacBook'unda gerçek terminal çıktısından/gözleminden bildirdiği, Claude'un yeniden üretemediği/bağımsız doğrulayamadığı sonuç; *[Türetilmiş]* = bildirilen sayılardan yapılan aritmetik; *[Bekliyor]* = yapılmamış, kullanıcı görevi.

---

## 1. Projenin devralınan durumu

| Konu | Başlangıç durumu *[Ölçüldü]* |
|---|---|
| Git | `main`, çalışma ağacı temiz, son commit `4b204a3 Complete Stage 3 camera infrastructure` |
| Aşama durumları | 1, 2, 3 `COMPLETED`; `configs/project_stages.yaml`'da 4 = `planned`; `human_detection/` yalnızca boş paket iskeleti (`"Henüz uygulanmadı"`) |
| Testler | `unittest discover -s tests`: **340 test, OK (skipped=3)** (3 isteğe bağlı GUI testi) |
| Aşama 2 bütünlük | `run_stage2c_checks.py`: **12/12** |
| Ortam | Python 3.11.17 `arm64`, torch 2.15.0 (`mps built=True`, `available=True`), torchvision 0.29.2, **ultralytics 8.4.210**, OpenCV 5.0.0, NumPy 2.4.6. `lap` ve `scipy` kurulu değil |
| Hazır model | Yok (`models/human_detection/` yalnızca `.gitkeep`) |
| Kamera altyapısı | `FrameSource.frames()` → `FramePacket` (kare, `frame_index`, monotonic `timestamp`), `FpsMeter`, `PreviewWindow` (macOS kırmızı düğme düzeltmeli), `run_preview` — **yeniden yazılmadı** |

Aşama 3 kapanış belgeleri okundu: kameranın bildirdiği FPS (15,0) gerçek hızı (~30) yansıtmıyor; bu yüzden bu aşamada hiçbir hesapta kaynak FPS kullanılmadı, `FramePacket.timestamp` ve monotonic zaman kullanıldı.

## 2. Kullanılan YOLO modeli ve sürümü

**Ultralytics YOLO11n** (`yolo11n.pt`), COCO üzerinde önceden eğitilmiş, 80 sınıflı nesne tespiti. Kurulu `ultralytics==8.4.210` ile **gerçekten yüklenip çalıştırılarak** doğrulandı *[Gerçek model]*.

| Alan | Değer *[Ölçüldü]* |
|---|---|
| Parametre sayısı | 2 624 080 |
| Dosya boyutu | 5 613 764 bayt (≈5,35 MiB) |
| SHA-256 | `0ebbc80d4a7680d14987a577cd21342b65ecfd94632bd9a8da63ae6417644ee1` |
| Eğitim verisi / `imgsz` | COCO (`coco.yaml`) / 640 (checkpoint `train_args`) |
| Checkpoint | `version=8.2.100`, `date=2024-09-25` (checkpoint içi metadata) |
| `person` sınıf kimliği | `0` — **`model.names` metadata'sından okundu**; kod `0`'ı varsaymaz, `person` yoksa/belirsizse modeli reddeder |
| Konum | `models/human_detection/yolo11n.pt` — `git check-ignore`: `.gitignore:75 *.pt` → **Git dışı** |

**Seçim gerekçesi ve dürüst sınır.** En küçük YOLO11 tespit modeli (düşük gecikme önceliği); klasik NMS tabanlı (`iou` eşiği anlamlı); kurulu sürümün resmî asset listesinde (`GITHUB_ASSETS_NAMES`) mevcut; CPU ve MPS'te çalıştığı gösterildi.
**Diğer modellerle ölçülmüş bir karşılaştırma yapılmadı**: kurulu sürüm `yolo26n.pt`, `yolov8n.pt` vb. adları da tanıyor, ancak bunlar indirilmedi/değerlendirilmedi. "En iyi model" iddiası yoktur.

**İndirme kaydı.** Kullanıcı talimatı kapsamında, yalnızca resmî kaynaktan, bu projenin izin-listeli indirme betiğiyle (`scripts/download_person_model.py`) tek kez indirildi:
kaynak `https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo11n.pt`; indirilen dosyanın SHA-256'sı, GitHub release API'sinin ilgili asset için verdiği `digest` değeriyle (`sha256:0ebbc80d…ee1`, `size=5613764`) **eşleşti**; ardından `weights.py`'de sabitlendi.
Bundan sonra betik yalnızca doğrular; var olan dosyanın üzerine yazmaz. İndirme dışında hiçbir ağ işlemi yapılmadı (aşağıda §5).

## 3. Modelin lisans ve kaynak bilgileri

* **Lisans: AGPL-3.0** — hem checkpoint metadata'sında (`license: AGPL-3.0 License (https://ultralytics.com/license)`) hem `ultralytics` paket metadata'sında (`License: AGPL-3.0`) *[Ölçüldü]*.
* Ticari/kapalı kaynak kullanım için Ultralytics'in ayrı (Enterprise) lisansı gerekir: <https://ultralytics.com/license>. AGPL'nin ağ üzerinden hizmet/dağıtım yükümlülükleri proje ürünleştirilirse yeniden değerlendirilmelidir. Bu bir öğrenci prototipidir; ağırlık Git'e eklenmez.
* Kaynak: Ultralytics resmî `assets` GitHub deposu (release `v8.4.0`). Eğitim verisi COCO'dur (genel sahneler; şantiye/baret odaklı değildir).
* Güvenlik notu: `.pt` dosyaları `pickle` içerir; yalnızca izin-listesi + sabit SHA-256 kabul edilir.

## 4. Oluşturulan / değiştirilen dosyalar

**Yeni (kaynak):** `src/isg_vision/human_detection/{errors,config,results,person_detector,visualization,pipeline,cli,weights,benchmark,evaluation}.py`
**Yeni (betik):** `scripts/{run_person_detection,download_person_model,benchmark_person_detection,evaluate_person_detection}.py`
**Yeni (belge):** `docs/person_detection.md`, `reports/detection/STAGE_4A_REPORT.md`
**Yeni (test):** `tests/person_detection_test_support.py` (yardımcı) ve `tests/test_person_detection_{config,results,visualization,pipeline,cli,benchmark,evaluation,real_model}.py`, `tests/test_person_detector.py`
**Yeni (Git dışı):** `models/human_detection/yolo11n.pt`; `results/person_detection/` (sentetik video + benchmark JSON'ları; görüntü içermez)

**Mevcut dosyalarda değişiklik (yalnızca bunlar):**

| Dosya | Değişiklik | Neden |
|---|---|---|
| `src/isg_vision/human_detection/__init__.py` | Boş iskelet → genel API dışa aktarımı | Aşama 4A paketi (yeni `detection/` paketi açılmadı; mevcut mimari korundu) |
| `src/isg_vision/utils/paths.py` | `REQUIRED_FILES`'a 5 yeni dosya yolu eklendi | Aşama 3'te yapılan uygulama ile aynı |
| `tests/test_no_side_effects.py` | (a) yeni modüller import-yan-etki listesine eklendi; (b) `models/` altında yasak ağırlık denetimine **tek onaylı istisna**: `models/human_detection/yolo11n.pt` | Test, indirilen onaylı ağırlığı "beklenmeyen model" sayıp düşerdi. Başka her `.pt/.onnx/.pth` hâlâ başarısız eder |
| `README.md`, `docs/architecture.md`, `docs/project_plan.md`, `configs/project_stages.yaml` | Aşama 4A durumu, alt aşamalar (`stage4_substages`), komutlar | Belge/yapılandırma tutarlılığı |

**Dokunulmadı** *[Ölçüldü: `git status`]*: `src/isg_vision/camera/*`, Aşama 1–3 kodları, `data/`, Aşama 2 raporları. Not: `run_stage2c_checks.py`, her çalıştığında kendi çıktısı `reports/dataset_audit/stage2c_integrity_checks.json` içindeki `date` satırını güncelliyor (sonuç aynı, 12/12);
bu tarih-only farkı geri aldım (`git checkout`), böylece Aşama 2 dosyası değişmemiş durumda.

## 5. PersonDetector mimarisi

Ayrım istendiği gibidir: **`PersonDetector`** (YOLO'yu yükler, tespit eder) · **`DetectionConfig`** (model, eşikler, cihaz, boyut) · **`PersonDetection`/`DetectionResult`** (sonuçlar) · **görselleştirme** (`visualization.py`) · **döngü** (`pipeline.py`).
Önemli kararlar ve **ölçülmüş** gerekçeleri:

* **Model bir kez yüklenir**; `load()` tekrar çağrısı etkisizdir (test: `load_count == 1`, hem mock hem gerçek model). Isınma yüklemeden ayrıdır.
* **`person` metadata'dan** doğrulanır; yalnızca `person` işlenir (model çağrısına `classes=[id]` + çıktıda ikinci süzme). Gerçek modelde süzülmemiş çıktıda `bus` görünür, sonuçta görünmez *[Gerçek model]*.
* **Cihaz:** `auto` (MPS varsa MPS, yoksa **uyarıyla** CPU), `cpu`, `mps`. Açıkça `mps` istenip kullanılamıyorsa `DeviceUnavailableError` (sessiz CPU geri dönüşü yok; CLI çıkış kodu 1, kamera hiç açılmaz).
* **Ağ yok — ölçülmüş bulgu:** `import ultralytics`, `YOLO_OFFLINE` ayarlanmadıkça DNS yoklaması yapıyor. `PersonDetector` modeli yüklemeden önce `YOLO_OFFLINE=1` ayarlar. Doğrulama: ağ çağrıları sahtelenmiş bir alt süreçte gerçek model yükleme + çıkarım çalıştı
  ve diske hiçbir dosya yazılmadı; `YOLO_OFFLINE` ayarı kodtan geçici kaldırılınca aynı test **başarısız oldu** (ağ çağrısı yakalandı) → test ayırt edici *[Gerçek model]*.
* **Süre ölçümü:** `time.perf_counter`; kapsam = ön işleme + ileri geçiş + NMS + sonuçların CPU'ya alınması; MPS'te bitişten önce `torch.mps.synchronize()`. Ultralytics'in kendi `speed` kırılımı MPS'te asenkron işi kapsamayabileceği için kullanılmadı.
* **Hatalar:** `InvalidFrameError`, `ModelLoadError` (dosya yok/bozuk/`person` yok), `DeviceUnavailableError`, `InferenceError`, `WeightsDownloadError`; yapılandırma hataları `DetectionConfigError`.

## 6. FramePacket entegrasyonu

`FrameSource.frames()` → `FramePacket` → `PersonDetector.detect(packet.frame, frame_index, timestamp)` → `DetectionResult` → `draw_person_detections` → `PreviewWindow`. Detector yalnızca BGR `ndarray` alır; kamera iç yapısına bağımlı değildir
(MacBook kamerası, USB kamera, video dosyası aynı kodla; IP/RTSP **geliştirilmedi**, eklenebilir biçimde tasarlandı). Döngü, kaynağı/pencereyi/modeli açıp kapatmaz (CLI'ın `finally` bloğu her koşulda serbest bırakır). `on_result(packet, result)` kancası 4B için bırakıldı.
CLI sırası: model yükle (başarısızsa kamera hiç açılmaz) → kaynak aç → gerçek kare boyutunda ısıt → pencere → döngü.

## 7. Tespit veri yapıları

`PersonDetection(class_id, class_name, confidence, bbox_xyxy, frame_index, timestamp)`, değiştirilemez. `bbox_xyxy = (x1, y1, x2, y2)`, **görüntü piksel koordinatları**, `x1<x2`, `y1<y2`, görüntü sınırlarına kırpılmış. `DetectionResult` güvene göre azalan sıralı, `person_count`, `inference_ms`, `device`, `dropped_boxes` taşır.
İnsan yoksa boş demet (hata değil). Geçersiz kutular (NaN/∞, dejenere, tamamen dışarıda, kırpma sonrası kenar < 1 px) atılır **ve `dropped_boxes` ile sayılır**; kenara taşan insanlar kırpılır. `track_id`/çalışan adı/yüz kimliği **yoktur** (testle de güvence altında).

## 8. Canlı görüntüleme özellikleri

Üst şerit: `PERSONS: n`, `SOURCE`, `RESOLUTION` (gerçek), `FPS` (ölçülen uçtan uca), `INFERENCE` (son 30 karenin ortalaması), `DEVICE`; kutu başına `PERSON 0.94` etiketi; alt şerit `Q / Esc / Close to exit`. İnsan yoksa `PERSONS: 0`. Çizim girdinin **kopyası** üzerindedir (testle doğrulandı); tüm kutular aynı renk (kimlik izlenimi verilmez).
**Doğrulama:** (a) piksel/metin testleri [Mock/birim]; (b) bu çizim fonksiyonuyla gerçek tespitlerden üretilen bir kareyi **gözle inceledim** (`bus.jpg`: 4 kutu, etiketler, şeritler doğru; otobüs kutulanmadı); (c) gerçek OpenCV penceresi Terminal panelinde (grafik oturumu) gerçek model + MPS ile sentetik videoyla açıldı, 90 kare işlendi, `exit=0`, kırmızı düğme uyarısı çıkmadı;
(d) `ISG_GUI_TESTS=1` ile gerçek pencerede kırmızı düğme tıklaması `pencere kapatıldı` ile döngüyü bitirdi ve kamerayı serbest bıraktı. (e) **Kullanıcı gerçek MacBook kamerasında pencereyi gözle doğruladı:** üç kişi kadraja girince `PERSONS: 3` göründü *[Kullanıcı]*; Q, Esc, kırmızı düğme ve Ctrl+C ile çıkışın gerçek kamerada sorunsuz çalıştığını bildirdi *[Kullanıcı]* (§19, Test 1 ve 6).

## 9. CLI komutları

```bash
.venv/bin/python scripts/run_person_detection.py --source camera
.venv/bin/python scripts/run_person_detection.py --source camera --camera-index 0
.venv/bin/python scripts/run_person_detection.py --source video --video-path /yol/video.mp4
.venv/bin/python scripts/run_person_detection.py --source camera --device mps
.venv/bin/python scripts/run_person_detection.py --source camera --device cpu
.venv/bin/python scripts/run_person_detection.py --source camera --conf 0.35
.venv/bin/python scripts/run_person_detection.py --source camera --no-display --max-frames 100
.venv/bin/python scripts/run_person_detection.py --help
```

Ek seçenekler: `--width/--height/--backend` (`run_camera.py` ile aynı), `--model`, `--iou`, `--imgsz`, `--max-det`. 22 hatalı argüman senaryosu çıkış kodu 2 + anlaşılır mesaj + `Traceback` yok olarak test edildi. `scripts/run_camera.py` bozulmadı (testle doğrulandı). Çıkış kodları: 0 / 1 / 2 / 130.
Kamera izni olmadan çalıştırıldığında *[Ölçüldü]*: `not authorized to capture video` → `[HATA] Kamera 0 açılamadı ... (izin ipucu)`, güvenli çıkış (kamera açılmadı; model serbest bırakıldı).

## 10. CPU / MPS performans sonuçları *[Ölçüldü]*

**Yöntem** (`benchmark.py`; ayrıntı ve tanımlar docs/person_detection.md §8–9): aynı video/model/eşikler/`imgsz`; yükleme, ısınma (2 boş kare) ve burn-in (10 kare) **ayrı**; 5 ölçülen geçiş, cihazlar **dönüşümlü** sırayla; pencere ve çizim yok.
**Ortam:** MacBook Pro M4 (`arm64`, macOS 26.5.2), Python 3.11.17, torch 2.15.0, ultralytics 8.4.210, OpenCV 5.0.0, 8 torch iş parçacığı. `yolo11n.pt`, `conf 0.25`, `iou 0.7`, `imgsz 640`.
**Video: SENTETİK** 1280×720, 300 kare (Ultralytics'in `bus.jpg` + `zidane.jpg` örnek görüntülerinden montaj). **Yalnızca hız içindir**; gerçek çalışma alanını temsil etmez; doğruluk ölçümünde kullanılamaz. Uygun, izinli, tekrarlanabilir bir gerçek çok kişili video elimde yoktu; dışarıdan rastgele video indirilmedi.

| Cihaz | Yükleme | Isınma | Uçtan uca FPS (medyan / aralık) | Çıkarım medyan (ms) (aralık) | Çıkarım ort (ms) | Çıkarım p95 (ms) | Kare→sonuç medyan (ms) |
|---|---|---|---|---|---|---|---|
| CPU | 0,55 sn | 0,59 sn | 51,7 (46,6–52,5) | 18,8 (18,7–20,4) | 19,4 | ≈19,3 | 18,8 |
| MPS | 0,06 sn | 0,32 sn | 104,2 (100,0–110,7) | 9,1 (8,5–9,7) | 9,2 | ≈10,5 | 9,1 |

* Ölçüm tanımları: **yakalama FPS'i** (modelsiz video okuma: 2521–3121 FPS; *dosya çözme hızıdır, kamera hızı değil*), **çıkarım gecikmesi** (yalnızca model çağrısı), **uçtan uca FPS** (okuma+çıkarım döngüsü; pencere/çizim yok), **kare→sonuç gecikmesi** (`FramePacket.timestamp` → tespit bitişi). Hepsi ayrı raporlanır; kaynağın bildirdiği FPS hiçbir yerde kullanılmadı.
* Bu koşulda MPS, çıkarımda CPU'dan yaklaşık 2 kat hızlı. CPU'da tek karelik sıçramalar (≈60 ms) ve biraz yavaş ilk geçiş (46,6 FPS) görüldü; ısıl/güç durumu ve arka plan süreçleri **kontrol edilmedi**.
* **Pencereli tek gözlem:** MPS, sentetik video, pencere açık (video kaynak hızına — 30 FPS — uydurulduğundan FPS tavanlıdır): 90 kare, çıkarım ort 12,9 ms (p95 17,0), FPS 29,6. Karşılaştırma değil, tek koşudur.
* **Bu bölümdeki tablo sentetik videodur.** Canlı kamera CPU/MPS ölçümleri kullanıcı tarafından yapılmıştır ve §19'da ayrı raporlanır; ikisi karıştırılmamalı, sentetik sonuçlar canlı performansın tahmini olarak kullanılmamalıdır (canlı MPS çıkarımı 13,4 ms medyanla sentetikteki 9,1 ms'den yüksektir; nedeni ayrıştırılmadı). 30 FPS zorunluluğu konmadı.
* MPS'te yeni girdi şekline ilk çıkarım belirgin yavaştı (bir probda 211 ms; sonraki iki çağrı 18,0 ve 14,7 ms) → ısınma gerçek kare boyutunda yapılır.

## 11. Gerçek model çıkarım testleri *[Gerçek model]*

`tests/test_person_detection_real_model.py` (13 test) + `test_person_detection_evaluation.py` içindeki 1 boru hattı testi, **mock değil, gerçek `yolo11n.pt`** ile çalışır. Ağırlık yoksa atlanır; indirme/internet gerekmez. Görüntüler `ultralytics` paketinin içindeki örnekler
(`bus.jpg`: 4 yaya + 1 otobüs; `zidane.jpg`: 2 kişi), kesitler bunlardan üretildi. Doğrulananlar: model bir kez yüklenir; `person` kimliği metadata'dan; yalnızca `person` döner (ham çıktıda `bus` var); kutular görüntü içinde ve geçerli; güven skorları 0..1 ve azalan sırada;
girdi görüntüsü değişmez; çıkarım süresi ölçülüyor; CPU↔MPS sonuçları uyuşuyor (aynı kişi sayısı; test toleransı kutu < 2 px ve güven < 0,02, ön denemede kutular birebir aynı çıktı); çıkarım sırasında ağ çağrısı ve dosya yazımı yok.

**Güven eşiğinin etkisi (`bus.jpg`, CPU)** — `--conf` değiştirildiğinde tespitler değişir:

| `conf` | 0,25 | 0,50 | 0,70 | 0,90 |
|---|---|---|---|---|
| Kişi sayısı | 4 | 4 | 3 | 0 |

(Güven değerleri 0,888 / 0,879 / 0,856 / 0,623.) `zidane.jpg`: `conf` 0,25 ve 0,70'te 2 kişi (0,843 / 0,771), 0,80'de 1 kişi.

## 12. Tek ve çoklu insan testleri

| Senaryo | Sonuç | Durum |
|---|---|---|
| **A — insan yok** | `bus.jpg`'nin insansız kesiti (bina + otobüs tepesi; gerçek görüntü), düz gri, rastgele gürültü → **0 kişi**, hata yok, `PERSONS: 0` çizimi sorunsuz | ✔ *[Gerçek model]* |
| **B — tek insan** | 4 kesit (`bus` sağdaki yaya 0,84; `bus` siyah ceketli yaya 0,87; `zidane` sol 0,92; `zidane` sağ 0,89) → her birinde **1 kişi**, geçerli kutu ve güven | ✔ *[Gerçek model]* |
| **C — çoklu insan** | `bus.jpg` → **4 kişi** (bağımsız kutular, sayı doğru); `zidane.jpg` → **2 kişi** (kontrollü test görüntüsü). **Canlı kamerada üç kişi aynı anda `PERSONS: 3` olarak görüldü** (§19 Test 1) | ✔ görüntü düzeyinde *[Gerçek model]* + canlı sahne *[Kullanıcı]*; kayıt yapılmadığından tekrar oynatılabilir çok kişili video yok |
| **D — kamera açısı** | Göz hizasına yakın iki görüntü (açı metadata'sı yok, görsel tahmin). **MacBook ≈2 m yüksekliğe yerleştirilerek yüksek açı ön testi yapıldı** (§19 Test 5). Gerçek duvar montajlı kamera: `DEFERRED_TO_SYSTEM_INTEGRATION` | ✔ ön test *[Kullanıcı]*; ⏳ gerçek kamera ertelendi (OT-4A-003) |
| **E — MacBook kamerası** | Claude ortamında kamera izni yok (`not authorized to capture video (status 0)`; Claude testi yapamadı, izin aşılmadı). **Kullanıcı canlı kamera testlerini kendi MacBook'unda yaptı** (§19) | ✔ *[Kullanıcı]* |

**Gizlenmeyen gözlemler.** (1) Bir kesitte (`bus` sol-orta) conf 0,25'te ikinci bir düşük güvenli (0,273) kutu çıktı — kesilmiş bir komşu gövde parçası; etiket olmadığı için "yanlış pozitif" demek de, değil demek de mümkün değil. (2) Sentetik video karesinde 6 ön plan kişisi + bulanık arka plan kalabalığı bulunuyor; model kare başına ortalama 6,52 (en çok 8) tespit verdi,
bazı kutular 0,31–0,37 güvenli ve bulanık arka plandaki kişiler/kenarda kırpılmış kişi üzerinde. Gerçek sayı etiketlenmediği için **doğru/yanlış söylenemez ve bu sayılar doğruluk olarak raporlanmaz**. (3) Bu örneklerde görünür bir **kaçırma** gözlenmedi, ancak iki görüntü ve birkaç kesit genelleme sağlamaz. (4) Gerçek kamerada ek gözlemler: yorgan yanlış pozitifi **FP-4A-001** ve en yüksek 7 tespit **OBS-4A-001** (§20).

## 13. Kamera açısı değerlendirmesi

Hedef kurulum: duvara 2–3 m yükseklikte monte tek (IP) kamera. Kod, açıyı/çözünürlüğü sabit varsaymaz (kare boyutu gerçek kareden okunur; `--imgsz`/`--conf` ayarlanabilir).
**Ön test (kullanıcı, MacBook ≈2 m yükseklikte, MPS, pencereli):** sistem kullanıcıyı bu konumdan başarıyla tespit etti; görüntünün köşesindeki bir yorgan kısa süreli insan sanıldı (**FP-4A-001**, §20) *[Kullanıcı]*. 331 kare, 18,4 FPS; bu FPS kamera yüksekliğine **atfedilmez** (kontrollü deney yok, §19).
Bu bir **ön testtir ve gerçek IP kamera doğrulaması değildir**: IP kamera henüz satın alınmadı, marka/model, RTSP bağlantısı, montaj ve nihai test alanı belirlenmedi.
Durum: ön test ✔ *[Kullanıcı]*; gerçek kamera `DEFERRED_TO_SYSTEM_INTEGRATION` (OT-4A-003). Önceki `PENDING_REAL_CAMERA_VALIDATION` etiketi bu ön test için geçerliliğini yitirmiş, gerçek kamera kısmı için yukarıdaki durumla değiştirilmiştir.
Bu açıda `imgsz`/`conf` ayarı gerekebilir; `conf 0.35` karşılaştırması kapanıştan sonra yapıldı; yorgan yanlış pozitifi çözülmedi, gerçek insan tespitinin bu eşikteki başarısı ölçülmedi (OT-4A-001, §23).

## 14. Otomatik test sonuçları *[Ölçüldü]* (geliştirme sonu; kapanış yeniden çalıştırması §22'de)

| Çalıştırma | Sonuç |
|---|---|
| `unittest discover -s tests` (varsayılan) | **596 test, OK, 5 atlandı** (atlananlar yalnızca isteğe bağlı gerçek-pencere testleri: Aşama 3'ten 3, Aşama 4A'dan 2) |
| `ISG_GUI_TESTS=1 ...` (Terminal paneli, grafik oturumu) | **596 test, OK, 0 atlandı** |
| `run_stage2c_checks.py` | **12/12** |
| `check_environment.py` | Başarılı (kritik hata 0) |

**Yeni testler: 256** (596 − 340). Dosya bazında: `config` 38 · `results` 27 · `detector` 42 · `visualization` 18 · `pipeline` 29 · `cli` 36 · `benchmark` 18 · `evaluation` 35 · `real_model` 13.
**Mock ve gerçek ayrı raporlanır:** 242 test mock/sahte/sentetik girdiyle çalışır (gerçek YOLO yok); **14 test gerçek `yolo11n.pt` çıkarımı** kullanır (13 gerçek-model + 1 değerlendirme boru hattı). **Mock testlerin geçmesi gerçek çıkarımın çalıştığını kanıtlamaz**; kanıt yalnızca §11'dedir.

İstenen 22 madde kapsamı: geçerli yapılandırma, geçersiz model yolu/`conf`/`iou`/`imgsz`, CPU/MPS seçimi, model yükleme hatası, **tek yükleme**, boş/geçersiz kare, insan yok / tek / çok, `person` dışı süzme, güven korunumu, kutu doğruluğu, görüntü dışı kutular, FPS ve çıkarım süresi hesabı,
overlay'in girdiyi değiştirmemesi, hatalı CLI argümanları, penceresiz çalışma, kullanıcı çıkışı ve kaynak temizliği, **import'ta kamera/model/ağın açılmaması** (alt süreç testi) — hepsi en az bir testle sınanır.

**Testlerin ısırdığının kanıtı (mutasyon):** 6 kritik davranış geçici bozuldu — kamera-model sırası, tek yükleme, `classes=` filtresi, `person` süzmesi, açık-MPS hatası, girdi kopyalama — her birinde ilgili testler başarısız oldu; kaynaklar özdeş geri alındı (`diff` boş). Yedinci mutasyon (`YOLO_OFFLINE` kaldırma) ağ testini düşürdü (§5).
**Geliştirme sırasında bulunup düzeltilen hatalar** (testleri gevşetmeden): (a) `test_no_side_effects` modülleri yeniden yüklediği için iki testim sınıf kimliği uyuşmazlığıyla tam pakette düşüyordu → importlar modül başına alındı; (b) benchmark'ta eksik ağırlık "cihaz atlandı" gibi sessizce geçiyordu → yalnızca `DeviceUnavailableError` cihazı atlatır, model hatası benchmark'ı durdurur;
(c) 5 karelik video oynatma testi yanlış beklenti içeriyordu (ilk kare son-teslim zamanı) → 30 kare ve gerekçeli tolerans.

## 15. Önceki aşamalardaki testlerin durumu

* Aşama 1–3'ün **340 testi** korunuyor ve geçiyor (kamera testleri 170/170, 3'ü yalnızca GUI gerektirdiği için varsayılanda atlanır; GUI ile 0 atlama).
* **Kamera kodu değişmedi** (`src/isg_vision/camera/` altında fark yok); `scripts/run_camera.py` testle doğrulandı.
* Aşama 2: bütünlük 12/12; `data/`, ham veri ve Aşama 2 raporları değişmedi (§4'teki tarih-only yan etki geri alındı). Belge/aşama-numarası tutarlılık denetimi (T2D.2) yeni belgelerle de geçti.
* `test_no_side_effects.py` (Aşama 1) yalnızca §4'te açıklanan iki gerekçeli değişiklikle güncellendi.

## 16. Bilinen hatalar ve sınırlamalar

1. **Doğruluk ölçülmedi.** Etiketli bağımsız insan test kümesi yok → Precision/Recall/F1/mAP raporlanmadı; güven skorları doğruluk olarak sunulmadı; **%85 hedefinin karşılandığı iddia edilmez.** Canlı kamera testleri işlevselliği ve işlem performansını doğrular, istatistiksel doğruluğu değil (OT-4A-002).
2. Performans: sentetik video ölçümleri (§10) pencere maliyeti içermez ve canlı performansın tahmini değildir; canlı kamera ölçümleri (§19) kullanıcı tarafından yapılmış tek oturumlardır; ısıl/güç kontrol edilmedi. Pencereli FPS (18,4–25,3; Test 7 dahil) penceresiz FPS'ten (30,0) düşüktür; nedeni ayrıştırılmadı (OBS-4A-002).
3. Model COCO ile eğitilmiş; yüksek açı için yalnızca MacBook ≈2 m ön testi var; gerçek duvar kamerası, şantiye/KKD'li kişiler için doğrulanmadı.
4. Düşük güvenli (0,27–0,37) sentetik/kesit kutularının doğru/yanlış olduğu bilinmiyor (§12).
5. **FP-4A-001:** yorgan/kumaş kısa süreli insan olarak tespit edildi (§20) — bilinen sınırlama; tek gözlem, genel başarısızlık sonucu çıkarılmaz; ByteTrack'in çözeceği iddia edilmez.
6. **OBS-4A-001:** en yüksek 7 tespit sayısı açıklanamıyor/doğrulanmadı (§20).
7. Gerçek IP kamera yok; RTSP kaynağı geliştirilmedi ve doğrulanmadı (OT-4A-003).
8. Lisans AGPL-3.0 (§3).
9. Aşama 3 backlog: B3-1, B3-2 (kare kaybı) ve B3-5 kapsanmadı. B3-3 için kare→sonuç/kare→görüntü gecikmesi kayıtlı videoda ve (kullanıcı) canlı kamerada ölçüldü; sensör/sürücü tamponlaması kapsam dışı; nihai ölçüm Aşama 12'de. B3-4 için modelli 300 karelik penceresiz canlı koşular 30,0 FPS verdi (kullanıcı); uzun süreli, bağımsız zaman referanslı doğrulama yapılmadı. **Hiçbiri tamamlanmış sayılmaz.**
10. Model yalnızca `person` sınıfıyla sınırlı; yüz/KKD yok (bilinçli kapsam dışı).
11. Ultralytics bağımlılıkları kendiliğinden kurmaya (ağ) çalışabilir; 4B'de ByteTrack için gereken paketler (`lap`/`scipy` kurulu değil) bilinçli kurulmalı ve `YOLO_OFFLINE` ile birlikte ele alınmalıdır.

## 17. Manuel doğrulamaların durumu

İlk teslimdeki manuel doğrulama listesinin güncel durumu (ayrıntı §19–§21):

| Madde | Durum |
|---|---|
| Pencere açılıyor, kutu + güven + `PERSONS: n` + FPS + INFERENCE görünüyor | ✔ *[Kullanıcı]* (Test 1, 2, 5) |
| Tek kişi / çok kişi (üç kişi) / kimse yokken sayı | ✔ *[Kullanıcı]* (Test 1, 2; Test 3–4'te 1 kişi) |
| Q, Esc, kırmızı düğme, Ctrl+C ile çıkış | ✔ *[Kullanıcı]* (Test 6); **çıkış kodları paylaşılmadı → gerçek kamerada ölçülmüş sayılmaz** |
| Canlı FPS/çıkarım/gecikme, CPU ve MPS ayrı | ✔ *[Kullanıcı]* (Test 3, 4) |
| Daha yüksek açıdan deneme | ✔ ön test, MacBook ≈2 m *[Kullanıcı]* (Test 5); gerçek duvar kamerası → `DEFERRED_TO_SYSTEM_INTEGRATION` |
| Yanlış pozitif/kaçırma gözlemi | ✔ FP-4A-001 ve OBS-4A-001 kaydedildi (§20); `conf` karşılaştırması → OT-4A-001: kapanıştan sonra yapıldı, yorgan yanlış pozitifi çözülmedi (§23) |
| Etiketli bağımsız test kümesi ile Precision/Recall/F1/mAP | ⏳ açık görev (OT-4A-002); altyapı hazır |
| Kayıtlı (tekrar oynatılabilir) gerçek ≥3 kişilik izinli video | Yapılmadı (canlı sahnede üç kişi doğrulandı, görüntü kaydedilmedi) |

## 18. Aşama 4B için hazır olan bileşenler

* `DetectionResult.detections`: kare başına `bbox_xyxy` (piksel), `confidence`, `frame_index`, `timestamp` — takipçi girdisi; takip kimliği alanı bilinçli olarak yok.
* `run_detection(..., on_result=...)` kancası; `FramePacket.timestamp` (monotonic) zaman tabanlı takip için.
* Ölçüm altyapısı (`LatencyStats`, benchmark, CPU/MPS karşılaştırması) takip eklenince gecikme artışını ölçmek için yeniden kullanılabilir.
* **Hazırlık notu:** 4B başlatma kararı kullanıcıya aittir; 4B yanlış pozitifleri (FP-4A-001) takip bağlamında yeniden değerlendirmelidir (etkisi ölçülmeden bilinmez). OT-4A-001'in sonucu 4B'de kullanılacak `conf` kararını etkileyebilir; varsayılan 0,25 değişmedi.
* **4B kararları (henüz verilmedi):** takipçi seçimi (Ultralytics ByteTrack/BoT-SORT vs. kendi uygulaması), ek bağımlılıklar (`lap`/`scipy`), düşük güvenli kutuların takipte kullanımı (ByteTrack bunları kullanır; mevcut varsayılan `conf` 0,25).

## 19. Kullanıcı tarafından gerçekleştirilen manuel doğrulamalar (2026-11-12) *[Kullanıcı]*

Bu bölümdeki tüm sayı ve gözlemler **kullanıcının kendi MacBook Pro M4'ünde, gerçek terminal çıktılarından ve kendi gözleminden** aktarılmıştır. Claude'un ortamında kamera izni olmadığından bunlar yeniden üretilmedi ve bağımsız doğrulanmadı;
Claude'un kendi çalıştırdığı otomatik testlerden (§11, §14, §22) ve sentetik video benchmark'ından (§10) **ayrıdır**. Ortak koşullar: `camera:0`, 1280×720, `yolo11n.pt`, `conf 0.25` (varsayılan), görüntü kaydedilmedi (video kanıtı yok).

**Pencereli oturumlar** (pencere açık: çizim + `imshow` + `waitKey` dâhil; hepsi MPS; komut `run_person_detection.py --source camera --camera-index 0 --device mps`):

| Test | Sahne | Kare | Süre | Uçtan uca FPS | Çıkarım ort / medyan / p95 (ms) | Kare→görüntü medyan | Kişi/kare ort | En çok | ≥1 kişi olan kare | Çıkış |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | Üç kişi kameranın önüne geçti | 800 | 34,3 sn | 23,3 | 15,5 / 14,8 / 21,0 | 30,8 ms | 2,32 | **7** (OBS-4A-001) | bildirilmedi | Q/Esc |
| 2 | Kamera boş → kullanıcı girdi → köşede başka kişi | 636 | 28,6 sn | 22,2 | 14,7 / 13,5 / 20,8 | 28,7 ms | 1,31 | 3 | 549/636 (**0 tespit: 87 kare**) | Q/Esc |
| 5 | MacBook ≈2 m yükseklikte (ön test) | 331 | 17,9 sn | 18,4 | 16,8 / 17,1 / 21,5 | 33,6 ms | 1,17 | 3 | 327/331 | Q/Esc |

**Penceresiz oturumlar** (`--no-display --max-frames 300`; yalnızca okuma + çıkarım; program `--max-frames` sınırında normal sonlandı):

| Test | Cihaz | Kare | Süre | Uçtan uca FPS | Çıkarım ort / medyan / p95 (ms) | Kare okuma ort / medyan (ms) | Kare→sonuç ort / medyan (ms) | Kişi/kare | En çok |
|---|---|---|---|---|---|---|---|---|---|
| 3 | MPS | 300 | 10,0 sn | **30,0** | 13,8 / 13,4 / 16,8 | 19,6 / 19,8 | 13,9 / 13,4 | 1,00 (300/300 karede ≥1 kişi) | 1 |
| 4 | CPU | 300 | 10,0 sn | **30,0** | 18,6 / 18,5 / 19,1 | 14,8 / 14,9 | 18,6 / 18,5 | 1,00 | 1 |

**Kullanıcı gözlemleri.**
* **Test 1:** üç kişi kameranın önüne geçtiğinde ekranda `PERSONS: 3` gösterildi; üç kişinin aynı anda algılandığı gözle doğrulandı.
* **Test 2:** ekran sonuçları gözlenen sahneyle uyumluydu (boş kadraj → kullanıcı → köşede ikinci kişi). 636 karenin 87'sinde hiç insan tespit edilmedi. **Bu 87 kare, modelin sıfır insan tespiti ürettiği karelerdir; her karenin bağımsız etiketlendiği bir doğruluk testi yapılmadı. Precision/recall veya doğruluk oranı olarak değerlendirilmez.**
* **Test 5:** sistem kullanıcıyı ≈2 m yükseklikteki kamera konumundan tespit etti; köşedeki bir yorgan kısa süreli insan sanıldı (**FP-4A-001**, §20).
* **Test 6 — çıkış yöntemleri (gerçek kamera):** Q tuşu ✔, Esc tuşu ✔, macOS kırmızı pencere kapatma düğmesi ✔, terminalde Ctrl+C ✔; oturumlar kullanıcı gözlemiyle sorunsuz kapandı. **Çıkış kodları paylaşılmadı; Ctrl+C dâhil hiçbir çıkış için gerçek kamerada belirli bir kod ölçüldüğü iddia edilmez** (130 yalnızca birim testleriyle doğrulanmıştır *[Mock]*).

**CPU/MPS canlı performans değerlendirmesi.**
* *[Kullanıcı]* Penceresiz canlı hız iki cihazda da 30,0 FPS. *[Türetilmiş]* Medyan çıkarım MPS 13,4 ms, CPU 18,5 ms → MPS çıkarımda ≈1,38 kat hızlı (18,5 / 13,4). Uçtan uca FPS aynıdır; çünkü her iki cihaz da kamera akışının hızına yetişir.
* *[Türetilmiş]* Kare okuma + çıkarım ≈33,4 ms her iki cihazda (MPS: 19,6 + 13,8; CPU: 14,8 + 18,6) ≈ 30 FPS kare aralığı: döngü kamera hızındadır, okuma süresi büyük ölçüde kameranın sonraki kareyi beklemesidir.
* MPS'nin ileride eklenecek modüllerle (takip, yüz vb.) avantaj sağlayabileceği öngörülebilir; ancak bu **henüz ölçülmemiştir** ve bir hız iyileştirmesi garanti edilmez.
* Canlı MPS çıkarım medyanı (13,4 ms) sentetik video benchmark'ından (9,1 ms; §10) yüksek, CPU'da fark yok (18,5 / 18,8 ms); CPU/MPS oranı sentetikte ≈2,07, canlıda ≈1,38. MPS farkının nedeni ayrıştırılmadı → sentetik sonuçlar canlı performansın tahmini değildir.
* Pencereli FPS (Test 1, 2, 5: 18,4–23,3; kapanış sonrası Test 7: 25,3 — §23) penceresiz 30,0'dan düşük (kare→görüntü medyanı 28,7–33,6 ms; Test 7: 28,5 ms; penceresiz kare→sonuç 13,4 / 18,5 ms). Pencere maliyeti ayrı ölçülmedi; sahne ve kişi sayısı oturumlar arasında kontrollü değil → **neden ayrıştırılmadı** (OBS-4A-002). **Test 5'teki düşük FPS kamera yüksekliğine atfedilmez.**
* Sınırlar: her satır tek oturum; ısıl/güç kontrolü yok; benzersiz kare sayımı ve bağımsız zaman referansı yok; zaman ölçümü `FramePacket.timestamp` ile okuma bitişinden başlar (sensör/sürücü tamponlaması kapsam dışı).

## 20. Bilinen sınırlama ve açık gözlemler

**FP-4A-001 — Blanket/Fabric Misclassified as Person** (bilinen sınırlama; izlenecek yanlış pozitif)

| Alan | Değer |
|---|---|
| Nesne | Yorgan / kumaş |
| Koşul | Yaklaşık 2 metre yüksekliğe yerleştirilmiş MacBook kamerası (Test 5), görüntünün köşesindeki yorgan |
| Model | YOLO11n (`yolo11n.pt`) |
| Confidence threshold | 0.25 |
| Sonuç | Kısa süreli yanlış insan tespiti |
| Güven skorunun kesin değeri | Bilinmiyor (eşik gereği ≥ 0,25 olmalıdır — *[Türetilmiş]*) |
| Hatanın oluştuğu kare sayısı | Bilinmiyor |
| Video kanıtı | Kaydedilmedi (varsayılan olarak görüntü kaydedilmez) |
| Önem | İzlenmesi ve sonraki aşamalarda değerlendirilmesi gereken yanlış pozitif |
| Güncelleme (OT-4A-001, `conf 0.35`, kapanış sonrası) | Yorgan `conf 0.35`'te de kısa süreliğine insan sanıldı (eşik gereği güven ≥ 0,35 — *[Türetilmiş]*); yanlış tespit süresi gözle görülür biçimde daha kısaydı, **sayısal ölçülmedi** (§23). **Sorun çözülmedi; kayıt açık kalır.** Varsayılan `conf` 0,25 |
| Yorum sınırı | **Tek gözlemdir; genel model başarısızlığı sonucu çıkarılmaz.** ByteTrack'in yanlış pozitifleri otomatik çözeceği **iddia edilmez** (etkisi 4B'de ölçülmeden bilinmez). Kamera açısının FPS düşüşüne neden olduğu söylenmez (kontrollü deney yok) |
| İlgili görev | OT-4A-001 (`conf 0.35`: yapıldı, çözmedi), OT-4A-002 (bağımsız değerlendirme) |

**OBS-4A-001 — Investigation / Unverified Peak Detection Count.** Test 1'de (800 kare, kare başına ortalama 2,32 kişi) en yüksek tespit sayısı **7** olarak raporlandı; aynı oturumda üç kişi kadraja girmişti. Bu 7 tespitin hangi karelerde ve hangi görüntü içeriğiyle oluştuğu **bilinmiyor**
(görüntü kaydedilmedi; CLI özeti yalnızca en yüksek sayıyı yazar, hangi karede olduğunu yazmaz — 4A kapsamında kod değiştirilmedi). **Kesin yanlış pozitif ya da kesin doğru tespit olarak sınıflandırılmaz.** Test 2 ve 5'te en yüksek sayı 3'tür. Durum: açık araştırma (OT-4A-004). Doğrulama yolu: aynı koşulları yeniden üretip canlı `PERSONS` sayacını izlemek veya tepe anını, yalnızca kullanıcının açık onayıyla, kaydetmek.

**OBS-4A-002 — Pencereli/penceresiz FPS farkı ve canlı/sentetik çıkarım farkı.** Bkz. §19; nedenler ayrıştırılmadı, neden iddiası yoktur. Gerekirse Aşama 12'deki canlı sistem testlerinde ölçülür.

## 21. Açık görevler ve ertelenenler

Makine tarafından okunur sürüm: `configs/project_stages.yaml` (`stage4a_open_items`, `stage4a_known_limitations`). OT-4A-001 test edilmiştir ancak sorunu çözmemiştir (§23); **diğerleri tamamlanmış sayılmaz.**

| ID | Görev | Durum | Not |
|---|---|---|---|
| **OT-4A-001** | `PENDING_MANUAL_TEST — CONFIDENCE_THRESHOLD_0_35` | **`TEST_PERFORMED`** (kullanıcı, kapanış sonrası) | **Test yapıldı; sorun çözülmedi:** yorgan 0,35'te de kısa süreliğine insan sanıldı (süre gözle daha kısaydı, sayısal ölçülmedi); gerçek insan tespitinin başarısı ölçülmedi; FP-4A-001 açık; varsayılan `conf` 0,25 değişmedi. Aşağıda |
| OT-4A-002 | Bağımsız etiketli değerlendirme kümesi ve resmî ölçümler (Precision, Recall, F1, mAP@0.5, mAP@0.5:0.95) | Açık | Altyapı hazır (`scripts/evaluate_person_detection.py`); veri kümesi yok. Zamanlama kullanıcı kararı (öneri: 4B takip ölçümleriyle birlikte veya Aşama 12 nihai değerlendirme) |
| OT-4A-003 | Gerçek IP kamera doğrulaması: kamera satın alma/montaj, RTSP/bağlantı doğrulaması (RTSP kaynak desteği geliştirilmedi), nihai açı, gecikme, görüntü kalitesi, nihai test alanı | **`DEFERRED_TO_SYSTEM_INTEGRATION`** | IP kamera satın alınmadı; marka/model, RTSP, test alanı (şantiye olması zorunlu değil) belirlenmedi; montaj yapılmadı. Hedef: sonraki entegrasyon ve sistem değerlendirme aşamaları (öneri: Aşama 11–12). **Bu yüzden Aşama 4A açık tutulmadı; MacBook testleri IP kamera doğrulaması olarak sunulmaz** |
| OT-4A-004 | OBS-4A-001 araştırması (en yüksek 7 tespit) | Açık araştırma | §20 |

#### OT-4A-001 — `PENDING_MANUAL_TEST — CONFIDENCE_THRESHOLD_0_35` (durum: `TEST_PERFORMED`, kullanıcı, kapanış sonrası)

Test, Aşama 4A kapandıktan sonra kullanıcı tarafından gerçekleştirildi; sonuçlar, soruların durumu ve kayıt şablonu **§23**'tedir. **Sorun çözülmüş sayılmaz (FP-4A-001 açık); varsayılan `conf` 0,25 olarak kalır.**
Test tasarımı (yapılmadan önce yazılmıştı) — komut:

```bash
.venv/bin/python scripts/run_person_detection.py --source camera --camera-index 0 --device mps --conf 0.35
```

Amaç, aynı yüksek kamera konumunda: (1) kullanıcı 0,35'te hâlâ doğru tespit ediliyor mu; (2) yorgan hâlâ insan olarak algılanıyor mu; (3) köşeden/kısmen görünen gerçek insanlar algılanmaya devam ediyor mu; (4) 0,25 ile 0,35 arasındaki fark nedir; (5) gerçek insanlar kaçırılıyor mu?
Prosedür: kamerayı mümkün olduğunca aynı konumda tutmak; aynı yorganı ve benzer aydınlatmayı kullanmak; `conf 0.25` sonuçlarını referans almak; `conf 0.35` ile gözlem yapıp kutuların gerçek insanları kapsayıp kapsamadığını karşılaştırmak; yanlış pozitif veya kaçırılan insan gözlemlerini kaydetmek.

## 22. Kapanış doğrulaması (2026-11-12, bu görevde Claude tarafından çalıştırıldı) *[Ölçüldü]*

*Bu bölüm kapanış anındaki durumu kaydeder; o sırada `conf=0.35` testi henüz yapılmamıştı. Kapanış sonrası güncelleme ve yeniden çalıştırılan kontroller §23'tedir.*

Bu görevde **yalnızca belge ve yapılandırma** güncellendi; kaynak kod, test ve betik değişmedi. Aşağıdaki sonuçlar gerçek çalıştırma çıktılarıdır.

| Kontrol | Sonuç |
|---|---|
| `unittest discover -s tests` (depo kökünden) | **596 test, OK, 5 atlandı** (atlananlar yalnızca isteğe bağlı gerçek-pencere testleri: Aşama 3'ten 3, Aşama 4A'dan 2) |
| Aynı komut başka çalışma dizininden (`-t` ile) | 596 test, OK, 5 atlandı |
| `ISG_GUI_TESTS=1 ...` (Terminal paneli, grafik oturumu) | **596 test, OK, 0 atlandı** |
| `run_stage2c_checks.py` (Aşama 2 bütünlük) | **12/12** (T2D.2 aşama-numarası tutarlılığı dâhil) |
| `check_environment.py` | Başarılı (kritik hata 0) |
| Aşama referans tutarlılığı (`stage_reference_problems`) | Sorun yok. Yaml: Aşama 3 `completed`, Aşama 4 `in_progress`; alt aşamalar 4A `completed`, 4B `planned`; açık görevler OT-4A-001 `pending_user_verification`, OT-4A-002 `open`, OT-4A-003 `deferred_to_system_integration`, OT-4A-004 `open_investigation`; FP-4A-001 `known_limitation` |
| Kod/veri değişmezliği (`git diff HEAD`) | `src/`, `scripts/`, `tests/`, `data/`, `models/`, `src/isg_vision/camera/`, `reports/dataset_audit/`, `reports/camera/` altında **fark yok**. Değişen dosyalar yalnızca: `README.md`, `configs/project_stages.yaml`, `docs/architecture.md`, `docs/person_detection.md`, `docs/project_plan.md`, `reports/detection/STAGE_4A_REPORT.md` |
| YOLO ağırlığı Git dışı | `git check-ignore`: `.gitignore:75 *.pt`; `git ls-files` içinde `.pt` dosyası **0**; dosya 5 613 764 bayt yerinde |
| `conf` varsayılanı | **0.25 değişmedi** (`DetectionConfig().conf = 0.25`; CLI `--conf` varsayılanı `0.25`). `0.35` hiçbir yerde varsayılan yapılmadı |
| Aşama 4B başlatılmadı | `src/isg_vision/tracking/__init__.py` yer tutucu (`"Henüz uygulanmadı"`) olarak değişmedi; takip/Track ID kodu yok (`track_id` yalnızca `results.py` açıklamasında "yoktur" notu ve yokluğunu doğrulayan testte geçer) |

**Not (Aşama 2 denetimi):** `run_stage2c_checks.py` her çalıştığında kendi çıktısı `reports/dataset_audit/stage2c_integrity_checks.json` içindeki yalnızca `date` satırını günceller (sonuç aynı); mevcut denetim yaklaşımına uygun olarak bu gereksiz fark geri alındı (`git checkout`), dosya değişmemiş durumda.

**Bu kapanışta yapılmayanlar:** kullanıcının MacBook sonuçları (§19) Claude tarafından yeniden üretilmedi (kamera izni yok); `conf=0.35` testi **çalıştırılmadı**; gerçek IP kamera testi yapılmadı; doğruluk ölçümü yapılmadı; commit/push yapılmadı; Aşama 4B başlatılmadı.

## 23. Kapanış sonrası güncelleme — OT-4A-001 (`conf=0.35`) test sonucu (2026-11-12) *[Kullanıcı]*

Aşama 4A kapandıktan sonra kullanıcı bekleyen OT-4A-001 manuel testini gerçekleştirdi ve sonucu bildirdi. **Aşama 4A `COMPLETED`, Aşama 4B `PLANNED`, Aşama 4 genel `IN_PROGRESS` durumları değişmedi**; kaynak kod ve varsayılan ayarlar değiştirilmedi. Sayılar kullanıcının terminal çıktısından aktarılmıştır; Claude kamera izni olmadığından bunları yeniden üretmedi.

| Alan | Değer *[Kullanıcı]* |
|---|---|
| Komut | `.venv/bin/python scripts/run_person_detection.py --source camera --camera-index 0 --device mps --conf 0.35` |
| Çözünürlük | 1280×720 |
| İşlenen kare / süre | 379 kare / 15,0 sn |
| Uçtan uca FPS | 25,3 (pencereli) |
| Çıkarım ort / medyan / p95 | 14,0 / 12,8 / 19,2 ms |
| Kare→görüntü medyan | 28,5 ms |
| Kişi/kare ort / en çok | 1,04 / 3 |
| Çıkış | Q/Esc ile başarılı (çıkış kodu bildirilmedi) |

**Kullanıcı gözlemi:** daha önce `conf=0.25` testinde insan olarak algılanan yorgan, `conf=0.35` testinde de kısa süreliğine insan olarak algılandı; ancak yanlış tespit süresi gözle görülür biçimde daha kısaydı. **Bu süre sayısal olarak ölçülmedi. Gerçek insan tespitinin bu eşikteki başarısı da ayrıca ölçülmedi.**

Test sorularının durumu:

| # | Soru | Sonuç |
|---|---|---|
| 1 | Kullanıcı 0,35'te hâlâ doğru tespit ediliyor mu? | **Ölçülmedi / bildirilmedi.** Kişi/kare ortalaması 1,04 ve en çok 3 bildirildi; bunlar etiketsiz sayımlardır, tespitin doğruluğunu göstermez |
| 2 | Yorgan hâlâ insan olarak algılanıyor mu? | **Evet, kısa süreliğine** (kullanıcı gözlemi). Eşik gereği bu tespitin güveni ≥ 0,35'tir *[Türetilmiş]*; kesin değer bilinmiyor |
| 3 | Köşeden/kısmen görünen gerçek insanlar algılanmaya devam ediyor mu? | **Ölçülmedi / bildirilmedi** |
| 4 | 0,25 ile 0,35 arasındaki fark nedir? | Nitel gözlem: yorgan yanlış tespit süresi gözle görülür biçimde daha kısaydı (**sayısal ölçülmedi**; tek oturum). Hız metriklerindeki farklar `conf` değişimine atfedilmez (aşağıda) |
| 5 | Gerçek insanlar kaçırılıyor mu? | **Ölçülmedi.** "Kaçırma yok" sonucu çıkarılmaz |

**Yorum ve sınırlar.**
* **OT-4A-001 durumu: `TEST_PERFORMED`.** Test gerçekleştirildi; **sorun çözülmüş sayılmaz.** `conf 0.35` bu tek gözlemde yanlış tespiti ortadan kaldırmadı, yalnızca (nitel olarak) kısalttı. **FP-4A-001 açık kalır.**
* **Varsayılan `conf` 0,25 olarak korunur** (kod ve yapılandırma değiştirilmedi). 0,35'in 0,25'ten daha iyi olduğu iddia edilmez: gerçek insanların bu eşikte tespit edilip edilmediği ve yanlış tespit süresinin sayısal değeri ölçülmedi; eşik kararı kullanıcıya aittir.
* **FPS/çıkarım karşılaştırması `conf`'a bağlanmaz.** Test 5 (`conf 0.25`: 18,4 FPS, çıkarım medyanı 17,1 ms) ile Test 7 (`conf 0.35`: 25,3 FPS, 12,8 ms) arasındaki farklar, `conf` dışındaki etkenleri de içerir: sahne ve kişi sayısı, ışık, pencere/sistem yükü; kamera konumunun ve aydınlatmanın Test 5 ile aynı olduğu ayrıca bildirilmemiştir. Oturumlar kontrollü bir karşılaştırma değildir. Pencereli FPS oturumlar arasında 18,4–25,3 aralığında değişir ve penceresiz 30,0'ın altındadır; nedeni ayrıştırılmamıştır (OBS-4A-002 geçerlidir).
* Ölçülmeyen kısımlar (gerçek insan tespitinin bu eşikteki başarısı, yanlış tespit süresi) bağımsız değerlendirme (OT-4A-002) veya yeniden test gerektirir; bu görevde yeni görev kimliği açılmadı.

Kayıt şablonu (kullanıcı bildirimlerine göre; "bildirilmedi" = bilgi yok, "olumsuz" anlamına gelmez):

| Gözlem | `conf 0.25` (Test 5) | `conf 0.35` (Test 7) |
|---|---|---|
| Kullanıcı tespit edildi mi? | Evet (kullanıcı gözlemi) | Bildirilmedi (kişi/kare 1,04) |
| Yorgan insan sanıldı mı? (süre/sıklık) | Evet, kısa süreli; süre/sıklık ölçülmedi | Evet, kısa süreli; süre gözle daha kısa; ölçülmedi |
| Köşede/kısmen görünen gerçek insan tespit edildi mi? | Bildirilmedi | Ölçülmedi / bildirilmedi |
| Kaçırılan gerçek insan var mı? | Bildirilmedi | Ölçülmedi |
| Diğer yanlış pozitifler | Bildirilmedi | Bildirilmedi |
| Çıkış özeti | 331 kare, 18,4 FPS, çıkarım 16,8 / 17,1 / 21,5 ms, en çok 3 | 379 kare, 25,3 FPS, çıkarım 14,0 / 12,8 / 19,2 ms, en çok 3 |

### Bu güncellemede çalıştırılan kontroller *[Ölçüldü]*

Bu güncellemede **yalnızca belge, rapor ve açık görev kayıtları** değişti; kaynak kod, test ve betik değişmedi.

| Kontrol | Sonuç |
|---|---|
| `unittest discover -s tests` (depo kökünden) | **596 test, OK, 5 atlandı** (isteğe bağlı gerçek-pencere testleri) |
| Aynı komut başka çalışma dizininden | 596 test, OK, 5 atlandı |
| `ISG_GUI_TESTS=1 ...` (Terminal paneli) | **596 test, OK, 0 atlandı** |
| `run_stage2c_checks.py` | **12/12** (T2D.2 dâhil); çıktı JSON'undaki tarih-only fark geri alındı |
| `check_environment.py` | Başarılı |
| Aşama referans tutarlılığı | Sorun yok. Yaml: Aşama 4 `in_progress`; 4A `completed`, 4B `planned`; OT-4A-001 `test_performed`, OT-4A-002 `open`, OT-4A-003 `deferred_to_system_integration`, OT-4A-004 `open_investigation`; FP-4A-001 `open_known_limitation` |
| Varsayılan `conf` | **0.25** (`DetectionConfig().conf = 0.25`; CLI `--conf` varsayılanı `0.25`) |
| Kod/veri değişmezliği (`git diff HEAD`) | `src/`, `scripts/`, `tests/`, `data/`, `models/`, `reports/dataset_audit/`, `reports/camera/` altında fark yok. Değişen dosyalar: `README.md`, `configs/project_stages.yaml`, `docs/architecture.md`, `docs/person_detection.md`, `docs/project_plan.md`, `reports/detection/STAGE_4A_REPORT.md` |
| YOLO ağırlığı | Git dışı (`.gitignore:75`); `git ls-files` içinde `.pt` yok |
| Aşama 4B | Başlatılmadı; `src/isg_vision/tracking/` değişmedi (yer tutucu) |

Kullanıcının `conf=0.35` sonuçları (yukarıda) Claude tarafından yeniden üretilmedi; commit/push yapılmadı.

## 24. Kapanış sonrası gözlem — yeni ortamda gerçek kamera testi (2026-11-12) *[Kullanıcı]*

4B poster tanı planının **T1** adımı için kullanıcı 4A betiğiyle (takipsiz) yeni bir gerçek kamera testi yaptı. **Test ortamı değiştiği için önceki poster aynı şartlarda yeniden kurulamadı; poster karşılaştırması (T1) tamamlanmış sayılmaz.** Aşama 4A `COMPLETED` durumu, önceki sonuçlar ve varsayılan ayarlar (`conf 0.25`) değişmedi; kaynak kod ve eşik değiştirilmedi. Ayrıntı ve yorum sınırları: [docs/person_detection.md §14.4](../../docs/person_detection.md). Güncel durum: **Aşama 4B `READY_FOR_MANUAL_VERIFICATION`**, Aşama 4 genel `IN_PROGRESS`.

| Alan | Değer *[Kullanıcı]* |
|---|---|
| Model / cihaz / çözünürlük | YOLO11n, MPS, 1280×720 (`conf`/`iou`/`imgsz` bildirilmedi; varsayılan varsayılıyor, doğrulanmadı) |
| Kare / süre / FPS | 1812 / 82,7 sn / **21,9** (türetilen: (1812−1)/82,7 = 21,9; periyot 45,7 ms) |
| Çıkarım | ort. **15,9 ms** (periyodun %35'i; kalan ≈ 29,8 ms bildirilmedi) |
| Kişi tespiti | kare başına ort. **2,89**, en çok **7** |
| Çıkış | Q, sorunsuz |

* **OBS-4A-003 (yeni ortam gözlemi):** ekran görüntüsü 1'de ön plandaki iki gerçek kişi **0,85 ve 0,87** ile ayrı algılandı; ekran görüntüsü 2'de **0,80 ve 0,66**; telefonda gösterilen insan fotoğrafı **ayrı tespit oluşturmadı** (tek kare; telefon fotoğrafı poster değildir, poster sorunu açık kalır — FP-4B-001). Zaman zaman arka planda gerçek kişiler de vardı: **ort. 2,89 ve en çok 7 yanlış pozitif olarak yorumlanmaz** (doğru olarak da sınıflandırılmaz; kare bazlı kayıt yok). "En çok 7", OBS-4A-001 ile aynı sayıdır; ilişki kurulmaz.
* **OBS-4A-004 (araştırılacak):** arka plandaki çok küçük görünen kişiler **kutulanmadı**; nedeni araştırılmadı. Kameasız ön prob (iyimser, kullanıcı sahnesi değil): `imgsz 640`'ta 720 px yüksekliğinde kişi boyu ≈ 90–120 px'in altına inince algılama belirgin düştü; `imgsz` artırmak yardımcı oldu ama çıkarımı yavaşlatır (MPS 7,7 → 11,8 → 18,2 ms; CPU 19,0 → 36,8 → 64,3 ms; 640/960/1280). Plan: **OT-4A-005** (docs §14.4); ayar değişikliği yapılmadı.
* **OBS-4A-002'ye veri noktası:** pencereli 21,9 FPS (takipsiz) — pencereli FPS düşüşünün takip olmadan da görüldüğünü gösterir; 4A'nın önceki 18,4–25,3 FPS aralığıyla uyumlu; nedeni ayrıştırılmamıştır.
* **Doğruluk iddiası yoktur:** Precision/Recall ölçülmedi (OT-4A-002); %85 hedefi karşılandı denmez. Commit/push yapılmadı.

---

## Kabul kriterleri (tek tek)

| # | Kriter | Durum | Kanıt |
|---|---|---|---|
| 1 | Mevcut proje yapısı korundu | ✔ | Mevcut `human_detection/` paketi kullanıldı; yeni kök klasör yok |
| 2 | Aşama 3 kamera modülü bozulmadı | ✔ | `camera/` altında fark yok; 170 kamera testi geçiyor |
| 3 | Hazır YOLO modeli başarıyla yükleniyor | ✔ | `yolo11n.pt`, 80 sınıf, 2 624 080 parametre *[Gerçek model]* |
| 4 | Gerçek modelle insan tespiti çalışıyor | ✔ | §11–12, CPU ve MPS |
| 5 | Yalnızca `person` sınıfı işleniyor | ✔ | ham çıktıda `bus` var, sonuçta yok *[Gerçek model]* |
| 6 | Tek insan destekleniyor | ✔ | §12-B |
| 7 | Birden fazla insan destekleniyor | ✔ | §12-C (görüntü düzeyinde *[Gerçek model]*) + canlı kamerada üç kişi `PERSONS: 3` *[Kullanıcı]* |
| 8 | İnsan bulunmayan kareler yönetiliyor | ✔ | §12-A |
| 9 | Bounding box ve confidence gösteriliyor | ✔ | çizim testleri + gözle incelenen kare + gerçek pencere koşusu; canlı kamerada kullanıcı gözlemi *[Kullanıcı]* |
| 10 | Toplam kişi sayısı gösteriliyor | ✔ | `PERSONS: n` (testler, gerçek model) |
| 11 | FramePacket entegrasyonu tamamlandı | ✔ | `run_detection`, `FrameSource.frames()` tüketir |
| 12 | Kamera ve video kaynakları destekleniyor | ✔ | Video: gerçek koşu ✔. Kamera: sahte kaynak + kod yolu ✔; **gerçek MacBook kamerası kullanıcı tarafından doğrulandı** *[Kullanıcı]* (§19). IP/RTSP kaynağı kapsam dışı (OT-4A-003) |
| 13 | FPS ve çıkarım gecikmesi ölçülüyor | ✔ | §10 (kayıtlı video, sentetik) ve canlı MacBook kamerası (§19, CPU ve MPS ayrı) *[Kullanıcı]* |
| 14 | MPS ve CPU davranışları test edildi / engeller belgelendi | ✔ | §10, §5, §19 (canlı CPU/MPS); Claude ortamındaki kamera izni engeli §12-E |
| 15 | Kamera hataları güvenli yönetiliyor | ✔ | testler + gerçek izin hatası (çıkış 1, mesaj, kaynak serbest) |
| 16 | Kullanıcı çıkışlarında kaynaklar serbest bırakılıyor | ✔ | Q/Esc/pencere/Ctrl+C testleri; gerçek pencerede kırmızı düğme; gerçek kamerada dört çıkış yöntemi sorunsuz *[Kullanıcı]* (çıkış kodları paylaşılmadı) |
| 17 | Yeni otomatik testler başarılı | ✔ | 256 yeni test geçiyor |
| 18 | Önceki aşamaların testleri başarılı | ✔ | 340 test korunuyor, 596/596 |
| 19 | Aşama 2 veri bütünlüğü korunuyor | ✔ | 12/12; `data/` ve raporlar değişmedi |
| 20 | Model performansı hakkında doğrulanmamış iddia yok | ✔ | doğruluk raporlanmadı, %85 hedefi karşılandı denmedi; hız yalnızca ölçülen koşulda, sınırlarıyla; nedeni ayrıştırılmayan farklar iddiasız bırakıldı |
| 21 | Dokümantasyon ve rapor oluşturuldu | ✔ | `docs/person_detection.md`, bu rapor, README/plan/mimari/yaml |
| 22 | Aşama 4B'ye geçilmedi | ✔ | takip/Track ID kodu yok; yaml: `4B: planned` |
| 23 | Manuel testler kaynaklarına göre sınıflandı | ✔ | *[Kullanıcı]* / *[Ölçüldü]* / *[Türetilmiş]* ayrımı (§19, başlık) |
| 24 | Açık görevler tanımlı ve izlenebilir | ✔ | OT-4A-001…004, FP-4A-001, OBS-4A-001/002; yaml `stage4a_open_items` |
| 25 | `conf` varsayılanı değişmedi; 0.35 testi doğru raporlandı | ✔ | `DEFAULT_CONF = 0.25` (§22, §23); OT-4A-001 `TEST_PERFORMED`: test kapanıştan sonra yapıldı, sorunu çözmedi, gerçek insan tespiti başarısı ölçülmedi (§23) |
| 26 | Gerçek IP kamera satın alınmış gibi davranılmadı | ✔ | OT-4A-003 `DEFERRED_TO_SYSTEM_INTEGRATION`; MacBook testi IP kamera doğrulaması olarak sunulmadı |

**Karar:** Mevcut Aşama 4A yazılım ve prototip kabul kriterleri karşılanmaktadır; kullanıcının MacBook'ta yaptığı manuel doğrulamalarla birlikte Aşama 4A, **yalnızca kapsam içi prototip geliştirme ve doğrulama olarak `COMPLETED`** yapıldı.
**Aşama 4 genel durumu `IN_PROGRESS`'tir; Aşama 4B `PLANNED`'dır ve başlatılmadı.** Gerçek IP kamera satın alınmadığı için Aşama 4A açık tutulmadı. Açık kalanlar (§21, §23): yorgan yanlış pozitifi FP-4A-001 (kapanıştan sonra yapılan `conf=0.35` testinde de görüldü, çözülmedi); bağımsız doğruluk ölçümü; gerçek IP kamera; OBS-4A-001. Commit/push yapılmadı.
