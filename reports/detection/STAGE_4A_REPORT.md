# Aşama 4A — Çoklu İnsan Tespiti: Sonuç Raporu

**Tarih:** 2026-11-12 · **Durum:** `READY_FOR_MANUAL_VERIFICATION` (Aşama 4 genel durumu `COMPLETED` **değildir**; 4B takip başlatılmamıştır)
**Teknik belge:** [docs/person_detection.md](../../docs/person_detection.md)

> **Özet.** Hazır `yolo11n.pt` ile Aşama 3 kare akışındaki her karede aynı anda birden fazla insan tespit edilir (yalnızca `person`); kutu, güven skoru ve kişi sayısı canlı gösterilir; FPS ve çıkarım gecikmesi ölçülür.
> Gerçek model çıkarımı CPU ve MPS'te çalıştırılarak doğrulandı. **Yapılamayanlar:** MacBook kamerasıyla canlı test (Claude ortamında kamera izni yok), yüksek açılı gerçek kamera testi ve doğruluk ölçümü (etiketli bağımsız test kümesi yok).
> Bu nedenle `COMPLETED` değil, `READY_FOR_MANUAL_VERIFICATION` kullanıldı.

**Kanıt türleri bu raporda ayrı tutulmuştur:** *[Ölçüldü]* = Claude'un bu oturumda çalıştırıp sonucunu gördüğü; *[Mock]* = sahte YOLO ile test; *[Gerçek model]* = gerçek `yolo11n.pt` çıkarımı; *[Bekliyor]* = kullanıcı doğrulaması gerekir, yapılmış sayılmaz.

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
(d) `ISG_GUI_TESTS=1` ile gerçek pencerede kırmızı düğme tıklaması `pencere kapatıldı` ile döngüyü bitirdi ve kamerayı serbest bıraktı. **Canlı pencerenin ekrandaki görünümünü kullanıcı gözle doğrulamadı** *[Bekliyor]*. Q/Esc/Ctrl+C davranışı korunmuştur (birim testleri + (d)); gerçek klavye basışı kullanıcı doğrulaması bekler.

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
* **Canlı kamerada CPU/MPS ölçümü yapılmadı** *[Bekliyor]*. 30 FPS zorunluluğu konmadı; kameranın ~30 FPS vermesi durumunda bu makinede model (penceresiz ölçümlere göre) darboğaz görünmüyor, ancak bu **canlı kamerada doğrulanmadı**.
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
| **C — çoklu insan** | `bus.jpg` → **4 kişi** (bağımsız kutular, sayı doğru); `zidane.jpg` → **2 kişi**. Kontrollü test görüntüsü (video değil) | ✔ görüntü düzeyinde; **gerçek ≥3 kişilik izinli video yok** *[Bekliyor]* |
| **D — kamera açısı** | Göz hizasına yakın iki görüntü (açı metadata'sı yok, görsel tahmin). Hafif/yüksek açı için uygun görüntü yok → **`PENDING_REAL_CAMERA_VALIDATION`** | ⏳ *[Bekliyor]* |
| **E — MacBook kamerası** | Claude ortamında kamera izni yok (`not authorized to capture video (status 0)`, bir deneme yapıldı, izin aşılmadı). **Yapılmış sayılmaz** | ⏳ *[Bekliyor]* — komutlar §17 |

**Gizlenmeyen gözlemler.** (1) Bir kesitte (`bus` sol-orta) conf 0,25'te ikinci bir düşük güvenli (0,273) kutu çıktı — kesilmiş bir komşu gövde parçası; etiket olmadığı için "yanlış pozitif" demek de, değil demek de mümkün değil. (2) Sentetik video karesinde 6 ön plan kişisi + bulanık arka plan kalabalığı bulunuyor; model kare başına ortalama 6,52 (en çok 8) tespit verdi,
bazı kutular 0,31–0,37 güvenli ve bulanık arka plandaki kişiler/kenarda kırpılmış kişi üzerinde. Gerçek sayı etiketlenmediği için **doğru/yanlış söylenemez ve bu sayılar doğruluk olarak raporlanmaz**. (3) Bu örneklerde görünür bir **kaçırma** gözlenmedi, ancak iki görüntü ve birkaç kesit genelleme sağlamaz.

## 13. Kamera açısı değerlendirmesi

Hedef kurulum: duvara 2–3 m yükseklikte monte tek kamera. Kod, açıyı/çözünürlüğü sabit varsaymaz (kare boyutu gerçek kareden okunur; `--imgsz`/`--conf` ayarlanabilir). **Yüksek açılı gerçek kamera testi yapılmadı**; model COCO ile eğitilmiştir ve bu açıdaki performansı bilinmemektedir.
Durum: `PENDING_REAL_CAMERA_VALIDATION`. Bu açıda küçük/uzak/üstten görünen kişiler için `imgsz` artırılması ve `conf` ayarı gerekebilir (denenmedi).

## 14. Otomatik test sonuçları *[Ölçüldü]*

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

1. **Doğruluk ölçülmedi.** Etiketli bağımsız insan test kümesi yok → Precision/Recall/F1/mAP raporlanmadı; güven skorları doğruluk olarak sunulmadı. %85 hedefinin hangi metrik için olduğu belirsiz.
2. Performans: tek makine, tek oturum, **sentetik** video, pencere maliyeti yok (tek pencereli gözlem hariç); canlı kamerada ölçülmedi; ısıl/güç kontrol edilmedi.
3. Model COCO ile eğitilmiş; yüksek açılı/şantiye/KKD'li kişiler için doğrulanmadı.
4. Düşük güvenli (0,27–0,37) kutuların doğru/yanlış olduğu bilinmiyor (§12).
5. Lisans AGPL-3.0 (§3).
6. Aşama 3 backlog: B3-1 ve B3-5 kapsanmadı; B3-2 (kare kaybı) ölçülmedi; B3-3 için yalnızca **kayıtlı videoda** kare→sonuç gecikmesi ölçüldü (canlı kamera yok); B3-4 (canlı kamerada ileri FPS) yapılmadı. **Hiçbiri tamamlanmış sayılmaz.**
7. Çalışan modeli yalnızca `person` sınıfıyla sınırlı; yüz/KKD yok (bilinçli kapsam dışı).
8. Ultralytics bağımlılıkları kendiliğinden kurmaya (ağ) çalışabilir; 4B'de ByteTrack için gereken paketler (`lap`/`scipy` kurulu değil) bilinçli kurulmalı ve `YOLO_OFFLINE` ile birlikte ele alınmalıdır.

## 17. Eksik kalan manuel doğrulamalar *[Bekliyor — yapılmış sayılmaz]*

Claude'un ortamında macOS kamera izni yoktur; bu testleri **siz** izinli terminalde çalıştırmalısınız:

```bash
.venv/bin/python scripts/run_person_detection.py --source camera --camera-index 0 --device mps
.venv/bin/python scripts/run_person_detection.py --source camera --camera-index 0 --device cpu
.venv/bin/python scripts/run_person_detection.py --source camera --camera-index 0 --no-display --max-frames 300 --device mps   # canlı hız özeti
```

Kontrol listesi: (1) pencere açılıyor ve kutu + `PERSON 0.xx` + `PERSONS: n` + FPS + INFERENCE görünüyor; (2) tek kişi, ardından ≥3 kişi (izinli) ile sayı doğru; kimse yokken `PERSONS: 0`; (3) `Q`, `Esc`, kırmızı düğme ve `Ctrl+C` ile çıkış (`exit=0`/`130`), kamera ışığı sönüyor;
(4) çıkış özetindeki canlı FPS/çıkarım/gecikme değerlerini not edin (CPU ve MPS ayrı); (5) mümkünse göz hizası ve daha yüksek açıdan deneyin (`PENDING_REAL_CAMERA_VALIDATION`); (6) gerçek yanlış pozitif/kaçırma görürseniz not edin ve `--conf`/`--imgsz` değiştirerek deneyin.
Etiketli bağımsız test kümesi hazırlanınca: `scripts/evaluate_person_detection.py` (altyapı hazır; veri özellikleri docs/person_detection.md §10).

## 18. Aşama 4B için hazır olan bileşenler

* `DetectionResult.detections`: kare başına `bbox_xyxy` (piksel), `confidence`, `frame_index`, `timestamp` — takipçi girdisi; takip kimliği alanı bilinçli olarak yok.
* `run_detection(..., on_result=...)` kancası; `FramePacket.timestamp` (monotonic) zaman tabanlı takip için.
* Ölçüm altyapısı (`LatencyStats`, benchmark, CPU/MPS karşılaştırması) takip eklenince gecikme artışını ölçmek için yeniden kullanılabilir.
* **4B kararları (henüz verilmedi):** takipçi seçimi (Ultralytics ByteTrack/BoT-SORT vs. kendi uygulaması), ek bağımlılıklar (`lap`/`scipy`), düşük güvenli kutuların takipte kullanımı (ByteTrack bunları kullanır; mevcut varsayılan `conf` 0,25).

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
| 7 | Birden fazla insan destekleniyor | ✔ | §12-C (görüntü düzeyinde); gerçek ≥3 kişilik video **bekliyor** |
| 8 | İnsan bulunmayan kareler yönetiliyor | ✔ | §12-A |
| 9 | Bounding box ve confidence gösteriliyor | ✔ | çizim testleri + gözle incelenen kare + gerçek pencere koşusu |
| 10 | Toplam kişi sayısı gösteriliyor | ✔ | `PERSONS: n` (testler, gerçek model) |
| 11 | FramePacket entegrasyonu tamamlandı | ✔ | `run_detection`, `FrameSource.frames()` tüketir |
| 12 | Kamera ve video kaynakları destekleniyor | ◐ | Video: gerçek koşu ✔. Kamera: sahte kaynak + kod yolu ✔; **gerçek kamera yapılmadı** *[Bekliyor]* |
| 13 | FPS ve çıkarım gecikmesi ölçülüyor | ✔ | §10 (kayıtlı video); canlı kamerada ölçülmedi |
| 14 | MPS ve CPU davranışları test edildi / engeller belgelendi | ✔ | §10, §5; kamera izni engeli §12-E |
| 15 | Kamera hataları güvenli yönetiliyor | ✔ | testler + gerçek izin hatası (çıkış 1, mesaj, kaynak serbest) |
| 16 | Kullanıcı çıkışlarında kaynaklar serbest bırakılıyor | ✔ | Q/Esc/pencere/Ctrl+C testleri; gerçek pencerede kırmızı düğme |
| 17 | Yeni otomatik testler başarılı | ✔ | 256 yeni test geçiyor |
| 18 | Önceki aşamaların testleri başarılı | ✔ | 340 test korunuyor, 596/596 |
| 19 | Aşama 2 veri bütünlüğü korunuyor | ✔ | 12/12; `data/` ve raporlar değişmedi |
| 20 | Model performansı hakkında doğrulanmamış iddia yok | ✔ | doğruluk raporlanmadı; hız yalnızca ölçülen koşulda, sınırlarıyla |
| 21 | Dokümantasyon ve rapor oluşturuldu | ✔ | `docs/person_detection.md`, bu rapor, README/plan/mimari/yaml |
| 22 | Aşama 4B'ye geçilmedi | ✔ | takip/Track ID kodu yok; yaml: `4B: planned` |

**Karar:** Zorunlu manuel doğrulamalar (gerçek kamera, yüksek açı, gerçek çok kişili sahne) yapılamadığı için Aşama 4A `COMPLETED` **ilan edilmedi**; durum **`READY_FOR_MANUAL_VERIFICATION`**. Aşama 4 genel durumu `in_progress`'tir. Commit/push yapılmadı. Aşama 4B başlatılmadı.
