# Çoklu İnsan Tespiti Modülü (Aşama 4A)

> **Durum: Aşama 4A `COMPLETED`** (2026-11-12; yalnızca kapsam içi prototip geliştirme ve doğrulama). Kod, otomatik testler, gerçek model doğrulamaları ve kullanıcının gerçek MacBook kamerasıyla yaptığı manuel testler tamamdır (§9.1, §14).
> Aşama 4 genel durumu **`IN_PROGRESS`**: takip (Aşama 4B) `PLANNED`, başlatılmamıştır. **Kapanışta açık kalanlar** (kapatılmış sayılmaz, §14): `conf=0.35` karşılaştırma testi (`PENDING_USER_VERIFICATION`), bağımsız doğruluk değerlendirmesi,
> gerçek IP kamera doğrulaması (`DEFERRED_TO_SYSTEM_INTEGRATION`), yorgan yanlış pozitifi (FP-4A-001) ve ilk oturumdaki en yüksek 7 tespit gözlemi (OBS-4A-001). Doğruluk ölçülmedi; %85 hedefinin karşılandığı iddia edilmez.
> Teknik sonuç raporu: [../reports/detection/STAGE_4A_REPORT.md](../reports/detection/STAGE_4A_REPORT.md).

## 1. Amaç ve kapsam

Aşama 3'ün kare akışından (`FrameSource.frames()` → `FramePacket`) gelen her karede, önceden eğitilmiş bir Ultralytics YOLO modeliyle **aynı anda birden fazla insanı** tespit etmek;
kutu, güven skoru ve kişi sayısını canlı görüntüde göstermek; FPS ve çıkarım gecikmesini ölçmek.

Bu aşamada **yoktur** (sonraki aşamaların işidir): takip / Track ID (4B), yüz tespiti (5), çalışan kaydı (6), kimlik tanıma (7), baret/yelek tespiti ve KKD modeli eğitimi (8+), hayvan tespiti,
sensör/mesai modu. Sonuç nesnelerinde `track_id`, çalışan adı veya yüz kimliği alanı **bulunmaz**. Model eğitilmez; veri setlerine dokunulmaz; görüntü kaydedilmez.

## 2. Dosyalar

```
src/isg_vision/human_detection/          (mevcut Aşama 1 iskeleti kullanıldı; ayrı bir "detection/" paketi açılmadı)
├── errors.py            hata sınıfları (yapılandırma / yükleme / cihaz / geçersiz kare / çıkarım / indirme)
├── config.py            DetectionConfig + cihaz seçimi (resolve_device)
├── results.py           PersonDetection, DetectionResult, kutu kırpma/doğrulama
├── person_detector.py   PersonDetector: YOLO'yu bir kez yükler, tek karede insanları tespit eder
├── visualization.py     draw_person_detections: kutu + güven + durum şeridi (kopya üzerinde)
├── pipeline.py          run_detection: FrameSource → PersonDetector → çizim → pencere; FPS/gecikme ölçümü
├── cli.py               komut satırı (scripts/run_person_detection.py bunu çağırır)
├── weights.py           resmî ağırlığın izin-listeli, SHA-256 doğrulamalı indirilmesi (projede ağ kullanan tek kod)
├── benchmark.py         CPU/MPS karşılaştırma altyapısı
└── evaluation.py        Precision/Recall/F1/mAP altyapısı (kendi başına sonuç üretmez)

scripts/run_person_detection.py          canlı/penceresiz insan tespiti
scripts/download_person_model.py         resmî ağırlığı açıkça indirir (tek ağ adımı)
scripts/benchmark_person_detection.py    CPU/MPS hız karşılaştırması (+ sentetik örnek video üretimi)
scripts/evaluate_person_detection.py     etiketli kümede Precision/Recall/F1/mAP (veri kümesi yok → sonuç yok)
```

Aşama 3 kamera kodu **yeniden yazılmadı**; `FrameSource`, `FramePacket`, `FpsMeter`, `PreviewWindow` (macOS kırmızı düğme düzeltmesiyle) ve `run_preview` içe aktarılarak kullanıldı.

## 3. Model

| Alan | Değer |
|---|---|
| Model | **Ultralytics YOLO11n** (`yolo11n.pt`), COCO üzerinde önceden eğitilmiş nesne tespiti, 80 sınıf |
| Ultralytics sürümü | `ultralytics==8.4.210` (kurulu; `requirements.txt`'te sabit). Ağırlık dosyası 8.2.100 ile eğitilmiş/kaydedilmiştir (checkpoint metadata'sı) |
| Kaynak (resmî) | `https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo11n.pt` (Ultralytics'in resmî `assets` deposu, release `v8.4.0`) |
| Dosya | `models/human_detection/yolo11n.pt` — **Git dışı** (`.gitignore: *.pt`), 5 613 764 bayt (≈5,35 MiB) |
| SHA-256 | `0ebbc80d4a7680d14987a577cd21342b65ecfd94632bd9a8da63ae6417644ee1` — GitHub release API'sindeki asset özetiyle eşleştiği doğrulandı; `weights.py`'de sabitlendi |
| Parametre sayısı | 2 624 080 (modelden ölçüldü) |
| Eğitim verisi | COCO (checkpoint `train_args.data = coco.yaml`, `imgsz=640`) — genel sahneler; şantiye/baret odaklı değildir |
| `person` sınıfı | Sınıf haritasından (`model.names`) okunur: bu modelde kimlik `0`. Kod `0`'ı varsaymaz; haritada `person` yoksa model reddedilir |
| Lisans | **AGPL-3.0** (checkpoint metadata'sı ve `ultralytics` paket metadata'sı). Kapalı kaynak/ticari dağıtım için Ultralytics'in ayrı (Enterprise) lisansı gerekir: <https://ultralytics.com/license>. Bu bir öğrenci prototipidir; ağırlıklar Git'e eklenmez. AGPL yükümlülükleri (özellikle ağ üzerinden hizmet/dağıtım) proje ürünleştirilirse yeniden değerlendirilmelidir |

**Seçim gerekçesi ve sınırı.** YOLO11 ailesinin en küçük (`n`) tespit modeli, düşük gecikme önceliğine uygundur; klasik NMS tabanlıdır (bu yüzden `iou` eşiği anlamlıdır); kurulu Ultralytics sürümünün resmî
asset listesinde bulunur; CPU ve MPS'te gerçekten çalıştırılarak doğrulanmıştır (§9, §11). **Diğer modellerle ölçülmüş bir karşılaştırma yapılmadı**: kurulu sürüm `yolo26n.pt`, `yolov8n.pt` gibi dosya adlarını da tanır, fakat
bunlar indirilmedi ve değerlendirilmedi; "en iyi model" iddiası yoktur. Başka bir COCO `person` modeli `models/human_detection/` altına konup `--model` ile denenebilir.

**İndirme prosedürü.** Model uygulama içinde **otomatik indirilmez**. Yalnızca açık adım:

```bash
.venv/bin/python scripts/download_person_model.py            # izin listesindeki resmî URL; SHA-256 doğrulanır; var olan dosyanın üzerine yazılmaz
.venv/bin/python scripts/download_person_model.py --verify-only
```

`.pt` dosyaları `pickle` içerir; güvenilmeyen kaynaktan yüklemek kod çalıştırabilir. Bu yüzden yalnızca izin-listesi + sabit SHA-256 kabul edilir.

## 4. Mimari

```
FrameSource ──► FramePacket ──► PersonDetector.detect(image) ──► DetectionResult(PersonDetection[]) ──► draw_person_detections ──► PreviewWindow
 (kamera)        (kamera)        (YOLO, person-only)             (piksel xyxy, güven)                    (kopya üzerinde)         (kamera)
                                          ▲
                              run_detection (pipeline.py) bunları birleştirir; her biri bağımsız değiştirilebilir
```

`FrameSource`, `FramePacket` ve `PreviewWindow` kamera paketindendir (Aşama 3'te tamamlandı); tespit, çizim ve döngü bu aşamada eklendi.

* **`PersonDetector`** kamerayı, pencereyi ve çizimi bilmez; girdi `uint8` BGR `numpy` görüntüsüdür. MacBook kamerası, USB kamera, video dosyası ve (ileride) IP/RTSP kaynağı aynı kodla çalışır. RTSP bu aşamada **geliştirilmedi**.
* Model **bir kez** yüklenir (`load`, tekrar çağrısı etkisizdir) ve her karede yeniden yüklenmez. Isınma (`warmup`) yüklemeden ayrıdır.
* **Yalnızca `person`**: model çağrısına `classes=[person_id]` verilir **ve** çıktı ayrıca süzülür (iki katman). Gerçek modelde süzülmemiş çıktıda `bus` görünür, sonuçta görünmez (test: `test_only_person_class_is_returned_even_though_model_sees_a_bus`).
* **Cihaz seçimi**: `auto` (MPS varsa MPS, yoksa **uyarıyla** CPU), `cpu`, `mps`. Açıkça `mps` istenip kullanılamıyorsa `DeviceUnavailableError` verilir; sessizce CPU'ya düşülmez ve "MPS çalışıyor" diye raporlanmaz.
* **Ağ yok**: Ultralytics `import` sırasında varsayılan olarak DNS yoklaması yapar; bu **ölçülmüştür** (test, `YOLO_OFFLINE` kaldırılınca ağ çağrısını yakalar). `PersonDetector` model yüklemeden önce `YOLO_OFFLINE=1` ayarlar. Gerçek model testi, ağ çağrıları sahtelenmiş bir alt süreçte çalışır ve diske dosya yazılmadığını doğrular.
* **Hata yönetimi**: geçersiz kare → `InvalidFrameError`; model dosyası yok/bozuk/`person` yok → `ModelLoadError` (kamera hiç açılmaz); MPS yok → `DeviceUnavailableError`; çıkarım hatası → `InferenceError`. CLI bunları çıkış kodu 1 ile bildirir.

### 4.1 Yapılandırma (`DetectionConfig`)

| Alan | Varsayılan | Kural | Not |
|---|---|---|---|
| `model_path` | `models/human_detection/yolo11n.pt` | `.pt` uzantısı; varlığı yüklemede denetlenir | |
| `conf` | `0.25` | `0 < conf ≤ 1` | Ultralytics varsayılanı. **Evrensel optimum değildir**; kamera açısı/sahneye göre ayarlanır. Düşük = daha çok (yanlış pozitif dâhil) tespit |
| `iou` | `0.7` | `0 < iou ≤ 1` | NMS eşiği (Ultralytics varsayılanı) |
| `imgsz` | `640` | 32–4096, 32'nin katı | Küçük = hızlı ama uzak/küçük insanları kaçırabilir. Sessizce yuvarlanmaz, reddedilir |
| `device` | `auto` | `auto` / `cpu` / `mps` | |
| `max_det` | `100` | 1–1000 | |

Görüntüleme açık/kapalı ve maksimum kare sayısı çalışma seçenekleridir (`--no-display`, `--max-frames`).

### 4.2 Veri yapıları (`results.py`)

`PersonDetection` (değiştirilemez): `class_id`, `class_name`, `confidence` (0..1), **`bbox_xyxy = (x1, y1, x2, y2)`**, `frame_index`, `timestamp`.
Koordinatlar **görüntünün piksel koordinatlarıdır**; başlangıç sol-üst köşe, `x1 < x2`, `y1 < y2`; kutular görüntü sınırlarına **kırpılır**. `width`, `height`, `area`, `center` yardımcı özelliklerdir.

`DetectionResult`: `detections` (güvene göre azalan), `person_count`, `image_size`, `frame_index`, `timestamp`, `inference_ms`, `device`, `dropped_boxes`. İnsan yoksa `detections` boş demettir (hata değildir).

Geçersiz kutular (NaN/sonsuz, `x1≥x2`, tamamen görüntü dışı, kırpma sonrası kenarı 1 pikselden küçük) **atılır ve `dropped_boxes` ile sayılır** (gizlenmez). Görüntü kenarına taşan insanlar atılmaz, kırpılır.

**Güven skoru kalibre edilmemiş bir model çıktısıdır; doğruluk olasılığı değildir.**

## 5. FramePacket entegrasyonu

```python
from isg_vision.camera import FrameSource, SourceConfig
from isg_vision.human_detection import DetectionConfig, PersonDetector

with FrameSource(SourceConfig.camera(0)) as source, PersonDetector(DetectionConfig(device="auto")) as detector:
    detector.warmup(source.actual_resolution or (640, 480))
    for packet in source.frames():
        result = detector.detect(packet.frame, frame_index=packet.frame_index, timestamp=packet.timestamp)
        for person in result.detections:
            print(person.bbox_xyxy, person.confidence)
```

Hazır döngü (`pipeline.run_detection`) pencere, Q/Esc/pencere kapatma/`Ctrl+C`, video sonu ve `--max-frames` çıkışlarını ve ölçümleri yönetir. `on_result(packet, result)` kancası, Aşama 4B takipçisinin aynı döngüye bağlanması için bırakılmıştır.

## 6. Canlı görüntüleme

Üst şeritte: `PERSONS: n`, kaynak, gerçek çözünürlük, `FPS` (ölçülen, uçtan uca, son 30 karenin ortalaması), `INFERENCE` (son 30 karenin ortalama çıkarım süresi), `DEVICE`; her kutuda `PERSON 0.94` gibi güven etiketi; altta `Q / Esc / Close to exit`.
İnsan yoksa `PERSONS: 0`. Çizim **girdinin kopyası** üzerindedir (kaynak kare değişmez). Tüm kutular aynı renktedir: kimlik/takip bu aşamada yoktur. Metinler ASCII'dir (OpenCV yazı tipi `ç ş ğ ı ö ü` çizemez).
Pencere `camera.display.PreviewWindow`'dur: macOS kırmızı kapatma düğmesi düzeltmesi, Q/Esc ve `Ctrl+C` davranışı korunmuştur (gerçek pencere testiyle doğrulandı, §11; gerçek kamerada kullanıcı tarafından da, §14.1).

## 7. Komut satırı

```bash
.venv/bin/python scripts/run_person_detection.py --source camera                                  # varsayılan kamera
.venv/bin/python scripts/run_person_detection.py --source camera --camera-index 0
.venv/bin/python scripts/run_person_detection.py --source video --video-path /yol/video.mp4
.venv/bin/python scripts/run_person_detection.py --source camera --device mps                     # MPS yoksa hata verir
.venv/bin/python scripts/run_person_detection.py --source camera --device cpu                     # CPU karşılaştırması
.venv/bin/python scripts/run_person_detection.py --source camera --conf 0.35                      # güven eşiği
.venv/bin/python scripts/run_person_detection.py --source camera --no-display --max-frames 100   # penceresiz
.venv/bin/python scripts/run_person_detection.py --help
```

Seçenekler: `--source`, `--camera-index`, `--width/--height`, `--video-path`, `--backend` (`run_camera.py` ile aynı) ve `--model`, `--conf`, `--iou`, `--imgsz`, `--max-det`, `--device`, `--no-display`, `--max-frames`.
Çıkış kodları: `0` normal, `1` kamera/video/pencere/model hatası, `2` hatalı argüman, `130` Ctrl+C. Sıra: model yüklenir (başarısızsa kamera **hiç açılmaz**) → kaynak açılır → gerçek kare boyutunda ısıtılır → pencere → döngü.
Model/cihaz/yükleme süresi, ısınma süresi ve oturum özeti (FPS, çıkarım ve gecikme istatistikleri, kişi sayıları) yazdırılır.

## 8. Performans ölçümünün tanımı

Dört ölçüm **birbirinden ayrıdır**:

| Ölçüm | Tanım | Nerede |
|---|---|---|
| **Yakalama FPS'i** | Kameradan/dosyadan tamamlanan okumaların hızı. Ardışık döngüde okuma çıkarımla iç içe olduğundan ayrı bir değer olarak modelsiz ölçülür | `run_preview` / benchmark yakalama geçişi; canlı kamera için Aşama 3 ölçümü (~30 FPS) |
| **Çıkarım gecikmesi** | Yalnızca `PersonDetector.detect` içindeki model çağrısı: ön işleme + ileri geçiş + NMS + sonuçların CPU'ya alınıp `PersonDetection`'a çevrilmesi. Kare okuma, çizim, gösterme **dâhil değildir** | `DetectionResult.inference_ms` |
| **Uçtan uca FPS** | Döngünün gerçekte işlediği hız: ilk-son kare arası `(N−1)/süre`; pencereli modda çizim + `imshow` + `waitKey(1)` dâhil, penceresizde değil | `DetectionRunSummary.end_to_end_fps` |
| **Kare → sonuç / kare → görüntü gecikmesi** | `FramePacket.timestamp` (okumanın bittiği an) ile tespitin bitişi / `window.show` dönüşü arası. **Dâhil değil:** kameranın pozlaması, sürücü/işletim sistemi tamponlaması (okumadan *önceki* süre) | `frame_to_result`, `frame_to_display` |

* Zaman: `time.perf_counter` (çıkarım) ve `time.monotonic` (döngü); hepsi monotoniktir. **Kaynağın bildirdiği FPS hiçbir yerde ölçülen FPS yerine kullanılmaz** (Aşama 3 bulgusu: kamera 15,0 bildirirken ~30 FPS okunuyor).
* **MPS asenkrondur**: çıkarım bitişinden önce `torch.mps.synchronize()` çağrılır; ayrıca sonuçlar CPU'ya alınırken de senkronizasyon gerçekleşir. Ultralytics'in kendi `speed` (ön/ileri/son işlem) kırılımı MPS'te asenkron işi kapsamayabileceği için **kullanılmaz**.
* **Yükleme, ısınma ve burn-in** sürekli performanstan ayrı raporlanır. Isınma boş karelerle yapılır; MPS'te gerçek içerikli ilk karelerde yine de ek maliyet görülebildiği için benchmark ayrıca ölçüme alınmayan 10 kare "burn-in" uygular.
* Video + pencere modunda oynatma kaynağın hızına uydurulur (`waitKey` beklemesi); bu bekleme "görüntü gecikmesi"ne **sayılmaz** (`None` bırakılır).

## 9. CPU / MPS karşılaştırması (ölçülmüş; yalnızca bu makine ve oturum)

Yöntem ve sınırlar: `human_detection/benchmark.py`. Ortam: MacBook Pro M4 (`arm64`, macOS 26.5.2), Python 3.11.17, torch 2.15.0, ultralytics 8.4.210, OpenCV 5.0.0, torch iş parçacığı 8. Model `yolo11n.pt`,
`conf 0.25`, `iou 0.7`, `imgsz 640`. Video: `scripts/benchmark_person_detection.py --make-sample-video` ile üretilen **SENTETİK** 1280×720, 300 kare (Ultralytics'in `bus.jpg` ve `zidane.jpg` örnek görüntülerinden montaj;
**yalnızca hız içindir**, gerçek bir çalışma alanını temsil etmez, doğruluk ölçümünde kullanılamaz). 5 geçiş, cihazlar dönüşümlü sırayla; pencere ve çizim **yok**.

| Cihaz | Model yükleme | Isınma (2 boş kare) | Uçtan uca FPS (5 geçiş: medyan / aralık) | Çıkarım medyan (geçiş medyanlarının medyanı / aralık) | Çıkarım p95 |
|---|---|---|---|---|---|
| CPU | 0,55 sn | 0,59 sn | 51,7 / 46,6–52,5 | 18,8 ms / 18,7–20,4 | ≈19,3 ms |
| MPS | 0,06 sn | 0,32 sn | 104,2 / 100,0–110,7 | 9,1 ms / 8,5–9,7 | ≈10,5 ms |

Modelsiz video okuma (yakalama) hızı 2521–3121 FPS'tir (dosya çözme; kamera hızı değildir). Bu koşulda MPS, CPU'ya göre çıkarımda yaklaşık 2 kat hızlıdır. CPU'da tek karelik sıçramalar (ör. 60 ms) görüldü; ilk CPU geçişi biraz yavaştır (46,6 FPS).
**Sınırlar:** güç/ısıl durum ve arka plan süreçleri kontrol edilmedi; tek makine, tek model, tek sentetik video; pencere/çizim maliyeti bu tabloda yok. Pencereli, MPS, kayıtlı (sentetik) video koşusu ayrıca bir kez gözlendi: çıkarım ortalaması 12,9 ms,
oynatma kaynak hızına (30 FPS) uydurulduğu için FPS 29,6 — bu tek gözlemdir, karşılaştırma değildir. **Canlı kamera ölçümleri kullanıcı tarafından yapılmıştır ve aşağıda (§9.1) ayrı raporlanır; bu tablodaki sentetik video sonuçlarıyla karıştırılmamalıdır.** 30 FPS hedefi zorunlu tutulmamıştır.

### 9.1 Canlı MacBook kamerası ölçümleri (kullanıcı tarafından, 2026-11-12) *[Kullanıcı]*

Bu bölümdeki sayılar, kullanıcının kendi MacBook Pro M4'ünde gerçek terminal çıktılarından aktardığı değerlerdir. Claude'un ortamında kamera izni olmadığından bunlar **yeniden üretilmedi ve bağımsız doğrulanmadı**; Claude'un kendi çalıştırdığı otomatik testlerden ve §9'daki sentetik video ölçümlerinden ayrıdır.
Ortak koşullar: `camera:0`, 1280×720, `yolo11n.pt`, `conf 0.25` (varsayılan), görüntü kaydedilmedi. Her satır **tek bir oturumdur**.

**Penceresiz** (`--no-display --max-frames 300`; yalnızca okuma + çıkarım döngüsü):

| Cihaz | Kare | Süre | Uçtan uca FPS | Çıkarım ort / medyan / p95 | Kare okuma ort / medyan | Kare→sonuç ort / medyan | Kişi/kare |
|---|---|---|---|---|---|---|---|
| MPS | 300 | 10,0 sn | **30,0** | 13,8 / 13,4 / 16,8 ms | 19,6 / 19,8 ms | 13,9 / 13,4 ms | 1,00 (en çok 1; 300 karenin tamamında ≥1 kişi) |
| CPU | 300 | 10,0 sn | **30,0** | 18,6 / 18,5 / 19,1 ms | 14,8 / 14,9 ms | 18,6 / 18,5 ms | 1,00 (en çok 1) |

**Pencereli** (pencere açık; çizim + `imshow` + `waitKey` dâhil; Q/Esc ile çıkış; hepsi MPS):

| Oturum | Kare | Süre | Uçtan uca FPS | Çıkarım ort / medyan / p95 | Kare→görüntü medyan | Kişi/kare ort (en çok) | ≥1 kişi olan kare |
|---|---|---|---|---|---|---|---|
| Test 1: üç kişi kameranın önüne geçti | 800 | 34,3 sn | 23,3 | 15,5 / 14,8 / 21,0 ms | 30,8 ms | 2,32 (**7**, bkz. OBS-4A-001) | bildirilmedi |
| Test 2: boş kadraj → kullanıcı girdi → köşede başka kişi | 636 | 28,6 sn | 22,2 | 14,7 / 13,5 / 20,8 ms | 28,7 ms | 1,31 (3) | 549/636 (**87 karede tespit yok**) |
| Test 5: MacBook ≈2 m yükseklikte | 331 | 17,9 sn | 18,4 | 16,8 / 17,1 / 21,5 ms | 33,6 ms | 1,17 (3) | 327/331 |

**Yorum (kaynağına göre).**
* *[Kullanıcı]* İki cihazda da penceresiz canlı hız 30,0 FPS'tir. *[Türetilmiş]* Medyan çıkarım MPS 13,4 ms, CPU 18,5 ms → MPS çıkarımda ≈1,38 kat hızlıdır (18,5/13,4). Uçtan uca FPS aynıdır, çünkü her iki cihaz da kamera akışının hızına yetişir.
* *[Türetilmiş]* Kare okuma + çıkarım süresi her iki cihazda ≈33,4 ms'dir (MPS: 19,6 + 13,8; CPU: 14,8 + 18,6) — yaklaşık 30 FPS kare aralığı. Döngü kamera hızındadır; okuma süresinin büyük kısmı kameranın bir sonraki kareyi beklemesidir.
* MPS'nin ileride eklenecek modüllerle (ör. takip, yüz) avantaj sağlayabileceği öngörülebilir; ancak bu **henüz ölçülmemiştir** ve bir hız iyileştirmesi garanti edilmez.
* Canlı MPS çıkarım medyanı (13,4 ms) sentetik video benchmark'ındakinden (9,1 ms) yüksektir; CPU'da fark yoktur (18,5 / 18,8 ms). MPS farkının nedeni ayrıştırılmadı (sahne içeriği, eşzamanlı kamera okuma, güç/ısıl durum kontrol edilmedi). CPU/MPS oranı sentetikte ≈2,07, canlıda ≈1,38'dir. **Sentetik sonuçlar canlı performansın tahmini olarak kullanılmamalıdır.**
* Pencereli FPS (18,4–23,3) penceresiz 30,0'dan düşüktür; kare→görüntü medyanı 28,7–33,6 ms'dir (kare→sonuç: 13,4 / 18,5 ms, penceresiz). Pencere maliyeti (çizim, `imshow`, `waitKey`) ayrı ölçülmedi; sahne, kişi sayısı ve oturumlar kontrollü değildir → **neden ayrıştırılmadı** (OBS-4A-002). Test 5'teki 18,4 FPS kamera yüksekliğine **atfedilmez**.
* Sınırlar: tek oturum, ısıl/güç kontrolü yok, benzersiz kare sayımı ve bağımsız zaman referansı yok. Zaman ölçümü `FramePacket.timestamp` ile okuma bitişinden başlar; sensör/sürücü tamponlaması kapsam dışıdır.

## 10. Değerlendirme: %85 hedefi ve metrikler

%85'in **hangi metrik için** beklendiği henüz belirlenmemiştir. Aşağıdaki ölçütler birbirinden ayrıdır ve karıştırılmamalıdır: Precision, Recall, F1, mAP@0.5, mAP@0.5:0.95 (doğruluk); FPS ve çıkarım gecikmesi (hız).
**Etiketli, bağımsız bir insan tespit test kümesi olmadığı için bu aşamada hiçbir Precision/Recall/F1/mAP değeri ölçülmemiş ve raporlanmamıştır.** Güven skorları doğruluk gibi raporlanmaz.

**Altyapı hazır:** `human_detection/evaluation.py` ve `scripts/evaluate_person_detection.py` etiketli küme verildiğinde bu metrikleri hesaplar (tanımlar modül açıklamasındadır). Matematik, elle hesaplanmış sonuçları olan sentetik örneklerle test edilmiştir
(ör. 2 etiketin 1'ini bulan kusursuz-precision tespitin AP'si = 51/101); bu testler hesaplamanın doğruluğunu kanıtlar, modelin doğruluğunu **kanıtlamaz**.

### Değerlendirme verisinin özellikleri (ileride hazırlanacak bağımsız test kümesi)

* **Bağımsızlık:** eğitim verisi (COCO) ve ayar yapılan görüntülerden ayrı; eşik/`imgsz` seçimi için ayrı bir doğrulama alt kümesi, nihai ölçüm için dokunulmamış (dondurulmuş) bir test alt kümesi.
* **Kamera açısı:** hedef kurulum duvara 2–3 m yükseklikte monte tek kameradır → göz hizası, hafif yüksek ve belirgin yüksek bakış açıları ayrı ayrı etiketlenmeli ve ayrı raporlanmalı.
* **Sahne:** tek kişili, çok kişili (yoğunluk artan), **insan olmayan** (yanlış pozitif ölçümü için) sahneler.
* **Zorluklar:** kısmen örtülü/kesik kişiler, farklı mesafeler (küçük/uzak kişiler), farklı aydınlatma (düşük ışık, kontrast), hareket bulanıklığı, kalabalık arka plan.
* **Etiket:** yalnızca `person`, her görünür kişi için kutu (YOLO `sınıf cx cy w h`); etiket kuralı (örtülü/çok küçük kişilerin sayılma ölçütü) baştan yazılı olmalı.
* **Hukuk/gizlilik:** kullanım izni olan görüntüler; kişisel veri koruma gereksinimleri (özellikle yüz) ayrıca ele alınmalı. Hazır kümeler lisans/hak doğrulaması olmadan kullanılmaz.
* **Raporlama:** görüntü ve etiket sayısı her sonuçla birlikte; küçük örneklem sonuçları genellenemez; mAP, Precision/Recall ve hız ayrı raporlanır; güven eşiği açıkça belirtilir.

## 11. Testler

| Dosya | Kapsam | Gerçek model? |
|---|---|---|
| `test_person_detection_config.py` | yapılandırma doğrulama, cihaz seçimi (MPS yok → hata), import yan etkisi yok, resmî ağırlık indirme/SHA-256 (ağ sahte) | Hayır |
| `test_person_detection_results.py` | veri yapıları, kutu kırpma/geçersiz kutu, `person` süzmesi, güven korunması | Hayır |
| `test_person_detector.py` | tek yükleme, tek/çok/sıfır insan, `person` dışı süzme, geçersiz kare, cihaz, süre ölçümü, hata sarmalama | **Mock YOLO** |
| `test_person_detection_visualization.py` | kutu/etiket/şerit çizimi, girdinin değişmemesi, kenar durumları | Hayır |
| `test_person_detection_pipeline.py` | döngü: FPS ve gecikme hesabı, Q/Esc/pencere/Ctrl+C, kaynak hataları; `ISG_GUI_TESTS=1` ile gerçek pencere | Hayır (sahte tespitçi) |
| `test_person_detection_cli.py` | argümanlar, çıkış kodları, penceresiz çalışma, kaynak temizliği, betik alt süreçleri | Hayır |
| `test_person_detection_benchmark.py` | karşılaştırma altyapısı (sahte tespitçi) | Hayır |
| `test_person_detection_evaluation.py` | Precision/Recall/F1/AP/mAP matematiği (sentetik, elle hesaplı) | Boru hattı testi hariç hayır |
| `test_person_detection_real_model.py` | **GERÇEK `yolo11n.pt` çıkarımı**: insan yok / tek insan / çok insan, `person` süzmesi, eşik etkisi, CPU↔MPS uyumu, ağ çağrısı ve dosya yazımı yok | **Evet** (ağırlık yoksa atlanır; indirme/internet gerekmez) |

**Mock testleri gerçek çıkarımın çalıştığını kanıtlamaz; gerçek çıkarım yalnızca `test_person_detection_real_model.py` ile doğrulanır.** Gerçek testler, `ultralytics` paketinin içindeki örnek görüntüleri kullanır
(`bus.jpg`: 4 yaya + otobüs, `zidane.jpg`: 2 kişi; dışarıdan görüntü indirilmez). Bu, küçük ve kontrollü bir örnektir; doğruluk ölçümü değildir.

```bash
.venv/bin/python -m unittest discover -s tests                           # tüm testler (internet/kamera gerekmez)
ISG_GUI_TESTS=1 .venv/bin/python -m unittest discover -s tests           # gerçek pencere testleri dâhil (grafik oturumu olan terminal)
.venv/bin/python -m unittest tests.test_person_detection_real_model -v   # yalnızca gerçek model testleri
```

## 12. Gizlilik ve güvenlik

Varsayılan olarak görüntü/video **kaydedilmez**; yüz arşivlenmez; kimlik veya biyometrik veri toplanmaz; kamera yalnızca siz komutu çalıştırınca açılır; model yüklenemezse kamera hiç açılmaz; ağ yalnızca
`download_person_model.py` çalıştırıldığında kullanılır. Benchmark/değerlendirme çıktıları yalnızca sayısal ölçümlerdir (`results/`, Git dışı).

## 13. Bilinen sınırlamalar

* Doğruluk ölçülmedi (etiketli bağımsız test kümesi yok); güven skoru doğruluk değildir.
* Sentetik sahnede 0,31–0,37 güvenli bazı `person` kutuları bulanık arka plan kalabalığında ve görüntü kenarında kırpılmış kişide görüldü; etiket olmadığı için bunların doğru/yanlış olduğu söylenemez. Düşük eşiklerde benzer düşük güvenli kutular beklenmelidir.
* Model COCO ile eğitilmiştir; yüksek açılı kamera için yalnızca MacBook ile ≈2 m'lik bir **ön test** yapılmıştır (§14); gerçek duvar montajlı kamera ve çalışma alanı kıyafetleri için doğrulanmamıştır.
* **FP-4A-001:** ≈2 m yükseklikten yapılan ön testte bir yorgan kısa süreli insan olarak algılandı (kullanıcı gözlemi; §14). Tek gözlemdir; genel model başarısızlığı sonucu çıkarılmaz ve ByteTrack'in bunu çözeceği varsayılmaz.
* **OBS-4A-001:** ilk canlı oturumda en yüksek tespit sayısı 7 raporlandı; bu tespitlerin hangi görüntülerde oluştuğu bilinmiyor, doğru/yanlış sınıflandırılmadı (§14).
* Performans: canlı kamera ölçümleri tek oturumlardır (§9.1); pencereli FPS penceresiz FPS'ten düşüktür ve nedeni ayrıştırılmadı (OBS-4A-002); ısıl/güç kontrol edilmedi. Sentetik video sonuçları (§9) canlı performansın tahmini değildir.
* Aşama 3 backlog durumu: B3-1 (yeniden bağlanma), B3-2 (kare kaybı) ve B3-5 (fiziksel bağlantı kopması) kapsanmadı. B3-3 (gecikme) için kare→sonuç/kare→görüntü gecikmesi kayıtlı videoda ve (kullanıcı) canlı kamerada ölçüldü; sensör/sürücü tamponlaması kapsam dışı, nihai ölçüm Aşama 12'de.
  B3-4 için modelli 300 karelik penceresiz canlı koşular 30,0 FPS verdi (kullanıcı); uzun süreli, bağımsız zaman referanslı doğrulama yapılmadı. **Hiçbiri tamamlanmış sayılmaz** (Aşama 3 backlog kayıtları değişmedi).
* AGPL-3.0 lisansı (bkz. §3).

## 14. Manuel doğrulamalar, gözlemler ve açık görevler

### 14.1 Kullanıcı tarafından yapılan manuel doğrulamalar (2026-11-12) *[Kullanıcı]*

Kullanıcının kendi MacBook Pro M4'ünde, gerçek kamerayla yaptığı testlerdir; sayılar §9.1'dedir. Claude bunları yeniden üretemedi (kamera izni yok); Claude'un kendi otomatik testlerinden ayrıdır.

| Test | Konu | Sonuç |
|---|---|---|
| 1 | Canlı kamera, MPS, **üç kişi** | ✔ Üç kişi kameranın önüne geçtiğinde ekranda `PERSONS: 3` görüldü; üç kişinin aynı anda algılandığı gözle doğrulandı. 800 kare, Q/Esc ile çıkış. Oturum istatistiğindeki en yüksek sayı 7 → **OBS-4A-001** |
| 2 | Boş ortam ve kadraja giriş | ✔ Kamera başta boştu; kullanıcı girdi, sonra köşede başka bir kişi belirdi; ekran sonuçları gözlenen sahneyle uyumluydu. 636 karenin 87'sinde hiç insan tespit edilmedi. **Bu, modelin 87 karede sıfır tespit ürettiğini gösterir; her karenin bağımsız etiketlendiği bir doğruluk testi değildir ve precision/recall olarak yorumlanmaz** |
| 3 | Penceresiz canlı performans, MPS | ✔ 300 kare, 30,0 FPS (§9.1) |
| 4 | Penceresiz canlı performans, CPU | ✔ 300 kare, 30,0 FPS (§9.1) |
| 5 | MacBook ≈2 m yükseklikte (yüksek açı **ön testi**) | ✔ Sistem kullanıcıyı bu konumdan tespit etti; bir yorgan kısa süreli insan sanıldı → **FP-4A-001**. 331 kare, 18,4 FPS (kamera yüksekliğine atfedilmez). Bu bir **IP kamera doğrulaması değildir** |
| 6 | Çıkış yöntemleri (gerçek kamera) | ✔ Q, Esc, macOS kırmızı pencere kapatma düğmesi ve terminalde Ctrl+C ile oturumlar sorunsuz kapandı (kullanıcı gözlemi). **Ctrl+C'nin (ve diğer çıkışların) çıkış kodu paylaşılmadı → gerçek kamerada ölçülmüş sayılmaz**; 130 değeri yalnızca birim testleriyle doğrulanmıştır |

### 14.2 Bilinen sınırlama ve açık gözlemler

**FP-4A-001 — Blanket/Fabric Misclassified as Person** (bilinen sınırlama, izlenecek yanlış pozitif)

| Alan | Değer |
|---|---|
| Nesne | Yorgan / kumaş |
| Koşul | Yaklaşık 2 metre yüksekliğe yerleştirilmiş MacBook kamerası (Test 5), görüntünün köşesindeki yorgan |
| Model / eşik | YOLO11n (`yolo11n.pt`), `conf 0.25` |
| Sonuç | Kısa süreli yanlış insan tespiti |
| Güven skorunun kesin değeri | Bilinmiyor (eşik gereği ≥ 0,25 olmalıdır — *[Türetilmiş]*) |
| Hatanın oluştuğu kare sayısı | Bilinmiyor |
| Video kanıtı | Kaydedilmedi (varsayılan olarak görüntü kaydedilmez) |
| Önem | İzlenmesi ve sonraki aşamalarda değerlendirilmesi gereken yanlış pozitif |
| Yorum sınırı | Tek gözlemdir; genel model başarısızlığı sonucu çıkarılmaz. ByteTrack'in yanlış pozitifleri otomatik çözeceği iddia edilmez (etkisi 4B'de ölçülmeden bilinmez). Kamera açısının FPS düşüşüne neden olduğu da söylenmez (kontrollü deney yok) |
| İlgili görev | OT-4A-001 (`conf 0.35` karşılaştırması), OT-4A-002 (bağımsız değerlendirme) |

**OBS-4A-001 — Investigation / Unverified Peak Detection Count.** Test 1'de (800 kare, kare başına ortalama 2,32 kişi) en yüksek tespit sayısı **7** olarak raporlandı; aynı oturumda üç kişi kadraja girmişti. Bu 7 tespitin hangi karelerde ve hangi görüntü içeriğiyle oluştuğu **bilinmiyor** (kayıt yok; CLI özeti yalnızca en yüksek sayıyı yazar, hangi karede olduğunu yazmaz).
Bu sayı **kesin yanlış pozitif ya da kesin doğru tespit olarak sınıflandırılmaz.** Durum: açık araştırma (OT-4A-004). Test 2 ve 5'te en yüksek sayı 3'tür.

**OBS-4A-002 — Pencereli/penceresiz FPS farkı ve canlı/sentetik çıkarım farkı.** Pencereli canlı FPS 18,4–23,3, penceresiz 30,0'dır; canlı MPS çıkarımı sentetik video benchmark'ından yavaştır (13,4 vs 9,1 ms). Nedenler ayrıştırılmadı; neden iddiası yoktur (§9.1). Gerekirse Aşama 12'deki canlı sistem testlerinde ölçülür.

### 14.3 Açık görevler ve ertelenenler

Makine tarafından okunur sürüm: `configs/project_stages.yaml` (`stage4a_open_items`). Hiçbiri tamamlanmış sayılmaz.

| ID | Görev | Durum | Not |
|---|---|---|---|
| **OT-4A-001** | `PENDING_MANUAL_TEST — CONFIDENCE_THRESHOLD_0_35` | **`PENDING_USER_VERIFICATION`** | **Henüz yapılmadı; kullanıcı daha sonra çalıştıracak.** Aşağıda |
| OT-4A-002 | Bağımsız, etiketli değerlendirme kümesi ve resmî ölçümler (Precision, Recall, F1, mAP@0.5, mAP@0.5:0.95) | Açık | Altyapı hazır (`evaluate_person_detection.py`); veri kümesi yok. Zamanlama kullanıcı kararı (öneri: 4B takip ölçümleriyle birlikte veya Aşama 12 nihai değerlendirme) |
| OT-4A-003 | Gerçek IP kamera doğrulaması: kamera satın alma ve montaj, RTSP/bağlantı doğrulaması (RTSP kaynak desteği geliştirilmedi), nihai açı, gecikme, görüntü kalitesi, nihai test alanı | **`DEFERRED_TO_SYSTEM_INTEGRATION`** | IP kamera henüz satın alınmadı; marka/model, RTSP, test alanı belirlenmedi. Hedef: sonraki entegrasyon ve sistem değerlendirme aşamaları (öneri: Aşama 11–12). **MacBook testleri IP kamera doğrulaması olarak sunulmaz** |
| OT-4A-004 | OBS-4A-001 araştırması (en yüksek 7 tespit) | Açık araştırma | Aynı koşulları yeniden üretip sayacı izleyerek veya tepe karenin kaydıyla (yalnızca kullanıcının açık onayıyla) yapılabilir; kod değişikliği 4A kapsamında yapılmadı |

#### OT-4A-001 — `PENDING_MANUAL_TEST — CONFIDENCE_THRESHOLD_0_35` (`PENDING_USER_VERIFICATION`)

**Bu test henüz gerçekleştirilmemiştir; hiçbir sonucu yoktur ve başarılıymış gibi okunmamalıdır.** Yorganın insan sanılması (FP-4A-001) nedeniyle güven eşiğini 0,25'ten 0,35'e çıkararak ek bir karşılaştırma yapılacaktır. Aşama 4A kapanışından **sonra da** uygulanabilecek bir iyileştirme/doğrulama görevidir.
**Varsayılan `conf` 0,25 olarak kalır** (kod ve yapılandırma değiştirilmedi); 0,35'in daha iyi olduğu ölçülmüş sonuç bulunmadan iddia edilmez.

Çalıştırılacak komut (kullanıcı):

```bash
.venv/bin/python scripts/run_person_detection.py --source camera --camera-index 0 --device mps --conf 0.35
```

Amaç — aynı yüksek kamera konumunda:
1. Kullanıcı 0,35 güven eşiğinde hâlâ doğru tespit ediliyor mu?
2. Yorgan hâlâ insan olarak algılanıyor mu?
3. Köşeden veya kısmen görünen gerçek insanlar algılanmaya devam ediyor mu?
4. 0,25 ile 0,35 arasındaki fark nedir?
5. Gerçek insanları kaçırma durumu oluşuyor mu?

Prosedür: kamerayı mümkün olduğunca aynı konumda tutun; aynı yorganı ve benzer aydınlatmayı kullanın; önce `conf 0.25` (mevcut) davranışını referans alın, sonra `conf 0.35` ile gözlem yapın; her iki durumda kutuların gerçek insanları kapsayıp kapsamadığını karşılaştırın; yanlış pozitif veya kaçırılan insan gözlemlerini kaydedin.

Kayıt şablonu (kullanıcı dolduracak; boş = yapılmadı):

| Gözlem | `conf 0.25` (referans) | `conf 0.35` |
|---|---|---|
| Kullanıcı tespit edildi mi? | | |
| Yorgan insan sanıldı mı? (süre/sıklık) | | |
| Köşede/kısmen görünen gerçek insan tespit edildi mi? | | |
| Kaçırılan gerçek insan var mı? | | |
| Diğer yanlış pozitifler | | |
| Çıkış özeti (FPS, çıkarım ms, en yüksek kişi sayısı) | | |

## 15. Aşama 4B için hazır bileşenler

* `DetectionResult` / `PersonDetection`: piksel `xyxy`, güven, `frame_index`, `timestamp` — takipçinin ihtiyaç duyduğu girdi (kare başına liste).
* `run_detection(..., on_result=...)` kancası ve `FramePacket.timestamp` (monotonic zaman) takipçiyi döngüye bağlamak için.
* Ölçüm altyapısı (`LatencyStats`, benchmark) takip ekleyince gecikme artışını ölçmek için yeniden kullanılabilir.
* **Notlar:** ByteTrack/BoT-SORT için Ultralytics izleyicileri ek bağımlılık (ör. `lap`) isteyebilir; ortamda `lap` ve `scipy` kurulu **değildir** — kurulum kararı 4B'de verilmelidir. Ultralytics bağımlılıkları kendiliğinden kurmaya (ağ) çalışabilir; 4B'de bu da `YOLO_OFFLINE` ile birlikte ele alınmalıdır.
  Takip, kimlik/kalıcı numara üretecektir; bu modül bunları üretmez.
