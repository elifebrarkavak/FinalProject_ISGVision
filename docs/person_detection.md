# Çoklu İnsan Tespiti Modülü (Aşama 4A)

> **Durum: Aşama 4A `READY_FOR_MANUAL_VERIFICATION`.** Kod, otomatik testler ve gerçek model doğrulamaları tamamdır; MacBook kamerasıyla canlı test ve yüksek açılı gerçek kamera
> testi **kullanıcı doğrulaması bekliyor** (bkz. §14). Aşama 4 genel durumu `COMPLETED` **değildir**: takip (Aşama 4B) başlatılmamıştır.
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
Pencere `camera.display.PreviewWindow`'dur: macOS kırmızı kapatma düğmesi düzeltmesi, Q/Esc ve `Ctrl+C` davranışı korunmuştur (gerçek pencere testiyle doğrulandı, §11).

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
oynatma kaynak hızına (30 FPS) uydurulduğu için FPS 29,6 — bu tek gözlemdir, karşılaştırma değildir. **Canlı kamerada CPU/MPS ölçümü yapılmadı** (kamera izni; §14). 30 FPS hedefi zorunlu tutulmamıştır.

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
* Model COCO ile eğitilmiştir; yüksek açılı (2–3 m, aşağı bakan) kamera ve çalışma alanı kıyafetleri için ayrıca doğrulanmamıştır.
* Performans tek makine/oturum/sentetik video ölçümüdür; canlı kamerada ölçülmedi.
* Aşama 3 backlog durumu: B3-1 (yeniden bağlanma) ve B3-5 (fiziksel bağlantı kopması) kapsanmadı; B3-2 (kare kaybı) ölçülmedi; B3-3 (gecikme) için yalnızca **kayıtlı videoda** kare→sonuç gecikmesi ölçüldü, canlı kamerada ölçülmedi (kısmi);
  B3-4 (canlı kamerada ileri FPS doğrulaması) yapılmadı. Hiçbiri tamamlanmış sayılmaz.
* AGPL-3.0 lisansı (bkz. §3).

## 14. Eksik kalan manuel doğrulamalar (`READY_FOR_MANUAL_VERIFICATION`)

1. **MacBook kamerasıyla canlı test** — Claude'un ortamında kamera izni yoktur (`not authorized to capture video`); bu test yapılmış sayılmaz. Komutlar:
   ```bash
   .venv/bin/python scripts/run_person_detection.py --source camera --camera-index 0 --device mps
   .venv/bin/python scripts/run_person_detection.py --source camera --camera-index 0 --device cpu
   .venv/bin/python scripts/run_person_detection.py --source camera --camera-index 0 --no-display --max-frames 300 --device mps   # canlı hız özeti
   ```
   Beklenen: tek ve birden fazla kişide kutu + güven, kişi sayısı, `Q`/`Esc`/kırmızı düğme/`Ctrl+C` ile çıkış (`exit=0`/`130`), kamera ışığının sönmesi. Çıkışta yazılan özetteki FPS/gecikme değerlerini not edin.
2. **Gerçek çok kişili (≥3) izinli video veya canlı sahne** — doğrulama şimdilik yalnızca Ultralytics örnek görüntüleriyle (4 ve 2 kişi) ve sentetik montajla yapıldı.
3. **Kamera açısı değerlendirmesi:** göz hizası dışındaki açılar için uygun görüntü yoktur → `PENDING_REAL_CAMERA_VALIDATION`. Duvar montajlı gerçek kamerayla (2–3 m) yapılacak testte `--conf`/`--imgsz` ayarı gerekebilir.
4. Etiketli bağımsız test kümesi hazırlanınca `scripts/evaluate_person_detection.py` ile Precision/Recall/F1/mAP ölçümü.

## 15. Aşama 4B için hazır bileşenler

* `DetectionResult` / `PersonDetection`: piksel `xyxy`, güven, `frame_index`, `timestamp` — takipçinin ihtiyaç duyduğu girdi (kare başına liste).
* `run_detection(..., on_result=...)` kancası ve `FramePacket.timestamp` (monotonic zaman) takipçiyi döngüye bağlamak için.
* Ölçüm altyapısı (`LatencyStats`, benchmark) takip ekleyince gecikme artışını ölçmek için yeniden kullanılabilir.
* **Notlar:** ByteTrack/BoT-SORT için Ultralytics izleyicileri ek bağımlılık (ör. `lap`) isteyebilir; ortamda `lap` ve `scipy` kurulu **değildir** — kurulum kararı 4B'de verilmelidir. Ultralytics bağımlılıkları kendiliğinden kurmaya (ağ) çalışabilir; 4B'de bu da `YOLO_OFFLINE` ile birlikte ele alınmalıdır.
  Takip, kimlik/kalıcı numara üretecektir; bu modül bunları üretmez.
