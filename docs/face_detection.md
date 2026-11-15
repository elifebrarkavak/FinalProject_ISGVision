# Çoklu Yüz Tespiti Modülü (Aşama 5A)

> **Durum: Aşama 5A `READY_FOR_MANUAL_VERIFICATION`** (2026-11-13) — **`COMPLETED` DEĞİLDİR.** Kod, otomatik testler (mock + gerçek model), kayıtlı video doğrulamaları ve performans ölçümleri tamamdır; **gerçek MacBook kamerasında YALNIZCA tek kişilik yüz tespiti kullanıcı tarafından nitel olarak doğrulandı** (kamera açıldı, canlı pencere çalıştı, gerçek yüz tespit edildi, yüz kutusu gösterildi; sayısal ölçüm yok — §13.1); çoklu yüz, mesafe, açı, kısmi görünüm, yanlış pozitif, ışık, 5A betiğiyle çıkış yolları, canlı FPS ve yüksek açı **yapılmadı** (§13 kullanıcı test planı). Aşama 5 genel durumu `IN_PROGRESS`; 5B ve 5C başlatılmadı. 5A/5B/5C ayrımı **öneridir** (kullanıcı kararı bekler).
> **Seçilen model:** YuNet (OpenCV `cv2.FaceDetectorYN`, MIT lisanslı ONNX, yalnızca CPU). Yeni paket **eklenmedi** (`requirements.txt`/`requirements-lock.txt` değişmedi).
> **Bu aşamada yoktur:** yüz tanıma, çalışan adı/kimliği, embedding/biyometrik şablon, yüz–Track ID eşleştirmesi (5B), kalıcı yüz veritabanı, görüntü kaydı. **Yüz tespiti bir kimlik veya canlı insan doğrulaması değildir.**
> Aşama 4B'nin açık sorunları (poster yanlış pozitifi FP-4B-001 dâhil) **çözülmüş değildir** ve bu aşamada değişmemiştir. Teknik sonuç raporu: [../reports/face_detection/STAGE_5A_REPORT.md](../reports/face_detection/STAGE_5A_REPORT.md).

**Kanıt türleri bu belgede ayrı tutulur:** *[Ölçüldü]* = Claude'un bu makinede çalıştırıp sonucunu gördüğü; *[Mock]* = sahte bağımlılıkla birim testi; *[Gerçek model]* = gerçek YuNet, örnek/türev görüntüler, kamerasız; *[Doküman]* = yayıncı/üretici belgesine dayanan, bu ortamda **doğrulanmamış** bilgi; *[Bekliyor]* = yapılmamış, kullanıcı görevi. Yapılmayan hiçbir test başarılı gösterilmemiştir.

## 1. Amaç ve kapsam

Tek kamera/video karesindeki **birden fazla insan yüzünü** tespit etmek; her yüz için piksel koordinatı (`bbox_xyxy`), güven skoru, kare numarası, zaman damgası ve görüntü boyutu üretmek; yüz kutularını canlı görüntüde göstermek; MacBook M4'te CPU'da çalışmak; Aşama 5B'nin yüz–Track ID ilişkilendirmesine **uygun** (ama onu yapmayan) çıktılar sunmak.

Kamera/video okuma ve pencere (`camera.capture.FrameSource`, `camera.display.PreviewWindow`) Aşama 3/4'te yazılmış kodla aynıdır; yeniden yazılmadı. Aşama 4A (`human_detection`) ve 4B (`tracking`) kaynakları, YOLO11n ağırlığı (SHA-256 `0ebbc80d…ee1`) ve ByteTrack eşikleri **değişmedi**.

## 2. Model seçimi ve gerekçe

### 2.1 Adaylar ve bu ortamda doğrulananlar

Ortam: MacBook Pro **Apple M4**, 10 çekirdek, 16 GiB, macOS 26.5.2, Python 3.11.17, OpenCV 5.0.0 (`opencv-python` + `opencv-contrib-python` 5.0.0.93), NumPy 2.4.6. Değerlendirme betikleri depoya eklenmemiştir (geçici; yöntem aşağıda tarif edilir); ham çıktılar `results/face_detection/` altındadır (Git dışı).

| Ölçüt | **YuNet** (OpenCV `FaceDetectorYN`) — **SEÇİLDİ** | MediaPipe BlazeFace short-range (Tasks API) | SCRFD (InsightFace) |
|---|---|---|---|
| Çoklu yüz | Evet; tüm karede çalışır, 16 px'ten 560 px yüze kadar bulundu *[Ölçüldü]* | Evet; **ama** girdi 128×128'e küçültülür: 1280×720 karede **≤ 128 px yüz bulunamadı**, yalnız ≥ 192 px *[Ölçüldü]* | Evet *[Doküman]*; bu ortamda **çalıştırılmadı** |
| Örnek görüntüler | zidane: 2 net yüz (0,94/0,90) + 1 bulanık arka plan yüzü (0,61); bus: 2/2 net yüz *[Ölçüldü]* | zidane: yalnızca bulanık arka plan yüzü (0,57), **iki net yüzü kaçırdı**; bus: **0/2** *[Ölçüldü]* | — |
| Apple Silicon / Python 3.11 / NumPy 2.4.6 | Çalışıyor (OpenCV 5.0 yerleşik) *[Ölçüldü]* | Çalışıyor (`mediapipe 1.2.2` zaten kurulu) *[Ölçüldü]* | `onnxruntime` gerekir; `pip install --dry-run` için `onnxruntime-1.32.0 + protobuf-7.36.2` uygun çıktı *[Ölçüldü: yalnızca çözümleme, kurulmadı]* |
| Hızlandırma (CPU/MPS/diğer) | **Yalnızca CPU.** OpenCL hedefi **etkisiz** (11,0–11,2 ms, hedef değişse de aynı); klasik DNN motoru bu modelle hata veriyor; yeni motor hedefi yok sayıyor *[Ölçüldü]* | CPU (XNNPACK). GPU temsilcisi 3 kanallı RGB girdiyle **hata verdi** (Metal: "yalnız 4 kanallı doku"); RGBA denenmedi *[Ölçüldü]* | CPU (onnxruntime) *[Doküman]*; CoreML/MPS denenmedi |
| Çıkarım süresi, 1280×720, CPU | **≈ 11,0–11,3 ms** (çok iş parçacıklı); `ultralytics` içe aktarılmışsa ≈ 30 ms (§9.4) *[Ölçüldü]* | ≈ 0,75 ms (ama büyük yüz sınırı) *[Ölçüldü]* | — |
| Model boyutu | 229.738 bayt | 229.746 bayt | Bu ortamda **indirilmedi; boyutu doğrulanmadı** |
| Çevrimdışı | Evet | Evet | Evet |
| Lisans (model) | **MIT** (OpenCV Zoo `LICENSE`, kaynak `libfacedetection.train` de MIT) *[Doğrulandı: dosyalar okundu]* | Paket **Apache-2.0** *[Ölçüldü: `pip show`]*; model kartı lisansı **doğrulanmadı** | Kod MIT, **hazır önceden eğitilmiş modeller "yalnızca ticari olmayan araştırma"** (InsightFace README, okundu) |
| Kurulum/bakım | Ek paket yok; tek küçük dosya | Paket kurulu; model dosyası **pakette yok** (Tasks API ayrı `.tflite` ister) | Yeni çıkarım çatısı (`onnxruntime`), ayrıştırma/anchor kodu yazılmalı |
| Ek bağımlılık | **Yok** | Yok (zaten pinli) | `onnxruntime`, `protobuf` |

**Karar: YuNet.** Gerekçe: (1) kamera karesinin tamamında, **farklı mesafelerdeki** (küçük/uzak ve büyük/yakın) yüzleri bulan tek aday (BlazeFace short-range 128×128 girdiyle uzak yüzleri kaçırır; çalışan takibi senaryosunda yüzler uzaktadır); (2) yeni paket, ikinci derin öğrenme çatısı veya büyük model gerektirmez; (3) MIT lisansı, küçük (230 KB) ve çevrimdışı; (4) yayıncının SHA-256 kaydıyla doğrulanabilir; (5) CPU'da 1280×720'de ≈ 11 ms. **Seçilmeyenler:** BlazeFace short-range (menzil sınırı ölçüldü; "full-range" modeli Tasks API'de bu ortamda **denenmedi**); SCRFD (hazır ağırlıkları ticari olmayan araştırma lisanslı, ek çalışma zamanı ister; **çalıştırılmadı, değerlendirmesi yalnızca dokümana dayanır**). Ultralytics tabanlı üçüncü taraf "yüz" ağırlıkları, OpenCV Haar/ResNet-SSD ve diğer adaylar **değerlendirilmedi**.

**Dürüstlük notları:** (a) YuNet README'si WIDER FACE AP değerlerini iki farklı yerde verir (metinde 0,834/0,824/0,708; tabloda 0,8844/0,8656/0,7503) ve modelin "yaklaşık 10×10 ile 300×300 piksel" yüzleri algıladığını söyler *[Doküman]*; bunlar **doğrulanmadı**. Bizim ölçümümüzde (tek görüntü ailesi) 560 px'e kadar bulundu, ancak bu genelleştirilemez. (b) Seçim, **iki örnek fotoğraf ve türevleri** üzerindeki kısıtlı ölçüme ve belgelere dayanır; bağımsız etiketli bir değerlendirme yoktur.

### 2.2 Ölçümlerin ayrıntısı *[Ölçüldü; sentetik/örnek görüntü; yalnızca bu makine]*

**Yüz boyutu taraması** (zidane.jpg yüzleri 1280×720 tuvale ölçeklenip merkeze konuldu; referans kutu = YuNet'in yerel boyuttaki kutusu, yani **sözde-GT**):

| Yüz yüksekliği (px) | 16 | 24 | 32 | 48 | 64 | 96 | 128 | 192 | 256 | 320 | 400 | 480 | 560 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| YuNet (IoU) | 0,89 | 0,86 | 0,86 | 0,91 | 0,94 | 0,92 | 0,96 | 0,92 | 0,92 | 0,91 | 0,89 | 0,86 | 0,85 |
| BlazeFace short-range | kaçırdı | kaçırdı | kaçırdı | kaçırdı | kaçırdı | kaçırdı | kaçırdı | 0,53 | 0,69 | 0,69 | 0,68 | 0,68 | 0,70 |

**Yüz içermeyen bölgeler** (bina, kaldırım, gövde/kravat, düz gri/siyah, gürültü, gradyan; 9 görüntü): eşik 0,30'da yalnızca **1** aday çıktı (bina kırpıntısında 12×15 px, skor 0,32); 0,4 ve üzeri yanlış pozitif yok. **Eşik duyarlılığı** (zidane: 0,94/0,90/0,61/0,38; bus: 0,92/0,91/0,36): 0,5–0,6'da 3+2, 0,7–0,9'da 2+2 yüz. Bu yüzden varsayılan `conf = 0,6` (evrensel optimum değildir; bulanık arka plan yüzünü de kabul eder).
**Düzlem içi dönme** (zidane, eşik 0,6; toplam tespit sayısı, beklenen 2 net yüz): 0° → 3, 15° → 2, 30° → 3, 45° → 2, **60° → 1 ve 90° → 1 (yalnızca düşük güvenli 0,61/0,62 tek tespit; iki net yüz kaçtı)**, 180° → 3. **Kısmen görünen yüz** (görüntü kaydırılarak): sol kenar yüzün ≈ %46'sını kesince (63/136 px) bulundu (skor 0,89, kutu kenara kırpıldı); ≈ %90'ını kesince bulunamadı; üst kenar yüzün ≈ %29'unu kesince bulundu, ≈ %75'ini kesince bulunamadı.
**Yöntem:** betikler depoya eklenmedi; aynı sonuçlar testlerde (§8) kalıcı olarak sınanır (boyut/çözünürlük/kenar/boş kare testleri).

## 3. Mimari ve dosya yapısı

`src/isg_vision/face_detection/` (mevcut boş iskelet kullanıldı; ikinci paket açılmadı). Sorumluluklar ayrıdır:

| Dosya | Sorumluluk |
|---|---|
| `errors.py` | `FaceConfigError` (hatalı kullanıcı değeri), `FaceDetectionError` ailesi: `FaceModelLoadError`, `FaceInvalidFrameError`, `FaceInferenceError`, `FaceWeightsDownloadError` |
| `config.py` | `FaceDetectionConfig` (doğrulama; model yükleme/ağ yok), sabitler (`DEFAULT_CONF 0.6`, `DEFAULT_NMS 0.3`, `DEFAULT_MAX_FACES 100`) |
| `results.py` | `FaceDetection`, `FaceDetectionResult`, `build_faces` (kırpma, kenar işareti, sıralama, kesme); yalnızca standart kütüphane |
| `weights.py` | Resmî model izin listesi, sabit commit URL'si, boyut+SHA-256 doğrulaması, indirme (**ağ kullanan tek modül**) |
| `detector.py` | `FaceDetector`: yükleme (bir kez, bütünlük denetimi), `detect`, `warmup`, süre ölçümü; kamera/çizim/pencere bilmez |
| `visualization.py` | `draw_face_detections` (kopya üzerinde çizim; "kimlik/canlılık değildir" uyarısı) |
| `pipeline.py` | `run_face_detection` döngüsü, `FaceRunSummary` (`LatencyStats` ve çıkış nedenleri 4A'dan yeniden kullanılır) |
| `cli.py` | Komut satırı (`camera.cli.config_from_args` yeniden kullanılır; çıkış kodları 0/1/2/130) |
| `benchmark.py` | Montaj üretimi, ölçüm yardımcıları, `run_benchmark` |
| `scripts/run_face_detection.py`, `download_face_model.py`, `benchmark_face_detection.py` | İnce başlatıcılar |

Akış: `FramePacket` → `FaceDetector.detect` (YuNet, CPU) → `FaceDetectionResult` → `draw_face_detections` (kopya) → `PreviewWindow`. `run_face_detection(..., on_result=lambda paket, sonuç: ...)` kancası 5B içindir. Değişen mevcut dosyalar yalnızca belgeler, `configs/project_stages.yaml` ve `tests/test_no_side_effects.py` (yeni modül adları + tek onaylı model dosyası eklendi; hiçbir kontrol gevşetilmedi).

## 4. Veri yapıları (`results.py`)

Koordinat kuralı `human_detection`/`tracking` ile **aynıdır**: `bbox_xyxy = (x1, y1, x2, y2)`, görüntü piksel koordinatları, sol-üst başlangıç, `x1<x2`, `y1<y2`, **görüntü sınırlarına kırpılmış** `float`.

* **`FaceDetection`**: `confidence` (0..1; kalibre değil), `bbox_xyxy`, `frame_index`, `timestamp` (`FramePacket.timestamp`), `image_size` `(genişlik, yükseklik)`, `face_index` (aynı karedeki sıra; güvene göre azalan), `at_border` (kutu kadraj kenarına ≤ 2 px: yüz kısmen görünüyor olabilir; **sezgiseldir**).
* **`FaceDetectionResult`**: `faces`, `image_size`, `frame_index`, `timestamp`, `inference_ms`, `device` (`cpu`), `dropped_boxes` (geçersiz kutular), `truncated_faces` (`max_faces` aşımı; sessizce atılmaz), `inference_size` (modelin çalıştığı boyut; `detect_width` kullanıldıysa farklı), `face_count`, `is_empty`.
* **Bilinçli olarak YOKTUR:** `track_id`, çalışan adı/kimliği, embedding, biyometrik şablon, **yüz işaret noktaları** (YuNet 5 nokta üretir; 5A bunları sonuca koymaz — kimlik tanıma hizalaması gerekirse Aşama 7'de bilinçli bir kararla eklenir), canlılık/yaşayan insan bilgisi. Bu, testle de güvence altındadır (alan adı denetimi).
* Hatalı/yüz bulunmayan kare **hata değildir:** boş sonuç döner. Geçersiz görüntü (`None`, yanlış tür/kanal/boyut) `FaceInvalidFrameError` verir (model **yanlış girdiyi hata vermeden kabul ettiği için** doğrulama bizdedir, *[Ölçüldü]*).
* Kutular görüntüyü aşıyorsa **atılmaz**, kırpılır ve `at_border` ile işaretlenir; tamamen dışarıdaki/dejenere kutular atılır ve `dropped_boxes`'a sayılır.

## 5. CLI kullanımı

```bash
.venv/bin/python scripts/download_face_model.py                    # modeli indirir (ağ kullanan TEK adım); --verify-only ile doğrular
.venv/bin/python scripts/run_face_detection.py --source camera --camera-index 0
.venv/bin/python scripts/run_face_detection.py --source video --video-path /yol/video.mp4
.venv/bin/python scripts/run_face_detection.py --source camera --conf 0.7 --detect-width 960
.venv/bin/python scripts/run_face_detection.py --source camera --no-display --max-frames 300
.venv/bin/python scripts/run_face_detection.py --help
.venv/bin/python scripts/benchmark_face_detection.py --gui          # sentetik montajlarla ölçüm; --import-ultralytics-first ortak süreç durumunu ölçer
```

Seçenekler: kaynak (`--source/--camera-index/--width/--height/--video-path/--backend`; diğer betiklerle aynı), model (`--model DOSYA.onnx`, `--conf` 0,6, `--nms` 0,3, `--max-faces` 100, `--detect-width PX`), çalışma (`--no-display`, `--max-frames`). **Cihaz seçeneği yoktur** (yalnızca CPU; §2.1). Çıkış: `Q`/`Esc`, pencereyi kapatma, `Ctrl+C`; kodlar 0 / 1 (kamera/video/pencere/model hatası) / 2 (hatalı argüman) / 130. Sıra: model yüklenir (başarısızsa kamera **hiç** açılmaz) → kaynak → ısınma → pencere → döngü; her koşulda kaynaklar serbest bırakılır. Pencerede: yüz kutuları + güven, **`FACES: n`**, FPS, `INFERENCE ms`, çözünürlük, her zaman görünen `FACE BOX = detection only (not identity, not liveness)`. Konsol özeti: toplam yüz tespiti (**benzersiz kişi sayısı DEĞİLDİR**), kare başına ortalama/en çok, çıkarım/okuma/kare→sonuç süreleri.

## 6. Bağımlılıklar ve lisanslar

* **Yeni paket yok.** `requirements.txt`/`requirements-lock.txt` **değişmedi** (OpenCV 5.0.0.93 zaten pinli). `mediapipe==1.2.2` Aşama 1'den beri pinli ve kuruludur ama **5A tarafından kullanılmaz**; kaldırma bu aşamada yapılmadı (kullanıcı kararı).
* **Model dosyası** (`models/face_detection/face_detection_yunet_2026may.onnx`, Git dışı — `.gitignore` `*.onnx`): kaynak OpenCV Zoo `models/face_detection_yunet`, **commit `48b10fa505aeb51fbd127fc60a5c660040e823b7`** (2026-10-20); URL `https://media.githubusercontent.com/media/opencv/opencv_zoo/<commit>/models/face_detection_yunet/face_detection_yunet_2026may.onnx`; boyut **229.738 bayt**; **SHA-256 `ebafce4e3c118d6554634be5c27ab333b4c047a9a8c3faf1d7cf93101c22f0f0`** (yayıncının Git LFS işaretçisiyle birebir; indirilen dosya da eşleşti *[Ölçüldü]*); lisans **MIT** (Copyright © 2020 Shiqi Yu). `2026may`, `2023mar` dosyasının dinamik girdi boyutlu yeniden dışa aktarımıdır (README); iki dosya bu projede aynı görüntülerde birebir aynı kutuları/skorları verdi *[Ölçüldü]*.
* **İndirme** yalnızca `scripts/download_face_model.py` ile, izin listesindeki adla ve sabit commit URL'sinden yapılır; boyut/SHA-256 uyuşmazlığında dosya reddedilir; var olan dosyanın üzerine `--force` olmadan yazılmaz. **Çalışma anında hiçbir indirme/ağ işlemi yoktur** (testle denetlenir). Yükleme sırasında resmî dosya adı için boyut+SHA-256 yeniden doğrulanır.
* AGPL-3.0 yükümlülükleri (Ultralytics/YOLO11n) 5A'ya ilişkin değildir; yüz modeli MIT'tir.

## 7. Güvenlik ve gizlilik

Görüntü varsayılan olarak **kaydedilmez/arşivlenmez** ve hiçbir servise gönderilmez (testle denetlenir: `imwrite`/`VideoWriter` çağrılmaz). Biyometrik şablon, embedding, çalışan profili ve kalıcı yüz veritabanı **yoktur**; sonuçlar bellekte kalır. Pencere uyarısı her karede görünür. `person_tracking.md` §18 ve `project_stages.yaml` `stage5_design_notes` kararları korunur: Track ID çalışan kimliği değildir; poster/basılı fotoğraf/ekrandaki yüz de tespit edilir; yüz tespiti canlı insan doğrulaması değildir; yüz bulunması o bölgenin gerçek çalışan olduğunu kanıtlamaz; yüz görünmediğinde kimlik hakkında çıkarım yapılmaz; 4B'nin açık yanlış pozitif sorunları çözülmüş gösterilmez.

## 8. Otomatik testler

`.venv/bin/python -m unittest discover -s tests`: **1145 test** = önceki **883** (4B kapanışındaki toplam; hiçbiri değişmedi/atlanmadı) + **262 yeni**. Varsayılanda **10 atlanır** (hepsi isteğe bağlı gerçek-pencere GUI testi: Aşama 3'ten 3, 4A'dan 2, 4B'den 2, **5A'dan 3**); `ISG_GUI_TESTS=1` ile **1145 test, OK, 0 atlanan** *[Ölçüldü]*. Başarısız/hatalı test yok. Mock ve gerçek model testleri **ayrı raporlanır**:

| Tür | Dosya (test sayısı) | Toplam |
|---|---|---|
| Mock/birim (sahte YuNet, sahte tespitçi/kamera/pencere) | `config` (13), `results` (29), `detector` (57), `weights` (20), `visualization` (17), `pipeline` (27; 2'si isteğe bağlı GUI), `cli` (37), `benchmark` (25) | **225** |
| **Gerçek model** (gerçek YuNet, örnek/türev görüntüler, kayıtlı video) | `real_model` (37; 1'i isteğe bağlı GUI) | **37** |

Kapsanan senaryolar (isteğin 12 maddesi): tek yüz, çok yüz (1–16 yüzlük montaj dâhil), yüz yok (7 farklı kare türü), kısmen görünen yüz (sol/üst kenar), farklı çözünürlükler (1×1'den 4000×3000'e), geçersiz/bozuk görüntü (11 çeşit), kadraj sınırındaki kutular, geçersiz yapılandırma, model yükleme/bütünlük hatası (eksik dosya, bozuk ONNX, kurcalanmış resmî dosya), kaynakların kapanması (Q/Esc/kırmızı düğme/Ctrl+C/model-kamera-pencere hataları), kayıtlı video üzerinde gerçek model, Aşama 1–4 regresyonu (883 test). `test_no_side_effects` yeni modülleri kapsar.

## 9. Gerçek model doğrulamaları ve performans

### 9.1 Gerçek model doğrulamaları *[Gerçek model]*
Gerçek YuNet yüklendi (SHA-256 doğrulandı); `zidane.jpg` (1280×720) ve `bus.jpg` (810×1080) — Ultralytics paketiyle gelen örnek fotoğraflar, 4A/4B testlerinde de kullanılmıştır. **Referans kutular bağımsız etiket değildir:** modelin kutuları görüntü üzerine çizilip **gözle** doğrulandı (zidane: Ancelotti ve Zidane'ın yüzleri; bus: iki yaya). Bulgular: zidane'da 2 net yüz IoU ≥ 0,6 ile; bus'ta 2/2; 1–16 yüzlük sentetik montajlarda **her sayıda tam o kadar yüz**; 4 ölçekte (640×360…1920×1080) 2 net yüz; 32–128 px küçük ve 300–600 px büyük yüzler; yüz içermeyen 7 karede 0 yüz; sınır kaydırmalarında tüm kutular kare içinde; `detect_width=640`'ta kutular özgün koordinata doğru dönüyor; kayıtlı sentetik montaj videosunda (4A'nın `make_synthetic_video`'su; bus + kayan zidane) **45/45 karede 4 yüz**, CLI `exit=0`, görüntü yazılmadı, ağ kullanılmadı.
**Ne gösterilmedi:** doğruluk (precision/recall/AP), profil/maskeli/gözlüklü/çocuk yüzleri, ten rengi/ışık çeşitliliği, gerçek kamera görüntüsü, yüksek açı, **gerçek poster/basılı fotoğraf** (yalnızca montajlardaki fotoğraf-içi-fotoğraf yüzlerin bulunması dolaylı işarettir). Test verisi **yetersizdir** (iki fotoğraf ve türevleri; tekrarlanan aynı iki yüz).

### 9.2 Performans *[Ölçüldü; yalnızca bu makine; sentetik montaj; Aşama 4B/4A sonuçları kullanılmadı]*
Yöntem: `scripts/benchmark_face_detection.py` (100 tekrar, 5 ısınma; çıkarım süresi = `FaceDetectionResult.inference_ms`). Ham çıktılar: `results/face_detection/benchmark_5A_*.json`.

**Koşu A — 5A tek başına (ultralytics içe aktarılmamış; OpenCV 10 iş parçacığı):**

| Ölçüm (1280×720, CPU) | Bulunan/beklenen yüz | Medyan ms | p95 ms | Çıkarım-yalnız FPS |
|---|---|---|---|---|
| zidane.jpg doğal fotoğraf | 3 / – | 11,23 | 12,33 | 89,0 |
| bus.jpg (810×1080) | 2 / – | 10,64 | 11,41 | 94,0 |
| 0 yüz (~96 px) | 0 / 0 | 11,03 | 11,88 | 90,6 |
| 1 yüz | 1 / 1 | 10,80 | 12,33 | 92,6 |
| 2 yüz | 2 / 2 | 11,15 | 12,77 | 89,7 |
| 4 yüz | 4 / 4 | 11,24 | 12,39 | 89,0 |
| 8 yüz | 8 / 8 | 10,80 | 11,74 | 92,6 |
| 16 yüz | 16 / 16 | 11,56 | 13,15 | 86,5 |

**Yüz sayısı** süreyi **neredeyse hiç etkilemez** (0–16 yüz: 10,8–11,6 ms; ağ tam-evrişimsel, maliyet girdi boyutuna bağlı).
**Çözünürlük** (4 yüzlü montaj yeniden boyutlandırıldı, 4/4 bulundu): 640×360 **3,72 ms (269 FPS)** · 960×540 **6,42 ms (156)** · 1280×720 **11,48 ms (87)** · 1920×1080 **24,87 ms (40)**.
**`--detect-width`** (4/4 bulundu): 1280×720 → 960: **8,11 ms**, → 640: **3,81 ms**; 1920×1080 → 960: **6,56 ms**, → 640: **4,08 ms** (kare: 23,42 ms). Bu montajda küçültme yüz kaybettirmedi; küçük/uzak yüzler için **genellenemez** (zidane'da 640'ta bulanık arka plan yüzü kayboldu).
**Model yükleme:** medyan **2,15 ms** (5 yükleme; SHA-256 denetimi dâhil); ısınma 1280×720: **14 ms**.
**Uçtan uca (gerçek CLI, kayıtlı 300 karelik video, penceresiz):** **88,1 / 87,7 / 88,0 FPS** (çıkarım 11,0 ms + okuma 0,3 ms); `--detect-width 960` ile **124,5 FPS** (çıkarım 7,7 ms), her ikisinde 300/300 karede 4 yüz. *Canlı kamera hızı ölçülmedi (kamera ≈ 30 FPS ile sınırlı olabilir; bkz. 4B raporu §25.2 — 5A için ayrıca doğrulanmadı).*
**Bellek** (`ps` anlık RSS, taze süreç, 1280×720): içe aktarma sonrası **53 MiB** → model yükü **+13** → ilk 1280×720 çıkarımı/ısınma **+85** → 300 çıkarım sonrası **153 MiB** (**+2**; büyüme yok). Benchmark'ın tepe değeri (1920×1080 ve çoklu boyut dâhil) **397 MiB** (`ru_maxrss`; 1280×720 sürekli kullanım değil).
**GUI maliyeti** (gerçek `PreviewWindow`, 1280×720, 200 tekrar): çizim (`draw_face_detections`) medyan **0,20 ms**; `show` (imshow + waitKey(1)) medyan **15,5 ms** (4B'de ölçülen ≈ 15–17 ms ile uyumlu). Penceresiz/pencereli **uçtan uca FPS farkı ayrıca ölçülmedi** (video+pencere modu kaynağın FPS'ine göre bilerek bekler).
**Hızlandırılmış çalışma:** **yok / ölçülemedi** (YuNet OpenCV'de yalnızca CPU; OpenCL hedefi etkisiz, §2.1).

### 9.3 Cihaz karşılaştırması
CPU ↔ MPS karşılaştırması **yapılamaz:** OpenCV DNN'de MPS hedefi yoktur ve OpenCL hedefi süreyi değiştirmedi (11,0–11,2 ms). Hiçbir hızlandırma değeri tabloya yazılmadı.

### 9.4 Aşama 5B için kritik gözlem: `ultralytics` içe aktarılınca YuNet ≈ 2,7× yavaşlar *[Ölçüldü]*
`ultralytics` içe aktarıldığında OpenCV süreç genelinde tek iş parçacığına düşer (`cv2.getNumThreads() == 1`) ve YuNet 1280×720'de **≈ 30 ms**'ye çıkar (koşu B: zidane 30,45 ms; 4 yüz 30,18 ms; 16 yüz 30,30 ms; `--detect-width 960` → 23,4 ms; 640 → 8,4 ms). Aynı sonuç yalnızca `cv2.setNumThreads(1)` ile de oluşur; `torch` tek başına etkilemez (11,5 ms). 5A betikleri `ultralytics` yüklemediği için **11 ms** ölçülür. **5B'de YOLO ve yüz tespiti aynı süreçte çalışacaksa** bu etki hesaba katılmalı (örn. içe aktarma sonrası `cv2.setNumThreads(<n>)` ile yeniden ayarlama değerlendirilmeli; 5A'da **yapılmadı**, çünkü süreç genelinde bir ayar ve 4A/4B davranışını etkileyebilir).

## 10. Bilinen sınırlamalar

1. **Yüz tespiti kimlik/canlılık değildir.** Poster, basılı fotoğraf, telefon/ekrandaki yüz de tespit edilir (beklenen; gerçek poster **test edilmedi**). Yüz görünmeyen kişi hakkında çıkarım yapılamaz.
2. **Doğruluk ölçülmedi:** etiketli veri yok; precision/recall/AP, "%85" gibi bir hedef için kanıt yok. YuNet'in yayıncı AP değerleri doğrulanmadı.
3. **Test verisi yetersiz:** iki örnek fotoğraf ve türevleri; profil, maske, gözlük, çocuk/yaşlı, farklı ten rengi ve ışık koşulları, hareket bulanıklığı yok.
4. **Döndürülmüş yüzler:** düzlem içi 60°/90° dönmede iki net yüz kaçtı (tek görüntüde ölçüldü; 0–45° ve 180°'de bulundu). Yan (yaw) ve eğik (pitch) yüzler için ölçüm **yok**.
5. **Kadraj kenarı:** yüzün ≈ %46'sı (sol kenar) veya ≈ %29'u (üst kenar) kesilince bulundu; ≈ %90 (sol) veya ≈ %75 (üst) kesilince bulunamadı (tek görüntü, kaydırma ile). `at_border` sezgiseldir; yüzün gerçekten kesildiğinin kanıtı değildir.
6. **Bulanık/arka plan yüzleri** `conf 0,6`'da kabul edilebilir (zidane'da 0,61); gerçek yüz mü yanlış pozitif mi **belirsizdir**.
7. **Yalnızca CPU;** hızlandırma yok. `ultralytics` aynı süreçteyse ≈ 2,7× yavaşlama (§9.4).
8. `--detect-width` küçük/uzak yüzleri kaçırabilir; etkisi sahneye bağlıdır.
9. İşaret noktası, poz, kalite, yaş/cinsiyet, canlılık yoktur; kareler arası tutarlılık/yüz izleme yoktur; sayı **benzersiz kişi sayısı değildir**.
10. **Gerçek kamerada yalnızca tek kişilik nitel doğrulama yapıldı (§13.1); çoklu yüz ve yüksek kamera açısı (≈ 2 m) testi yapılmadı;** yukarıdan bakışta yüzlerin görünürlüğü ve eğimi bilinmiyor.
11. 5A'da **yüz ↔ kişi/Track ID eşleştirmesi yoktur;** bu 5B'nin işidir. Aşama 4B sınırlamaları (ID değişimleri, poster yanlış pozitifi FP-4B-001 vb.) aynen geçerlidir.
12. Kullanılmayan bağımlılık: `mediapipe==1.2.2` pinli ama 5A tarafından kullanılmaz.

## 11. Aşama 5B'ye aktarılacak çıktılar

* `FaceDetectionResult` / `FaceDetection` (alanlar §4) ve `run_face_detection(..., on_result=lambda paket, sonuç: ...)`; aynı karenin `TrackingResult`'ı ile ilişkilendirme anahtarı **`(frame_index, image_size)`** (ikisi eşit olmalı; `timestamp` de aynı `FramePacket`'ten gelir). `FaceDetection.face_index`, `TrackedPerson.detection_index` gibi karedeki sıradır. **Eşleştirme algoritması 5A'da YOKTUR.**
* Koordinatlar `PersonDetection`/`TrackedPerson` ile aynı sistemdedir; yüz merkezi/kutusu ile kişi kutusunun örtüşmesi 5B'de tanımlanmalıdır.
* **Aşama 5 tasarım notları** (`docs/person_tracking.md` §18, `stage5_design_notes`): Track ID ≠ çalışan kimliği; ID değişimlerinde yanlış eşleştirme önlenmeli (çok kareli doğrulama, belirsizlikte eşleştirmeme); yanlış pozitif iz üzerindeki yüz bir çalışana bağlanmamalı; ID değişimi sıklığı bilinmiyor. 5A'dan eklenen girdiler: **(a)** `ultralytics` ile ortak süreçte YuNet yavaşlaması (§9.4); **(b)** poster/ekrandaki yüz de tespit edilir ve poster iz-kutusu içinde bir yüz üretebilir; **(c)** `at_border` ve bulanık yüzler (düşük güven) eşleştirmede düşük güvenle ele alınmalı; **(d)** yüz tespiti ve insan tespiti kare başına bağımsız çalışır (aynı kareyi paylaşırlar; çıkarım bütçesi: YuNet ≈ 11 ms CPU + YOLO).
* **5B'den önce gerekenler:** kullanıcının gerçek kamera testleri (§13), özellikle çoklu kişi/mesafe/açı ve **yüksek kamera açısında yüz görünürlüğü**; ortak süreç iş parçacığı kararı; yüz–kişi ilişkilendirme kuralının ve belirsizlik politikasının tanımı; etiketli küçük bir değerlendirme kümesi (izinli) ihtiyacının kararı.

## 12. Kabul kriterleri ve kanıt düzeyi (5A)

✔ = karşılandı (kanıt tam); ✔* = kısmen karşılandı / kanıt sınırlı (eksik kısım açıkça yazılı); ✖ = karşılanmadı.

| # | Kriter | Düzey | Kanıt / eksik kısım |
|---|---|---|---|
| 1 | Gerçek yüz tespit modeli yüklenebilmeli | ✔ | Gerçek YuNet yüklendi, SHA-256 doğrulandı; gerçek model testleri (36 çalıştı) gerçek YuNet'i kullanır *[Gerçek model]* |
| 2 | Tek ve çoklu yüz tespiti çalışmalı | ✔* | Tek yüz (2 kırpma + tuval), çoklu (zidane 2, bus 2, montaj 1–16, video 4/kare) çalışıyor *[Gerçek model]*. **Eksik:** gerçek kamera; yalnızca iki fotoğraf ailesi |
| 3 | Koordinatlar ve güven değerleri doğrulanmalı | ✔* | Referans kutulara IoU ≥ 0,6, geometrik değişmezler, eşik tekdüzeliği, sınır kırpma *[Gerçek model]*. **Eksik:** referanslar bağımsız etiket değil (gözle doğrulandı); güven kalibre değil |
| 4 | Kamerasız gerçek model testleri geçmeli | ✔ | 36 çalıştı/geçti, 1'i isteğe bağlı GUI (`ISG_GUI_TESTS=1` ile o da geçti) |
| 5 | GUI ve CLI entegrasyonu test edilmeli | ✔* | CLI: sahte + gerçek model + alt süreç; GUI: sahte pencere + gerçek Cocoa penceresi (kırmızı düğme dâhil, `ISG_GUI_TESTS=1`). **Eksik:** gerçek kamerada GUI/CLI |
| 6 | Aşama 1–4 regresyonu olmamalı | ✔ | Önceki 883 test geçti; 4A/4B/kamera kaynakları, model, ham veri, eşikler değişmedi |
| 7 | Atlama/başarısızlık raporlanmalı | ✔ | 10 atlama (hepsi GUI, nedeniyle); 0 başarısız; geliştirme sırasında bulunan test hataları ve düzeltmeleri raporda |
| 8 | Canlı testler belirlenmeli | ✔ | §13 |
| 9 | Model lisans/kaynak bilgisi belgelenmeli | ✔ | §6 (commit, URL, boyut, SHA-256, MIT). Eksik: MediaPipe model kartı lisansı doğrulanmadı (seçilmedi) |
| 10 | 5B'ye uygun arayüz hazır olmalı | ✔* | §11; veri yapıları + `on_result` kancası + sözleşme testleri. **Eksik:** gerçek `TrackingResult` ile birlikte çalıştırılmadı (yalnız sahte tespitçiyle sözleşme testi) |

**Sonuç:** teknik olarak **`READY_FOR_MANUAL_VERIFICATION`**; `COMPLETED` iddia **edilmez** (gerçek kamera testleri ve doğruluk ölçümü yok).

## 13. Manuel MacBook kamera test planı (kullanıcı; Claude ortamında kamera izni yoktur)

Hazırlık: kamera iznini Terminal/VS Code'a verin; `.venv/bin/python scripts/download_face_model.py --verify-only`; `.venv/bin/python scripts/run_face_detection.py --help`. Pencereli test: `--source camera --camera-index 0`; sayısal ölçüm: `--no-display --max-frames 300`. **Gözlem yapılmadan sonuç yazmayın; bildirilmeyen test yapılmamış sayılır.** Her koşuda ortamı (ışık, mesafe, arka plan, kaç kişi) ve konsol özetini kaydedin.

| # | Test | Yapılacak | Kontrol listesi |
|---|---|---|---|
| 1 | Tek yüz | ≈ 0,5–1 m, kameraya bakın, sağa/sola hareket | `FACES: 1`? Kutu yüzü sarıyor mu? Güven (≈ 0,8+)? Kutu kayıyor mu? |
| 2 | Çoklu yüz | 2, sonra 3+ kişi yan yana | `FACES` kişi sayısına eşit mi? Her yüz ayrı kutu mu? Kaçırılan/çift kutu var mı? |
| 3 | Mesafe | 1, 2, 3, 4+ m (ve mümkünse ≈ 2 m yükseklikten) | Yüzün kaybolduğu en uzak mesafe? Güven mesafeyle nasıl değişiyor? `--detect-width 960` ile fark? |
| 4 | Açı | Yüzü 15°, 30°, 45°, 60° ve profil çevirin; yukarı/aşağı bakın; başı yana eğin (30°) | Hangi açıda kutu kayboluyor? Güven düşüyor mu? |
| 5 | Kısmi görünüm | Yüzü kadraj kenarına taşıyın; elle/maske/gözlük/şapkayla kısmen kapatın | Kutu çıkıyor mu? `EDGE` etiketi görünüyor mu? |
| 6 | Yüz yok | Boş oda, sırtınız dönük | `FACES: 0`, hata yok? |
| 7 | **Yanlış pozitif** | Telefonda/ekranda bir yüz fotoğrafı, basılı fotoğraf (varsa poster), yüze benzeyen nesne, bulanık arka plan | Hangileri tespit ediliyor, hangi güvenle? **(Beklenti: fotoğraf/ekran yüzü tespit edilir; bu yanlış değil, tasarım sınırıdır.)** Ekran görüntüsüyle kaydedin |
| 8 | Işık | Loş oda, arkadan ışık | Güven/kayıp nasıl? |
| 9 | Çıkışlar | `Q`, `Esc`, macOS kırmızı düğme, terminalde `Ctrl+C` | Pencere kapandı mı? Kamera ışığı söndü mü? Çıkış kodu (`echo $?`): 0/0/0/130 |
| 10 | Performans | `--no-display --max-frames 300`; pencereli koşu | "uçtan uca FPS", "Çıkarım süresi", "Kare okuma", "Kare → sonuç gecikmesi" satırları; pencereli/penceresiz fark |
| 11 | Karşılaştırma | `--detect-width 960` ve `640` | Hız kazancı / kaçırılan yüz |

### 13.1 Kullanıcı tarafından bildirilen canlı kamera doğrulaması (nitel; kayıt tarihi 2026-11-15)
**Kaynak/tür:** kullanıcı bildirimi; gerçek MacBook kamerası, tek kişi. **Tam konsol çıktısı, sayısal ölçüm (güven, FPS, süre) ve ekran görüntüsü paylaşılmadı; kullanılan betik (5A betiği mi, 5B hattı mı) bildirilmedi.** Bu bildirim bağımsız doğruluk ölçümü değildir.

| # | Test | Kullanıcı bildirimi | Durum |
|---|---|---|---|
| 1 | Tek yüz | Kamera başarıyla açıldı; canlı görüntüleme penceresi çalıştı; gerçek insan yüzü tespit edildi; yüz kutusu ekranda gösterildi | **Kısmen doğrulandı (nitel).** Mesafe, güven (≈ 0,8+), kutunun yüzü sarması/kayması ve hareket ayrıntıları bildirilmedi |
| 2–11 | Çoklu yüz, mesafe, açı, kısmi görünüm, yüz yok, yanlış pozitif, ışık, çıkışlar, performans, `--detect-width` | **Bildirilmedi** | **Yapılmadı sayılır** (`OT-5A-001` kısmen, `OT-5A-004`, `OT-5A-005` açık) |

Notlar: (i) Gerçek kamerada **iki/üç kişiyle çoklu yüz testi yapılmadı**; kamerasız gerçek model ve sentetik montaj testleri (§9) ayrı kanıttır ve bu bildirimle eşdeğer değildir. (ii) 5B'nin canlı ekran görüntülerinde (`docs/face_association.md` §13.5) YuNet ≈ 14 ms görüldü; bu **5B hattının** anlık overlay değeridir, 5A betiğinin canlı performansı (`OT-5A-004`) ölçülmüş sayılmaz. (iii) 5B'nin Bölüm A kontrolleri (baş çevirme, yüz kapatma vb.) 5B hattı üzerinden bildirildi; 5A test listesinin 4/5/6. testlerinin yerine geçmez.

## 14. Başlıca bulgular (geliştirme sırasında) ve yapılmayanlar

* **Test hataları (kaynak değil, beklenti hataları; testler gevşetilmeden düzeltildi):** `detect_width=50` geçersiz (alt sınır 64); kamera bitince `SourceDisconnectedError` (4B'de de görülen tuzak) → `max_frames`; `FakeCapture` `KeyboardInterrupt` fırlatmaz; sahte tespitçinin sayı listesi son değeri tekrarlar; bozulmuş en-boy oranı (810×1080'e ezilmiş zidane) yüzleri kaçırır (bu durum sınanmaz, test düzeltildi); `test_no_side_effects` yeni onaylı model dosyası için güncellendi.
* **Yapılmadı:** gerçek kamera testi, doğruluk ölçümü, SCRFD/BlazeFace-full-range değerlendirmesi, MPS/hızlandırma, yüz izleme, işaret noktası, 5B eşleştirmesi, iş parçacığı ayarı değişikliği.
