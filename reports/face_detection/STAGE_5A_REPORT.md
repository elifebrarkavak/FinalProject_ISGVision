# Aşama 5A — Çoklu Yüz Tespiti: Sonuç Raporu

**Tarih:** 2026-11-13 · **Durum: `READY_FOR_MANUAL_VERIFICATION`** — **`COMPLETED` DEĞİLDİR.** Kod, otomatik testler (mock + gerçek model), kamerasız gerçek model doğrulamaları, kayıtlı video ölçümleri ve belgeler tamam; **ilk teslimde gerçek MacBook kamerası ile hiçbir test yapılmamıştı** (Claude ortamında kamera izni yok); **sonradan kullanıcı tek kişilik canlı yüz tespitini nitel olarak doğruladı (§13: kamera açıldı, pencere çalıştı, gerçek yüz tespit edildi, yüz kutusu gösterildi; sayısal ölçüm yok)**; çoklu yüz, mesafe, açı, canlı FPS ve yüksek açı **yapılmadı**, **doğruluk (precision/recall/AP) ölçülmedi**, test verisi yetersiz.
**Aşama 5 genel durumu: `IN_PROGRESS`** (5B/5C başlatılmadı; 5A/5B/5C ayrımı **öneridir**). Aşama 4 (`COMPLETED`) ve 4B kapanış kayıtları/açık sınırlamaları **değişmedi.** · **Teknik belge:** [docs/face_detection.md](../../docs/face_detection.md) · **Git:** commit/push yapılmadı.

> **Özet.** Seçilen model **YuNet** (OpenCV `cv2.FaceDetectorYN`, MIT lisanslı 230 KB ONNX, yalnızca CPU); **yeni paket eklenmedi.** `FaceDetector` her kare için `bbox_xyxy`, `confidence`, `frame_index`, `timestamp`, `image_size` üretir; yüz bulunmayan kareyi hatasız işler; kutular `human_detection`/`tracking` ile **aynı koordinat sisteminde** ve görüntü sınırlarına kırpılmıştır. Canlı pencerede kutu + güven + toplam yüz sayısı + FPS + çıkarım süresi gösterilir. Gerçek model 1280×720'de ≈ **11 ms/kare** (CPU; kayıtlı videoda penceresiz ≈ **88 FPS**) ve yüz sayısından neredeyse bağımsızdır.
> **Yüz tespiti kimlik veya canlı insan doğrulaması DEĞİLDİR:** poster/basılı fotoğraf/ekrandaki yüz de tespit edilir. Yüz tanıma, çalışan kimliği, embedding/biyometrik veri ve **yüz–Track ID eşleştirmesi (5B) YOKTUR.** Aşama 4B'nin açık yanlış pozitif sorunları (FP-4B-001 vb.) çözülmüş değildir.

## Gerçekte ne var? (kanıt türlerine göre ayrılmış özet)

| Başlık | İçerik |
|---|---|
| **Gerçekte çalışanlar** | `FaceDetector` (gerçek YuNet), `FaceDetectionConfig`, sonuç veri yapıları, `draw_face_detections` (kopya üzerinde çizim, "kimlik/canlılık değildir" uyarısı), `run_face_detection` döngüsü (kamera/video, 4A/4B ile aynı çıkış kodları ve kaynak serbest bırakma), `scripts/run_face_detection.py`, `download_face_model.py` (resmî, sabit commit, boyut+SHA-256), `benchmark_face_detection.py` |
| **Test edilenler** | **262 yeni test:** 225 mock/birim + **37 gerçek model** (gerçek YuNet; örnek fotoğraflar ve türevleri; kayıtlı video; CLI/betik uçtan uca). Önceki **883 test korundu** (toplam **1145**). Varsayılanda 10 atlanır (hepsi isteğe bağlı GUI), `ISG_GUI_TESTS=1` ile 0 atlanan |
| **Ölçülenler** | Model seçimi karşılaştırması (YuNet ↔ BlazeFace, 1280×720'de 16–560 px yüz taraması), yüz içermeyen bölgelerde yanlış pozitif, eşik duyarlılığı, düzlem içi dönme, kısmi yüz; model yükleme/ısınma, çıkarım süresi (0–16 yüz, 4 çözünürlük, `--detect-width`), uçtan uca CLI FPS, bellek, gerçek pencerede `show` maliyeti; OpenCV DNN hedefleri (OpenCL etkisiz) ve `ultralytics`'in OpenCV iş parçacığı etkisi |
| **Yapılamayanlar** | Gerçek MacBook kamerası testlerinin çoğu (ilk teslimde izin yoktu; sonradan yalnızca tek kişilik nitel yüz tespiti kullanıcı tarafından doğrulandı, §13); canlı FPS/gecikme; yüksek kamera açısı; doğruluk ölçümü (etiketli veri yok); gerçek poster/basılı fotoğraf testi; SCRFD ve BlazeFace-full-range çalıştırma; hızlandırılmış (MPS/OpenCL) çalışma karşılaştırması (OpenCV'de hedef yok/etkisiz) |
| **Bilinen sınırlamalar** | docs/face_detection.md §10 (`KL-5A-001…010`): kimlik/canlılık değildir; veri yetersiz; 60°+ dönmüş yüz kaçar; kadrajda ~%75+ kesik yüz kaçar; bulanık arka plan yüzü `conf 0,6`'da kabul edilir; yalnızca CPU; `ultralytics` aynı süreçteyse ≈ 2,7× yavaşlama |
| **Kullanıcının yapacağı manuel testler** | docs/face_detection.md §13 (11 test: tek/çoklu yüz, mesafe, açı, kısmi görünüm, yüz yok, yanlış pozitif, ışık, çıkışlar, performans, `--detect-width`) |

**Kanıt türleri:** *[Ölçüldü]* = Claude'un bu makinede çalıştırıp gördüğü; *[Mock]* = sahte bağımlılık; *[Gerçek model]* = gerçek YuNet, örnek/türev görüntü, kamerasız; *[Doküman]* = yayıncı belgesine dayanan, doğrulanmamış; *[Bekliyor]* = yapılmamış, kullanıcı görevi. Yapılmayan hiçbir test başarılı gösterilmemiştir.

---

## 1. Başlangıç denetimi

| Konu | Durum *[Ölçüldü]* |
|---|---|
| Git | `main`, HEAD `d675529` ("cheapter 4b completed"), çalışma ağacı temiz |
| Aşama durumları (yaml) | 1–4 `completed`, 4A/4B `completed`, **5 `planned`**; `stage5_design_notes` N5-1…3 mevcut; 5A/5B/5C ayrımı **repoda yoktu** (öneri) |
| Başlangıç testleri | `unittest discover -s tests`: **883 test, OK, 7 atlandı** (isteğe bağlı GUI) |
| Mevcut yüz kodu | `src/isg_vision/face_detection/__init__.py` yalnızca "Henüz uygulanmadı." docstring'i (boş iskelet); `models/face_detection/` yalnızca `.gitkeep` |
| Bağımlılıklar | `requirements.txt`/lock: `opencv-python==5.0.0.93`, `opencv-contrib-python==5.0.0.93`, `numpy==2.4.6`, **`mediapipe==1.2.2` (Aşama 1'den pinli ve kurulu, kullanılmıyordu)**, `ultralytics==8.4.210`, `lap==0.5.14`; `psutil==7.2.2` yalnızca `requirements-lock.txt`'de (dolaylı bağımlılık), `requirements.txt`'de yok; 5A kullanmadı (bellek `ps` ve `resource` ile ölçüldü) |
| Model dosyası politikası | `.gitignore` `*.onnx` ve `/models/**/*.onnx`'i Git dışında tutar; `test_no_side_effects` yalnızca onaylı ağırlıklara izin verir |
| Yeniden kullanılan altyapı | `camera.capture.FrameSource`, `camera.display.PreviewWindow`, `camera.cli.config_from_args`, `human_detection.pipeline.LatencyStats`, `human_detection.results.clip_box_xyxy`, `human_detection.benchmark.make_synthetic_video` (yalnızca test/ölçüm) — **hiçbiri değiştirilmedi** |

## 2. Model seçimi

**YuNet (OpenCV `FaceDetectorYN`) seçildi.** Ayrıntılı karşılaştırma ve kanıt ayrımı: [docs/face_detection.md §2](../../docs/face_detection.md).

| Aday | Bu ortamda doğrulanan | Sonuç |
|---|---|---|
| **YuNet** | Çalışıyor; zidane: 2 net yüz + 1 bulanık arka plan yüzü, bus: 2/2; **16–560 px** yüz bulundu; 1280×720'de ≈ 11 ms; yeni paket yok; MIT; 230 KB; yalnızca CPU | **Seçildi** |
| MediaPipe BlazeFace short-range | Çalışıyor (pakette zaten kurulu); ama girdi 128×128'e küçüldüğü için 1280×720'de **≤ 128 px yüz bulunamadı** (net yüzler kaçtı; bus 0/2); ≈ 0,75 ms; GPU temsilcisi RGB girdiyle hata verdi; model dosyası pakette yok | Menzil yetersiz (çalışan takibinde yüzler uzak) |
| SCRFD (InsightFace) | **Çalıştırılmadı.** *[Doküman]* hazır ağırlıklar "yalnızca ticari olmayan araştırma"; `onnxruntime` + `protobuf` gerekir (`pip --dry-run`: uygun) | Lisans ve ek çalışma zamanı nedeniyle seçilmedi |
| Diğerleri (BlazeFace full-range, üçüncü taraf YOLO-yüz, Haar/SSD) | Değerlendirilmedi | — |

**Dürüstlük:** seçim iki örnek fotoğraf ailesi ve yayıncı belgelerine dayanır; YuNet README'sinin AP değerleri doğrulanmadı; bağımsız etiketli değerlendirme yoktur.

## 3. Mimari ve dosyalar

**Oluşturulan** (Git'e eklenmedi, commit yok):
`src/isg_vision/face_detection/{errors,config,results,weights,detector,visualization,pipeline,cli,benchmark}.py`; `scripts/{run_face_detection,download_face_model,benchmark_face_detection}.py`; `tests/{face_detection_test_support, test_face_detection_config,results,detector,weights,visualization,pipeline,cli,benchmark,real_model}.py`; `docs/face_detection.md`; bu rapor. **Git dışı:** `models/face_detection/face_detection_yunet_2026may.onnx`, `results/face_detection/*` (ölçüm çıktıları).
**Değiştirilen:** `src/isg_vision/face_detection/__init__.py` (iskelet → paket API'si); `configs/project_stages.yaml` (Aşama 5 `in_progress`; `stage5_substages`, `stage5a_*`, `stage5_design_notes` N5-4…6 **eklendi**; mevcut 4B kayıtları ve N5-1…3 dokunulmadı); `README.md`, `docs/project_plan.md`, `docs/architecture.md` (5A durumu; 4B bloğunun "Mevcut durum:" **etiketi** "Aşama 4B —" yapıldı, içerik aynı); `tests/test_no_side_effects.py` (+10 modül adı, +1 onaylı model dosyası; **hiçbir kontrol gevşetilmedi**: diğer her model dosyası hâlâ reddedilir).
**Değişmedi:** 4A/4B/kamera kaynakları, `models/human_detection/yolo11n.pt` (SHA-256 `0ebbc80d…ee1`), ByteTrack adaptörü ve eşikleri, ham veri, `requirements*.txt`, 4A/4B/Aşama 3 raporları, `docs/person_detection.md`, `docs/person_tracking.md`.

Mimari: model, yapılandırma, sonuç yapıları, görselleştirme ve döngü ayrı modüllerdir; `FaceDetector` yalnızca "görüntü içeri, yüz kutuları dışarı" yapar; çizim girdinin **kopyasında** yapılır (testle denetlenir); `on_result(paket, sonuç)` kancası 5B içindir. `FaceDetection.image_size` + `frame_index` + `timestamp` + `face_index` + `at_border` ile **yüz–Track ID eşleştirmesi yapılmadan** 5B'ye hazırlanmıştır.

## 4. Bağımlılık, kaynak ve lisans kaydı

* **Yeni paket yok;** `requirements.txt`/`requirements-lock.txt` değişmedi. `mediapipe==1.2.2` 5A tarafından kullanılmaz (`OT-5A-007`: kaldırma kararı kullanıcıya ait).
* **Model:** `face_detection_yunet_2026may.onnx`; OpenCV Zoo `models/face_detection_yunet`, **commit `48b10fa505aeb51fbd127fc60a5c660040e823b7`** (2026-10-20); URL `https://media.githubusercontent.com/media/opencv/opencv_zoo/<commit>/models/face_detection_yunet/…`; **229.738 bayt**; **SHA-256 `ebafce4e3c118d6554634be5c27ab333b4c047a9a8c3faf1d7cf93101c22f0f0`** (yayıncının LFS işaretçisiyle birebir; indirilen dosya eşleşti); lisans **MIT** (`LICENSE` okundu; kaynak `libfacedetection.train` da MIT).
* **İndirme:** yalnızca `download_face_model.py` (izin listesi + sabit commit + boyut/SHA-256; mevcut dosyanın üzerine `--force` olmadan yazmaz); çalışma anında indirme/ağ **yoktur** (testle denetlenir); yükleme sırasında resmî dosya adı için boyut+SHA-256 yeniden doğrulanır (kurcalanmış kopya ayrıştırıcıya verilmez; testli). Değerlendirme için ayrıca indirilen iki aday dosya (`face_detection_yunet_2023mar.onnx`, `blaze_face_short_range.tflite`) yalnızca geçici klasörde kullanıldı, depoya girmedi; ikisinin de boyut+özeti yayıncı kaydıyla (YuNet) / sunucu başlığıyla (BlazeFace: 229.746 bayt, ETag MD5 eşleşti) doğrulandı.

## 5. Otomatik testler

| Çalıştırma | Sonuç *[Ölçüldü]* |
|---|---|
| Başlangıç (geliştirme öncesi) | **883 test, OK, 7 atlandı** |
| Son — varsayılan | **1145 test, `OK (skipped=10)`** → 1135 çalıştı ve geçti, **0 başarısız, 0 hata**, 10 atlandı |
| Son — `ISG_GUI_TESTS=1` | **1145 test, OK, 0 atlandı** (gerçek Cocoa pencereleri; kamera açılmadı) |
| Aşama 2 bütünlük denetimi | **12/12** (T2D.2 aşama numaraları dahil); `check_environment.py` **BAŞARILI** |

**Atlanan 10 test ve nedeni:** hepsi isteğe bağlı gerçek-pencere GUI testi (`macOS + ISG_GUI_TESTS=1 + grafik oturumu` gerekir; kamera açmazlar): Aşama 3'ten 3, 4A'dan 2, 4B'den 2, **5A'dan 3** (`RealWindowFaceLoopTest` 2, `RealWindowRealModelTest` 1). Koşullu atlamalar (model dosyası/örnek görüntü yoksa) bu makinede **tetiklenmedi:** 36 gerçek model testi çalıştı.

| Tür | Dosya (test) | Toplam |
|---|---|---|
| **Mock/birim** (sahte YuNet, sahte tespitçi/kamera/pencere) | `config` 13, `results` 29, `detector` 57, `weights` 20, `visualization` 17, `pipeline` 27 (2 GUI), `cli` 37, `benchmark` 25 | **225** |
| **Gerçek model** (gerçek YuNet) | `real_model` 37 (1 GUI) | **37** |

İsteğin 12 senaryosu: **(1)** tek yüz (`SingleFaceTest`, mock'ta tek kutu) · **(2)** çoklu yüz (zidane 2, bus 2, montaj 1/2/4/8/16, video 4/kare) · **(3)** yüz yok (7 kare türü gerçek modelde; mock'ta `None`/boş dizi) · **(4)** kısmen görünen yüz (sol/üst kenar kaydırma; mock'ta kırpma/`at_border`) · **(5)** farklı çözünürlük (1×1…4000×3000; 640×360…1920×1080 ölçek; 32–600 px yüz) · **(6)** geçersiz/bozuk görüntü (11 çeşit; model çağrılmadan reddedilir) · **(7)** sınıra yakın kutular (4 kenar, tolerans sınırı, tamamen dışarıdaki/ince kutu) · **(8)** geçersiz yapılandırma (conf/nms/max_faces/detect_width/model yolu/bayrak) · **(9)** model yükleme/bütünlük hatası (eksik dosya, bozuk ONNX, kurcalanmış resmî dosya, arayüz eksik, yeniden deneme) · **(10)** kaynak kapanışı (Q/Esc/kırmızı düğme/Ctrl+C/model-kamera-pencere hataları; model yüklenemezse kamera hiç açılmaz) · **(11)** kayıtlı video üzerinde gerçek model (45 kare, CLI ve betik alt süreci) · **(12)** Aşama 1–4 regresyonu (önceki 883 test).

**Mock ve gerçek ayrımı:** mock testler yalnızca `FaceDetector` mantığını (doğrulama, koordinat dönüşümü, kırpma, hata yönetimi, tek yükleme, süre ölçümü) sınar ve gerçek yüz tespitini **kanıtlamaz**; gerçek tespit yalnızca `test_face_detection_real_model.py` ile kanıtlanır. Hiçbir test gerçek kamera, doğruluk veya canlı performans kanıtı değildir.

## 6. Gerçek model testlerinin sonuçları *[Gerçek model]*

Girdiler: `zidane.jpg` (1280×720) ve `bus.jpg` (810×1080) — Ultralytics paketiyle gelen örnek fotoğraflar (indirme yok; 4A/4B testlerinde de kullanıldı) ve onlardan türetilen kırpma/ölçekleme/kaydırma/montajlar; 4A'nın sentetik montaj videosu. **Referans kutular bağımsız etiket değildir;** modelin kutuları görüntüde **gözle** doğrulandı.

| Doğrulama | Sonuç |
|---|---|
| Model yükleme | Gerçek YuNet yüklendi; boyut+SHA-256 doğrulandı; tek yükleme; kurcalanmış kopya ve bozuk ONNX reddedildi |
| Tek yüz | Ancelotti kırpması ve Zidane kırpması: tam 1 yüz, IoU ≥ 0,6, güven ≥ 0,8; büyük tuvalde de |
| Çoklu yüz | zidane: 2 net yüz (0,94/0,90) IoU ≥ 0,6 (+ 1 bulanık arka plan yüzü 0,61); bus: 2/2 net yüz; 1/2/4/8/16 yüzlük montaj: **her sayıda tam o kadar yüz** |
| Yüz yok | 7 kare türünde (gri/siyah/beyaz/gürültü/bina/kaldırım/gövde) 0 yüz, hata yok |
| Kısmi yüz | Sol kenarda ≈ %46 kesik yüz bulundu (`at_border=True`, kutu kırpıldı); üst kenarda işaretlendi; ≈ %90 kesik yüz bulunamadı; tüm kaydırmalarda kutular kare içinde |
| Çözünürlük/boyut | 4 ölçekte 2 net yüz; 32–128 px küçük ve 300–600 px büyük yüzler bulundu; 1×1…4000×3000 çökmeden boş sonuç |
| `--detect-width 640` | Net yüzler bulundu, kutular özgün koordinata doğru dönüyor |
| Eşik | Yüksek eşik hiçbir zaman yüz eklemiyor (0,3→0,97 azalan); `max_faces` en güvenli yüzü tutuyor ve kesmeyi bildiriyor |
| Kayıtlı video | 45/45 karede 4 yüz; `dropped=0`; CLI `exit=0`; betik alt süreci `exit=0`; görüntü yazılmadı; ağ kullanılmadı |
| Dayanıklılık | Deterministik; girdi değişmiyor; salt-okunur/bitişik olmayan görüntüler; ardışık boyut değişimi |

**Ne gösterilmedi:** doğruluk (precision/recall/AP), profil/maskeli/gözlüklü/çocuk yüzleri, ten rengi/ışık çeşitliliği, gerçek kamera, yüksek açı, **gerçek poster/basılı fotoğraf** (yalnızca montajdaki fotoğraf-içi-fotoğraf yüzlerin bulunması dolaylı işarettir). **Test verisi yetersizdir:** iki fotoğraf ve türevleri; montajlar aynı iki yüzü tekrarlar.

## 7. Performans *[Ölçüldü; yalnızca bu makine: MacBook Pro Apple M4, 10 çekirdek, 16 GiB, macOS 26.5.2; sentetik montaj; Aşama 4B/4A sonuçları 5A sonucu olarak sunulmamıştır]*

Ham çıktılar: `results/face_detection/` (Git dışı). Ayrıntılı tablolar: docs/face_detection.md §9.

| Metrik | Değer |
|---|---|
| **Model yükleme** | medyan **2,15 ms** (5 yükleme; SHA-256 denetimi dâhil); ısınma 1280×720 **14 ms** |
| **Çıkarım, 1280×720, CPU** (tek başına süreç) | **≈ 10,8–11,6 ms** (medyan; p95 11,7–13,2); doğal fotoğraflar 11,23 (zidane) / 10,64 ms (bus 810×1080) |
| **Farklı yüz sayıları** (96 px yüz, 1280×720 montaj) | 0 yüz 11,03 · 1 yüz 10,80 · 2 yüz 11,15 · 4 yüz 11,24 · 8 yüz 10,80 · **16 yüz 11,56 ms** — yüz sayısına neredeyse bağımsız; bulunan = beklenen (0/1/2/4/8/16) |
| **Çözünürlük** (4 yüzlü montaj yeniden boyutlu, 4/4 bulundu) | 640×360 **3,72 ms (269 FPS)** · 960×540 **6,42 (156)** · 1280×720 **11,48 (87)** · 1920×1080 **24,87 (40)** (çıkarım-yalnız FPS) |
| **`--detect-width`** | 1280→960: **8,11 ms**; 1280→640: **3,81**; 1920→960: **6,56**; 1920→640: **4,08** (kare 23,42) — 4/4 bulundu (genellenemez: zidane'da 640'ta bulanık yüz kayboldu) |
| **Uçtan uca FPS** (gerçek CLI, kayıtlı 300 karelik video, penceresiz) | **88,1 / 87,7 / 88,0** (çıkarım 11,0 + okuma 0,3 ms); `--detect-width 960` **124,5 FPS** (7,7 ms); 300/300 karede 4 yüz |
| **Bellek** (taze süreç, 1280×720, anlık RSS) | 53 MiB → model yükü **+13** → ilk 1280×720 çıkarım **+85** → 300 çıkarım sonrası **153 MiB (+2; büyüme yok)**; benchmark tepe değeri (1920×1080 dâhil) **397 MiB** |
| **GUI maliyeti** (gerçek `PreviewWindow`, 1280×720, 200 tekrar) | çizim medyan **0,20 ms**; `show` (imshow+waitKey(1)) medyan **15,5 ms** (4B'de ölçülen ≈ 15–17 ms ile uyumlu). Pencereli/penceresiz uçtan uca FPS farkı **ayrıca ölçülmedi** |
| **Hızlandırılmış çalışma** | **Yok/ölçülemedi:** OpenCV 5.0 yeni DNN motoru hedefi yok sayar, OpenCL hedefi süreyi değiştirmedi (11,0–11,2 ms), klasik motor bu modelle hata verir; MPS hedefi yok. CPU↔MPS karşılaştırması **yapılamaz** |
| **Canlı kamera FPS/gecikme** | **Ölçülmedi** (kamera yok; kullanıcı görevi) |

**`ultralytics` etkisi *[Ölçüldü]*:** `ultralytics` içe aktarıldığında OpenCV tek iş parçacığına düşer (`getNumThreads()==1`) ve YuNet 1280×720'de **≈ 30 ms**'ye çıkar (≈ 2,7×); aynı etki `cv2.setNumThreads(1)` ile de oluşur; `torch` tek başına etkilemez. 5A betikleri `ultralytics` yüklemez (11 ms). **5B'de YOLO ile aynı süreçte çalışırken** önemlidir (`OT-5A-006`, `stage5_design_notes` N5-4). 5A'da süreç genelinde ayar **yapılmadı**.

## 8. Yapılamayan testler ve açık sorunlar

| Unsur | Durum | Kayıt |
|---|---|---|
| Gerçek MacBook kamerası testleri (tek/çoklu yüz, mesafe, açı, kısmi, ışık, çıkışlar) | İlk teslimde **yapılmadı** (izin yok); **güncel: tek kişilik yüz tespiti kullanıcı tarafından nitel doğrulandı (§13), geri kalanı yapılmadı** | `OT-5A-001` (`partially_performed`) |
| Doğruluk (precision/recall/AP), etiketli değerlendirme | **Ölçülmedi** | `OT-5A-002`, `KL-5A-002` |
| Test verisi çeşitliliği; gerçek poster/basılı fotoğraf testi | **Yapılmadı** | `OT-5A-003`, `KL-5A-001/003` |
| Canlı performans (FPS, kare→sonuç, pencereli/penceresiz) | **Ölçülmedi** | `OT-5A-004` |
| Yüksek kamera açısı (≈ 2 m) | **Yapılmadı** | `OT-5A-005`, `KL-5A-010` |
| Ortak süreç (YOLO + yüz) OpenCV iş parçacığı kararı | **5B'ye ertelendi** | `OT-5A-006` |
| Kullanılmayan `mediapipe` bağımlılığı | **Karar bekliyor** | `OT-5A-007` |
| Alternatif modeller (BlazeFace full-range, SCRFD) | **Değerlendirilmedi** | `OT-5A-008` |
| Hızlandırılmış (MPS/OpenCL) çalışma | **Desteklenmiyor/ölçülemedi** | `KL-5A-007` |
| Aşama 4B açık sorunları (FP-4B-001, ID değişimleri vb.) | **Çözülmedi; değişmedi** | `stage4b_*` |

## 9. Kabul kriterleri ve kanıt düzeyi

✔ = karşılandı (kanıt tam) · ✔* = kısmen karşılandı / kanıt sınırlı · ✖ = karşılanmadı. **✖ yok; ✔* olanların eksik kısmı açıkça yazılıdır.**

| # | Kriter | Düzey | Kanıt / eksik kısım |
|---|---|---|---|
| 1 | Gerçek yüz tespit modeli yüklenebilmeli | ✔ | Gerçek YuNet yüklendi, SHA-256 doğrulandı *[Gerçek model]* |
| 2 | Tek ve çoklu yüz tespiti çalışmalı | ✔* | Tek/çoklu (zidane 2, bus 2, montaj 1–16, video 4/kare) çalışıyor. **Eksik:** gerçek kamera; yalnızca iki fotoğraf ailesi |
| 3 | Koordinatlar ve güven doğrulanmalı | ✔* | Referans kutulara IoU ≥ 0,6, geometrik değişmezler, eşik tekdüzeliği, sınır kırpma. **Eksik:** referanslar bağımsız etiket değil; güven kalibre değil |
| 4 | Kamerasız gerçek model testleri geçmeli | ✔ | 36/36 çalıştı ve geçti (+1 isteğe bağlı GUI testi `ISG_GUI_TESTS=1` ile geçti) |
| 5 | GUI ve CLI entegrasyonu test edilmeli | ✔* | CLI: sahte + gerçek model + alt süreç; GUI: sahte pencere + gerçek Cocoa penceresi (kırmızı düğme dâhil). **Eksik:** gerçek kamerada GUI/CLI |
| 6 | Aşama 1–4 regresyonu olmamalı | ✔ | Önceki 883 test geçti; kaynaklar/model/eşikler değişmedi |
| 7 | Atlama ve başarısızlıklar raporlanmalı | ✔ | §5 (10 atlama nedeniyle; 0 başarısız) ve §10 (geliştirme hataları) |
| 8 | Canlı testler belirlenmeli | ✔ | docs/face_detection.md §13 |
| 9 | Lisans/kaynak bilgisi belgelenmeli | ✔ | §4 (commit, URL, boyut, SHA-256, MIT). Seçilmeyen MediaPipe model kartı lisansı doğrulanmadı |
| 10 | 5B'ye uygun sonuç arayüzü | ✔* | Veri yapıları + `on_result` + ilişkilendirme anahtarı + sözleşme testleri. **Eksik:** gerçek `TrackingResult` ile birlikte çalıştırılmadı (sahte tespitçiyle sözleşme testi) |

**Sonuç:** 5A **teknik olarak hazır (`READY_FOR_MANUAL_VERIFICATION`)**, ancak gerçek kamera testleri ve doğruluk ölçümü olmadan **`COMPLETED` ilan edilmez.** Aşama 5'i `COMPLETED` yapmak da gerekçelendirilemez (5B/5C yok).

## 10. Geliştirme sırasında bulunan hatalar (dürüst kayıt)

* **Ölçüm hatası (düzeltildi):** model seçimi ölçümlerimin ilki 1280×720'de YuNet için ≈ 29 ms verdi; kayıtlı video CLI'sında ≈ 11 ms çıktı. Neden: ilk betiklerim örnek görüntü yolu için `ultralytics` içe aktarıyordu (OpenCV'yi tek iş parçacığına düşürüyor). Doğrulandı ve **iki durum ayrı** raporlandı (§7); benchmark artık `ultralytics`'i içe aktarmadan yol bulur (testli).
* **Test beklentisi hataları (kaynak değil; test gevşetilmeden düzeltildi):** `detect_width=50` geçersiz (alt sınır 64); sahte kamera bitince `SourceDisconnectedError` (4B'de de görülen tuzak) → `max_frames`/video kaynağı; `FakeCapture` `KeyboardInterrupt` fırlatmaz; sahte tespitçinin sayı listesi son değeri tekrarlar; en-boy oranı bozulmuş (810×1080'e ezilmiş) zidane yüzleri kaçar (bu durum sınanmaz).
* **Mevcut test güncellemesi:** `test_no_side_effects` yeni onaylı model dosyası için güncellendi (başka model dosyası hâlâ reddedilir).
* **Belge düzeltmeleri (teslimden önce):** kısmi yüz kesik oranları (%46/%90, %29/%75) ve SCRFD hücresindeki doğrulanmamış boyut ifadesi düzeltildi.

## 11. Aşama 5B'ye geçmeden önce gerekenler

1. **Kullanıcının gerçek kamera testleri** (docs/face_detection.md §13), özellikle çoklu kişi/mesafe/açı ve **yüksek kamera açısında yüz görünürlüğü**; sonuçlar `OT-5A-001/004/005`'e işlenmeli.
2. **Karar:** YOLO + yüz tespiti aynı süreçte çalışacaksa OpenCV iş parçacığı yönetimi (`OT-5A-006`).
3. **Tanım:** yüz kutusu ↔ kişi/iz kutusu ilişkilendirme kuralı ve **belirsizlik politikası** (belirsizde eşleştirmeme); çok kareli doğrulama; Track ID ≠ çalışan kimliği; yanlış pozitif iz üzerindeki yüzün çalışana bağlanmaması (`stage5_design_notes`).
4. **Karar:** küçük, izinli, etiketli bir yüz/ID değerlendirme kümesi gerekip gerekmediği (`OT-5A-002/003`); ID değişimi sıklığı 4B'de ölçülmedi (`OT-4B-003`).
5. **Karar:** kullanılmayan `mediapipe` bağımlılığı (`OT-5A-007`); 5A/5B/5C ayrımının onayı.

## 12. Güncel durum

* **Durum: `READY_FOR_MANUAL_VERIFICATION`** (5A) · Aşama 5 `IN_PROGRESS` · 5B/5C `planned` (başlatılmadı) · Aşama 6/7 işleri yapılmadı.
* Kod/belge değişiklikleri **commit'lenmedi**; kapanış ve Git işlemleri için kullanıcı kararı beklenir.

## 13. Güncelleme: kullanıcı tarafından bildirilen canlı kamera doğrulaması (kayıt tarihi 2026-11-15)
**Kaynak/tür:** kullanıcı bildirimi (nitel). Gerçek MacBook kamerasında YuNet yüz tespiti çalıştırıldı: **kamera başarıyla açıldı; canlı görüntüleme penceresi çalıştı; gerçek insan yüzü tespit edildi; yüz kutusu ekranda gösterildi.** Sayısal değer (güven, FPS, çıkarım süresi), ekran görüntüsü ve kullanılan betik bildirilmedi; bağımsız doğruluk ölçümü değildir.
* **Doğrulananlar:** kamera erişimi + canlı pencere + tek gerçek yüzün tespiti (docs/face_detection.md §13.1 test 1, **kısmen**).
* **Hâlâ yapılmadı/bildirilmedi:** gerçek kamerada **çoklu yüz (2–3 kişi)**, mesafe, açı, kısmi görünüm, yüz yok, yanlış pozitif (fotoğraf/ekran/poster), ışık, 5A betiğiyle çıkış yöntemleri (Q/Esc/kırmızı düğme/Ctrl+C), `--detect-width` karşılaştırması, canlı FPS/kare→sonuç gecikmesi (`OT-5A-004`), yüksek kamera açısı (`OT-5A-005`), doğruluk ölçümü (`OT-5A-002/003`).
* Kamerasız gerçek model ve sentetik montaj kanıtları (§6) **ayrı tutulur**; gerçek kamerada çoklu yüz testi yapılmış gibi sunulmaz. 5B canlı ekran görüntülerindeki YuNet ≈ 14 ms **5B hattının** değeridir; 5A betiğinin canlı performansı ölçülmüş sayılmaz.
* **Durum değişmedi:** 5A `READY_FOR_MANUAL_VERIFICATION`, **`COMPLETED` DEĞİL**; Aşama 5 `IN_PROGRESS`. `OT-5A-001` durumu `partially_performed`; diğer 5A açık görevleri ve `KL-5A-001…010` aynen geçerlidir. Kabul kriterleri (§9) düzeyleri değişmedi (kriter 2 ve 5'teki "gerçek kamera" eksikleri kısmen azaldı; çoklu yüz/çıkış yolları eksik kaldı).
