# Aşama 4B — ByteTrack ile Çoklu İnsan Takibi: Sonuç Raporu

**Tarih:** 2026-11-12 · **Durum: `READY_FOR_MANUAL_VERIFICATION`** (kod, otomatik testler, gerçek ByteTrack ve gerçek YOLO + ByteTrack entegrasyon testleri, kayıtlı/sentetik video ölçümleri tamam; **gerçek MacBook kamerası ile manuel testler yalnızca kısmen yapıldı** — §24; üç kişi, örtüşme, yer değiştirme, ID switch, kadrajdan çıkma, yüksek açı ve Esc/kırmızı düğme/Ctrl+C çıkışları yapılmadı) — `COMPLETED` değildir.
**2026-11-12 güncellemesi (§24–§28):** kullanıcının gerçek kamera sonuçları eklendi (pencereli MPS 2924 kare 16,3 FPS; penceresiz CPU ve MPS 29,9 FPS); **panodaki basılı fotoğraf insan sanıldı ve Track ID aldı (FP-4B-001)**, iki gerçek kişi varken dört aktif iz ve çakışan büyük kutular görüldü; kaynak (YOLO mu ByteTrack mi) **henüz ayrıştırılmadı**, tanı planı hazırlandı; **hiçbir eşik/varsayılan değiştirilmedi**.
**Aşama 4 genel durumu: `IN_PROGRESS`** (4A `COMPLETED`; 4B kullanıcı doğrulamasını bekliyor) · **Teknik belge:** [docs/person_tracking.md](../../docs/person_tracking.md) · **Git:** commit/push yapılmadı.

> **Özet.** Aşama 4A'nın `PersonDetector` çıktıları (`DetectionResult`), kurulu `ultralytics==8.4.210`'daki **gerçek ByteTrack** (`BYTETracker`) algoritmasına küçük bir adaptörle (`PersonTracker`) verilir ve her aktif insana **geçici, oturum kapsamlı Track ID** atanır.
> **YOLO her karede yalnızca bir kez çalışır** (takip ek çıkarım yapmaz); canlı pencerede kutu + `ID n`, `DETECTIONS` / `ACTIVE TRACKS` / `LOST TRACKS` ayrı sayılarıyla gösterilir. Track ID bir çalışan kimliği değildir.
> Yeni 287 otomatik test geçti (toplam **883**: önceki 596 korundu). Gerçek YOLO11n + gerçek ByteTrack zinciri CPU ve MPS'te çevrimdışı çalıştırıldı. ByteTrack güncellemesi ≈ **0,4 ms/kare** (YOLO çıkarımının %2–4'ü).

## Gerçekte ne var? (kanıt türlerine göre ayrılmış özet)

| Başlık | İçerik |
|---|---|
| **Gerçekte çalışanlar** | `PersonTracker` + gerçek ByteTrack; kare başına tek YOLO çağrısı; Track ID yaşam döngüsü (onay, kayıp tamponu, oturum/sıfırlama); canlı görselleştirme; `run_person_tracking.py` CLI (kamera/video, CPU/MPS, penceresiz); `lap` eksikse temiz hata (otomatik kurulum yok); iz özeti çıktısı; benchmark betiği |
| **Test edilenler** | 287 yeni test: 203 mock/birim + 62 gerçek ByteTrack (sahte tespit) + 20 gerçek YOLO + ByteTrack (örnek görüntü/sentetik montaj; CPU 12, MPS 8) + 2 isteğe bağlı gerçek pencere. Aşama 2 bütünlüğü 12/12, ortam denetimi başarılı |
| **Ölçülenler** | Kayıtlı (sentetik montaj) videoda CPU/MPS: FPS, çıkarım, **ByteTrack süresi**, kare→sonuç gecikmesi, çizim; ByteTrack'in kişi sayısıyla ölçeklenmesi; altı eşik yapılandırmasının gerçek YOLO ile karşılaştırması |
| **Yapılamayanlar** | Claude ortamında gerçek kamera testi (izin yok). Kullanıcı tarafında **yapılmayanlar:** üç kişi, örtüşme, yer değiştirme, ID switch, kadrajdan çıkma, yüksek açı, Esc/kırmızı düğme/Ctrl+C çıkışları, yorgan testi. Ayrıca IDF1/HOTA/MOTA (etiketli veri yok); canlı kare→sonuç gecikmesi ve konsolun okuma/çizim/gösterme satırları (bildirilmedi); ≥3 kişilik gerçek video |
| **Kullanıcının canlı testleri (2026-11-12)** *[Kullanıcı]* | Pencereli MPS 1280×720, 2924 kare, 178,9 sn, **16,3 FPS**, YOLO 17,7 ms, ByteTrack 0,5 ms, 30 farklı ID, Q ile sorunsuz çıkış; penceresiz CPU 29,9 FPS (YOLO 18,5 / ByteTrack 0,2 ms) ve MPS 29,9 FPS (14,7 / 0,4 ms), her ikisinde 2 ID. **Poster yanlış pozitifi (ID 167), dört aktif iz, çakışan büyük kutular.** Ayrıntı §24–§28 |
| **Bilinen sınırlamalar** | §20 — özellikle: ID'ler konuma yapışır (görünüm kullanılmaz), kalıcı yanlış pozitif iz alır, kenardan çıkıp dönen kişi yeni ID alır, düşük güven kurtarması varsayılanda fiilen kapalı |
| **Kullanıcının yapacağı manuel testler** | §21 (durum sütunuyla) ve poster/çakışan kutu tanı planı §26 |

**Kanıt türleri bu raporda ayrı tutulmuştur:** *[Mock]* = sahte bağımlılıklarla birim testi; *[Gerçek ByteTrack]* = gerçek `BYTETracker`, sahte (betiklenmiş) tespit; *[Gerçek YOLO+ByteTrack]* = gerçek `yolo11n.pt` + gerçek ByteTrack, örnek görüntü veya sentetik montaj; *[Ölçüldü]* = Claude'un çalıştırıp sonucunu gördüğü; *[Sentetik]* = Ultralytics örnek görüntülerinden montaj/betikli dizi (hız/entegrasyon amaçlı, **doğruluk çıkarılamaz**);
*[Bekliyor]* = yapılmamış, kullanıcı görevi. Yapılmayan hiçbir test başarılı gösterilmemiştir.

---

## 1. Aşama 4B'nin amacı ve kapsamı

Amaç: 4A'nın insan tespit sonuçlarını ByteTrack'e aktarıp kamera/video karelerindeki birden fazla insanı kareler arasında takip etmek (geçici Track ID, ID sürekliliği, kısa kayıp/örtüşme yönetimi, kadraja giriş/çıkış, gerçek zamanlı çalışma, Aşama 5'e aktarılabilir veri yapıları).
**Kapsam dışı (yapılmadı):** yeni model eğitimi, yüz tespiti/tanıma, çalışan kaydı/kimlik tanıma, KKD, hayvan tespiti, sensör/mesai modu, güvenlik ihlali kararı, veritabanı/panel, RTSP/IP kamera, BoT-SORT, görüntü kaydı, Aşama 5.

## 2. Devralınan Aşama 4A durumu

| Konu | Durum *[Ölçüldü]* |
|---|---|
| Git | `main`, HEAD `c51ae14`, çalışma ağacı temiz (başlangıçta) |
| Aşama durumları | 1, 2, 3, 4A `COMPLETED`; 4B `PLANNED`; Aşama 4 `IN_PROGRESS` (`configs/project_stages.yaml`, `reports/detection/STAGE_4A_REPORT.md`, `docs/*`) |
| Testler (başlangıç) | `unittest discover -s tests`: **596 test, OK, 5 atlandı** (isteğe bağlı GUI) |
| Ortam | Python 3.11.17, torch 2.15.0 (MPS), ultralytics 8.4.210, NumPy 2.4.6, OpenCV 5.0.0; **`lap` ve `scipy` kurulu değildi** |
| Model | `models/human_detection/yolo11n.pt`, SHA-256 `0ebbc80d…ee1`, Git dışı (`.gitignore:75`) |
| 4A API | `run_detection(source, detector, window, *, max_frames, meter, clock, on_result)`; `on_result(paket, sonuç)` `None` döndürür, çizim `draw_person_detections`'a sabit (§6) |
| Açık görevler (aynen korundu) | `FP-4A-001` yorgan yanlış pozitifi (açık, bilinen sınırlama); `OT-4A-002` bağımsız doğruluk değerlendirmesi (açık); `OT-4A-003` gerçek IP kamera (`DEFERRED_TO_SYSTEM_INTEGRATION`); `OT-4A-004 / OBS-4A-001` ilk oturumdaki 7 tespit gözlemi (açık). `OT-4A-001` (`conf=0.35`) kullanıcı tarafından yapıldı (`TEST_PERFORMED`), yeniden bekleyen olarak gösterilmedi |

4A raporundaki tamamlanmış sonuçlar değiştirilmedi.

## 3. Kullanılan ByteTrack sürümü ve API'si

Ultralytics **8.4.210** (`ultralytics.trackers.byte_tracker.BYTETracker`), kaynak kodundan okundu. Ayrıntılı tablo: [docs/person_tracking.md §3](../../docs/person_tracking.md). Öne çıkanlar: `update(boxes)` girdisi `conf/cls/xywh` veren NumPy tabanlı nesnedir (`ultralytics.engine.results.Boxes`);
çıktı `(N,8)` `[x1,y1,x2,y2,track_id,score,cls,idx]`, **iz yoksa 1B `(0,)`**; çıktı kutusu Kalman tahminidir; `reset()` kimlikleri 1'den başlatır; kimlik sayacı takipçiye özeldir; `matching.py` modül düzeyinde `import lap` yapar ve yoksa **`pip` ile otomatik kurmaya çalışır**; `scipy` gerekmez.
`BYTETracker` dahili bir API'dir (sürüm bağımlılığı riski: `ultralytics==8.4.210` sabitli, farklı sürümde uyarı, beklenmeyen çıktıda hata).

## 4. Bağımlılık denetimi

| Paket | Durum | Kayıt |
|---|---|---|
| `lap==0.5.14` | **Kuruldu**; `requirements.txt` + `requirements-lock.txt` | PyPI `cp311-cp311-macosx_11_0_arm64` wheel; SHA-256 `9c99688a55a0f4a676d86786080ae192272b9f951cf9e4e4deada43b6927d47d`; BSD-2-Clause; tek bağımlılık `numpy>=1.21.6`. Wheel önce yalnızca indirilip `METADATA`'sı okundu; aynı dosyadan `pip install --no-index --no-deps` ile kuruldu; `pip check` temiz; NumPy 2.4.6 ile `lap.lapjv` çalıştı *[Ölçüldü]* |
| `scipy` | **Eklenmedi** | Gerekmez (kaynak kodda doğrulandı); gereksiz bağımlılık yok |

Ağ **yalnızca** bu kurulumda kullanıldı (sürüm sorgusu + wheel indirme). Takip çalışırken ağ/indirme/kurulum yoktur: `check_tracker_dependencies()` Ultralytics'i import etmeden önce `lap`'ı denetler (eksikse kurulum komutuyla temiz hata; otomatik pip yolu hiç açılmaz), `YOLO_OFFLINE=1` ve `YOLO_AUTOINSTALL=False` ayarlanır.
`socket.connect`/`subprocess` yasaklanmış hâlde gerçek takipçi, gerçek YOLO+ByteTrack ve "lap yokmuş" CLI testleri geçti *[Gerçek ByteTrack / Gerçek YOLO+ByteTrack]*. Ağ erişimi olsaydı da hata gizlenmez/sahte başarı üretilmez. `yolo11n.pt` aynı (SHA-256 testle doğrulanır), başka ağırlık yüklenmez.

## 5. Mimari ve dosya yapısı

Mevcut `src/isg_vision/tracking/` iskeleti kullanıldı (ikinci paket açılmadı): `errors.py`, `config.py`, `results.py`, `bytetrack_adapter.py`, `metrics.py`, `visualization.py`, `pipeline.py`, `cli.py`, `benchmark.py`; betikler `scripts/run_person_tracking.py`, `scripts/benchmark_person_tracking.py`.
Tespit, takip, görselleştirme ve döngü ayrıdır. Değişen mevcut dosyalar yalnızca: `requirements.txt`, `requirements-lock.txt` (+ `lap`), `src/isg_vision/tracking/__init__.py` (yer tutucu → paket), `tests/test_no_side_effects.py` (yeni modül adları listeye eklendi; hiçbir kontrol gevşetilmedi) ve belgeler.
`git diff HEAD` Aşama 1–4A kaynakları, betikleri, `data/`, `models/` ve 4A/3/2 raporlarında **boştur** *[Ölçüldü]*.

## 6. YOLO → ByteTrack entegrasyonu

`YOLO.track()` **kullanılmadı**: o, `predict`'e geri çağrımlarla bağlanır ve 4A'nın `PersonDetector.detect`'iyle aynı kareyi **ikinci kez** YOLO'ya göndermek anlamına gelirdi. Bunun yerine `BYTETracker` doğrudan kullanılır; `DetectionResult` → `Boxes[N,6]` dönüşümü adaptördedir, görüntü verilmez (kopyalanmaz).

```text
FramePacket → PersonDetector.detect() [YOLO: kare başına TEK çağrı] → DetectionResult → PersonTracker.update() [ByteTrack] → TrackingResult → draw_person_tracks → PreviewWindow
```

**4A `on_result` kancası** doğrulandı ve yeterli bulunmadı: `(paket, sonuç)` alır, `None` döndürür, çizim sabittir, özet takip süresini taşımaz. 4A koduna dokunmadan, aynı yapıda kardeş döngü `run_tracking` yazıldı (aynı `FpsMeter`, `LatencyStats`, çıkış tuşu/pencere mantığı yeniden kullanıldı); 4A betiği ve `run_detection` değişmedi.
**Kare başına tek YOLO çağrısı iki seviyede sınandı:** gerçek `PersonDetector` + sahte model (`predict` sayısı = kare sayısı) *[Mock]* ve gerçek YOLO (`predict` sarmalanıp sayıldı: 45 kare → 45 çağrı) *[Gerçek YOLO+ByteTrack]*; ayrıca bir `ast` testi adaptörün modeli çağırmadığını/import etmediğini koddan doğrular.

## 7. Track ID yaşam döngüsü

Ayrıntı: [docs/person_tracking.md §7](../../docs/person_tracking.md). Gerçek ByteTrack'te **ölçülen** davranışlar *[Gerçek ByteTrack]*: yeni iz ardışık **iki karede** eşleşmeden görüntülenmez (oturumun 1. karesi istisna); onaysız silinen iz de numara tüketir; kayıp izler `LostTrack` olarak tutulur ve **aktif sayılmaz/çizilmez**;
etkin kurtarma penceresi `track_buffer + 1` kare (`track_buffer + 2` kayıpta yeni ID); kayıp iz Kalman tahminiyle son hızıyla ilerler; oturum sıfırlamada (kaynak, çözünürlük, elle) ID'ler 1'den başlar ve `session_id` artar (benzersiz anahtar `(session_id, track_id)`); tekrar/geri giden kare `FrameOrderError` verir ve durumu değiştirmez; aynı karede yinelenen `track_id` çalışma anında hata sayılır.

## 8. Takip yapılandırma eşikleri

Varsayılanlar kurulu `bytetrack.yaml` ile birebir aynı (testle doğrulanır): `track_high_thresh 0.25`, `track_low_thresh 0.1`, `new_track_thresh 0.25`, `track_buffer 30 (kare)`, `match_thresh 0.8`, `fuse_score True`. Evrensel optimum değil, makul başlangıç. 4A `DetectionConfig.conf=0.25` değişmedi; `0.35` varsayılan yapılmadı.
CLI'de doğrulanmış aralıklarla açıktır (`--track-high-thresh`, `--track-low-thresh`, `--new-track-thresh`, `--track-buffer`, `--match-thresh`, `--no-fuse-score`); `low >= high` veya `new < high` gibi tutarsızlıklar çıkış kodu 2 ile reddedilir (yeni iz yalnızca yüksek güvenli tespitten doğduğu için `new < high` ayarı sessizce etkisiz kalırdı).

## 9. Düşük confidence tespitleriyle ilgili karar

**Sorular ve yanıtlar** (ayrıntı: [docs/person_tracking.md §6.1](../../docs/person_tracking.md)):

1. **`conf=0.25` düşük güvenli eşleştirmeyi nasıl etkiliyor?** YOLO `conf` altını ByteTrack'e hiç vermez; `conf >= track_high_thresh` ise ikinci (düşük güven) aşama **fiilen kapalıdır**. Varsayılan (0,25/0,25) tam bu durumdadır; CLI bunu her başlangıçta yazar.
2. **Takip eşiği gösterim eşiğinden ayrı olmalı mı?** Tek YOLO çağrısı = tek `conf`; "gösterim" zaten onaylanmış aktif izlerdir. Ayrı bir ayar eklenmedi; `--conf` ile `--track-*` eşikleri birlikte ayarlanır ve etkileşim çalışma anında açıklanır.
3. **Düşük eşik yanlış pozitifi artırır mı?** Tespit sayısı artar (6,52 → 8,58/kare); düşük güvenli tespit yeni iz doğurmaz ama kısa/şüpheli izlerin ömrünü uzatır (131 → 180 aktif kare). Yararı bu videoda gösterilemedi → varsayılan yapılmadı.
4. **Yorgan riski nasıl izlenecek?** CLI iz özeti (ilk-son kare, aktif kare, süre, güven); kısa/düşük güvenli izler aday. Gerçek yorgan testi kullanıcıda (§21).
5. **Yeni iz eşiği ↔ sürdürme eşiği:** ayrı ayarlar (`--new-track-thresh` / `--track-high-thresh`); iz 0,9 ile başlayıp 0,3 ile sürebilir, 0,4'lük tespit `new=0.5` iken iz başlatamaz *[Gerçek ByteTrack]*.

**Gerçek YOLO ile karşılaştırma** *[Gerçek YOLO+ByteTrack, Sentetik; CPU; 300 kare; etiket YOK; yalnızca gözlem]* — 4A'nın sentetik montaj videosu (kare başına ~6,5 kişi):

| Yapılandırma (CLI karşılığı) | Tespit/kare | Farklı ID | Uzun iz (≥250 kare) | Kısa iz (<100 kare) | Kısa izlerin toplam kare | Düşük güven aşaması |
|---|---|---|---|---|---|---|
| A varsayılan | 6,52 | 10 | 6 | 4 | 131 | kapalı |
| B `--conf 0.35 --track-high-thresh 0.35 --new-track-thresh 0.35` | 6,23 | 9 | 6 | 3 | 68 | kapalı |
| C `--new-track-thresh 0.6` | 6,52 | 6 | **5** | 1 | 27 | kapalı |
| D `--conf 0.10` | 8,58 | 10 | 6 | 4 | 180 | AÇIK |
| E `--conf 0.10 --track-high-thresh 0.35 --new-track-thresh 0.35` | 8,58 | 9 | 6 | 3 | 107 | AÇIK |
| F `--conf 0.10 --track-high-thresh 0.50 --new-track-thresh 0.60` | 8,58 | 6 | 5 | 1 | 37 | AÇIK |

Komutlar: `.venv/bin/python scripts/run_person_tracking.py --source video --video-path results/person_detection/sentetik_cok_kisili_1280x720.mp4 --device cpu --no-display <seçenekler>`; C satırı CLI'den yeniden üretilip doğrulandı (6 ID).
**Karar:** varsayılan yapılandırma **değiştirilmedi** (güvenli, anlaşılır: 4A `conf` + Ultralytics varsayılanları). **Yorum sınırı:** B ve C kısa izleri azaltır ama yok etmez (B, 4A `conf=0.35` gözlemiyle uyumlu: sorun çözülmedi); **C ayrıca 6 kararlı izden birini (kenarda kısmen görünen, ortalama güveni 0,37 olan kişi) kaybettirir** — yanlış pozitif azaltmanın gerçek kişi kaçırma maliyeti vardır.
"Kısa iz" yanlış pozitif **demek değildir** (ID 22/32 gözle bakıldı: zidane resminin bulanık arka planında küçük, düşük güvenli kutular; gerçek bulanık kişi mi yanlış pozitif mi **doğrulanamadı**; ID 11/26 yalnızca kutu konumundan incelendi). Hiçbir eşik "optimum" ilan edilmedi; tek örnekten eşik çıkarılmadı.

## 10. Canlı görüntüleme

`draw_person_tracks` (girdinin **kopyasına** çizer): üst şeritte `DETECTIONS`, `ACTIVE TRACKS`, `LOST TRACKS` (ayrı satırlar; kayıp iz soluk), `SOURCE`, `RESOLUTION`; sağda `FPS`, `INFERENCE`, `TRACKING` (ayrı süreler), `DEVICE`; aktif her kişi için kutu + `ID n 0.87`; alt şeritte çıkış ipucu ve `ID = temporary track number (not an identity)`.
Renk `track_id`'ye göre sabit ama **kimlik/doğruluk anlamı taşımaz** (12 renk). Kayıp/onaysız izler çizilmez. Boş kareler, çok iz, kenarlardaki kutular, küçük görüntüler hata vermez *[Mock]*. Sentetik montajda çizilmiş kareler gözle incelendi: sayılar/ID'ler beklenen biçimde (DETECTIONS 7 / ACTIVE 7 / LOST 0 vb.).
Q / Esc / kırmızı kapatma / Ctrl+C: kamera paketinin `PreviewWindow`'u yeniden kullanıldı; sahte pencereyle Q, Esc, kapatma, `KeyboardInterrupt` yolları ve gerçek Cocoa penceresinde **kırmızı düğme tıklaması** (isteğe bağlı GUI testi, `ISG_GUI_TESTS=1`) 4B döngüsünü bitirdi ve kaynaklar serbest kaldı *[Ölçüldü]*. Gerçek MacBook kamerasında dört çıkış yöntemi **henüz kullanıcı tarafından denenmedi** (§21).

## 11. CLI komutları

```bash
.venv/bin/python scripts/run_person_tracking.py --source camera --camera-index 0 --device mps
.venv/bin/python scripts/run_person_tracking.py --source camera --camera-index 0 --device cpu
.venv/bin/python scripts/run_person_tracking.py --source video --video-path /yol/video.mp4 --device mps
.venv/bin/python scripts/run_person_tracking.py --source camera --camera-index 0 --device mps --no-display --max-frames 300
.venv/bin/python scripts/run_person_tracking.py --source camera --camera-index 0 --device mps --conf 0.25
.venv/bin/python scripts/run_person_tracking.py --help
```

Çıkış kodları 4A ile aynı (0 / 1 / 2 / 130). Kaynak/model seçenekleri `run_person_detection.py` ile birebir aynıdır (4A parser'ı yeniden kullanılır); 4A betiğinin yardım metni ve davranışı değişmedi (testle doğrulanır). Sıra: YOLO yükle → takipçi yükle → kaynak aç → ısın → pencere → döngü; model/takipçi başarısızsa kamera hiç açılmaz.
**Gerçek koşular** *[Gerçek YOLO+ByteTrack, Sentetik]*: `--source video ... --device mps|cpu --no-display` her ikisi `exit=0`, iz tabloları iki cihazda birebir aynı (10 ID). Kamera komutları Claude ortamında **çalıştırılmadı** (izin yok, aşılmaya çalışılmadı).

## 12. Otomatik testler

`.venv/bin/python -m unittest discover -s tests`: **883 test, OK, 7 atlandı** (hepsi isteğe bağlı GUI: Aşama 3'ten 3, 4A'dan 2, 4B'den 2); `ISG_GUI_TESTS=1` ile **883 test, OK, 0 atlandı**. Önceki 596 test korundu *[Ölçüldü]*. Yeni **287** test, tür ayrımıyla:

| Tür | Dosya (test) | Toplam |
|---|---|---|
| Mock/birim | config (17), results (19), adapter (51), visualization (21), pipeline (30), cli (47), benchmark (18) | **203** |
| Gerçek ByteTrack (sahte tespit) | bytetrack_real (59), pipeline (2), benchmark (1) | **62** |
| Gerçek YOLO + ByteTrack | real_model (CPU 12, MPS 8) | **20** |
| İsteğe bağlı gerçek pencere | pipeline (2) | **2** |

İlkeler: birim testleri internet/kamera izni gerektirmez; mock ve gerçek testler ayrı dosyalarda ve ayrı raporlanır; `lap`/ağırlık/MPS yoksa ilgili testler **açık nedenle atlanır** (sahte başarı yok); testler yalnızca geçsin diye gevşetilmedi. Beklentiler ölçümle belirlendi; ölçüm beklentimi yalanladığında **beklenti** düzeltilip bulgu belgelendi (kenardan çıkıp aynı kenardan dönen kişinin "aynı ID alır" varsayımı yanlıştı → sınırlama olarak kaydedildi; kayıp kurtarma penceresi `track_buffer+1` ölçüldü). Gevşek bir eşik (Kalman kutusu sapması, ölçülen ≈0,2 kare) sıkılaştırıldı; docstring'i taramaya takılan zayıf bir metin-tarama testi `ast` tabanlı teste çevrildi; adaptör sınıf kimliğine bağlı `isinstance` yerine öznitelik tabanlı doğrulamaya geçirildi.
Zorunlu mimari testi: aynı kare için YOLO yalnızca bir kez çağrılır (§6). `test_no_side_effects.py` yeni modülleri kapsar (import: kamera/ağ/torch/ultralytics/lap yok).

## 13. Gerçek ByteTrack testleri

*[Gerçek ByteTrack]* — `tests/test_person_tracking_bytetrack_real.py` (59 test; gerçek `BYTETracker`, sahte tespit dizileri, YOLO/kamera yok) ve gerçek YOLO+ByteTrack: `tests/test_person_tracking_real_model.py` (20 test).
Gerçek YOLO + ByteTrack: `bus.jpg` (4 kişi: güven 0,89/0,88/0,86/0,62) ve `zidane.jpg` (2 kişi) sabit görüntülerinde ID'ler kareler boyunca değişmedi, tespit sayısıyla aktif iz sayısı eşit, takip kutusu ham kutudan <3 px sapıyor; gerçek kişi kesitlerinin sentetik tuvalde hareketi (1 ve 2 kişi, zıt yönde, kısa boş karelerden sonra) ID'lerini korudu;
boş sahne hatasız; kaynak değişimi yeni oturum; sentetik montaj videosunda 45 kare = 45 YOLO çağrısı. **Bu testler gerçek kamera, IDF1/HOTA veya "%85 hedefi" kanıtı değildir.** ≥3 kişilik gerçek/izinli video Claude ortamında yoktu: test yapılmış gösterilmedi, rastgele görüntü indirilmedi; kullanıcı testi olarak §21'e yazıldı.

## 14. Tek kişi ve çoklu kişi sonuçları *[Gerçek ByteTrack; idealize sentetik]*

| Senaryo | Sonuç |
|---|---|
| A — Tek kişi hareket (40 kare, 5 px/kare) | ID 1 tüm karelerde; ID switch 0; takip kutusu hareketle aynı yönde, en büyük sapma ≈ 0,2 kare hareketi (oturduktan sonra ≈ 0,01). Sonradan beliren kişi ilk görüldüğü karede onaysız (tentative), ikinci karede ID alır |
| B — Üç kişi | 3 bağımsız iz, 3 farklı ID, aynı karede yinelenen ID yok; tespit sırası her karede döndürülse de ID'ler kişiye bağlı kalır (eşleme `detection_index` ile); **ID'ler soldan sağa değil**, ilk karedeki tespit (güven) sırasıyla verilir; 30 kişi de benzersiz ID aldı |
| H — Yorgan türü sahte yanlış pozitif | §18 |
| Gerçek YOLO + gerçek kişi kesiti | 1 kişi: 30/30 karede aynı ID; 2 kişi (dikey ayrık, zıt yön): 50/50 karede (ID 1, 2), ID switch 0 |

## 15. ID switch ve örtüşme gözlemleri *[Gerçek ByteTrack; idealize sentetik]*

| Senaryo | Sonuç |
|---|---|
| C — Yer değiştirme, düzgün hareket (farklı derinlik; aynı satırda 3 px gürültüyle) | ID switch **0** (bu **idealize** dizilerde; gerçek videoda switch olabilir) |
| C — **Anlık yer değiştirme** (iki kişi ardışık iki karede konum değiştirir) | **2 ID switch + 2 devralma**; metrik yakalar, aynı karede yinelenen ID yok. ByteTrack görünüm kullanmaz: numaralar konuma yapışır |
| D — Biri diğerinin önünden geçer (ikisi de tespit ediliyor) | Takip sürdü, switch 0 |
| D — Örtülen kişi 8 kare tespit edilemiyor | Örtülen **aktif sayılmaz** (`LOST TRACKS: 1`, `frames_since_seen` 1…8), örtü bitince **aynı ID**; switch 0 |
| D — NMS iki kutuyu tek kutuya indiriyor | Kaybolan iz aynı ID ile döndü |
| E — Kısa kayıp (≤ `track_buffer`) ve aynı konumda dönüş | Aynı ID |
| E — `track_buffer+2` veya daha uzun kayıp | Yeni ID (ID switch olarak raporlanır, gizlenmez); kayıp sayısı tampon dolunca 0 |
| E — Uzakta beliren yeni kişi | Yeni ID (eski izi devralmaz) |
| E — **Tampon içinde kaybolanın konumuna giren yeni kişi** | **Eski ID'yi devralır** (bilinen sınırlama; takeover olarak sayılır) |
| E — **Kadrajdan çıkıp aynı kenardan dönen kişi** | Tampon dolmadan bile **yeni ID** (2–10 px/kare × 6/15/30 kare kayıp, hepsinde aynı); nedeni: kayıp iz son hızıyla kadraj dışına ilerler (bilinen sınırlama) |
| E — Hareket ederken ortada kaybolup tahmin edilen konumda dönen | ID korunur (3/6/10 px/kare × 5/15 kare); ters yönde dönen yeni ID alır |
| F — Hiç insan yok (50 kare) | Hata yok; hepsi 0; kimse yokken `ACTIVE TRACKS: 0`, kayıp izler ayrı sayılır |
| G — 1, 2, 3, 8 karelik kaçırma | Aynı ID, yeni ID doğmaz; kaçırılan karede iz **aktif gösterilmez** (kayıp tamponunda), eşleşince döner |
| I — Kaynak/oturum değişimi | Önceki iz ve kayıp tampon **taşınmaz**; ID'ler 1'den, `session_id` artar, `reset_reason` işaretlenir; farklı çözünürlük de sıfırlar; ikinci takipçi örneği ID sayacını paylaşmaz |
| J — Hatalı girdi | Görüntü dışı kutu/geçersiz boyut/yapı: `InvalidTrackingInputError`; tekrar/geri kare: `FrameOrderError`; ikisinde de durum değişmez (reddedilen kare izleri yaşlandırmaz); backend çökmesi/beklenmeyen çıktı: `TrackerBackendError`; eksik `lap`: `TrackerDependencyError`; geçersiz `bbox`/`confidence` zaten `PersonDetection` oluşturulurken reddedilir |

**Hızlı hareket:** ardışık karelerde kutular yeterince örtüşmezse (kare başına kutu genişliği kadar kayma) iz hiç oluşmadı (idealize dizi). 30 FPS'te yürüyen kişide bu beklenmez; düşük FPS veya kameraya çok yakın hızlı hareket için gerçek testte gözlenmelidir.

## 16. CPU/MPS performansı *[Ölçüldü, Sentetik video; yalnızca bu makine ve oturum]*

Aynı video (4A'nın sentetik montajı: 300 kare, 1280×720, kare başına ort. 6,52 tespit), aynı model/eşikler, 5 geçiş, cihaz ve mod sırası döndürülür; yükleme/ısınma/burn-in ayrı (yöntem: `tracking/benchmark.py`). Medyanlar (`results/person_tracking/benchmark_20261112_101837.json`, Git dışı):

| Cihaz | FPS yalnızca-tespit | FPS tespit+takip | FPS +çizim | YOLO çıkarım (ms) | **ByteTrack (ms)** | Çizim (ms) | Kare→sonuç (ms) | Model yükleme / ısınma (sn) |
|---|---|---|---|---|---|---|---|---|
| CPU | 51,3 | 49,3 | 49,1 | 18,6 | **0,41** | 0,45 | 19,1 (takipsiz 18,7) | 0,45 / 0,51 |
| MPS | 101,2 | 92,2 | 89,1 | 9,9 | **0,40** | 0,41 | 10,4 (takipsiz 9,4) | 0,06 / 0,31 |

* **ByteTrack güncellemesi ≈ 0,4 ms/kare** (geçişler arası 0,32–0,42 ms, kararlı): YOLO çıkarımının **%2–4'ü**. Kare okuma medyanı ≈ 0,3 ms. Takipçi yükleme ≈ 5 ms (Ultralytics zaten yüklüydü; yeni süreçte ilk import ≈ 0,5 sn).
* FPS farkından türetilen ek yük (CPU 0,78, MPS 0,96 ms/kare) doğrudan ölçülen 0,4 ms'den büyüktür; geçişler arası kare süresi gürültüsü (CPU yalnızca-tespit 18,8–20,0 ms; MPS 9,2–10,5 ms) bu fark mertebesindedir. **Güvenilir sayı doğrudan ölçülen ByteTrack süresidir**; FPS farkı gürültü içindedir.
* **Ölçeklenme** (YOLO yok, CPU, betiklenmiş tespit; iki ölçüm): 1 kişi 0,10 ms · 5: 0,20–0,21 · 10: 0,32–0,34 · 20: 0,55–0,56 · 50: 1,23–1,26 · 100: 2,40–2,48 ms (≈ doğrusal, ≈ 0,024 ms/kişi).
* Gerçek CLI, penceresiz, tek koşu: MPS 103,9 FPS (çıkarım 9,1 / ByteTrack 0,3 / toplam 9,4 ms), CPU 51,7 FPS (18,3 / 0,4 / 18,7 ms). **Canlı kamera performansı ölçülmedi** (§21). Kesin 30 FPS zorunluluğu konmadı; darboğaz YOLO'dur, takip değil → optimizasyon yapılmadı.

## 17. Aşama 4A ile performans karşılaştırması

* **Aynı oturumda, aynı video:** takip ek yükü ≈ 0,4 ms/kare (§16) — YOLO'nun yanında ihmal edilebilir; CPU ve MPS'te uçtan uca FPS farkı gürültü düzeyinde.
* **4A'nın kayıtlı benchmark'ı** (farklı oturum; aynı video/model): CPU 46,6–52,5 FPS (çıkarım medyan 18,7–20,4 ms), MPS 100,0–110,7 FPS (8,5–9,7 ms). 4B değerleri aynı aralıklardadır.
* **4A'nın canlı kamera sonuçları** (penceresiz 30,0 FPS CPU/MPS; MPS medyan 13,4 ms, CPU 18,5 ms; pencereli 18,4–25,3) **kullanıcının testleridir ve 4B performansı olarak sunulmaz**; 4B için canlı ölçüm kullanıcıdadır.

## 18. Yorgan yanlış pozitifiyle ilgili değerlendirme (FP-4A-001)

Gerçek yorgan görüntüsü kaydedilmedi; **olay birebir yeniden üretilmedi**. Kontrollü sahte tespit dizileriyle **gerçek ByteTrack** *[Gerçek ByteTrack]*:

| Desen | Sonuç |
|---|---|
| Tek karelik yanlış tespit (oturumun 1. karesi değil) | **Hiç iz görüntülenmez** (güven 0,3/0,6/0,95 fark etmez); ID numarası tüketilir |
| Oturumun **1. karesinde** tek karelik | Bir kare görüntülenir (istisna), sonra tamponda kaybolur |
| 2–3 kare süren | 1–2 kare görüntülenir; `track_buffer` kare kayıp tamponunda kalır, sonra silinir |
| 1-0-1-0 titreşim | Hiç onaylanmaz |
| 2 var / 1 yok tekrarlı | Onaylanır, **tek kararlı ID** ile sürer |
| **Uzun süre devam eden statik** | **Sürekli ID alır** (60 karede kesintisiz); ByteTrack kalıcı yanlış pozitifi ELEMEZ |
| Düşük güvenli (0,15) | Yeni iz doğurmaz; mevcut izi sürdürür |

**Cevap:** *Bu tespitler Track ID oluşturabiliyor mu?* — Yalnızca **≥ 2 ardışık karede** (30 FPS'te ≥ ~67 ms) tespit edilirse (ve oturumun ilk karesinde 1 karede); tek karelik tespitler iz almaz. *Hangi koşullarda oluşur/silinir?* — Yukarıdaki tablo; kayıp iz `track_buffer(+1)` kare sonra silinir.
**ByteTrack yanlış pozitifi tamamen engellemez**; kullanıcının gözlemindeki "kısa süreliğine" yorgan algısı ≥ 2 ardışık karede ise ID alır; kalıcıysa kalıcı ID tutar. Gerçek ortamdaki etkisi ve `--new-track-thresh 0.6`'nın gerçek yorgan/gerçek kişi üzerindeki sonucu **ölçülmedi** (§21, Manuel Test 7). Gerçek YOLO ile sentetik montajda kısa, düşük güvenli izler gözlendi (§9); doğrulanamadığı için yanlış pozitif sayılmadı.

## 19. Yüksek kamera açısı durumu

**Ölçülmedi.** Gerçek kamera ve ≈ 2–3 m yükseklik testi yapılamadı (Claude ortamında kamera izni yok; kullanıcı testi bekleniyor, Manuel Test 6). Kodda kamera açısı/çözünürlüğü sabit varsayılmaz (çözünürlük değişimi oturumu sıfırlar); sentetik testler açı etkisini temsil edemez. 4A'daki ≈ 2 m ön testi yalnızca tespit içindi. Yüksek açıda ID sürekliliği, kutu şekli değişimi ve kısmi vücut görünümü etkileri bilinmiyor; kenarda kısmen görünen kişilerin ID kararlılığı §15'teki sınırlamadan etkilenebilir.

## 20. Bilinen sınırlamalar

1. **Track ID kimlik değildir**; ByteTrack görünüm kullanmaz → hızlı yer değiştirmede numaralar konuma yapışır (ID switch), tampon içinde kaybolanın konumuna giren eski ID'yi devralır.
2. **Kalıcı yanlış pozitif iz alır** (FP-4A-001 açık kalır; gerçek yorganla davranış ölçülmedi).
3. Kadrajdan çıkıp **aynı kenardan** dönen kişi tampon dolmadan bile yeni ID alır.
4. Düşük güven kurtarması varsayılanda **fiilen kapalı**; açmak yanlış pozitif ömrünü uzatır, yararı gösterilemedi.
5. `track_buffer` kare sayısıdır (FPS'e bağlı süre); oturumun 1. karesi onaysız iz alır; çıktı kutusu Kalman tahminidir; hızlı hareket (IoU düşük) iz oluşturmaz.
6. `BYTETracker`/`Boxes` dahili Ultralytics API'sidir (sürüm riski, §3). AGPL-3.0 yükümlülükleri ürünleştirmede yeniden değerlendirilmelidir.
7. Kanıt sınırı: gerçek-ByteTrack senaryoları **idealize sentetik**; gerçek-model testleri örnek görüntülerden. **IDF1/HOTA/MOTA ölçülmedi**, gerçek kamerada ID sürekliliği ölçülmedi, "%85 hedefi karşılandı" iddiası yoktur.
8. Backlog (Aşama 3): B3-3 (gecikme) yalnızca **kayıtlı videoda** ölçüldü; canlıda kare→sonuç satırı bildirilmedi. B3-4 (ileri FPS doğrulaması) **kısmen**: kullanıcı 179 sn'lik pencereli ve 300 karelik penceresiz canlı koşuları bildirdi (bağımsız zaman referansı yok). B3-2 (kare kaybı) yapılmadı. Hepsi ertelenmiş kalır.
9. Açık görevler: FP-4A-001, OT-4A-002, OT-4A-003 (`DEFERRED_TO_SYSTEM_INTEGRATION`), OT-4A-004/OBS-4A-001. Gerçek IP kamera 4B kapanış koşulu yapılmadı.
10. **Basılı insan fotoğrafı (pano/poster) insan sanılır ve Track ID alır (FP-4B-001, kullanıcının canlı gözlemi);** çakışan büyük kutular gözlendi (OBS-4B-001); kaynak ayrıştırılmadı (§26). Eşikle çözüm denenmedi/uygulanmadı (§27).
11. **Pencereli mod ağırdır:** `imshow+waitKey(1)` bu makinede tek başına ≈ 15–17 ms ölçüldü; kullanıcının pencereli canlı koşusu 16,3 FPS, penceresiz 29,9 FPS (OBS-4B-002); pencereli FPS'in tam nedeni ölçülmedi (§25.4).

## 21. Bekleyen manuel testler (kullanıcı; MacBook kamerası) — durum 2026-11-12

Ayrıntılı kontrol listeleri: [docs/person_tracking.md §17](../../docs/person_tracking.md). Kamera izni Terminal/VS Code/Claude'a verilmelidir. Gözlem yapılmadan sonuç yazılmamalıdır. **Bildirilmeyen test yapılmamış sayılır.**

| Görev | Test | Durum (kullanıcı bildirimine göre) | Komut / yöntem |
|---|---|---|---|
| `OT-4B-001` | 1 Tek kişi | **Kontrol listesi doldurulmadı** (sahnede iki gerçek kişi vardı) | `.venv/bin/python scripts/run_person_tracking.py --source camera --camera-index 0 --device mps` |
| `OT-4B-001` | 2 Üç kişi | **YAPILMADI** | aynı komut |
| `OT-4B-001` | 3 Yer değiştirme / ID switch | **YAPILMADI** | aynı komut |
| `OT-4B-001` | 4 Kısmi örtüşme | **YAPILMADI** (çakışan büyük kutular gözlendi; bu bir örtüşme senaryosu testi değildir) | aynı komut |
| `OT-4B-001` | 5 Kadrajdan çıkma | **YAPILMADI** | aynı komut |
| `OT-4B-001` | 6 Yüksek açı (≈ 2 m) | **Bildirilmedi** | 1–3 testlerini tekrarla |
| `OT-4B-002` | 7 Yorgan yanlış pozitifi (+ `--new-track-thresh 0.6`) | **YAPILMADI**; yerine **poster** yanlış pozitifi gözlendi (FP-4B-001) | aynı komut + eşik; "İz özeti" |
| `OT-4B-001` | 8 Çıkışlar | **Q: sorunsuz** *[Kullanıcı]*; Esc, kırmızı düğme, Ctrl+C, çıkış kodu, kameranın serbest kalması **bildirilmedi** | `echo $?` (0/0/0/130) |
| `OT-4B-004` | Canlı performans | Pencereli MPS ve penceresiz CPU/MPS **yapıldı**; okuma/çizim/gösterme/kare→sonuç satırları ve pencereli CPU **bildirilmedi** | `... --no-display --max-frames 300 --device cpu|mps`; pencereli çalıştırmada konsol satırları |
| `OT-4B-005` | Poster/çakışan kutu tanısı (YOLO mu ByteTrack mı) | **Plan hazır, koşulmadı** (§26) | T0–T7 (§26) |
| `OT-4B-003` | Takip doğruluğu (ID switch, IDF1/HOTA) | **Ölçülmedi** | Etiketli/izinli küçük değerlendirme videosu (docs/person_tracking.md §14) |

## 22. Aşama 5'e aktarılacak veri yapıları

`TrackingResult` / `TrackedPerson` (`track_id`, `session_id`, `bbox_xyxy`, `confidence`, `frame_index`, `timestamp`, `detection_index`, `age_frames`), `LostTrack`, benzersiz anahtar `(session_id, track_id)`, `run_tracking(..., on_result=lambda paket, tespit, takip: ...)` (görüntü + ham tespit + takip birlikte). Yüz tespiti yüz kutularını kişi izlerine kutu içerme/örtüşmeyle ilişkilendirirken `(session_id, track_id)` kullanmalı; kimlik tanıma sonuçları `session_id` değişiminde ve iz silinince geçersiz sayılmalı;
ID devralma/switch olasılığı nedeniyle tek kare yerine çok kareli yüz eşleşmesiyle doğrulanmalıdır. Biyometrik veri bu aşamada yoktur; ID'ler bellekte kalır, diske yazılmaz.

## 23. Tamamlama kriterleri ve sonuç

| # | Kriter | Durum | Kanıt / not |
|---|---|---|---|
| 1 | Aşama 4A kaynak kodları korunuyor | ✔ | `git diff HEAD`: `human_detection/`, `camera/`, 4A betikleri, raporlar boş *[Ölçüldü]* |
| 2 | ByteTrack mevcut Ultralytics sürümüyle çalışıyor | ✔ | 8.4.210; §13 *[Gerçek ByteTrack, Gerçek YOLO+ByteTrack]* |
| 3 | Bağımlılıklar kontrollü kurulmuş | ✔ | `lap==0.5.14` (kaynak, hash, lisans kayıtlı); `scipy` gereksiz, eklenmedi |
| 4 | Ağ olmadan takip çalışıyor | ✔ | ağ/pip yasaklı testler + `YOLO_OFFLINE`; ağ yalnızca `lap` kurulumunda kullanıldı |
| 5 | YOLO her karede yalnızca bir kez | ✔ | mock model + gerçek model `predict` sayımı + `ast` testi |
| 6 | DetectionResult → ByteTrack | ✔ | `PersonTracker.update` |
| 7 | Her aktif takip için geçici Track ID | ✔ | §7 |
| 8 | Aynı karede ID benzersiz | ✔ | çalışma anı değişmezi + testler |
| 9 | Tek insan takibi | ✔* | *sentetik/örnek görüntü; gerçek kamera bekliyor |
| 10 | Çoklu insan takibi | ✔* | *aynı kısıt |
| 11 | Kısa örtüşme değerlendirildi | ✔* | §15 *idealize sentetik; gerçek video yok |
| 12 | Kadraja giriş/çıkış yönetiliyor | ✔* | §15; **kenardan dönen yeni ID alır** (sınırlama) |
| 13 | Kayıp/aktif iz ayrımı | ✔ | `LostTrack` ayrı, çizilmez/sayılmaz |
| 14 | Tracker reset doğru | ✔ | oturum/kaynak/çözünürlük/elle; testlerle |
| 15 | İnsan olmayan kareler | ✔ | §15-F |
| 16 | Eşik davranışları analiz edildi | ✔ | §8–9; gerçek YOLO karşılaştırması |
| 17 | Yorgan riski değerlendirildi | ✔* | §18 *kontrollü sahte dizi; gerçek yorgan bekliyor; **çözülmedi/engellenmedi** |
| 18 | Bounding box ve Track ID görüntüleniyor | ✔* | testler + gözle incelenen kareler; **canlı kamerada kullanıcı ekran görüntüsünde kutular ve ID'ler göründü** *[Kullanıcı]* (poster/çakışan kutu sorunuyla birlikte, §24) |
| 19 | FPS ve takip gecikmesi ölçülüyor | ✔* | kayıtlı videoda ölçüldü; **canlıda FPS/YOLO/ByteTrack süreleri kullanıcı tarafından bildirildi** (§24); **canlı kare→sonuç gecikmesi bildirilmedi** |
| 20 | Kamera/video kaynakları destekleniyor | ✔* | video gerçek koşu ✔; **gerçek MacBook kamerası kullanıcı tarafından çalıştırıldı** *[Kullanıcı]*; IP/RTSP kapsam dışı |
| 21 | Q/Esc/kırmızı düğme/Ctrl+C korunuyor | ✔* | sahte pencere + gerçek Cocoa penceresinde kırmızı düğme *[Ölçüldü]*; **gerçek kamerada yalnızca Q doğrulandı** *[Kullanıcı]*; Esc/kırmızı düğme/Ctrl+C bekliyor |
| 22 | Gerçek ByteTrack entegrasyonu doğrulanmış | ✔ | 62 + 20 test |
| 23 | Önceki 596 test korunuyor | ✔ | 883 test OK |
| 24 | Aşama 2 bütünlük kontrolleri başarılı | ✔ | 12/12; `check_environment.py` başarılı |
| 25 | Ham veri ve model ağırlıkları değişmemiş | ✔ | 12/12 T6.8 (ham veri zip ile boyut+CRC32); model SHA-256 aynı, Git dışı |
| 26 | Teknik rapor hazırlanmış | ✔ | bu rapor, `docs/person_tracking.md`, README/mimari/plan/yaml |
| 27 | Eksik manuel testler açıkça belirtilmiş | ✔ | §21 (durum sütunuyla) |
| 28 | Aşama 5'e geçilmemiş | ✔ | yok |
| 29 | Canlı yanlış pozitif gözlemi kayıtlı ve çözülmüş gösterilmemiş | ✔ | FP-4B-001 / OBS-4B-001 açık; çözüm adayları uygulanmadı (§27) |

(✔* = otomatik/kayıtlı-sentetik düzeyde karşılandı; gerçek MacBook kamerası doğrulaması **kısmen** yapıldı (§24), kalanı kullanıcıda.)

**Karar:** Otomatik ve kayıtlı/sentetik video doğrulamaları başarılı, **gerçek MacBook kamera testleri yalnızca kısmen yapıldığı ve canlıda yeni bir yanlış pozitif (FP-4B-001) çözülmeden kaldığı için** Aşama 4B **`READY_FOR_MANUAL_VERIFICATION`** durumundadır; `COMPLETED` iddia edilmez. **Aşama 4 genel durumu `IN_PROGRESS`.**
Gerçek IP kamera 4B kapanış koşulu yapılmadı (OT-4A-003 sistem entegrasyonuna ertelendi). Açık: `FP-4A-001`, `OT-4A-002`, `OT-4A-003`, `OT-4A-004/OBS-4A-001`; yeni: `OT-4B-001…005` (manuel testler, doğruluk ölçümü, tanı planı), canlı gözlemler `FP-4B-001`, `OBS-4B-001`, `OBS-4B-002` ve bilinen sınırlamalar `KL-4B-001…008` (`configs/project_stages.yaml`).
Aşama 5'e geçilmedi; Git commit/push yapılmadı; kapanış kararı ve sonraki aşamaya geçiş kullanıcıya aittir.

---

## 24. Gerçek kamera testleri — kullanıcı sonuçları (2026-11-12) *[Kullanıcı]*

MacBook Pro M4 gerçek kamerası, `run_person_tracking.py`. Claude bu koşuları yeniden üretemez (kamera izni yok); sayılar kullanıcının bildirdiği değerlerdir (ortalama mı medyan mı belirtilmedi).

| | 1) Pencereli, MPS | 2) Penceresiz, CPU | 3) Penceresiz, MPS |
|---|---|---|---|
| Çözünürlük / kare | 1280×720 / 2924 | — / 300 | — / 300 |
| Süre | 178,9 sn | — | — |
| Uçtan uca FPS | **16,3** | **29,9** | **29,9** |
| YOLO çıkarımı | 17,7 ms | 18,5 ms | 14,7 ms |
| ByteTrack | 0,5 ms | 0,2 ms | 0,4 ms |
| YOLO + ByteTrack | — | 18,8 ms | 15,1 ms |
| Farklı Track ID | **30** | 2 (ikisi de 300 kare) | 2 (300 ve 299 kare) |
| Çıkış | Q, sorunsuz | — | — |

**Görsel gözlem (ekran görüntüsü):** iki gerçek kişi varken **dört aktif Track ID**; panodaki **basılı erkek fotoğrafı ID 167 olarak** algılandı; gerçek kişilerin üzerinde **çakışan büyük takip kutuları** oluştu.
**Bildirilmeyenler:** ekrandaki `DETECTIONS`/`ACTIVE`/`LOST` değerleri ve kutu güvenleri; konsolun "Kare okuma / Çizim / Pencereye gösterme / Kare→görüntü(sonuç) / Tespit: kare başına ortalama / kayıp iz en çok" satırları ve iz özeti; üç koşunun sahnesinin aynı olup olmadığı. **Bu satırlar olmadan §25'teki yorumlar sınırlıdır.**
**Türetilmiş doğrulama:** (2924−1)/178,9 = 16,34 FPS (bildirilenle tutarlı). Kare periyodu: pencereli 61,3 ms; penceresiz 33,4 ms.
**Kayıt:** FP-4B-001 (poster yanlış pozitifi), OBS-4B-001 (iki gerçek kişi, dört aktif iz, çakışan büyük kutular), OBS-4B-002 (pencereli 16,3 ↔ penceresiz 29,9 FPS) — `configs/project_stages.yaml`.

## 25. Analiz

### 25.1 CPU ↔ MPS (penceresiz canlı, koşu 2 ve 3)

| Büyüklük | CPU | MPS | Fark |
|---|---|---|---|
| FPS | 29,9 | 29,9 | 0 |
| YOLO çıkarımı | 18,5 ms | 14,7 ms | MPS −3,8 ms (−20,5%; 1,26×) |
| ByteTrack | 0,2 ms | 0,4 ms | +0,2 ms (§25.5) |
| YOLO + ByteTrack | 18,8 ms | 15,1 ms | −3,7 ms (−19,7%) |
| Hesaplama-yalnız FPS üst sınırı *[Türetilmiş]* | 53,2 | 66,2 | — |
| Periyodun hesaplamaya giden payı *[Türetilmiş]* | %56 | %45 | — |
| Periyodun geri kalanı *[Türetilmiş]* | ≈ 14,6 ms | ≈ 18,3 ms | — |

MPS canlıda ≈ %20 hızlıdır (kayıtlı videodaki ≈ 1,9×'dan çok daha az, §25.3). 4A canlı sonuçlarıyla (CPU 18,5 / MPS 13,4 ms medyan) karşılaştırma: CPU aynı, MPS +1,3 ms (farklı oturum; ortalama/medyan farkı olabilir). Aktif kare farkı (300 / 299) canlı karelerin farklı olmasından beklenir, cihaz etkisi olarak yorumlanmaz.

### 25.2 İki testte FPS neden aynı?
**Desteklenen:** hiçbir koşu hesaplamaya bağlı değildi — hesaplama (18,8 / 15,1 ms) kare periyodunun (33,4 ms) altında; hesaplamaya bağlı olsalar CPU ≤ 53, MPS ≤ 66 FPS'e kadar ayrışırlardı. İki cihaz da 29,9'a oturduğuna göre FPS'i cihazı ayırt etmeyen bir **kare teslim üst sınırı** belirliyor; cihaz farkı FPS'te değil çıkarım süresinde ve boşta kalan payda görünür.
**Bu sınırın kamera olduğunu destekleyen (kesin olmayan) kanıtlar:** (1) Aşama 3'te kullanıcı, işlem yokken ham kamerayı **29,98 FPS** ölçmüştü; (2) kameasız model deneyi *[Ölçüldü; dosya kaynağı + yapay 33,3 ms teslim, canlı kamera DEĞİL]*: kayıtlı hızda CPU ≈ 50 / MPS ≈ 104–107 FPS iken 33,3 ms aralıkla **ikisi de 30,0 FPS**; okuma beklemesi CPU ≈ 14 ms, MPS ≈ 17–18,5 ms (kullanıcıdan türetilen 14,6 / 18,3 ms ile uyumlu).
**Desteklenmeyen:** canlıdaki gerçek okuma beklemesi (bildirilmedi), kare atlama olup olmadığı ve 29,9'un donanım tavanı mı otomatik pozlama sınırı mı olduğu. "Kamera 30 FPS ile sınırlıdır" **güçlü çıkarımdır, doğrulanmış ölçüm değildir**; doğrulama: aynı oturumda `run_camera.py --no-display --max-frames 300` ve iki koşunun "Kare okuma süresi" satırları.

### 25.3 MPS çıkarımı canlıda neden kayıtlı videodan yavaş? *[Ölçüldü, kameasız]*
MPS, penceresiz, 150 kare; kareler arası yapay boşluk arttıkça çıkarım medyanı: 0 ms → **9,2**, 15 → 10,6, 25 → 11,9, **33,3 → 13,8**, 50 → 17,4, 100 → 23,0 ms (ByteTrack da 0,3 → 1,0 ms). CPU kontrolü: 19,1 → 18,9–19,1 ms (değişmedi). **Gözlem:** boşta kalma payı arttıkça MPS çıkarımı (ve CPU'daki ByteTrack süresi) uzuyor; 33,3 ms'de 13,8 ms, kullanıcının canlı 14,7 ms'siyle uyumlu. **Yorum sınırı:** kamera ile yapılmış ölçüm değil, mekanizma (güç/saat durumu, uyanma gecikmesi) ölçülmedi; ilişki gözlemdir, kanıtlanmış neden değil.

### 25.4 Pencereli 16,3 FPS
Hesaplama 18,2 ms (periyodun %30'u); kalan ≈ 43 ms'nin dökümü bildirilmedi. **Doğrudan ölçülen** *[kameasız, bu makine, gerçek pencere]*: `imshow + waitKey(1)` tek başına **ort. 17,3 ms (1280×720), 15,0 ms (640×480)**; çizim 0,4–0,7 ms. Kamera-benzeri 33,3 ms tempolu gerçek pencereli koşuda **iki cihaz da 30,0 FPS** verdi (toplam MPS ≈ 29,5, CPU ≈ 32,4 ms < 33,3 ms). Kullanıcının canlı koşusunda ise çıkarım 17,7 ms (kameasız 12,8 ms'nin ≈ 5 ms üstü) ve toplam yaklaşık 35–36 ms *[gösterme süresi bildirilmediğinden yaklaşık]*; gözlenen 61,3 ms periyodu bu toplamdan çok uzundur.
**Hipotezler (doğrulanmadı):** H1 bütçe aşılınca kamera teslimi iki periyoda kuantize oluyor; H2 canlı pencereli çalışmada `imshow`/pencere yöneticisi maliyeti ölçülenden büyük; H3 kamera/ISP yükü; H4 sahne farkı (daha çok tespit). Ayırt edici test: §26 T7 (özellikle pencereli `--imgsz 320` tanı koşusu). 4A'nın pencereli 18,4–25,3 FPS'i de aynı ailedendir ve ayrıştırılmamıştır. Takibin eklediği 0,5 ms bu farkı açıklayamaz.

### 25.5 ByteTrack süresi
ByteTrack CPU'dadır, cihazdan bağımsızdır; iki kişide ölçeklenme tablomuz (§16) ≈ 0,15–0,2 ms verir: CPU koşusu (0,2) uyumlu. MPS koşularındaki 0,4–0,5 ms için olası etkenler (doğrulanmadı): daha çok tespit (poster/çakışan kutu), boşta kalma (§25.3'te ByteTrack süresi de boşlukla uzadı), ortalama/medyan farkı. Her durumda kare periyodunun %1,5'inden azdır.

### 25.6 "30 farklı ID" ve ID 167
Kimlik sayacı iz **oluşturulunca** artar (onay beklemez): ID 167 bu oturumda en az 167 iz oluşturulduğunu, yalnızca 30 farklı ID görüntülendiğine göre *[Türetilmiş]* **≥ 137 izin hiç onaylanmadığını** (tek karelik/kararsız tespitler) gösterir; ByteTrack'in iki-kare onayı bunları eledi, **kalıcı** poster ise süzgeçten geçti (§18 tablosuyla uyumlu). 30 ID'nin kaçı gerçek kişi parçalanması, kaçı yanlış pozitif **ayrıştırılmadı**. Konsol iz özeti en çok 20 izi gösterir: 10 iz (ID 167 satırı dahil olabilir) görünmez.

## 26. Poster / çakışan kutu: YOLO mu, ByteTrack mi? — bilinenler ve test planı

**Şimdiden söylenebilenler** (ayrıntı: [docs/person_tracking.md §21](../../docs/person_tracking.md)):
1. **`ACTIVE ≤ DETECTIONS` değişmezi** (kodda ve gerçek-model testlerinde denetlenir): dört aktif iz, o karede en az dört YOLO kutusu demektir; ByteTrack kutu uyduramaz. (Ekrandaki `DETECTIONS` değeri bildirilmedi.)
2. ByteTrack çakışan/iç içe kutuları **birleştirmez/bastırmaz** *[Gerçek ByteTrack, betiklenmiş]*: IoU 0,49 (iç içe büyük kutu) ve 0,60 (yana kaymış kutu) ayrı, kararlı ID'ler aldı; 3 karenin 2'sinde gelen kararsız ikinci kutu da onaylanıp ID'sini korudu (aktif iz 1 ↔ 2 salındı). YOLO'nun kararsız/yinelenen kutusu → kalıcı ek ID.
3. Takip kutusu ↔ eşleşen ham kutu IoU'su *[Ölçüldü; sentetik montaj, 1931 iz-karesi]*: medyan 0,998, %1 dilimi 0,884, **en düşük 0,824** — Kalman süzgeci bu örnekte kutuyu büyütmedi (kullanıcı sahnesinde ölçülmedi).
4. Ham YOLO kutuları normal kişilerde bile her karede çakışıyor (en büyük IoU 0,34, içerme 0,76) *[aynı montaj]*: çakışma tek başına arıza belirtisi değildir.
**Sonuç:** kaynağın YOLO tespit katmanı olması **beklenir**, ancak poster kutusunun ve büyük çakışan kutuların kullanıcı sahnesinde gerçekten ham YOLO çıktısında bulunduğu **doğrulanmamıştır**. ByteTrack'in rolü: ≥ 2 karede görülen her kutuya kalıcı ID vermek.

**Test planı (kullanıcı koşar; kamera görüntüleri depoya girmez, kayıt alınırsa izinle, yerelde, Git dışı, iş bitince silinir):**

| Adım | Amaç | Komut / yöntem | Karar kuralı |
|---|---|---|---|
| T0 | Sayıları oku | `run_person_tracking.py --source camera --camera-index 0 --device mps`; sorunlu karenin ekran görüntüsü | `DETECTIONS`, `ACTIVE`, `LOST`, her kutunun `ID n güven`; `ACTIVE ≤ DETECTIONS` |
| T1 | YOLO mu (takipsiz) | aynı sahne/cihaz/`--conf`: `run_person_detection.py --source camera --camera-index 0 --device mps` | Poster ve çakışan büyük kutular burada da varsa → **YOLO kaynaklı** (birkaç saniye izle; aralıklı algılanabilir). **Durum: KISMEN yapıldı, TAMAMLANMADI** (aşağıda) |
| T2 | Poster güven dağılımı | yalnız poster, 60 sn: `... run_person_tracking.py ... --no-display --max-frames 1800` | Poster izi: aktif kare, ort./en yüksek güven |
| T3 | Gerçek kişi güven dağılımı | aynı mesafe/ışık, poster yok, 60 sn (hareketli + durağan) | Gerçek kişi izlerinin güveni; poster **maks.** güveni gerçek kişi **alt ucundan** belirgin düşükse eşik düşünülebilir (alt uç için kare bazlı log gerekir) |
| T4 | Tekrarlanabilir klip | 20–30 sn klip → `run_person_detection.py` ve `run_person_tracking.py` `--source video --video-path KLİP --device mps|cpu --no-display` | Aynı kareler: cihaz/takip etkisi ayrışır; eşik denemeleri bu klipte tekrarlanır |
| T5 | Büyük kutu Kalman kaynaklı mı | Kare bazlı log (araç eklenirse): izin kutusu ↔ ham kutu IoU | ≈ 1 → YOLO kutusu; ≪ 1 → Kalman/ByteTrack |
| T6 | ID kaynağı | T4'te 4A `PERSONS` oynaklığı ↔ 4B iz özeti | Tespit sayısı sabit, ID çok → parçalanma; tespit oynak → kararsız tespit |
| T7 | Pencereli FPS tanısı | (a) `run_camera.py --source camera --no-display --max-frames 300`; (b) pencereli ve `--no-display` koşuların konsol satırları; (c) pencereli `--imgsz 320` (yalnız tanı) | (a) ham kamera tavanı; (b) okuma/gösterme süreleri; (c) bütçe ≤ 25 ms'de FPS ≈ 30'a dönerse H1 |

**T1 durumu (2026-11-12):** kullanıcı 4A betiğiyle **yeni ortamda** gerçek kamera testi yaptı (1812 kare, 21,9 FPS, ort. çıkarım 15,9 ms, ort. 2,89 / en çok 7 kişi; ön plandaki iki kişi 0,85/0,87 ve 0,80/0,66; telefondaki insan fotoğrafı ayrı tespit oluşturmadı; arka plandaki çok küçük kişiler kutulanmadı) — [4A raporu §24](../detection/STAGE_4A_REPORT.md). **Ortam değiştiği için önceki poster aynı şartlarda yeniden kurulamadı; T1 poster karşılaştırması tamamlanmış sayılmaz, H1/H2 doğrulanmadı.** Telefondaki fotoğraf poster değildir; 2,89 / 7 tespit yanlış pozitif sayılmaz. T0, T2–T7 yapılmadı.

**Not (araç):** T3/T5 için kare bazlı sayısal log, ham tespitleri gösteren `--show-detections` ve iz özetinde tüm izleri gösteren bir seçenek faydalı olur; bunlar yeni kod ve test demektir ve **eklenmemiştir** (onaya bağlı).

## 27. Yanlış pozitifleri azaltma seçenekleri — değerlendirme (**hiçbiri uygulanmadı**)

Gerçek kişi kaybına yol açabilecek hiçbir eşik/filtre doğrulama olmadan uygulanmadı; varsayılanlar (`conf 0.25`, `track_high 0.25`, `new_track 0.25`, `iou 0.7`) **değişmedi**. **Kabul ölçütü önerisi (onaya tabi):** ≥ 3 sahnede her aday için yanlış pozitif iz-karesi, gerçek kişi kapsamı, ID switch ve FPS ölçülür; **kabul:** gerçek kişi kapsamı varsayılana göre ≤ 2 yüzde puan düşer **ve** yanlış pozitif iz-karesi ≥ %50 azalır.

| Seçenek | Kazanç | Gerçek kişi riski | Kanıt | Karar |
|---|---|---|---|---|
| `--new-track-thresh` ↑ (0,5–0,6) | Düşük güvenli yeni iz azalır | **Yüksek** | Montajda 0,6: kısa izler 4→1 **ama kararlı bir zayıf-güvenli gerçek iz kayboldu** | T2/T3/T4 ölçülmeden **uygulanmaz** |
| YOLO `--conf` ↑ (0,35/0,5) | Düşük güvenli kutular atılır | Orta–yüksek | 4A: yorgan 0,35'te de görüldü; montajda kısa izler azaldı, yok olmadı | Aynı |
| NMS `--iou` ↓ (0,5) | Çakışan/yinelenen kutular azalır | Orta (yakın iki kişiden biri bastırılabilir; örtüşme testleri yapılmadı) | Montajda ham çakışma ≤ 0,34: bu videoda etkisiz; kullanıcı sahnesi bilinmiyor | Yalnız T4 klibinde **tanı** |
| Kutu içerme/geometri filtresi | İç içe/devasa kutular | Orta–yüksek (normal kişilerde içerme 0,76) | Yok | Ayrı onay; 4B kapsamı dışı |
| Daha uzun onay (N kare) | Kısa titreşim | Düşük (N/30 sn gecikme) | **Kalıcı poster/yorganı elemez** | Yalnız titreşim görülürse |
| Hareketsiz iz **bayrağı** (silme değil) | Aşama 5+ ipucu | Silme → durağan gerçek çalışan kaybı | Yok | Bayrak olarak düşünülebilir, filtre olarak değil |
| "Yok sayılan bölge" | Bilinen poster alanı | Bölgede kişi görünmez; sahneye bağımlı | Yok | Varsayılan olmaz |
| Daha büyük YOLO modeli | Ayrım iyileşebilir | Düşük | Yok; FPS/lisans/ağ | Kapsam dışı (ayrı karar) |
| Zor negatif veriyle ince ayar | Kök neden | Düşük | Yok | Aşama 8+ |
| Yüz+hareket+poz doğrulaması | Basılı yüz ↔ canlı kişi | — | **Poster de yüz içerir; yüz tek başına ayırmaz** | İleri aşama |

**Önerilen sıra:** T0 → T1 → T2/T3 → T4 → ancak sonra aday ayarların klipte tanı denemesi → kabul ölçütü sağlanırsa kullanıcı onayıyla uygulama.

## 28. Güncel durum ve sonraki adımlar

* **Durum: `READY_FOR_MANUAL_VERIFICATION` korunuyor.** `COMPLETED` değil. Yapılmayan testler (üç kişi, örtüşme, yer değiştirme, ID switch, kadrajdan çıkma, yüksek açı, Esc/kırmızı düğme/Ctrl+C, yorgan) başarılı gösterilmemiştir.
* **Yeni açık kayıtlar:** FP-4B-001 (poster yanlış pozitifi, `open_known_limitation`), OBS-4B-001 (dört aktif iz/çakışan kutu, `open_investigation`), OBS-4B-002 (pencereli FPS düşüşü, `open_investigation`), OT-4B-005 (tanı planı). OT-4B-001 ve OT-4B-004 `partially_performed`.
* **Kod/test değişikliği yok:** bu güncelleme yalnızca belge ve yapılandırma kayıtlarıdır; 883 test sonucu değişmedi (§12). Kameasız deneyler (pacing, kutu sapması, çakışma) depoya eklenmeyen betiklerle yapıldı; yöntemleri yukarıda tarif edildi.
* **Kullanıcıdan istenenler** (yorumu kesinleştirir): (1) sorunlu karenin `DETECTIONS`/`ACTIVE`/`LOST` değerleri ve her kutunun güveni; (2) pencereli koşunun konsol özetindeki "Kare okuma süresi", "Çizim süresi", "Pencereye gösterme süresi", "Kare → görüntü gecikmesi", "Tespit: kare başına ortalama / en çok" satırları; (3) iki penceresiz koşunun "Kare okuma süresi" ve "Tespit" satırları; (4) üç koşunun sahnesinin aynı olup olmadığı; (5) T1 sonucu — yeni ortamda kısmen bildirildi (4A raporu §24); **önceki poster/çakışan kutular için aynı şartlarda T1 hâlâ bekliyor**.
* **2026-11-12 (yeni ortam):** 4A betiğiyle yapılan T1 gözlemi kaydedildi (OBS-4A-003, OBS-4A-004, OT-4A-005; 4A raporu §24, docs/person_detection.md §14.4); durum değişmedi, eşik/algoritma değiştirilmedi. Sonraki adım (kullanıcı): aynı yeni ortamda 4B ile iki kişinin ID kararlılığı ve örtüşme testi — kayıt şablonu: docs/person_tracking.md §17.1.
* Aşama 5'e geçilmedi; commit/push yapılmadı.
