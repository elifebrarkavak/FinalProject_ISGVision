# Çoklu İnsan Takibi Modülü (Aşama 4B)

> **Durum: Aşama 4B `READY_FOR_MANUAL_VERIFICATION`** (2026-11-12). Kod, otomatik testler (mock, gerçek ByteTrack, gerçek YOLO + ByteTrack, sentetik/kayıtlı örnek görüntü) ve performans ölçümleri tamamdır;
> **gerçek MacBook kamerası ile manuel testlerin yalnızca bir kısmı yapılmıştır** (§17, §19): pencereli MPS oturumu, penceresiz CPU ve MPS koşuları ve Q ile çıkış *[Kullanıcı]*. Üç kişi, örtüşme, yer değiştirme, ID switch, kadrajdan çıkma, yüksek açı ve diğer çıkış yöntemleri **yapılmadı**. `COMPLETED` iddiası yoktur. Aşama 4 genel durumu **`IN_PROGRESS`**.
> **Canlı testte yeni yanlış pozitif gözlemi (FP-4B-001):** panodaki basılı erkek fotoğrafı insan olarak algılanıp Track ID aldı (ID 167); iki gerçek kişi varken dört aktif iz ve çakışan büyük kutular görüldü. **Kaynak (YOLO mu ByteTrack mi) henüz ayrıştırılmadı**; tanı planı §21, çözüm adaylarının değerlendirmesi §22. **Hiçbir eşik/varsayılan değiştirilmedi.**
> **Ölçülmeyenler:** takip doğruluğu (IDF1 / HOTA / MOTA), gerçek kamerada ID sürekliliği/ID switch, canlı gecikme (kare→sonuç), yorgan ve poster yanlış pozitiflerinin kaynağı. "%85 hedefi karşılandı" iddiası yoktur.
> **Track ID bir kimlik değildir:** geçici ve oturum kapsamlıdır; çalışan adı, personel numarası veya biyometrik kimlik DEĞİLDİR.
> Teknik sonuç raporu: [../reports/tracking/STAGE_4B_REPORT.md](../reports/tracking/STAGE_4B_REPORT.md). Önceki aşama: [person_detection.md](person_detection.md).

## 1. Amaç ve kapsam

Aşama 4A'nın `PersonDetector` çıktılarını (`DetectionResult`) Ultralytics **ByteTrack** algoritmasına vererek kamera/video karelerindeki birden fazla insana **geçici Track ID** atamak ve ID'yi kareler arasında
mümkün olduğunca korumak. YOLO her karede **yalnızca bir kez** çalışır; takip ek çıkarım yapmaz.

Bu aşamada **yoktur**: yeni model eğitimi, yüz tespiti/tanıma, çalışan kaydı, kimlik tanıma, KKD, hayvan tespiti, sensör/mesai modu, RTSP/IP kamera desteği, veritabanı/panel, görüntü kaydı, BoT-SORT karşılaştırması.

## 2. Dosyalar

```
src/isg_vision/tracking/                  (mevcut Aşama 1 iskeleti kullanıldı; ikinci bir takip paketi açılmadı)
├── errors.py             hata sınıfları (yapılandırma / bağımlılık / backend / geçersiz girdi / kare sırası)
├── config.py             TrackingConfig + analyze_thresholds (YOLO conf ↔ ByteTrack eşikleri)
├── results.py            TrackedPerson, LostTrack, TrackingResult (yalnızca standart kütüphane)
├── bytetrack_adapter.py  PersonTracker: DetectionResult → ByteTrack → TrackingResult; bağımlılık denetimi
├── metrics.py            count_id_switches (ID switch / devralma sayımı; IDF1/HOTA DEĞİL)
├── visualization.py      draw_person_tracks: kutu + "ID n" + sayılar (kopya üzerinde)
├── pipeline.py           run_tracking: FrameSource → PersonDetector → PersonTracker → çizim → pencere; ölçümler
├── cli.py                komut satırı (scripts/run_person_tracking.py bunu çağırır)
└── benchmark.py          takip ek yükü ölçümü (CPU/MPS, yalnızca-tespit ↔ tespit+takip) ve ByteTrack ölçeklenmesi
scripts/run_person_tracking.py            çalıştırma betiği (ince başlatıcı)
scripts/benchmark_person_tracking.py      performans ölçümü betiği
```

Aşama 4A (`human_detection/`), kamera paketi, `scripts/run_person_detection.py` ve ağırlık dosyası **değiştirilmemiştir** (`git diff HEAD` bu yollarda boştur).

## 3. ByteTrack: sürüm, API ve entegrasyon kararı

Kurulu **`ultralytics==8.4.210`** kaynak kodundan okundu (tahmin değildir):

| Konu | Doğrulanan davranış |
|---|---|
| Sınıf | `ultralytics.trackers.byte_tracker.BYTETracker(args)`; `args`: `track_high_thresh, track_low_thresh, new_track_thresh, track_buffer, match_thresh, fuse_score` (`cfg/trackers/bytetrack.yaml` ile aynı anahtarlar) |
| Güncelleme | `rows = tracker.update(boxes)`; `boxes`: `.conf`, `.cls`, `.xywh` veren, boolean maske ile indekslenebilen NumPy tabanlı nesne. Kullanılan: `ultralytics.engine.results.Boxes(np.ndarray[N,6]=[x1,y1,x2,y2,conf,cls], (yükseklik, genişlik))` — `YOLO.track()`'in içeride `result.boxes.cpu().numpy()` ile verdiği nesne türü |
| Çıktı | `(N, 8)` `float32`: `[x1, y1, x2, y2, track_id, score, cls, idx]`; `idx` = verilen tespit dizisindeki sıra (tespit ↔ iz eşlemesi buradan). **Hiç iz yoksa çıktı 1B `(0,)` gelir** (adaptör `(0, 8)`'e normalleştirir). Kutu **Kalman tahminidir**, ham YOLO kutusu değil |
| Durum | `tracked_stracks`, `lost_stracks`, `removed_stracks`, `frame_id`; `tracker.reset()` havuzları boşaltır, Kalman süzgecini yeniler, kimlikleri **1'den** başlatır. Kimlik sayacı takipçi örneğine özeldir (iki takipçi birbirini etkilemez) |
| Görüntü | `update(results, img=None)`: ByteTrack hareket telafisi (GMC) veya ReID kullanmaz; kare verilmez, kopyalanmaz |
| Bağımlılık | `ultralytics.trackers.utils.matching` modül düzeyinde `import lap` yapar; **yoksa `pip` ile otomatik kurmaya çalışır** (§4). `scipy` **gerekmez**: atama çözücüsü `lap.lapjv`'dir; Ultralytics'in `scipy`'siz NumPy yedeği vardır |

**Mimari karar: `YOLO.track()` KULLANILMADI.** `YOLO.track()` bir `predict` çağrısına takipçiyi geri çağrımlarla bağlar; yani onu kullanmak 4A'nın `PersonDetector.detect`'i ile **aynı kareyi ikinci kez** YOLO'ya göndermek anlamına gelirdi.
Bunun yerine `BYTETracker` doğrudan kullanılır ve `DetectionResult` ona küçük bir adaptörle (`PersonTracker`) beslenir:

```text
FramePacket → PersonDetector.detect()  [YOLO: kare başına TEK çağrı] → DetectionResult → PersonTracker.update()  [ByteTrack] → TrackingResult → draw_person_tracks → PreviewWindow
```

**Sürüm bağımlılığı riski.** `BYTETracker` ve `Boxes` Ultralytics'in *dahili* sınıflarıdır; kararlılık sözü yoktur. `tracked_stracks`, `lost_stracks`, `start_frame` gibi öznitelikler kaynak kodda belgelenmiştir ama sürümler arasında değişebilir. Önlemler: `requirements.txt`
`ultralytics==8.4.210`'a sabitli; farklı sürümde `PersonTracker.backend_info["version_note"]` ve CLI uyarı verir; beklenmeyen çıktı biçiminde `TrackerBackendError` fırlatılır (sessizce yanlış sonuç üretilmez). Sürüm yükseltilirse
`tests/test_person_tracking_bytetrack_real.py` yeniden çalıştırılmalıdır (bu testler bu sürümde *ölçülen* davranışı da sabitler).

**Neden `run_detection(..., on_result=...)` kancası kullanılmadı?** Kanca `(paket, sonuç)` ile çağrılır ve `None` döndürür; çizim `draw_person_detections`'a sabit bağlıdır ve özet takip süresini taşımaz. Takip kancayla çalıştırılabilirdi ama sonucu ekrana
çizdiremez ve `tracking` süresini ölçüme katamazdı. 4A koduna dokunmadan bunu sağlamak için `run_tracking` aynı yapıda kardeş döngü olarak yazıldı (aynı `FpsMeter`, `LatencyStats`, `REASON_*`, çıkış tuşu ve pencere mantığı yeniden kullanıldı).
`run_tracking`'in kendi `on_result(paket, tespit, takip)` kancası vardır (Aşama 5 için).

## 4. Bağımlılıklar, ağ ve çevrimdışı çalışma

Denetim (2026-11-12): `lap` ve `scipy` kurulu değildi (4A kapanış raporuyla aynı). ByteTrack'in **gerçekten** ihtiyaç duyduğu tek ek paket **`lap`**'tır.

| Paket | Karar | Gerekçe / kayıt |
|---|---|---|
| `lap==0.5.14` | **Kuruldu**, `requirements.txt` ve `requirements-lock.txt`'e eklendi | PyPI; `cp311-cp311-macosx_11_0_arm64` hazır wheel (1,5 MB); SHA-256 `9c99688a55a0f4a676d86786080ae192272b9f951cf9e4e4deada43b6927d47d`; lisans BSD-2-Clause; tek bağımlılığı `numpy>=1.21.6` (kurulu 2.4.6). Ultralytics en az `0.5.12` ister. Wheel önce yalnızca indirilip `METADATA`'sı okundu (çalıştırılmadan), sonra **aynı dosyadan** `pip install --no-index --no-deps` ile kuruldu; `pip check`: sorun yok; NumPy 2.4.6 ile `lap.lapjv` çalıştı |
| `scipy` | **Eklenmedi** | Gerekmez (yukarıda). Gereksiz bağımlılık eklenmedi |

Ağ yalnızca bu kurulum adımında kullanıldı (PyPI sürüm sorgusu ve wheel indirme). **Takip çalışırken ağ kullanılmaz, model indirilmez, paket kurulmaz:**

* `check_tracker_dependencies()` `lap`/`ultralytics`'in varlığını `importlib.metadata` ile **Ultralytics'i import etmeden ÖNCE** denetler; `lap` yoksa açıklayıcı `TrackerDependencyError` verir (kurulum komutuyla) ve Ultralytics'in otomatik pip kurulumuna hiç yol vermez.
* `YOLO_OFFLINE=1` (4A korumasıyla aynı) ve `YOLO_AUTOINSTALL=False` ayarlanır.
* Testler: `socket.connect` ve `subprocess.Popen/check_output` yasaklanmış hâlde (a) gerçek takipçi yüklenip çalıştırılır, (b) gerçek YOLO + ByteTrack zinciri yüklenip çalıştırılır, (c) `lap` yokmuş gibi CLI başlatılır: ağ/pip girişimi olursa test düşer; (c)'de kamera hiç açılmaz ve çıkış kodu 1'dir.
* `yolo11n.pt` aynı kalır: SHA-256 `0ebbc80d4a7680d14987a577cd21342b65ecfd94632bd9a8da63ae6417644ee1` (testle doğrulanır), Git dışı (`.gitignore:75`), lisans AGPL-3.0 (bkz. 4A raporu §3). Başka YOLO ağırlığı yüklenmez.

## 5. Mimari

* **Adaptör (`PersonTracker`) politika sahibidir, backend algoritmadır.** Backend (`UltralyticsByteTrackBackend`) ince bir sarmalayıcıdır; politika (girdi doğrulama, kare sırası, oturum, eşleme, hata sarmalama) backend'den bağımsızdır ve sahte backend ile test edilebilir.
* Tespit, takip, görselleştirme ve döngü ayrıdır; birbirlerini bilmezler. `PersonTracker` yalnızca `DetectionResult` alır (görüntü almaz) ve modeli çağırmaz (bir `ast` testi bunu kodda doğrular).
* Her işlenen karede `update` **tam bir kez** çağrılır — **tespitsiz kareler dahil** (boş girdi ile); aksi halde kayıp izler yaşlanmaz.
* Kaynak (kamera/video) kodu yeniden yazılmadı; IP/RTSP eklenmesini engelleyen bir şey yoktur (RTSP desteği geliştirilmedi).
* Doğrulama **öznitelik tabanlıdır** (sınıf kimliğine bağlı değil): `detections`, `image_size`, `frame_index`, `timestamp` ve her tespitte `bbox_xyxy/confidence/class_id/class_name` aranır; eksik/geçersiz yapı `InvalidTrackingInputError` ile reddedilir (durum değişmez).

## 6. Yapılandırma ve eşikler

`TrackingConfig` varsayılanları **kurulu Ultralytics'in `bytetrack.yaml` dosyasıyla birebir aynıdır** (bir test dosyayı okuyup karşılaştırır): `track_high_thresh=0.25`, `track_low_thresh=0.1`, `new_track_thresh=0.25`, `track_buffer=30`, `match_thresh=0.8`, `fuse_score=True`.
Bunlar evrensel optimum değil makul bir başlangıçtır. 4A'nın `DetectionConfig.conf=0.25` **değiştirilmedi**; `conf=0.35` kendiliğinden varsayılan yapılmadı. Doğrulama: `track_low_thresh < track_high_thresh <= new_track_thresh`, `track_buffer` 1–1000 tam sayı (**kare sayısı**), eşikler `(0, 1]`.

**Eşiklerin gerçek anlamı** (`BYTETracker.update` kaynağından): skoru `s` olan tespit — `s >= track_high_thresh` → birinci eşleştirme (Kalman tahmini ile IoU, `match_thresh`); eşleşmeyen ve `s >= new_track_thresh` olan **yeni iz doğurur**.
`track_low_thresh < s < track_high_thresh` → yalnızca ikinci eşleştirmeye (yalnızca IoU, eşik 0,5) girer: **mevcut** izi sürdürür/kurtarır, **asla yeni iz doğurmaz**. `s <= track_low_thresh` yok sayılır. Yeni iz yalnızca yüksek güvenli tespitten doğduğu için `new_track_thresh < track_high_thresh` anlamsızdır (reddedilir).

### 6.1 Tespit eşiği (YOLO `conf`) ve ByteTrack: sorulara yanıtlar

1. **`conf=0.25`, ByteTrack'in düşük güvenli davranışını nasıl etkiliyor?** YOLO, `conf` altındaki kutuları ByteTrack'e **hiç vermez**. `conf >= track_high_thresh` ise ikinci (düşük güven) aşama **fiilen devre dışıdır**; varsayılan yapılandırma (`0.25` / `0.25`) tam olarak bu durumdadır. Bu bilinçli, güvenli bir tercihtir ama ByteTrack'in kurtarma özelliğini kullanmaz.
   CLI bunu her başlangıçta yazar (`Eşik notu: ... fiilen KAPALI`).
2. **Takip için tespit eşiği görsel gösterim eşiğinden ayrı olmalı mı?** Bu mimaride tek YOLO çağrısı vardır, dolayısıyla tek `conf` vardır; "gösterim" eşiği ise zaten takibin kendisidir (yalnızca onaylanmış aktif izler çizilir). Ayrı bir "takip-tespit conf" ayarı eklemek yerine `--conf` ile `--track-high-thresh`/`--new-track-thresh` birlikte (ve çalışma anında açıklanarak) ayarlanır.
3. **Daha düşük YOLO eşiği yanlış pozitifleri artırır mı?** Tespit sayısı artar (ölçüldü: sentetik videoda kare başına 6,52 → 8,58 tespit, `conf 0.25 → 0.10`). Düşük güvenli tespitler yeni iz doğurmaz, ama **mevcut kısa/şüpheli izlerin ömrünü uzatır** (aynı videoda 4 kısa izin toplam aktif karesi 131 → 180). Yararı bu videoda gösterilemedi (uzun izler zaten kesintisizdi). Bu yüzden varsayılan yapılmadı.
4. **Yorgan gibi nesnelere geçici Track ID atanma riski nasıl izlenecek?** CLI her çalıştırmada **iz özeti** yazar (her iz için ilk-son kare, aktif kare sayısı, süre, ortalama/en yüksek güven); kısa ve düşük güvenli izler yanlış pozitif adayıdır. Davranışı §12 açıklar. Gerçek yorgan testi kullanıcıya bırakıldı (§17, Manuel Test 7).
5. **Yeni iz oluşturma eşiği ile sürdürme eşiği nasıl ayrılmalı?** `--new-track-thresh` (yeni iz başlatma) ve `--track-high-thresh` (sürdürme, birinci aşama) ayrı ayarlardır. Yeni iz eşiğini yükseltmek yanlış pozitiften iz doğmasını azaltır ama gerçek, zayıf tespitli kişinin iz almasını da engeller (§12 ölçümü).

**Eşiklerin gerçek YOLO ile karşılaştırması** (4A'nın sentetik montaj videosu, CPU, 300 kare, **etiket yok**; yalnızca gözlem; her satır CLI ile yeniden üretilebilir, bkz. rapor §9):

| Yapılandırma | Tespit/kare | Farklı ID | Uzun iz (≥250 kare) | Kısa iz (<100 kare) | Kısa izlerin toplam aktif karesi | Düşük güven aşaması |
|---|---|---|---|---|---|---|
| A: varsayılan (`conf .25`, high `.25`, new `.25`) | 6,52 | 10 | 6 | 4 | 131 | kapalı |
| B: `conf .35`, high=new=`.35` | 6,23 | 9 | 6 | 3 | 68 | kapalı |
| C: `conf .25`, `new-track-thresh .60` | 6,52 | 6 | 5 | 1 | 27 | kapalı |
| D: `conf .10`, high `.25` | 8,58 | 10 | 6 | 4 | 180 | AÇIK |
| E: `conf .10`, high=new=`.35` | 8,58 | 9 | 6 | 3 | 107 | AÇIK |
| F: `conf .10`, high `.50`, new `.60` | 8,58 | 6 | 5 | 1 | 37 | AÇIK |

Okuma (yalnızca bu videoda ve etiketsiz): B ve C kısa izleri azaltır ama yok etmez (B, kullanıcının 4A `conf=0.35` gözlemiyle uyumlu: sorun **çözülmedi**). **C aynı zamanda 6 kararlı izden birini (görüntü kenarında kısmen görünen, ortalama güveni 0,37 olan kişi) kaybettirir:** yanlış pozitif azaltmanın gerçek kişi kaçırma
maliyeti vardır. D–F: düşük güven aşaması ID sayısını değiştirmedi, kısa izleri uzattı. "Kısa iz" = yanlış pozitif DEĞİLDİR (görsel incelemede ID 22/32 bulanık arka plan bölgesinde küçük, düşük güvenli kutulardı; gerçek bulanık kişi mi yanlış pozitif mi **doğrulanamadı**). Hiçbir eşik "optimum" ilan edilmedi.

## 7. Track ID yaşam döngüsü (8.4.210'da ÖLÇÜLEN davranış)

* **Onay:** yeni iz, **ardışık iki karede** eşleşmeden görüntülenmez (`TrackingResult.tentative_count`). Tek karelik tespit (güven skoru ne olursa olsun) iz olarak gösterilmez. **İstisna:** oturumun (her `reset` sonrası da) **1. karesindeki** tespitler onay beklemeden iz alır.
* **Numara tüketimi:** onaylanmadan silinen geçici iz de numara tüketir; ID'ler bitişik olmak zorunda değildir (1, 2, 5, 6 normaldir).
* **Kayıp tamponu:** eşleşmeyen aktif iz kayıp olur (`LostTrack`), ekranda ve aktif sayıda **gösterilmez**. `track_buffer` **karedir**, süre değil. Ölçülen etkin kurtarma penceresi `track_buffer + 1` karedir (silinen iz havuzda bir kare daha kalır); `track_buffer + 2` kare kayıpta iz silinir ve kişi yeni ID alır.
  Kayıp iz, Kalman tahminiyle **son hızıyla** ilerlemeyi sürdürür: kişi tahmin edilen konumda dönerse ID korunur (15 kare kayıpta da ölçüldü); ters yönde dönerse veya kadrajdan çıkıp aynı kenardan girerse **yeni ID alır** (tampon dolmadan bile).
* **Yer değiştirme:** ByteTrack yalnızca konum/IoU kullanır, **görünüm kullanmaz**. İki kişi ardışık iki karede yer değiştirirse (kare kaybı, çok düşük FPS) numaralar KONUMA yapışır: ID switch oluşur. Tampon süresi içinde kaybolan birinin son konumuna başka biri girerse **eski ID'yi devralır**. Metrik (`count_id_switches`) bunları yakalar, gizlenmez.
* **Hızlı hareket:** ardışık karelerde kutular yeterince örtüşmezse (IoU ≈ 0,2'nin altı; ör. kare başına kutu genişliği kadar kayma) iz oluşmaz/kopar.
* **Oturum:** takipçi sıfırlandığında (yeni kaynak, yeni oturum, çözünürlük değişimi, elle `reset()`) ID'ler 1'den başlar; bu yüzden her sonuçta `session_id` taşınır ve benzersiz anahtar **`(session_id, track_id)`**'dir. `run_tracking` döngü başında `reset()` çağırır.
  `update(..., source=...)` kaynak etiketi önceki karelerden farklıysa veya görüntü boyutu değişirse takipçiyi **otomatik** sıfırlar (`TrackingResult.reset_reason`).
* **Kare sırası:** `frame_index` kesin artan olmalıdır; tekrar eden/geri giden kare (veya geri giden zaman damgası) `FrameOrderError` verir ve durum **değişmez** (kare iki kez yaşlandırılmaz). Atlanan numaralar kabul edilir, `stats.gap_frames` olarak sayılır (ByteTrack boşluktaki kareleri ardışık sayar).
  Yeni oturumda numara 0'a dönebilir.
* **Aynı karede benzersizlik:** aynı karede aynı `track_id` iki kişiye verilemez; iç tutarsızlık olursa `TrackerBackendError` fırlatılır (testle sınanır).
* `track_id` ByteTrack çıktısında `float32` olduğundan `2**24`'ten büyük değerler reddedilir (takipçiyi sıfırlayın).

## 8. Veri yapıları (`results.py`)

* **`TrackedPerson`**: `track_id`, `class_id`, `class_name`, `confidence` (eşleşen YOLO tespitinin skoru; **takip doğruluğu değildir**), `bbox_xyxy` (takipçinin Kalman kutusu, görüntüye kırpılmış piksel), `frame_index`, `timestamp` (`FramePacket.timestamp`, monotonic), `session_id`, `detection_index` (aynı karenin `DetectionResult.detections` içindeki ham tespit), `age_frames`. `key = (session_id, track_id)`.
* **`LostTrack`**: `track_id`, `frames_since_seen` (görünmez; yalnızca tampon durumunu izler).
* **`TrackingResult`**: `tracks` (aktif; `track_id`'ye göre sıralı, benzersiz), `lost_tracks`, `tentative_count`, `detection_count`, `image_size`, `frame_index`, `timestamp`, `session_id`, `tracking_ms`, `dropped_tracks`, `reset_reason`; özellikler `active_count`, `lost_count`, `track_ids`.
  **`detection_count` ≠ `active_count` ≠ `lost_count`**; karıştırılmamalıdır.
* **`TrackerStats`** (`PersonTracker.stats`): `frames`, `sessions`, `resets`, `gap_frames`, `dropped_tracks`, `distinct_tracks`, `max_active`.

## 9. Canlı görüntüleme

`draw_person_tracks` girdi görüntüsünün **kopyasına** çizer. Üst şerit: `DETECTIONS`, `ACTIVE TRACKS`, `LOST TRACKS` (ayrı), `SOURCE`, `RESOLUTION`; sağda `FPS`, `INFERENCE`, `TRACKING` (ayrı süreler), `DEVICE`. Her aktif kişinin kutusu ve `ID n 0.87` etiketi çizilir; kayıp/onaysız izler çizilmez.
Renk `track_id`'ye göre sabittir ama **kimlik/doğruluk anlamı taşımaz** (12 renk; iki farklı kişi aynı renkte olabilir). Alt şeritte `Q / Esc / Close to exit` ve `ID = temporary track number (not an identity)`. Pencere ve macOS kırmızı kapatma düğmesi düzeltmesi kamera paketinin `PreviewWindow` sınıfındandır (yeniden yazılmadı).
Gerçek pencereli isteğe bağlı testle (`ISG_GUI_TESTS=1`) kırmızı düğme tıklaması 4B döngüsünü bitirdi ve kaynaklar serbest kaldı. Çizilmiş örnek karelere (sentetik montaj) gözle bakıldı: sayılar ve ID'ler beklenen biçimde göründü; bu bir görsel kontroldür, otomatik doğruluk ölçümü değildir.

## 10. Komut satırı

```bash
.venv/bin/python scripts/run_person_tracking.py --source camera --camera-index 0 --device mps
.venv/bin/python scripts/run_person_tracking.py --source camera --camera-index 0 --device cpu
.venv/bin/python scripts/run_person_tracking.py --source video --video-path /yol/video.mp4 --device mps
.venv/bin/python scripts/run_person_tracking.py --source camera --camera-index 0 --device mps --no-display --max-frames 300
.venv/bin/python scripts/run_person_tracking.py --source camera --camera-index 0 --device mps --conf 0.25
.venv/bin/python scripts/run_person_tracking.py --help
```

Kaynak/model seçenekleri `run_person_detection.py` ile birebir aynıdır (4A'nın `build_parser`'ı yeniden kullanılır). Takip eşikleri doğrulanmış aralıklarla açıktır: `--track-high-thresh`, `--track-low-thresh`, `--new-track-thresh` (0 < x ≤ 1), `--track-buffer` (1–1000 kare), `--match-thresh`, `--no-fuse-score`;
tutarsız kombinasyonlar çıkış kodu 2 ile reddedilir. Çıkış kodları: 0 normal (Q/Esc/pencere kapatma/video sonu/`--max-frames`), 1 kamera/video/pencere/model/takip hatası, 2 hatalı argüman, 130 `Ctrl+C`.
Sıra: YOLO yüklenir → takipçi yüklenir (bağımlılık denetimi) → kaynak açılır → ısıtma → pencere → döngü; model veya takipçi başarısızsa **kamera hiç açılmaz**. Bitişte iz özeti ve ayrı süre istatistikleri yazılır; görüntü kaydedilmez.

## 11. Performans ölçümünün tanımı ve sonuçları (ölçülmüş; yalnızca bu makine ve oturum)

Tanımlar (`tracking/pipeline.py` ve `tracking/benchmark.py` açıklamaları): `read`, `inference` (YOLO; MPS'te senkronize), `tracking` (`PersonTracker.update`: dönüştürme + ByteTrack + sonuç), `detect_plus_track`, `frame_to_result` (`FramePacket.timestamp` → takip sonucu), `draw`, `show`, uçtan uca FPS (`(N-1)/süre`, ölçülen monotonic zaman; kameranın bildirdiği FPS kullanılmaz).
Model yükleme, takipçi yükleme, ısınma ve burn-in sürekli işlemden **ayrı** raporlanır. Yöntem: aynı video/model/çözünürlük/eşikler; her geçişte `detect_only` (4A döngüsü), `track`, `track_draw` (boş pencere: çizim maliyeti, `imshow` değil) modları ve cihazlar **dönüşümlü** sırayla koşulur.

**Ortam:** MacBook M4, macOS 26.5.2, Python 3.11.17, torch 2.15.0, ultralytics 8.4.210, OpenCV 5.0.0. **Video:** 4A'nın sentetik montajı (`results/person_detection/sentetik_cok_kisili_1280x720.mp4`, 300 kare, 1280×720, kare başına ort. 6,52 / en çok 8 tespit; **Ultralytics örnek görüntülerinden montaj, gerçek bir çalışma alanı değildir**).
5 geçiş, medyanlar (`results/person_tracking/benchmark_20261112_101837.json`):

| Cihaz | FPS yalnızca-tespit | FPS tespit+takip | FPS tespit+takip+çizim | YOLO çıkarım medyan (ms) | **ByteTrack medyan (ms)** | Çizim medyan (ms) | Kare→sonuç medyan (ms) | Takip ek yükü, FPS farkından (ms/kare) |
|---|---|---|---|---|---|---|---|---|
| CPU | 51,3 | 49,3 | 49,1 | 18,6 | **0,41** | 0,45 | 19,1 (takipsiz: 18,7) | 0,78 |
| MPS | 101,2 | 92,2 | 89,1 | 9,9 | **0,40** | 0,41 | 10,4 (takipsiz: 9,4) | 0,96 |

* **ByteTrack güncellemesi ≈ 0,4 ms/kare** (6–8 kişide; geçişler arasında 0,32–0,42 ms, kararlı) — YOLO çıkarımının (CPU ≈ 18,6 ms, MPS ≈ 9,9 ms) **%2–4'ü**. Model yükleme: CPU 0,45 sn, MPS 0,06 sn; takipçi yükleme ≈ 5 ms (Ultralytics, model yüklenirken zaten import edilmişti; yeni bir süreçte ilk `import` ≈ 0,5 sn sürer); ısınma CPU 0,51 / MPS 0,31 sn.
* **FPS farkından türetilen ek yük (0,8–1,0 ms) doğrudan ölçülen 0,4 ms'den büyüktür:** geçişler arası kare süresi gürültüsü (CPU yalnızca-tespit 18,8–20,0 ms; MPS 9,2–10,5 ms) bu farkla aynı mertebededir; fark ölçüm gürültüsü + döngü işleri içindedir. Güvenilir sayı doğrudan ölçülen ByteTrack süresidir. Güç/ısıl durum ve arka plan süreçleri kontrol edilmedi.
* **Ölçeklenme (YOLO yok, CPU, betiklenmiş tespitler; iki ölçüm):** 1 kişi 0,10 ms; 5 kişi 0,20–0,21; 10 kişi 0,32–0,34; 20 kişi 0,55–0,56; 50 kişi 1,23–1,26; 100 kişi 2,40–2,48 ms — yaklaşık doğrusal (≈ 0,024 ms/kişi).
* Gerçek CLI, penceresiz, tek koşu: MPS 103,9 FPS (çıkarım medyan 9,1; ByteTrack 0,3; toplam 9,4 ms), CPU 51,7 FPS (18,3; 0,4; 18,7 ms).
* **4A ile karşılaştırma** (farklı oturum; aynı video, aynı model): 4A kayıtlı benchmark'ı CPU 46,6–52,5 FPS / çıkarım medyan 18,7–20,4 ms, MPS 100,0–110,7 FPS / 8,5–9,7 ms. 4B değerleri aynı aralıklardadır; takip yükü bu gürültünün içinde kalan ≈ 0,4 ms'dir. 4A'nın **canlı kamera** sonuçları (30,0 FPS penceresiz; pencereli 18,4–25,3) kullanıcının testleridir ve 4B performansı olarak sunulmaz.
* **Ölçülmeyenler:** canlı kamerada FPS/gecikme (kamera izni Claude ortamında yoktur; bkz. §17), `imshow`/pencere maliyeti (gerçek pencerede kullanıcının makinesinde), kare kaybı, uzun süreli (>5 dk) kararlılık. Kesin 30 FPS zorunluluğu konmadı; darboğaz YOLO çıkarımıdır, takip değil, bu yüzden optimizasyon yapılmadı.

## 12. Yanlış pozitifler ve yorgan (FP-4A-001)

Gerçek yorgan görüntüsü kaydedilmedi; **olay birebir yeniden üretilmedi.** Aşağıdakiler kontrollü sahte tespit dizileriyle **gerçek ByteTrack** üzerinde ölçülen davranıştır (testlerde sabitlenmiştir):

| Yanlış pozitif deseni | ByteTrack davranışı (8.4.210) |
|---|---|
| Tek karelik (oturumun 1. karesi değil) | **İz görüntülenmez** (güven skoru 0,3–0,95 fark etmez); ID numarası tüketilir; kayıp iz de kalmaz |
| Oturumun **1. karesinde** tek karelik | Bir kare **görüntülenir** (onay beklenmez), sonra tamponda kaybolur |
| 2–3 ardışık kare | 1–2 kare görüntülenir; sonra `track_buffer` kare boyunca kayıp olarak tutulur |
| 1-0-1-0 titreşim | **Hiç onaylanmaz** |
| 2 kare var, 1 kare yok (tekrarlı) | Onaylanır ve **tek, kararlı ID** ile sürer |
| Kalıcı/statik | **Sürekli ID alır** (60 karede kesintisiz). ByteTrack kalıcı yanlış pozitifi ELEMEZ |
| Düşük güvenli (0,10–0,25) | Yeni iz doğurmaz; ama mevcut izi sürdürür |

**Sonuç:** ByteTrack yanlış pozitifi **engellemez**; yalnızca ardışık iki karede görülmeyen tespitlere iz vermez. Yorgan gibi **sürekli** tespit edilen bir nesne Track ID alır ve tutar; kısa süreli gözlem (kullanıcının raporuna göre "kısa süreliğine") yalnızca ≥ 2 ardışık karede (30 FPS'te ≥ ~67 ms) tespit edildiyse ID alır.
Takip tek başına çözüm değildir; çözüm tespit kalitesinde veya sonraki doğrulamalardadır (yüz doğrulaması, KKD, hareket/boyut süzgeçleri). Takibin FP-4A-001'e etkisi gerçek yorganla **ölçülmemiştir** (§17, Manuel Test 7).

## 13. Testler

`.venv/bin/python -m unittest discover -s tests` — **883 test** (4A kapanışındaki 596 + 287 yeni). Varsayılanda **7 atlanır** (hepsi isteğe bağlı gerçek-pencere GUI testleri: Aşama 3'ten 3, 4A'dan 2, 4B'den 2); `ISG_GUI_TESTS=1` ile 883 test, 0 atlanan. Gerçek kamera açılmaz, izin istenmez, ağ kullanılmaz.
Yeni 287 test, kanıt türüne göre **ayrı** raporlanır:

| Tür | Dosya (test sayısı) | Ne kanıtlar / kanıtlamaz |
|---|---|---|
| Mock/birim (sahte backend, tespitçi, takipçi, kamera, pencere) | `test_person_tracking_config` (17), `_results` (19), `_adapter` (51), `_visualization` (21), `_pipeline` (30), `_cli` (47), `_benchmark` (18) = **203** | Politika, doğrulama, kare sırası, oturum, hata ve bağımlılık yönetimi, ölçüm yöntemi, çıkış yolları, kaynak temizliği. **Algoritmanın veya gerçek takibin çalıştığını kanıtlamaz** |
| Gerçek ByteTrack, sahte (betiklenmiş) tespit | `test_person_tracking_bytetrack_real` (59) + pipeline (2) + benchmark (1) = **62** | Senaryolar A–J: tek/çoklu kişi, yer değiştirme, örtüşme, çıkış-giriş, boş kare, kaçırma, yanlış pozitif, eşikler, kaynak değişimi, hatalı girdi. **YOLO yok, kamera yok; doğruluk kanıtı değildir** |
| Gerçek YOLO11n + gerçek ByteTrack | `test_person_tracking_real_model` (**20**: 12 CPU, 8 MPS) | Gerçek ağırlıkla zincir; kare başına tam bir YOLO çıkarımı (model `predict` çağrıları sayılır); çevrimdışı; ağırlık SHA-256 değişmedi; gerçek betik alt süreçte. **Örnek görüntü/sentetik montaj; gerçek kamera/IDF1 kanıtı değildir** |
| İsteğe bağlı gerçek pencere | `test_person_tracking_pipeline` (2) | Kırmızı kapatma düğmesi 4B döngüsünü bitirir (`ISG_GUI_TESTS=1`) |

"Aynı kare için YOLO yalnızca bir kez çağrılır" mimari güvencesi iki seviyede sınanır: gerçek `PersonDetector` + sahte YOLO modeli (`predict` sayısı = kare sayısı) ve gerçek YOLO modeli (`predict` sarmalanıp sayılır). Eksik paket/model nedeniyle atlanan testler (`lap`, ağırlık, MPS yoksa) açıkça atlama nedeniyle raporlanır.
Aşama 2 bütünlük denetimi 12/12, `check_environment.py` başarılı.

## 14. Takip başarı metrikleri

| Ölçüt | Durum |
|---|---|
| ID switch sayısı | Kimlik **bilinen** kontrollü dizilerde sayılır (`count_id_switches`; testler ve §7). Gerçek kamerada/etiketsiz videoda **ölçülmedi** |
| IDF1 / MOTA / HOTA | **Ölçülmedi** (etiketli takip verisi yok; hesaplayan kod da yazılmadı) |
| Takip sürekliliği | Yalnızca betiklenmiş/sentetik dizilerde gözlendi; gerçek ortamda **ölçülmedi** |
| FPS ve takip gecikmesi | Kayıtlı videoda ölçüldü (§11); canlı kamerada **ölçülmedi** |

`distinct_tracks` (görülen farklı ID sayısı) ve iz özeti ID parçalanmasına dair **ipucudur**, doğruluk ölçüsü değildir. YOLO güven skoru takip doğruluğu değildir. %85 başarı hedefinin karşılandığı iddia edilmez.
**İleride küçük, kontrollü, izinli bir değerlendirme videosu hazırlamak için:** (1) katılımcıların yazılı izniyle, yalnızca bu amaçla 1–3 dakikalık, 3 kişilik video kaydedin (başka kimse görünmesin; dosya `data/` altında, Git dışında kalsın, paylaşılmasın); (2) senaryoları sabitleyin: yürüme, yer değiştirme, birinin önünden geçme, kadrajdan çıkıp girme;
(3) her kişiye kare aralıkları için elle gerçek kimlik (A/B/C) atayın (yalnızca kimlik değişimi anlarını işaretlemek yeterlidir); (4) `run_tracking`'in `on_result` kancasıyla kare başına `(gerçek_kimlik, track_id)` eşlemesini alıp `count_id_switches`'e verin. IDF1/HOTA için kutu etiketleri de gerekir ve ayrı bir iş olarak planlanmalıdır; kapsam büyük veri toplama projesine dönüştürülmemelidir.

## 15. Gizlilik ve güvenlik

Görüntü varsayılan olarak **kaydedilmez ve arşivlenmez**; hiçbir görüntü/iz ağa gönderilmez. Track ID'ler bellekte tutulur, diske yazılmaz; `(session_id, track_id)` oturum sonunda anlamını yitirir. Benchmark JSON'ları (`results/person_tracking/`, Git dışı) yalnızca sayısal ölçümler içerir.
Sentetik montaj ve örnek görüntüler Ultralytics paketinin kendi dosyalarıdır. Yeni bağımlılık (`lap`) kaynağı ve hash'iyle kaydedildi (§4). Model ağırlığı Git'e eklenmez.

## 16. Bilinen sınırlamalar

1. **Track ID kimlik değildir**; ID switch olabilir. ByteTrack görünüm (ReID) kullanmaz: hızlı yer değiştirmede numaralar konuma yapışır; tampon içinde kaybolan kişinin konumuna giren biri eski ID'yi devralır.
2. **Kalıcı yanlış pozitif iz alır** (§12); ByteTrack tek karelik tespitleri eler, kalıcı olanları elemez. Gerçek yorganla davranış ölçülmedi.
3. **Kadrajdan çıkıp aynı kenardan giren kişi, tampon dolmadan bile yeni ID alır**; hareket ederken ortada kaybolup tahmin edilen konumda dönen kişi ID'sini korur.
4. **Düşük güven kurtarması varsayılan olarak fiilen kapalıdır** (`conf 0.25 ≥ track_high_thresh 0.25`); açmak yanlış pozitif ömrünü uzatır, bu videoda yararı gösterilemedi.
5. **`track_buffer` kare sayısıdır**; pencereli düşük FPS'te (4A: 18–25 FPS) saniye cinsinden daha uzun, yüksek FPS'te daha kısadır.
6. Oturumun 1. karesindeki tespitler onaysız iz alır; kutular Kalman tahminidir, ham YOLO kutusuyla birebir aynı değildir. İdealize (gürültüsüz, sabit hızlı) sentetik harekette ölçülen en büyük sapma ≈ 0,2 kare hareketi (oturduktan sonra ≈ 0,01); **gürültülü gerçek tespitlerde ölçülmedi**.
7. `BYTETracker`/`Boxes` Ultralytics'in dahili API'sidir; sürüm değişirse davranış değişebilir (§3).
8. Tüm gerçek-ByteTrack senaryoları **idealize sentetik** hareketlerdir (gürültüsüz/düzgün); "ID switch yok" sonuçları gerçek videoyu temsil etmez. Gerçek model testleri örnek görüntülerden üretilmiştir.
9. Canlı kamera performansı, gerçek IP kamera (OT-4A-003, `DEFERRED_TO_SYSTEM_INTEGRATION`), kare kaybı ve uzun süreli kararlılık ölçülmedi.
10. AGPL-3.0 (Ultralytics/YOLO11n): ürünleştirilirse lisans yükümlülükleri yeniden değerlendirilmelidir (4A raporu §3).
11. **Basılı insan fotoğrafı (pano/poster) insan sanılabilir ve Track ID alır** (FP-4B-001, kullanıcının canlı gözlemi; §19). Çakışan büyük kutular gözlendi, kaynağı ayrıştırılmadı (§21).
12. **Pencereli mod ağırdır:** bu makinede `imshow+waitKey(1)` tek başına ≈ 15–17 ms ölçüldü (§20.4); kullanıcının pencereli canlı koşusu 16,3 FPS verdi, penceresiz 29,9 FPS. Pencereli FPS'in tam nedeni ölçülmedi.

## 17. Manuel MacBook test planı (kullanıcı tarafından yapılacak; Claude ortamında kamera izni yoktur)

**Durum (2026-11-12, kullanıcı bildirimine göre; bildirilmeyen = yapılmadı sayılır):**

| # | Test | Durum |
|---|---|---|
| 1 | Tek kişi | Kontrol listesi **doldurulmadı** (bildirilen sahnede iki gerçek kişi vardı) |
| 2 | Üç kişi | **Yapılmadı** |
| 3 | Yer değiştirme / ID switch | **Yapılmadı** |
| 4 | Kısmi örtüşme | **Yapılmadı** (çakışan büyük kutular gözlendi, bu bir örtüşme senaryosu testi değildir; §19) |
| 5 | Kadrajdan çıkma | **Yapılmadı** |
| 6 | Yüksek kamera açısı | **Bildirilmedi** (yapılmadı sayılır) |
| 7 | Yorgan yanlış pozitifi | **Yapılmadı**; bunun yerine **poster** yanlış pozitifi gözlendi (FP-4B-001, §19) |
| 8 | Çıkışlar | **Q: sorunsuz** *[Kullanıcı]*. Esc, kırmızı düğme, Ctrl+C, çıkış kodu ve kameranın serbest kalması **bildirilmedi** |
| — | Canlı performans | Pencereli MPS (2924 kare) ve penceresiz CPU/MPS (300 kare) **yapıldı**; okuma/çizim/gösterme/kare→sonuç satırları ve pencereli CPU **bildirilmedi** (§19) |

**Hazırlık.** Kamera iznini Terminal/VS Code/Claude uygulamasına Sistem Ayarları → Gizlilik ve Güvenlik → Kamera'dan verin. Tüm testlerde önce kısa bir denetim: `.venv/bin/python scripts/run_person_tracking.py --help`.
Pencereli test: `... --source camera --camera-index 0 --device mps`; penceresiz sayısal ölçüm: `... --no-display --max-frames 300`. Sonuçları **gerçek gözleminizle** doldurun; gözlem yapılmadan sonuç yazmayın.

| # | Test | Yapılacak | Kontrol listesi |
|---|---|---|---|
| 1 | Tek kişi | Kameranın önünde sağa, sola, ileri, geri yürüyün | Track ID atandı mı? Hareket boyunca korundu mu? Kutu kişiyle birlikte hareket etti mi? |
| 2 | Üç kişi | Üç kişi aynı anda görüntüde | `ACTIVE TRACKS: 3`? Üç farklı ID? Aynı karede aynı ID iki kişide var mı? |
| 3 | Yer değiştirme | Kişiler yer değiştirsin (önce yavaş, sonra hızlı) | ID'ler korundu mu? ID switch (iki kişinin numaraları değişti mi)? Hangi hızda oldu? |
| 4 | Kısmi örtüşme | Biri diğerinin önünden geçsin | Takip sürdü mü? Kimlik karıştı mı? Örtülen kişi `LOST TRACKS`'te göründü mü, aynı ID ile döndü mü? |
| 5 | Kadrajdan çıkma | Kişi çıkıp tekrar girsin: (a) ~0,5 sn sonra, (b) ~3 sn sonra, (c) aynı kenardan, (d) başka kenardan | Hangi koşullarda **yeni ID** oluştu? (Beklenti: aynı kenardan dönüşte tampon dolmadan bile yeni ID olabilir, §7) |
| 6 | Yüksek kamera açısı | MacBook'u güvenli biçimde ≈ 2 m yüksekliğe yerleştirin; 1–3 testi tekrarlayın | ID'ler korundu mu? Yüksek açı sürekliliği etkiledi mi? `INFERENCE` ve `TRACKING` süreleri? |
| 7 | **Yorgan yanlış pozitifi** | 4A'daki koşullarda (≈ 2 m, köşedeki yorgan) varsayılan ayarla, sonra `--new-track-thresh 0.6` ile | Yorgan insan algılandı mı? **Track ID aldı mı?** Kaç kare/saniye sürdü (konsol "İz özeti")? Ayarlar arasında fark? Gerçek kişi `--new-track-thresh 0.6`'da hâlâ iz alıyor mu? |
| 8 | Çıkışlar | `Q`, `Esc`, macOS kırmızı kapatma düğmesi, terminalde `Ctrl+C` | Pencere kapandı mı? Kamera ışığı söndü mü? Konsol "Bitiş nedeni" doğru mu? Çıkış kodu (`echo $?`): 0 / 0 / 0 / 130 |

Sayısal ölçüm (her cihaz için): `.venv/bin/python scripts/run_person_tracking.py --source camera --camera-index 0 --device mps --no-display --max-frames 300` ve `--device cpu`; çıktıdaki "uçtan uca FPS", "YOLO çıkarım süresi", "ByteTrack takip süresi", "Kare -> takip sonucu gecikmesi", "Görülen farklı Track ID" ve iz özetini not edin.
Pencereli çalıştırmada "Pencereye gösterme süresi" `imshow` maliyetini verir. 4A'daki pencereli FPS düşüşü (18,4–25,3) nedeni ayrıştırılmamıştı; 4B'de bunu da gözlemleyin.
**Çıkış doğrulaması:** her çıkış yolundan sonra kameranın serbest kaldığını (yeşil ışık/menü çubuğu göstergesi) ve ikinci çalıştırmanın kamerayı açabildiğini kontrol edin.

### 17.1 Kayıt şablonu — iki kişi, ID kararlılığı ve örtüşme (Manuel Test 1, 3, 4; yeni ortamda)

Amaç: iki gerçek kişide Track ID'nin korunup korunmadığını ve örtüşmede ne olduğunu **gözleme dayalı** kaydetmek. Bu şablon testlerin yerine geçmez; yalnızca sonucun yorumlanabilir olmasını sağlar. Gözlem yapılmadan hiçbir sonuç doldurulmamalıdır.

**Hazırlık (her koşu için ortak):** `.venv/bin/python scripts/run_person_tracking.py --source camera --camera-index 0 --device mps`; ortamı yazın (ışık, kameraya mesafe, arka planda poster/telefon/başka kişi var mı); iki kişiyi ayırt edecek bir işaret belirleyin (örn. A kırmızı, B siyah üst) ve **koşu başında ekran görüntüsüyle A=ID?, B=ID?** kaydedin.

| Senaryo | Ne yapılır (≈ 20–30 sn) | Kaydedilecek |
|---|---|---|
| S1 İki kişi durağan | İkisi yan yana, kıpırtısız | A/B ID'leri; ID değişti mi, `ACTIVE` sürekli 2 mi |
| S2 Yan yana yürüme | İkisi birlikte yavaşça sola-sağa | ID'ler korundu mu; `ACTIVE`/`DETECTIONS` |
| S3 Önünden geçme | B, A'nın önünden yavaşça geçer (kısmi örtüşme) | Örtüşme sırasında `ACTIVE` ve `LOST`; örtüşme sonrası A/B ID'leri aynı mı |
| S4 Yer değiştirme | Yavaş, sonra hızlı yer değiştirme | ID'ler korundu mu / **ID switch (A↔B numaraları değişti mi) kaç kez** |
| S5 Kadrajdan çıkıp girme | Biri çıkar, 1–2 sn / 5 sn sonra geri girer | Geri girişte aynı ID mi yeni ID mi; hangi kenardan |

**Her koşu sonunda (Q ile çıkıp) konsoldan şu satırları kopyalayın:** "İşlenen kare … uçtan uca FPS", "YOLO çıkarım süresi", "ByteTrack takip süresi", "Kare okuma süresi", "Çizim süresi", "Pencereye gösterme süresi", "Kare -> görüntü gecikmesi", "Tespit: kare başına ortalama / en çok … aktif iz en çok … kayıp iz en çok", "Görülen farklı Track ID" ve **iz özeti** tablosu (A ve B'nin ID'lerinin kaç kare aktif kaldığı; kısa, düşük güvenli ek izler var mı). Beklenmedik bir kutu (poster, telefon, çok küçük kişi) görürseniz ekran görüntüsüyle ID'sini ve güvenini not edin.
Sonuç ifadeleri: "ID korundu / ID switch N kez / yeni ID aldı" gibi **gözleminizi** yazın; sayısal doğruluk ölçümü (IDF1/HOTA) değildir.

## 18. Aşama 5'e aktarılacak veri yapıları

* `TrackingResult` / `TrackedPerson`: kare başına kişi kutuları (`bbox_xyxy`, piksel), `track_id`, `session_id`, `frame_index`, `timestamp`; yüz tespiti yüz kutularını kişi izlerine **kutu içerme/örtüşmeyle** ilişkilendirirken `(session_id, track_id)` anahtarını kullanmalıdır.
* `run_tracking(..., on_result=lambda paket, tespit, takip: ...)` kancası: `FramePacket` (görüntü), `DetectionResult` (ham tespitler) ve `TrackingResult` aynı kare için birlikte verilir.
* `TrackedPerson.detection_index`: ham YOLO tespitine dönüş. `LostTrack`: kimlik atamalarının ne zaman bekletileceği/silineceği için kayıp tampon bilgisi.
* **Uyarılar:** ID'ler oturum kapsamlıdır; kimlik tanıma sonuçları bir `track_id`'ye bağlanıyorsa `reset_reason`/`session_id` değişiminde ve iz silinince geçersiz sayılmalıdır. `track_id` bir kimliğe dönüştürülürken ID devralma/switch olasılığı (§7, §16) göz önünde tutulmalı, tek bir kare yerine birden fazla karedeki yüz eşleşmesiyle doğrulanmalıdır.

## 19. Gerçek MacBook kamerası gözlemleri (kullanıcı, 2026-11-12) *[Kullanıcı]*

Claude bu koşuları yeniden üretemez (kamera izni yok); sayılar kullanıcının bildirdiği değerlerdir. Ortalama mı medyan mı oldukları belirtilmedi (4A medyan raporlamıştı); aşağıda "sayı" olarak anılır.

| | 1) Pencereli, MPS | 2) Penceresiz, CPU | 3) Penceresiz, MPS |
|---|---|---|---|
| Çözünürlük / kare | 1280×720 / 2924 | — / 300 | — / 300 |
| Süre | 178,9 sn | — | — |
| Uçtan uca FPS | **16,3** | **29,9** | **29,9** |
| YOLO çıkarımı | 17,7 ms | 18,5 ms | 14,7 ms |
| ByteTrack | 0,5 ms | 0,2 ms | 0,4 ms |
| YOLO + ByteTrack | — | 18,8 ms | 15,1 ms |
| Farklı Track ID | **30** | 2 (ikisi de 300 kare aktif) | 2 (300 ve 299 kare aktif) |
| Çıkış | Q, sorunsuz | — | — |

**Görsel gözlem (ekran görüntüsü, kullanıcı):** sahnede iki gerçek kişi varken **dört aktif Track ID**; panodaki **basılı bir erkek fotoğrafı ID 167 olarak** algılandı; gerçek kişilerin üzerinde de **çakışan büyük takip kutuları** oluştu.
**Bildirilmeyenler (yorumu sınırlar):** ekrandaki `DETECTIONS` / `ACTIVE TRACKS` / `LOST TRACKS` değerleri ve kutuların güvenleri; konsol özetindeki "Kare okuma süresi", "Çizim süresi", "Pencereye gösterme süresi", "Kare → görüntü / takip sonucu gecikmesi", "Tespit: kare başına ortalama / en çok", "kayıp iz en çok" satırları ve iz özeti; koşulardaki sahnenin aynı olup olmadığı; kamera/ışık koşulları.

**Doğrulanabilen aritmetik** *[Türetilmiş]*: (2924−1)/178,9 = 16,34 FPS (bildirilen 16,3 ile tutarlı). Kare periyodu: pencereli 61,3 ms; penceresiz 33,4 ms (29,9 FPS).

## 20. Analiz

### 20.1 CPU ↔ MPS karşılaştırması (penceresiz canlı koşular 2 ve 3)

| Büyüklük | CPU | MPS | Fark |
|---|---|---|---|
| Uçtan uca FPS | 29,9 | 29,9 | 0 |
| YOLO çıkarımı | 18,5 ms | 14,7 ms | MPS −3,8 ms (**−20,5%**; oran 1,26×) |
| ByteTrack | 0,2 ms | 0,4 ms | MPS koşusunda +0,2 ms (§20.5) |
| YOLO + ByteTrack | 18,8 ms | 15,1 ms | MPS −3,7 ms (−19,7%) |
| Yalnız hesaplama için üst sınır FPS (1000/ms) *[Türetilmiş]* | 53,2 | 66,2 | — |
| 33,4 ms'lik kare periyodunun hesaplamaya giden payı *[Türetilmiş]* | %56 | %45 | — |
| Periyodun geri kalanı (okuma bekleme + diğer) *[Türetilmiş]* | ≈ 14,6 ms | ≈ 18,3 ms | MPS'te daha çok boşta/bekleme |
| Aktif kareler | 300, 300 | 300, 299 | 1 karelik fark canlı karelerin farklı olmasından beklenir; cihaz etkisi olarak yorumlanmaz |

* **YOLO çıkarımı:** MPS CPU'dan canlıda ≈ %20 hızlıdır. 4A'nın canlı sonuçlarıyla (CPU 18,5 / MPS 13,4 ms, medyan) karşılaştırma: CPU aynı; MPS 4B'de 1,3 ms (≈ %10) yüksek (farklı oturum, ortalama/medyan farkı olabilir; 4B'nin YOLO'ya etkisi yoktur, takip ayrı çalışır).
* **Kayıtlı videodaki (§11) fark çok daha büyüktü** (CPU 18,6 / MPS 9,9 ms; oran ≈ 1,9×). Canlıda MPS avantajının neden daraldığı §20.3'te incelenir.
* Hesaplama süresinin (≤ 18,8 ms) kare periyodundan (33,4 ms) küçük olması iki cihazın da hesaplamaya bağlı **olmadığı** anlamına gelir.

### 20.2 İki testte FPS neden aynı (29,9)?

**Ölçümlerin desteklediği sonuç:** hiçbir koşu hesaplamaya bağlı değildi. Hesaplama süreleri (CPU 18,8; MPS 15,1 ms) 33,4 ms'lik periyodun altındadır; hesaplamaya bağlı olsalardı CPU ≤ 53, MPS ≤ 66 FPS'e kadar farklı olabilirlerdi. İkisi de 29,9'da kaldığı için FPS, **iki cihazı ayırt etmeyen bir üst sınırla** (kare teslim hızı) belirlenmiştir. Dolayısıyla bu yapılandırmada cihaz farkı FPS'te değil **çıkarım süresinde ve boşta kalan süre payında** görünür.
**Bu üst sınırın kameraya ait olduğunu destekleyen kanıtlar (kesin kanıt değil):**
1. Kamera paketinin kendi ölçümü: **işlem yokken ham kamera 29,98 FPS** (kullanıcı, 300 kare / 9,97 sn; medyan kare aralığı 33,3 ms) — bkz. `docs/camera_module.md`.
2. Kameasız model deneyi *[Ölçüldü; dosya kaynağı + yapay 33,3 ms teslim, canlı kamera DEĞİL]*: aynı video, penceresiz, 300 kare: kayıtlı hızda CPU ≈ 50, MPS ≈ 104–107 FPS; **33,3 ms aralıkla her iki cihaz da 30,0 FPS**; okuma beklemesi CPU ≈ 13,9–14,1 ms, MPS ≈ 17,3–18,5 ms — bu, kullanıcı koşularından türetilen 14,6 / 18,3 ms ile uyumludur.
**Desteklenmeyen / kesin olarak söylenemeyenler:** canlı koşularda gerçek okuma bekleme süresi (bildirilmedi); kamera/AVFoundation'ın kare atlatıp atlatmadığı; 29,9'un donanım tavanı mı otomatik pozlama/ışık kaynaklı bir sınır mı olduğu (kamera aynı oturumda ham ölçülmedi). Bu yüzden "kamera 30 FPS ile sınırlıdır" ifadesi **güçlü bir çıkarımdır, doğrulanmış bir ölçüm değildir**. Doğrulama: aynı oturumda `run_camera.py --no-display --max-frames 300` ve iki koşunun "Kare okuma süresi" satırları (§21, T7).

### 20.3 MPS çıkarımı canlıda neden kayıtlı videodan yavaş? *[Ölçüldü, kameasız model deneyi]*

Aynı video, MPS, penceresiz, 150 kare; kareler arası yapay boşluk (kamera teslim hızını taklit eden bekleme) arttıkça:

| Kare aralığı | FPS | YOLO çıkarım medyan | ByteTrack medyan | Okuma (bekleme) ort |
|---|---|---|---|---|
| 0 (beklemesiz) | 102,3 | 9,2 ms | 0,3 ms | 0,3 ms |
| 15 ms | 66,8 | 10,6 | 0,4 | 3,8 |
| 25 ms | 40,0 | 11,9 | 0,5 | 12,5 |
| **33,3 ms** | **30,0** | **13,8** | **0,7** | 18,7 |
| 50 ms | 20,0 | 17,4 | 0,8 | 31,6 |
| 100 ms | 10,0 | 23,0 | 1,0 | 75,7 |

CPU kontrolü (33,3 ms aralık): çıkarım 19,1 → 18,9–19,1 ms, **değişmedi**. **Gözlem:** MPS çıkarım süresi (ve CPU'da çalışan ByteTrack süresi bile) kareler arası boşlukla düzenli artıyor; 33,3 ms aralıkta 13,8 ms, kullanıcının canlı 14,7 ms'siyle ve 4A'nın 13,4 ms'siyle uyumlu.
**Yorum sınırı:** bu, kamera ile yapılmış bir ölçüm değildir ve mekanizmayı (boşta kalma sonrası güç/saat durumu, zamanlayıcı uyanma gecikmesi vb.) **ölçmez**; yalnızca "boşta kalma payı arttıkça çıkarım uzuyor" ilişkisini gösterir. Canlıda MPS avantajının daralması bu ilişkiyle **uyumludur**, nedeni olduğu kanıtlanmış değildir (kamera/ISP yükü gibi başka etkenler ayrıştırılmadı).

### 20.4 Pencereli koşuda 16,3 FPS neden? *[kısmen ölçüldü; neden doğrulanmadı]*

* **Hesaplama toplamı yalnızca 18,2 ms'dir** (17,7 + 0,5; periyodun %30'u); kalan ≈ 43 ms'nin dökümü bildirilmedi *[Türetilmiş]*.
* **Doğrudan ölçülen (kameasız, bu makine, gerçek OpenCV penceresi):** yapay karelerle `imshow + waitKey(1)` tek başına **ortalama 17,3 ms (1280×720) ve 15,0 ms (640×480)** (p95 18,8 / 17,0 ms). Maliyet çözünürlüğe neredeyse bağlı değil (sabit bir maliyet gibi davranıyor; 60 Hz'de bir yenileme periyodu 16,7 ms'dir, ama **nedeni ölçülmedi**). Çizim süresi ≈ 0,4–0,7 ms (ihmal edilebilir).
* Kamera-benzeri tempolu (33,3 ms) gerçek pencereli koşu *[Ölçüldü; kamera YOK]*: **iki cihaz da 30,0 FPS**; gösterme süresi MPS 15,6, CPU 14,1 ms; çıkarım MPS 12,8, CPU 17,6 ms. Yani toplam (çıkarım + takip + çizim + gösterme: MPS ≈ 29,5, CPU ≈ 32,4 ms) 33,3 ms'lik kare periyodunun altında kaldığında pencere FPS'i düşürmedi (CPU'da bütçe sınıra yakındı).
* Kullanıcının canlı koşusunda ise çıkarım 17,7 ms (kameasız deneydeki 12,8 ms'nin ≈ 5 ms üstünde) ve toplam yaklaşık 35–36 ms *[yaklaşık, gösterme süresi bildirilmedi; yalnızca ölçülen ≈ 15–17 ms varsayılarak]*, yani periyodun sınırında/üstündedir. Gözlenen 61,3 ms periyodu (≈ 2 × 33,3 ms'ye yakın) bu toplamdan **beklenen ≈ 36 ms'den çok daha uzundur**.
**Hipotezler (hiçbiri doğrulanmadı):** (H1) kare bütçesi aşılınca kamera kare teslimi iki periyoda kuantize oluyor; (H2) canlı pencereli çalışmada `imshow`/`waitKey` veya pencere yöneticisi maliyeti ölçülenden büyük; (H3) kamera/ISP yükü çıkarımı ve okumayı uzatıyor; (H4) sahne farkı (daha çok tespit) çıkarımı/NMS'yi uzatıyor. **Ayırt edici testler:** §21, T7 (özellikle `--imgsz 320` pencereli tanı koşusu: bütçe 25 ms altına inince FPS ≈ 30'a dönerse H1; dönmezse H2/H3 öne çıkar). 4A'nın pencereli 18,4–25,3 FPS gözlemi de aynı ailedendir ve **hâlâ ayrıştırılmamıştır**. Takibin eklediği yük (0,5 ms) bu farkı açıklayamaz.
**Yeni veri noktası (takipsiz, yeni ortam, 4A betiği, MPS, 1812 kare; [person_detection.md §14.4](person_detection.md)) *[Kullanıcı]*:** 21,9 FPS, ort. çıkarım 15,9 ms → periyot 45,7 ms, çıkarım %35, kalan ≈ 29,8 ms *[Türetilmiş]*. Pencereli FPS düşüşü **takip olmadan da** görülüyor (ByteTrack nedeni değil). Farklı oturum/ortam olduğundan 4B'nin 16,3 FPS'iyle doğrudan karşılaştırılmaz; nedeni hâlâ ayrıştırılmadı (yalnızca ölçülen `imshow+waitKey(1)` ≈ 15–17 ms, bütçeyi kısıtlayan bilinen bir bileşen).

### 20.5 ByteTrack süresi: 0,2 ↔ 0,4 ↔ 0,5 ms
ByteTrack her zaman CPU'da çalışır; cihaz seçiminden bağımsızdır. İki kişilik sahnede ölçeklenme tablomuz (§11) ≈ 0,15–0,2 ms verir: CPU koşusu (0,2) bununla uyumlu. MPS koşularındaki 0,4–0,5 ms için olası etkenler (hiçbiri doğrulanmadı): tespit sayısının daha yüksek olması (poster/çakışan kutular ek tespit ve onaysız iz demektir), boşta kalma payının yüksek olması (§20.3'te ByteTrack süresi de boşlukla 0,3 → 0,7 ms'ye çıktı), ortalama/medyan farkı. Tüm durumlarda ByteTrack, kare periyodunun %1,5'inden azdır; performans darboğazı değildir.

### 20.6 "30 farklı Track ID" ve ID 167
Ekran görüntüsünde bir izin **ID 167** olması, bu oturumda o ana kadar **en az 167 iz oluşturulduğunu** gösterir (kimlik sayacı iz oluşturulunca artar, onay beklemez). Oturum boyunca yalnızca 30 farklı ID görüntülendiğine göre *[Türetilmiş]* yaklaşık **≥ 137 iz hiç onaylanmadı** (tek karelik/kararsız tespitler): ByteTrack'in iki-kare onay süzgeci bunları ekrandan eledi. Poster ise tespitin **kalıcı** olduğu bir örnektir ve süzgeçten geçti (§12 tablosuyla uyumlu). 30 ID'nin kaçının gerçek kişi parçalanması, kaçının yanlış pozitif olduğu **ayrıştırılmadı**. Not: konsol iz özeti en çok 20 izi gösterir; 30 izli bir oturumda 10 iz görünmez (ID 167 satırı tabloda olmayabilir).

## 21. Poster / çakışan kutu tanısı: YOLO mu, ByteTrack mi?

### 21.1 Şimdiden söylenebilenler
1. **Değişmez:** `ACTIVE TRACKS ≤ DETECTIONS`. Takipçi yalnızca bu karede bir tespitle eşleşmiş izleri döndürür (çıktı satırlarındaki `idx` tespit dizisine işaret eder); adaptör geçersiz `idx`'i ve yinelenen ID'yi hata sayar; gerçek-model testleri değişmezi denetler. **Dört aktif iz, o karede en az dört YOLO kutusu demektir.** (Ekrandaki `DETECTIONS` değerini bilmiyorum; kullanıcıdan istenir.) Yani ByteTrack kutu **uyduramaz**.
2. **ByteTrack çakışan/iç içe kutuları birleştirmez, bastırmaz:** gerçek ByteTrack + betiklenmiş tespit *[Ölçüldü]*: bir kişiyi çevreleyen büyük ikinci kutu (IoU = 0,49) ve 15 px yana kaymış ikinci kutu (IoU = 0,60) **ayrı, kararlı ID'ler** aldı; ikinci kutu 3 karenin 2'sinde gelirse bile onaylandı ve ID'sini korudu (aktif iz sayısı 1 ↔ 2 salındı). Yani YOLO'nun kararsız/yinelenen bir kutusu ByteTrack'te **kalıcı bir ek ID'ye** dönüşebilir.
3. **Takip kutusu, eşleştiği ham YOLO kutusuna yakındır** *[Ölçüldü; sentetik montaj, MPS, 1931 iz-karesi]*: IoU medyan 0,998, %1'lik dilim 0,884, **en düşük 0,824**. Bu örnekte Kalman süzgeci kutuyu en çok ≈ %18 IoU kaybı kadar oynattı; "büyük kutular" üretmedi. **Kullanıcı sahnesinde ölçülmedi.**
4. **Ham YOLO kutuları normal kişilerde bile çakışır** *[Ölçüldü; aynı montaj]*: 300 karenin tamamında çakışan çift var; en büyük IoU 0,34, en büyük içerme oranı (kesişim/küçük alan) 0,76 (uzanmış kol, yan yana kişiler). Yani çakışan büyük kutuların varlığı tek başına arıza belirtisi değildir; ID almaları ve gerçek kişi kutusunu örtmeleri sorun olabilir.
**Sonuç (kullanıcı sahnesi için):** çakışan/ek kutuların **kaynağının YOLO tespit katmanı olması beklenir** (1–4); ancak (a) poster kutusunun ve büyük çakışan kutuların gerçekten ham YOLO çıktısında bulunduğu ve (b) Kalman süzgecinin kutuları büyütmediği, kullanıcının sahnesinde **doğrulanmamıştır**. ByteTrack'in sorundaki rolü: ≥ 2 karede görülen her kutuya kalıcı ID vermek (yanlış pozitifleri süzmemek).

### 21.2 Hipotezler ve ayırt edici tahminler

| # | Hipotez | Doğruysa gözlenecek |
|---|---|---|
| H1 | Poster ham YOLO çıktısında insan olarak var | 4A betiğinde (takipsiz) poster kutusu, güvenle birlikte görünür |
| H2 | Büyük çakışan kutular ham YOLO kutularıdır (NMS `iou=0.7` yinelenen/iç içe kutuya izin verir; grup kutusu; poster+kişi örtüşmesi) | 4A betiğinde aynı kutular var; takip kutusu ↔ ham kutu IoU ≈ 1 |
| H3 | Büyük kutular Kalman/ByteTrack kaynaklı (örn. örtüşme sonrası yeniden eşleşme) | 4A betiğinde yok; takip kutusu ↔ ham kutu IoU ≪ 1 (§21.1 madde 3 bunu zayıflatır) |
| H4 | Ek ID'ler (30, 167) YOLO tespitinin kararsızlığından | Takipsiz sahnede tespit sayısı oynak; iz özetinde çok sayıda kısa iz |
| H5 | Ek ID'ler gerçek kişilerin parçalanmasından (ID switch/yeniden başlatma) | Gerçek kişi alanında art arda biten/başlayan izler; tespit sayısı sabit, ID değişir |

### 21.3 Test planı (kullanıcı koşar; Claude kamera erişemez)

Her adımda ekran görüntüsü veya konsol çıktısı paylaşılır. Kamera görüntüleri depoya girmez; kayıt alınacaksa katılımcıların izniyle, yalnızca yerelde (`data/` altında, Git dışı) tutulur ve iş bitince silinir.

| Adım | Amaç | Nasıl (komut) | Rapor edilecek | Karar kuralı |
|---|---|---|---|---|
| **T0** | Mevcut sahnede sayıları oku (kod yok) | `.venv/bin/python scripts/run_person_tracking.py --source camera --camera-index 0 --device mps` ile aynı sahneyi tekrar kur; sorunlu karede ekran görüntüsü al | `DETECTIONS`, `ACTIVE`, `LOST`; her kutunun `ID n güven` | `ACTIVE ≤ DETECTIONS` doğrulanır; hangi kutu poster, hangisi çakışan/ek kutu işaretlenir |
| **T1** | YOLO mu? (takipsiz karşılaştırma, kod yok) | Aynı sahne/cihaz/`--conf`: `.venv/bin/python scripts/run_person_detection.py --source camera --camera-index 0 --device mps` | Poster ve çakışan büyük kutular görünüyor mu, güvenleri; `PERSONS: n` | **Varsa → kaynak YOLO tespiti** (H1/H2). Posteri aralıklı algılanıyor olabilir: birkaç saniye izleyin, birkaç ekran görüntüsü alın; tek karede görünmemesi "yok" demek değildir. Uzun gözlemde 4A'da hiç yok ama 4B'de var ise §21.1 değişmeziyle çelişir; öncelikli hata incelemesi |
| **T2** | Poster güven dağılımı (yalnız poster, kimse yok) | Kamerayı posterin görüleceği şekilde sabitle; `... run_person_tracking.py --source camera --camera-index 0 --device mps --no-display --max-frames 1800` (≈ 60 sn) | Poster izi: kaç kare aktif, ort./en yüksek güven; tespit sayısı ort./en çok | Posterin güven aralığını verir (eşik kararının girdisi) |
| **T3** | Gerçek kişi güven dağılımı (aynı mesafe/ışık, poster yok veya görünmüyor) | Aynı komut; kişi(ler) hem hareketli hem durağan, ayrı ayrı 60 sn | Gerçek kişi izleri: ort./en yüksek güven | Posterin **en yüksek** güveni gerçek kişilerin **alt uç** güveninden **belirgin** düşükse eşikle ayrım düşünülebilir. Bugün yalnızca iz başına ortalama/en yüksek güven raporlanır; alt uç (p5) için kare bazlı log gerekir (§22, "araç" notu) |
| **T4** | Tekrarlanabilir klip | 20–30 sn'lik klibi (poster + 2 kişi + çakışma anları) QuickTime/Photo Booth ile kaydet (MP4/MOV) → `.venv/bin/python scripts/run_person_detection.py --source video --video-path KLİP --device mps --no-display` ve `... run_person_tracking.py --source video --video-path KLİP --device mps --no-display`; ardından `--device cpu` | İki betiğin kişi/iz sayıları; iz özetleri; cihazlar arası fark | Aynı kareler: cihaz ve takip etkisi ayrıştırılır; sonraki eşik denemeleri bu klipte tekrarlanabilir. (Not: kayıt sıkıştırması canlı kareden hafifçe farklıdır) |
| **T5** | Büyük kutu Kalman/ByteTrack kaynaklı mı? | Kare bazlı log (§22 araç notu) ile her izin kutusu ↔ eşleşen ham kutu IoU'su | IoU dağılımı, en düşük değerler, hangi karelerde | IoU ≈ 1 → YOLO kutusu (H2); belirgin düşük → Kalman/ByteTrack (H3). Araç eklenmeden: yalnızca T1 ile görsel ayrım |
| **T6** | ID kaynağı: kararsız tespit mi parçalanma mı? | T4 klibinde 4A betiğinin `PERSONS` oynaklığı ile 4B iz özeti (ID sayısı, kısa/uzun iz) | Tespit sayısı sabitse ID sayısı | H4 ↔ H5 ayrımı (§21.2) |
| **T7** | Pencereli FPS tanısı | (a) `... run_camera.py --source camera --no-display --max-frames 300` (ham kamera); (b) aynı sahnede `run_person_tracking.py` pencereli ve `--no-display` (MPS); konsolda "Kare okuma / Çizim / Pencereye gösterme / Kare→görüntü" satırlarını kaydet; (c) tanı: pencereli `--imgsz 320` (yalnızca tanı) | Satırlar; FPS | (a) ham kameranın tavanı; (b) gösterme süresi; (c) bütçe ≤ 25 ms'de FPS ≈ 30'a dönerse H1, dönmezse H2/H3 |

**T1 durumu (2026-11-12): KISMEN yapıldı, TAMAMLANMADI.** Kullanıcı 4A betiğiyle **yeni bir ortamda** gerçek kamera testi yaptı ([person_detection.md §14.4](person_detection.md)): ön plandaki iki kişi 0,85/0,87 ve 0,80/0,66 ile ayrı algılandı; telefonda gösterilen insan fotoğrafı ayrı tespit oluşturmadı; arka plandaki çok küçük kişiler kutulanmadı. **Ortam değiştiği için önceki poster aynı şartlarda yeniden kurulamadı; bu nedenle poster ve çakışan büyük kutuların ham YOLO çıktısında olup olmadığı (H1, H2) hâlâ doğrulanmamıştır.**
Telefondaki fotoğraf poster değildir (küçük ışık yayan ekran ↔ büyük basılı pano) ve tek karelik gözlem zaman içindeki yokluğu göstermez. T0, T2–T7 yapılmadı. Yeni ortamdaki 2,89 / en çok 7 kişi tespiti yanlış pozitif olarak yorumlanmaz (arka planda gerçek kişiler vardı).

Manuel Test 1–6 ve 8 (üç kişi, örtüşme, yer değiştirme, ID switch, kadrajdan çıkma, yüksek açı, Esc/kırmızı düğme/Ctrl+C) **ayrıca ve henüz yapılmamış olarak** §17'de bekler; bu tanı planı onların yerine geçmez.

## 22. Yanlış pozitifleri azaltma seçenekleri — değerlendirme (**hiçbiri uygulanmadı**)

Kural: gerçek kişi kaybına yol açabilecek hiçbir eşik/filtre, aşağıdaki doğrulama yapılmadan varsayılan yapılmaz veya uygulanmaz. **Kabul ölçütü önerisi (kullanıcı onayına tabi):** en az 3 farklı sahnede (poster/yorgan içeren ve içermeyen), her aday için (i) yanlış pozitif iz-karesi sayısı, (ii) her gerçek kişi için "izin aktif olduğu kare oranı" (gerçek kişi kapsamı), (iii) ID switch (kimlik bilinen senaryolarda), (iv) FPS/süre; **kabul:** gerçek kişi kapsamı varsayılana göre ≤ 2 yüzde puan düşer **ve** yanlış pozitif iz-karesi ≥ %50 azalır. Aksi hâlde uygulanmaz.

| # | Seçenek | Neyi hedefler | Gerçek kişi için risk | Mevcut kanıt | Uygulamadan önce gereken doğrulama |
|---|---|---|---|---|---|
| 1 | `--new-track-thresh` yükseltme (örn. 0,5–0,6) | Düşük güvenli tespitten yeni iz doğmasını önler | **Yüksek:** zayıf tespitli gerçek kişi iz alamaz | Sentetik montajda `0.6`: kısa izler 4 → 1, **ama 6 kararlı izden biri (kenarda kısmen görünen kişi, ort. güven 0,37) kayboldu** | T2/T3 güven dağılımı + T4 klibinde kapsam ölçümü; posterin güveni gerçek kişilerin alt ucundan belirgin düşük değilse **önerilmez** |
| 2 | YOLO `--conf` yükseltme (0,35 / 0,5) | Tüm düşük güvenli kutuları atar | **Orta–yüksek:** uzak/kısmi/örtülü kişiler | 4A: yorgan 0,35'te de görüldü; montajda 0,35 kısa izleri azalttı ama yok etmedi | Aynı (T2/T3/T4); poster/yorgan güveni ölçülmeden anlamsız |
| 3 | NMS `--iou` düşürme (0,7 → 0,5) | Çakışan/yinelenen kutuları azaltır | **Orta:** yakın/örtüşen iki gerçek kişiden birini bastırır (örtüşme testleri henüz yapılmadı) | Montajda ham kutu çakışması en çok IoU 0,34 (<0,5): bu videoda etkisiz kalırdı; kullanıcı sahnesinde bilinmiyor | T4 klibinde yalnızca tanı olarak `--iou 0.5`; üç kişi/örtüşme testleri sonrasında |
| 4 | Kutu içerme/geometri filtresi (bir kutu başkasını büyük oranda kapsıyorsa/aşırı büyükse ele) | İç içe/devasa kutuları | **Orta–yüksek:** yakın duran, kucaklaşan, kameraya yakın kişiler | Normal kişilerde içerme oranı 0,76'ya kadar çıktı (§21.1) → saf eşik gerçek kişileri vurur | Yeni karar mantığı + T4/T5 verisi; 4B kapsamını aşar, ayrı onay |
| 5 | Görüntülemeden önce daha uzun onay (N ardışık kare) | Kısa titreşen yanlış pozitifleri keser | **Düşük:** gerçek kişi N/30 sn geç görünür | ByteTrack zaten 2 kare ister; **poster/yorgan gibi kalıcı yanlış pozitifleri ELEMEZ** | Kazanç küçük; yalnız titreşim sorunu T6'da görülürse |
| 6 | Hareketsiz iz **bayrağı** (silme değil) | Aşama 5+ doğrulamaya ipucu verir | Bayrak yalnızca bilgidir; **silme** durağan gerçek çalışanı (iş yaparken) kaybettirir | Yok | T4'te poster ile durağan kişi ayrımı ölçülürse; bayrak olarak, kesinlikle filtre olarak değil |
| 7 | Sabit kamera için "yok sayılan bölge" (poster alanı) | Bilinen poster/ekran bölgesi | **Yüksek (bölgede):** bölgeye giren kişi görünmez; kameraya/sahneye bağımlı (projenin "belirli sahneye bağımlı tasarlama" kuralına aykırı) | Yok | Yalnızca isteğe bağlı kurulum ayarı olarak düşünülebilir, varsayılan olmaz |
| 8 | Daha büyük/farklı YOLO modeli | Basılı fotoğraf ayrımı iyileşebilir | Düşük (doğruluk); **FPS maliyeti** | Yok (karşılaştırma yapılmadı) | Kapsam dışı: ağırlık değiştirme/indirme (ağ, AGPL, SHA-256) ayrı karar |
| 9 | Zor negatif (poster/fotoğraf/yorgan) içeren veriyle ince ayar | Kök neden | Düşük | Yok | Aşama 8+ işi; veri toplama gerekir |
| 10 | Çok sinyalli doğrulama (yüz + hareket + poz, Aşama 5+) | Basılı yüzü canlı kişiden ayırma | — | **Poster de yüz içerir; tek başına yüz tespiti ayırmaz** | İleri aşama; bugün değerlendirilemez |

**Önerilen sıra:** T0 → T1 (kaynağı ayır) → T2/T3 (güven dağılımı; eşik kararının tek geçerli girdisi) → T4 (klip) → ancak sonra seçenek 1/2/3'ün klip üzerinde **tanı** olarak denenmesi → kabul ölçütü sağlanırsa kullanıcı onayıyla uygulama.
**Araç notu (eklenmedi; onayınıza bağlı):** T3/T5 için kare bazlı sayısal log (kare, tespit sayısı, ham kutu, iz kutusu, ID, güven; görüntü içermez) ve ham tespitleri ince çizgiyle gösteren bir `--show-detections` seçeneği, ayrıca iz özetinde tüm izleri/süreye göre sıralamayı sağlayan küçük bir seçenek faydalı olur. Bunlar yeni kod ve test demektir; istenirse ayrı bir adım olarak eklenir.
