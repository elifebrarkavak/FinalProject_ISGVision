# Aşama 5B — Yüz–Kişi (Track ID) İlişkilendirme ve Çoklu İnsan Takibi Entegrasyonu: Sonuç Raporu

**Tarih:** 2026-11-15 · **Durum: `READY_FOR_MANUAL_VERIFICATION`** — **`COMPLETED` DEĞİLDİR.** Kod, otomatik testler (mock + gerçek model), kayıtlı-sentetik video doğrulamaları, ölçümler ve belgeler tamam; **ilk teslimde (2026-11-15) gerçek MacBook kamerasıyla hiçbir test yapılmamıştı; sonradan kullanıcı tek kişilik (Bölüm A) manuel kontrolleri yaptı ve altısını BAŞARILI BİLDİRDİ (§19; nitel bildirim, tam konsol çıktısı/sayısal ölçüm yok, bağımsız doğruluk ölçümü değil)**; **Bölüm B (2–3 kişi) yapılmadı**, **gerçek çoklu-kişi videosu yoktur**, ilişkilendirme ve ID switch **doğruluğu gerçek veride ölçülmedi.** Canlı ekran görüntülerinde tek kişi için **ek insan kutuları ve `ambiguous` yüz** gözlendi; kök neden ayrıştırılmadı, **çözülmüş değildir** (§19.3).
**Aşama 5 genel durumu: `IN_PROGRESS`** (5A ve 5B `READY_FOR_MANUAL_VERIFICATION`; 5C başlatılmadı). Aşama 4 (`COMPLETED`) ile 4B kapanış kayıtları/açık sınırlamaları ve 5A'nın açık görevleri **değiştirilmedi.** · **Teknik belge:** [docs/face_association.md](../../docs/face_association.md) · **Git:** commit/push yapılmadı.

> **Özet.** YOLO11n insan tespiti, ByteTrack takibi ve YuNet yüz tespiti **aynı süreçte, her model kare başına yalnızca bir kez** çalışır. Her yüz, **geometrik kanıt güvenilirse** ilgili **geçici Track ID**'ye bağlanır; **belirsizde eşleştirilmez** ("karşılıklı en iyi + belirsizlik payı"). Bir yüz aynı karede en fazla bir kişiye, bir kişi en fazla bir yüze atanır (yapısal garanti + sonuç değişmezi denetimi).
> ID switch/devralma/örtüşme/oturum değişiminde eski eşleşme **taşınmaz** (eşleşme her karede yalnızca o karenin geometrisinden kurulur); izlere `epoch` + `trusted` verilir ki sonraki aşamalar kanıtı yanlış kişiye biriktirmesin. Yüz kaybı takipten çıkarmaz.
> **Gerçek çalışan kimliği, yüz tanıma, embedding/biyometrik veri YOKTUR; `track_id` kimliğe dönüştürülmez; yüz–kişi bağı canlı insan doğrulaması DEĞİLDİR** (poster/ekrandaki yüz bir kişi kutusunun kafa bölgesine denk gelirse bağlanabilir). OpenCV–Ultralytics iş parçacığı sorunu araştırıldı ve ölçüldü; **varsayılan DEĞİŞTİRİLMEDİ, karar kullanıcıda.**

## Gerçekte ne var? (kanıt türlerine göre ayrılmış)

| Başlık | İçerik |
|---|---|
| **Gerçekte çalışanlar** | `face_association` paketi (`FacePersonAssociator`, `ContinuityMonitor`, sonuç yapıları, `run_face_association` döngüsü, görselleştirme, CLI, `--opencv-threads`), `scripts/run_face_person_association.py`; gerçek YOLO11n + gerçek ByteTrack + gerçek YuNet aynı süreçte (CLI, betik alt süreci, benchmark) |
| **Test edilenler** | **259 yeni test:** 218 mock/birim + 18 sahne/değerlendirme (örnek fotoğraflar, model yok) + **23 gerçek model**. Önceki 1145 test korundu (toplam **1404**) |
| **Ölçülenler** | Kayıtlı sentetik video performansı (2 YOLO cihazı × 2 iş parçacığı ayarı × 3 tekrar), OpenCV iş parçacığı sondaları (6 `setNumThreads` değeri), sentetik senaryo doğruluğu (10 senaryo) ve eşik duyarlılığı, ilişkilendirme süresinin ölçeklenmesi |
| **Gerçek kamera (kullanıcı bildirimi; §19)** | Tek kişilik Bölüm A: birleşik görüntü, baş çevirme, kadrajdan çıkış/yeni ID, yüz kapatma/`NO FACE`, keep/restore denemesi, çıkış yolları **başarılı bildirildi** (nitel); iki ekran görüntüsü ve iki iz özeti paylaşıldı |
| **Yapılamayanlar / eksik** | Bölüm B (2–3 kişi), gerçek çoklu-kişi videosu, etiketli doğruluk ve ID switch oranı, gerçek poster/telefon (A10), A4/A6, yüksek kamera açısı, **canlı `keep`/`restore` sayısal özetleri**, çıkış kodları, kare→sonuç gecikmesi |
| **Bilinen sınırlamalar** | `stage5b_known_limitations` (`KL-5B-001…008`); 4B (`KL-4B-009/010`, `FP-4B-001`, `FP-4A-001`) ve 5A (`KL-5A-001…010`) aynen geçerlidir |
| **Kullanıcının yapacağı** | docs/face_association.md §13 (A1–A10, B1–B8; BOŞ kayıt şablonu) + `--opencv-threads` varsayılan kararı |

**Kanıt türleri:** *[Ölçüldü]* = bu makinede çalıştırılıp görülen; *[Mock]* = sahte bağımlılık; *[Gerçek model]* = gerçek YOLO/ByteTrack/YuNet, kamerasız (gerçek fotoğraflar ve **sentetik montaj**); *[Bekliyor]* = yapılmamış, kullanıcı görevi. **Yapılmayan hiçbir test başarılı gösterilmemiştir.**

## 1. Başlangıç denetimi *[Ölçüldü]*
| Konu | Durum |
|---|---|
| Git | `main`, HEAD `d675529`; çalışma ağacında 5A'nın **commit'siz** değişiklikleri vardı (6 değişmiş takipli dosya + 5A'nın takipsiz dosyaları). Bu oturumun başında `git status`, `git diff`, takipsiz dosya listesi, `project_stages.yaml` kopyası ve model SHA-256'ları kaydedildi |
| Başlangıç testleri | `unittest discover -s tests`: **1145 test, `OK (skipped=10)`** (5A raporundaki sayıyla aynı) |
| Doğrulanan 5A iddiaları | YuNet CPU'da ≈ 11 ms; `ultralytics` içe aktarılınca ≈ 30 ms ve `getNumThreads()==1` **yeniden üretildi** (§9); `ultralytics/utils/__init__.py:186`'da `cv2.setNumThreads(0)` bulundu |
| Aşama durumları (yaml) | 1–4 `completed`, 5 `in_progress`, 5A `ready_for_manual_verification`, 5B/5C `planned`; `stage5_design_notes` N5-1…6 mevcut. 5A `COMPLETED` yapılmadı |
| Kod incelemesi | `run_tracking(..., on_result=...)` kancası, `TrackingResult`/`TrackedPerson` (`bbox_xyxy`, `session_id`, `frame_index`, `timestamp`), `FaceDetectionResult`/`FaceDetection` (`face_index`, `at_border`, `image_size`), `FrameSource`/`PreviewWindow`, `FpsMeter`, test destek dosyaları okundu; **hiçbir API tahmin edilmedi** |

## 2. Mimari (Soru 1)
Yeni paket `src/isg_vision/face_association/`; Aşama 11'e ayrılmış `integration` paketi dokunulmadan bırakıldı. Akış (her model kare başına bir kez): `FrameSource` → `FramePacket` → `PersonDetector` (YOLO11n) → `PersonTracker` (ByteTrack adaptörü) → `FaceDetector` (YuNet) → `FacePersonAssociator` → `draw_face_associations` → pencere.
Sorumluluklar ayrıdır: `geometry` (saf skor), `matching` (durumsuz tek-kare kararı), `continuity` (ID güveni; eşleşme KURMAZ), `associator` (doğrulama + birleştirme), `results` (değişmez denetimli yapılar), `pipeline`/`visualization`/`cli` (kamera/pencere altyapısı **yeniden yazılmadan** `FrameSource`/`PreviewWindow`/`FpsMeter` kullanılır). `run_tracking`'in `on_result` kancası kullanılmadı (çizim `draw_person_tracks`'e bağlı, özet yüz/ilişkilendirme sürelerini taşımıyor — 4B'nin aynı gerekçesi); kancanın bu döngüyle **birebir aynı eşleşmeleri** verdiği testle gösterildi. Ayrıntı: docs §2.

## 3. Yüz–kişi eşleştirme algoritması (Soru 2)
**Sert elemeler** (biri bile sağlanmazsa çift aday değildir): yüz alanının ≥ %80'i kişi kutusunda; yüz merkezi kişi kutusunun üst %55'inde; yüz/kişi yükseklik oranı 0,03–0,8. **Skor** (0..1): içerme (0,30), dikey konum (0,35), yatay merkezlilik (0,20), boyut tipikliği (0,15) ağırlıklı ortalaması; kadraj kenarı yüzü ×0,9; **yüz tespit güveni skora girmez.**
**Karar:** `(yüz f, kişi p)` ancak `p` f'nin en iyi kişisi, `f` p'nin en iyi yüzü, her iki taraftaki ikinci aday ile fark ≥ 0,15 ve skor ≥ 0,5 ise eşleşir → **tek-birebirlik yapısal olarak garanti**, eşit skor = belirsiz. Eşleşmeyenler: kalan adayı varsa `ambiguous`, yoksa `no_face`/`unmatched_face`. **Yapılmayanlar:** eleme yoluyla atama, Macar (global) atama (zayıf çifti zorla eşleştirir), çok kareli oylama (ID switch'te yanlış taşır). Gerekçeli alternatif karşılaştırması: docs §3.1. **Ağırlıklar/eşikler gerçek veriyle ayarlanmadı** (`OT-5B-007`).

## 4. Dosyalar (Soru 3)
**Oluşturulan (takipsiz, commit yok):** `src/isg_vision/face_association/{__init__,errors,config,geometry,results,matching,continuity,associator,threads,visualization,pipeline,cli,scenes,evaluation,benchmark}.py` (15); `scripts/{run_face_person_association,benchmark_face_person_association,evaluate_face_association_scenarios}.py`; `tests/face_association_test_support.py` + `tests/test_face_association_{config,geometry,matching,results,continuity,associator,visualization,pipeline,cli,scenes,benchmark,real_model}.py` (12); `docs/face_association.md`; bu rapor. **Git dışı (ölçüm çıktıları):** `results/face_association/benchmark_5B_full.json`, `scenario_evaluation_cpu.json`.
**Değiştirilen (5A'nın commit'siz değişikliklerinin ÜZERİNE):** `configs/project_stages.yaml` (5B `ready_for_manual_verification`; `stage5b_open_items/observations/known_limitations` eklendi; OT-5A-006 durumu `investigated_in_5b_decision_pending_user`; **4A/4B kayıtlarına dokunulmadı** — oturum başı kopyasıyla satır satır karşılaştırıldı), `README.md`, `docs/architecture.md`, `docs/project_plan.md` (5B durumu), `tests/test_no_side_effects.py` (+15 modül adı; **hiçbir kontrol gevşetilmedi**).
**Değişmedi:** `camera/`, `human_detection/`, `tracking/`, `face_detection/` kaynakları, `models/*` (SHA-256 doğrulandı), `requirements*.txt`, ByteTrack eşikleri, YOLO/YuNet varsayılanları (testle denetlenir), 4A/4B/5A betikleri ve CLI seçenekleri. **Yeni paket/model eklenmedi.**

## 5. YOLO, ByteTrack ve YuNet gerçekten birlikte çalıştı mı? (Soru 4)
**Evet** *[Gerçek model]*: gerçek `PersonDetector` (`yolo11n.pt`), gerçek `PersonTracker` (Ultralytics `BYTETracker`) ve gerçek `FaceDetector` (YuNet ONNX) aynı süreçte `run_face_association` döngüsünden geçti: `zidane.jpg`/`bus.jpg`, 10 sentetik montaj senaryosu, mp4 kaydı (`FrameSource` video modu), `cli.main` ve betik alt süreci (çıkış kodu 0, çalışma dizinine dosya yazılmadı). **Kare başına çağrı sayısı** `YOLO.predict`, `FaceDetectorYN.detect` ve ByteTrack `update` seviyesinde **tam N** olarak sayıldı (N kare).

## 6. Aynı karede birden fazla kişi ve yüz eşleştirildi mi? Kanıt (Soru 5)
**Evet, kamerasız kanıtla** *[Gerçek model]*. **Gerçek fotoğraflar:** `zidane.jpg`'de iki kişi (kutuları küçük kutunun %72'si ölçüsünde örtüşür) → **2 doğru eşleşme + 1 eşleşmeyen arka plan yüzü**; `bus.jpg`'de 4 kişi/2 yüz → **2 eşleşme, 2 `no_face`.** **Sentetik montaj (T1–T9b):** 806 görünür (kişi,kare) için **0 yanlış çift**, 766 doğru bağlanma, 40 `ambiguous` (T9b), 0 kaçan (aşağıdaki tablo).

| Senaryo | Yanlış çift | Doğru bağlanma / beklenen | ID switch (ByteTrack; AYRI) |
|---|---|---|---|
| T1 tek kişi | 0 | 40/40 | 0 |
| T2 iki yan yana | 0 | 80/80 | 0 |
| T3 üç kişi | 0 | 120/120 | 0 |
| T4 yer değiştirme | 0 | 110/110 | 0 |
| T5 önünden geçme | 0 | 107/107 | **1** (B 2→3) |
| T6 çıkıp geri girme | 0 | 129/129 | **1** (A 1→4) |
| T7 arkasını dönme (taklit) | 0 | 100/100 | 0 |
| T8 yüz görünmüyor (taklit) | 0 | 40/40 | 0 |
| T9 poster/fotoğraf | 0 (80 sahte-yüz çifti: sahte/sahipsiz izlerle) | 40/40 | 0 |
| T9b fotoğraf tutan kişi | 0 (40 sahte yüz gerçek kişinin parça-izine, `trusted=False`) | **0/40 (`ambiguous`)** | 0 |

**Kanıt sınırı:** MONTAJ (tek fotoğraftan kırpılmış üç gerçek kişi, piksel-aynı kareler, kaydırma; "arkasını dönme" yüz maskeleme taklidi; "poster" kırpıntı). **Gerçek farklı kişilerin hareket ettiği video ile eşdeğer DEĞİLDİR.** Bağımsız gerçek çoklu-kişi test verisi **bulunmadığı** için bu kanıt eksikliği açıkça kayıtlıdır (`OT-5B-003`, KL-5B-005).

## 7. Örtüşmede ve ID switch durumunda ne oluyor? (Soru 6)
* **Eşleşme geçmişi taşınmaz:** eşleşme her karede yalnızca o karenin geometrisinden kurulur → "eski kişinin yüzü yeni ID'ye yapışması" mümkün değil. ID değiş tokuşunda (birim testi) yüz o karedeki **kutudaki** kişiye bağlanır; iki iz `jump` ile yeni `epoch` alır, `trusted=False`.
* **Yeni Track ID** önceki kişinin eşleşmesini devralmaz: `new_track`, `epoch=1`, `matched_streak` ≤ 1, ilk karede güvensiz (gerçek ByteTrack ID değişimlerinde — T5/T6 — testli).
* **Kaybolanın yerine giren** (KL-4B-001): uzun boşluk → `gap`, yeni epoch, güvensiz. **Oturum/kaynak değişimi:** tüm geçmiş silinir (`session_reset`).
* **Örtüşme:** örtüşen izler (ortak alan/küçük kutu ≥ 0,25) **güvensizdir**; örtüşme bitince (10 sakin kare) `overlap_end` ile yeni epoch (örtüşmeden ÖNCEKİ kanıt epoch'u korur). `zidane.jpg`'deki iki örtüşen kişi doğru eşleşti ama `trusted=False`.
* **ID switch ile yanlış eşleştirme ayrı raporlanır** (`id_report` ↔ `wrong_pairs`). **Bu, ByteTrack'in ID switch'i önlediği anlamına gelmez** (KL-4B-001/009/010 açık): ByteTrack T5 ve T6'da ID değiştirdi; `swap_places`'te değiştirmedi — bu tek bir sentetik örnektir, oran değildir.

## 8. Belirsiz eşleşmeler nasıl yönetiliyor? (Soru 7)
Eşit/yakın skor (< 0,15 fark), yinelenen/iç içe kutu, bir kişi için iki eşit yüz, düşük skor (< 0,5) → **`ambiguous`, HİÇBİR eşleşme yapılmaz**; yüz `unmatched_face`/`ambiguous` iken Track ID **yazılmaz**; yüzsüz kişi `no_face` olarak kalır ve takipten çıkarılmaz; tutarsız girdi (kare/boyut/zaman damgası uyuşmuyor, kutu görüntü dışı, yinelenen ID, kare sırası bozuk) → `invalid_input`, hiçbir eşleştirme yapılmaz ve süreklilik güncellenmez (döngü çökmez). **Bilinçli olarak "eleme yoluyla atama" yoktur** (tahmin sayılır); maliyeti bazı gerçek eşleşmelerin `ambiguous` kalmasıdır: T9b'de B'nin gerçek yüzü YOLO'nun iki üst-gövde kutusu arasında çekiştiği için 40/40 karede eşleşmedi; aynı karelerde sahte yüz alt-gövde parçasına bağlandı (`trusted=False`). Bu bilinen sınırdır (`OT-5B-009`, KL-5B-001/003).

## 9. Ultralytics–OpenCV iş parçacığı sorununa karar (Soru 8)
**Bulgu:** `ultralytics` içe aktarımı `cv2.setNumThreads(0)` çağırır → OpenCV paralel bölgeleri kapanır → YuNet ≈ 11 → **≈ 30 ms**. **Deneyler** (M4; her koşu yeni süreç): (1) aynı süreçte ultralytics öncesi/sonrası/`setNumThreads(N)` sonrası YuNet+YOLO süreleri ve `cv2.getNumThreads()`: `setNumThreads(0)` ve `(1)` **yavaş kalır**, **`N ≥ 2` ve `−1` ≈ 11 ms'ye döndürür**; `getNumThreads()` güvenilir değil (`setNumThreads(1)` sonrası 10 gösterir). (2) Entegre hat (kayıtlı sentetik video, 3 tekrar): **FPS MPS 22,1 → 43,1; CPU 19,5 → 32,1**; YuNet 30,7 → 11,0 ms; **YOLO gerilemedi** (MPS 13,3 → 10,9; CPU 19,5 → 18,7); ByteTrack aynı (≈ 0,3 ms); **12/12 koşuda sonuç imzası (Track ID, eşleşen yüz, durum) aynı** → ayar davranışı değiştirmedi. CPU/MPS (YOLO cihazı) ayrı raporlandı; YuNet her zaman CPU.
**Karar:** **varsayılan değiştirilmedi** — `--opencv-threads keep` (hiçbir şey yapma; `getNumThreads()==1` ise uyarı + önerilen komut). İsteğe bağlı `--opencv-threads restore|N` eklendi (yalnızca 5B CLI'sinde; süreç geneli; `ultralytics` yüklendikten SONRA uygulanır; eğitim aşamalarında kullanılmamalı). **Öneri (ölçüm kanıtlı): canlı kullanımda `restore`**; varsayılan yapma ve gerçek kamerada teyit (A8) kullanıcı kararıdır (`OT-5B-004`, `OT-5B-008`). Kayıtlı videodan ölçülen FPS **canlı FPS gibi sunulmamıştır.** Ayrıntı: docs §10–11.

## 10. Test sayıları (Soru 9)
| Çalıştırma | Sonuç *[Ölçüldü]* |
|---|---|
| Başlangıç (geliştirme öncesi) | **1145 test, OK, 10 atlandı** |
| **Son — varsayılan** | **1404 test, `OK (skipped=10)`** → **toplam 1404 · geçen 1394 · atlanan 10 · başarısız 0 · hata 0** |
| Yeni testler | **259**: mock/birim 218 (config 16, geometry 20, matching 21, results 19, continuity 28, associator 31, visualization 18, pipeline 21, cli 29, benchmark 15) · sahne/değerlendirme 18 · **gerçek model 23** |
| Atlanan 10 | Hepsi isteğe bağlı gerçek-pencere GUI testi (`ISG_GUI_TESTS=1` + macOS grafik oturumu): 3 Aşama 3, 2 4A, 2 4B, 3 5A. **5B bunları yeniden çalıştırmadı** (kamera/pencere kodu değiştirilmedi) |
Mock testler gerçek tespit/takibin çalıştığını **kanıtlamaz**; gerçek zincir yalnızca `test_face_association_real_model.py` ile kanıtlanır. Hiçbir test gerçek kamera, doğruluk veya canlı performans kanıtı değildir.

## 11. Gerçek model entegrasyonunun sonuçları (Soru 10)
Bkz. §5–6 ve docs §9: gerçek fotoğraflar, 10 sentetik senaryo (0 yanlış çift), mp4 kaydı (24/24 kare, 3 iz, eşleşme ≥ %90), CLI/betik (çıkış 0), kare başına tek çağrı denetimi, MPS değişmez testi (0 yanlış çift). **Performans** (kayıtlı sentetik video, penceresiz): `keep` MPS 22,1 / CPU 19,5 FPS, `restore` MPS 43,1 / CPU 32,1 FPS; YOLO 10,9–19,5 ms, ByteTrack ≈ 0,3 ms, YuNet 11,0–30,7 ms, **ilişkilendirme ≈ 0,1 ms** (60 kişi/60 yüzde ≈ 3,8 ms). **Canlı kamera performansı ölçülmedi.**

## 12. Henüz yapılmayan gerçek kamera testleri (Soru 11)
*(İlk teslimdeki metin; **güncel durum §19'dadır:** Bölüm A kısmen kullanıcı tarafından yapıldı ve başarılı bildirildi; kalanlar §19.6.)* İlk teslimde **hiçbir gerçek kamera testi yapılmamıştı.** Bölüm A (tek kişi, A1–A10: insan+yüz kutuları, Track ID–yüz ilişkisi, yüzü çevirme, uzaklaşma, yüzü kapatma, arkasını dönme, kadrajdan çıkıp-giriş, performans keep/restore, çıkış yöntemleri, sahte yüz) → `OT-5B-001`. Bölüm B (2–3 kişi, B1–B8: yan yana, birlikte yürüme, yer değiştirme, önünden geçme, kısmi/tam örtüşme, üç kişi, ID değişimi ve yanlış eşleştirme, ≈ 2 m yüksek açı) → `OT-5B-002`: **kullanıcının yanında ikinci kişi olmadığı için açık bırakıldı; kullanıcı gerçekleştirmeden başarılı sayılmaz.** Komutlar ve BOŞ kayıt şablonu: docs/face_association.md §13. Ayrıca: gerçek poster/telefon (`OT-5B-005`), yüksek açı (`OT-5B-006`), canlı FPS/gecikme (`OT-5B-004`).

## 13. Aşama 4B'den kalan çoklu insan takibi testlerinin durumu (Soru 12)
`OT-4B-001` (gerçek kamera manuel testleri) ve `OT-4B-003` (ID switch oranı/IDF1/HOTA/MOTA) **olduğu gibi AÇIK; statüleri değiştirilmedi, geriye dönük başarılı işaretlenmedi.** Bağımsız ByteTrack tarafı (4B'nin 62 gerçek-ByteTrack testi) değiştirilmedi ve T1–T9 için yeniden kullanıldı; 5B entegrasyon testleri eklendi:

| T | Bağımsız ByteTrack (4B, mevcut) | 5B mock/birim | 5B gerçek model | Gerçek kamera |
|---|---|---|---|---|
| T1 tek kişi | ✔ | ✔ | ✔ montaj | **bekliyor** (A1–A2) |
| T2 iki yan yana | ✔ | ✔ | ✔ montaj | **bekliyor** (B1) |
| T3 üç kişi | ✔ | ✔ | ✔ montaj | **bekliyor** (B6) |
| T4 yer değiştirme | ✔ | ✔ | ✔ montaj | **bekliyor** (B3) |
| T5 önünden geçme | ✔ | ✔ | ✔ montaj | **bekliyor** (B4–B5) |
| T6 çıkıp geri girme | ✔ | ✔ | ✔ montaj | **bekliyor** (A7) |
| T7 arkasını dönme | — | ✔ | ✔ **taklit** | **bekliyor** (A6) |
| T8 yüz görünmüyor | — | ✔ | ✔ **taklit** | **bekliyor** (A5) |
| T9 poster/telefon/fotoğraf | ✔ (statik FP) | ✔ | ✔ **sahte yüz montajı**, gerçek poster DEĞİL | **bekliyor** (A10) |
| T10 yüksek açı | — | — | **YAPILAMAZ** (montaj açı simüle etmez) | **bekliyor** (B8) |

## 14. Aşama 5A'nın açık görevleri korundu mu? (Soru 13)
**Evet.** `OT-5A-001…005, 007, 008` ve `KL-5A-001…010` **değişmedi** (5A `READY_FOR_MANUAL_VERIFICATION`, `COMPLETED` yapılmadı). Tek durum değişikliği: `OT-5A-006` (ortak süreçte OpenCV iş parçacığı kararı — 5A'da açıkça "5B'ye ertelendi" idi) → `investigated_in_5b_decision_pending_user`: **araştırıldı/ölçüldü, isteğe bağlı bayrak eklendi, varsayılan değişmedi, karar kullanıcıda** (`OT-5B-008`). Çözülmüş sayılmaz.

## 15. Kabul kriterleri (Soru 14) — *ilk teslim; güncel düzey §19.5'tedir*
✔ = tam · ✔* = kısmi (eksik kısım yazılı) · ✖ = karşılanmadı. **14 kriterin 9'u tam, 5'i sınırlı kanıtlıdır; ✖ yok.**

| # | Kriter | Düzey | Eksik kısım |
|---|---|---|---|
| 1 | YOLO11n + ByteTrack + YuNet aynı süreçte | ✔ | — |
| 2 | Aynı karede insan ve yüz kutuları | ✔ | — |
| 3 | Güvenilir eşleşmeler doğru | ✔* | gerçek kamera/gerçek çoklu-kişi verisi/etiketli doğruluk yok |
| 4 | Bir yüz iki kişiye atanmıyor | ✔ | — |
| 5 | Belirsizde yanlış eşleştirme yok | ✔* | gerçek örtüşme verisi yok; T9b'de sahte yüz parçaya bağlandı (`trusted=False`) |
| 6 | Yüz kaybolunca takip sürüyor | ✔* | yüz kaybı maskeleme TAKLİDİ; gerçek A5/A6 bekliyor |
| 7 | Oturum/Track ID değişimi güvenli | ✔* | gerçek kamerada ID değişimi/yanlış bağ gözlenmedi; ID switch oranı ölçülmedi (OT-4B-003) |
| 8 | Çoklu kişi/yüz senaryoları otomatik testlerle | ✔* | sentetik montaj; gerçek çoklu-kişi videosu yok; T10 yapılamadı |
| 9 | Gerçek model entegrasyonu çalışıyor | ✔ | — |
| 10 | OpenCV iş parçacığı + performans araştırıldı/kaydedildi | ✔ | varsayılan kararı kullanıcıda |
| 11 | Aşama 1–5A testleri bozulmuyor | ✔ | — |
| 12 | Manuel test planı hazır | ✔ | — |
| 13 | Yapılamayan çoklu-kişi testleri açık kayıt | ✔ | — |
| 14 | Teknik rapor/belgeler | ✔ | — |

## 16. 5B'yi kapatmadan önce kullanıcının yapması gerekenler (Soru 15) — *ilk teslim; güncel liste §19.6'dadır*
1. **Bölüm A** (docs §13.1; tek kişi, ≈ 30 dk): A1–A10; özellikle A2 (ID–yüz ilişkisi), A6/A7 (arkasını dönme, kadrajdan çıkıp-giriş: yeni ID `FACE*` ile başlıyor mu), A8 (**`keep` ↔ `restore` canlı FPS**), A10 (sahte yüz) → `OT-5B-001`, `OT-5B-004`, `OT-5B-005`.
2. **Bölüm B** (ikinci/üçüncü kişi bulunduğunda; B1–B8; ID değişimi ve yanlış yüz bağlantısını **ayrı** sayın) → `OT-5B-002` (+ 4B `OT-4B-001/003` bilgisi); mümkünse ≈ 2 m yüksek açı (B8) → `OT-5B-006`.
3. **Kararlar:** `--opencv-threads` varsayılanı (`OT-5B-008`); izinli etiketli küçük çoklu-kişi değerlendirme videosu gerekir mi (`OT-5B-003`, `OT-5B-007`); kutu bölünmesi/yinelenen kutu için araştırma (`OT-5B-009`); 5C kapsamı; `mediapipe` (`OT-5A-007`).
4. Sonuçları yaml'a işleyin; **`COMPLETED` kararı yalnızca kullanıcı onayıyla.** Commit/push yapılmadı.

## 17. Geliştirme sırasında bulunan hatalar (dürüst kayıt)
* **Kaynak hatası (düzeltildi):** tutarsız girdide (yinelenen Track ID) `invalid_input` sonucu, sonuç değişmezi denetimi tarafından reddediliyordu (döngü çökerdi); değişmezin yinelenen-ID denetimi yalnızca geçerli girdiye uygulanacak biçimde düzeltildi (testli).
* **Sahne hatası (düzeltildi):** `pass_in_front` senaryosunda ön kişi arkadakinin yüzünü örtmüyordu (yüzü üstte kalıyordu); önde duran kişi yukarı alınarak gerçekten örtüşecek biçimde düzeltildi, test beklentisi gevşetilmedi.
* **Test beklentisi hataları (kaynak değil; gevşetilmeden düzeltildi):** fikstür kutuları görüntü dışında kaldı (kişi merkezi 60 → `invalid_input`), "çocuk önde" fikstürünün skor farkı payın altındaydı, `cv2.FaceDetectorYN.detect` salt okunur (sayaç için sarmalayıcı kullanıldı), `torch_mps_status()` demet döndürür, bağımsız test görüntüsünde yüz kutusu küçük görüntü boyutuna sığmadı, geçişli ID sıçramaları (örtüşme testlerinde ışınlanma yerine düzgün hareket).
* **Ölçüm notu:** keşif sırasında `getNumThreads()`'in yavaşlamanın göstergesi OLMADIĞI bulundu (`setNumThreads(1)` sonrası 10) → karar süre ölçümüne dayandırıldı.
* **Gözlenen ve çözülmeyen:** fotoğraf tutan kişinin YOLO'da iki kutuya bölünmesi gerçek yüzü `ambiguous` bırakıyor (`OT-5B-009`); kadraj kenarında yanlış `jump` alarmı (KL-5B-004).

## 18. Git durumu ve ayrıştırma (Soru 16)
Commit/push **yapılmadı.** Oturum başındaki `git status`/`git diff`/takipsiz liste kaydı ile karşılaştırıldı:
* **Önceden mevcut (5A, commit'siz) ve dokunulmayan:** `src/isg_vision/face_detection/*` (tüm 5A dosyaları), `docs/face_detection.md`, `reports/face_detection/`, `scripts/{run,benchmark}_face_detection.py`, `scripts/download_face_model.py`, `tests/*face_detection*` — **bu oturumda değiştirilmedi** (dosya zamanları ve takipsiz liste karşılaştırması).
* **Önceden değişmiş, bu oturumda ÜZERİNE eklenen takipli dosyalar:** `README.md` (+26 satır), `configs/project_stages.yaml` (+38), `docs/architecture.md` (+14), `docs/project_plan.md` (+4), `tests/test_no_side_effects.py` (+16; yalnızca modül adları). `src/isg_vision/face_detection/__init__.py`'ye dokunulmadı.
* **Bu oturumda oluşturulan 32 takipsiz dosya:** `src/isg_vision/face_association/` (15), `scripts/` (3), `tests/` (13), `docs/face_association.md`; ve bu rapor (`reports/face_association/`).
* `models/*` SHA-256, `requirements*.txt`, `data/` **değişmedi**; yeni paket yok.

## 19. Gerçek kamera sonuçlarının kaydı (kullanıcı bildirimi) — güncelleme
**Kayıt tarihi:** 2026-11-15 (testlerin tarih/saati ve süreleri bildirilmedi) · **Yeni geliştirme, eşik veya algoritma değişikliği YAPILMADI;** yalnızca kullanıcı bildirimleri belgelendi. `src/`, `scripts/`, `tests/`, `models/`, `data/`, `requirements*.txt` değişmedi (dosya özetleriyle doğrulandı, §19.7). Hiçbir aşama `COMPLETED` yapılmadı; commit/push yok.

### 19.1 Başarılı bildirilen manuel kontroller (nitel; bağımsız ölçüm DEĞİL)
Komut: `.venv/bin/python scripts/run_face_person_association.py --source camera --camera-index 0 --device mps --opencv-threads restore` (gerçek MacBook kamerası, tek gerçek kişi, Bölüm A).

| # | Kontrol (plan) | Kullanıcı sonucu | Kaydedilmeyen |
|---|---|---|---|
| 1 | Birleşik görüntü (A1) | İnsan kutusu, yüz kutusu ve Track ID birlikte gösterildi — **başarılı** | Ek insan kutuları da görüldü (§19.3) |
| 2 | Baş çevirme (A3, kısmi) | Yüz kaybolduğunda insan takibi devam etti — **başarılı** | Açı değerleri, `NO FACE` süreleri |
| 3 | Kadrajdan çıkış (A7) | Geri girişte yeni ID ve `FACE*` davranışı — **başarılı** | Süreler, ID değişim sayısı |
| 4 | Yüz kapatma (A5) | Yüz bulunmayınca `NO FACE` — **başarılı** | Sahte yüz eşleşmesi olup olmadığı |
| 5 | Performans karşılaştırması (A8) | `keep` ve `restore` denendi — **başarılı** | **Sayısal FPS/süre özetleri paylaşılmadı** |
| 6 | Pencere ve çıkış (A9) | Q, Esc, kırmızı düğme, Ctrl+C — **başarılı** | **Çıkış kodları paylaşılmadı** |

**5A ile ilgili (kullanıcı bildirimi):** YuNet yüz tespiti gerçek MacBook kamerasında çalıştırıldı: kamera açıldı, canlı pencere çalıştı, gerçek insan yüzü tespit edildi, yüz kutusu gösterildi (nitel; sayısal değer yok). Gerçek kamerada iki/üç kişiyle çoklu yüz testi **yapılmadı.**

### 19.2 İlk ve ikinci canlı koşu arasındaki fark (iz özetleri; sayaçlar)
| | 1. koşu | 2. koşu |
|---|---|---|
| Farklı Track ID | **7** (1, 2, 3, 5, 14, 15, 20) | **2** (1, 6) |
| Ana iz (ID 1) aktif kare | 457 | 719 (kare aralığı 0–739; 740 karelik aralık) |
| Ana iz yüz eşleşmesi | 448 (≈ %98,0 aktif karenin) | 645 (≈ %89,7) |
| Ana iz güvenilir eşleşme | 299 (güvenilir/eşleşen ≈ %66,7; **149 eşleşme güvensiz**) | 643 (güvenilir/eşleşen ≈ %99,7; güvenilir/aktif ≈ %89,4) |
| Ana iz epoch | 7 | 2 |
| Diğer izler | ID 2/3/5/15: 0 eşleşme; ID 14: 3, ID 20: 4 eşleşme (hiçbiri güvenilir değil) | ID 6: 1 kare, 0 eşleşme |
| Süreklilik olayları | `gap 5`, `new_track 7`, `overlap_end 6` (iç tutarlı: 7 = ID sayısı; Σ(epoch−1) = 11 = 5 + 6) | **paylaşılmadı** |

Yorum: 2. koşu daha kararlı bir gözleme **işaret eder**; ancak iki koşunun ortamı, hareketi ve test içeriği **birebir karşılaştırılabilir olarak doğrulanmamıştır**, 1. koşunun toplam kare sayısı ve gerçek kişi sayısı iz özetiyle birlikte bildirilmemiştir. **Algoritmanın iyileştiği iddia EDİLMEZ**; fark ortamdan, hareketten, `keep`/`restore` farkından veya şanstan kaynaklanabilir. Bu değerler gerçek kimlik tanıma doğruluğu, precision/recall, IDF1 veya HOTA DEĞİLDİR.

### 19.3 Ek insan kutuları ve yüz eşleştirme belirsizliği (ekran görüntüleri)
| | Görüntü 1 | Görüntü 2 |
|---|---|---|
| PERSONS / FACES / MATCHED | 2 / 1 / 1 (trusted 1) | 3 / 1 / **0** (trusted 0) |
| NO FACE / AMBIGUOUS | 1 / — | 1 / **2** |
| FPS · YOLO (MPS) · ByteTrack · YuNet · ASSOC | 16,7 · 15 ms · 0,3 · 14 · 0,09 | 16,6 · 15 ms · 0,4 · 14 · 0,10 |
| Gözlem | Yüz `ID 27` ile eşleşti; kaldırılmış el bölgesinde `ID 43 NO FACE` ek kutusu | Gövde ve öndeki diz bölgesinde birden fazla örtüşen kutu; yüz `FACE ambiguous`, zorla eşleştirilmedi |

* Tek gerçek kişi için **ek insan izleri ve belirsiz eşleşme bulunduğu** kaydedildi. Ek kutuların kaynağı **ham YOLO çıktısıyla aynı karelerde karşılaştırılmadı**; yanlış pozitifin kök nedeni **belirlenmiş değildir** → `OT-5B-012` (araştırma, 4B `OT-4B-005` ailesi). **Çözüldü işaretlenmedi.**
* Farklı ID numaraları (27, 43; iz özetlerindeki ID'ler en çok 20) **ID switch sayısı değildir**; ekran görüntüleri ile iz özetleri aynı koşuya ait olmayabilir.
* Görüntü 2, belirsizde eşleştirmeme davranışının gerçek kamerada **gözlendiğini** destekler; ancak ek kutular gerçek yüzü kullanılamaz bırakabilir (`MATCHED: 0`) — yanlış eşleştirme değil, **kullanım riski** (`OT-5B-009`); sıklığı ölçülmedi.
* Ekran görüntülerinin hangi `--opencv-threads` ayarıyla alındığı yazılmadı (komut `restore`; YuNet ≈ 14 ms `restore` ile uyumlu bir okuma, doğrulanmış bilgi değil).

### 19.4 `keep` / `restore` kanıt durumu
* **Var:** kayıtlı sentetik videoda `restore` MPS 22,1 → 43,1, CPU 19,5 → 32,1 FPS, sonuç imzası aynı (§9); gerçek kamerada `restore` ile sistem **başarıyla çalıştı** (kullanıcı); iki ekran görüntüsünde pencereli ≈ 16,6–16,7 FPS (anlık overlay; 4B'nin pencereli 16,3 FPS'iyle aynı aile; kayıtlı penceresiz video ile karşılaştırılamaz).
* **Eksik:** canlı kamerada `keep` ve `restore` için aynı ortamda **sayısal** FPS/süre özetleri; pencereli/penceresiz fark; kare→sonuç gecikmesi → `OT-5B-004`, `OT-5B-011`.
* **Karar:** `--opencv-threads` varsayılanı **değiştirilmedi** (`keep`); `OT-5B-008` kullanıcı onayına açık. Kayıtlı video ölçümleri canlı ölçüm olarak sunulmaz.

### 19.5 14 kabul kriterinin güncel kanıt düzeyi
**9 tam kanıtlı · 5 sınırlı kanıtlı · 0 karşılanmadı** (sınıflar değişmedi; 3, 5, 6, 7 için gerçek tek-kişi kanıtı eklendi, eksikler güncellendi).

| # | Kriter | Güncel | Gerçek kamera ile eklenen / kalan eksik |
|---|---|---|---|
| 1 | YOLO+ByteTrack+YuNet aynı süreçte | Tam | Canlıda çalıştı (kullanıcı) |
| 2 | Aynı karede insan ve yüz kutuları | Tam | Canlı ekran görüntüleri |
| 3 | Güvenilir eşleşmeler doğru | **Sınırlı** | Canlıda tek kişide yüz `ID 27` ile eşleşti (nitel). Eksik: bağımsız doğruluk, gerçek çoklu-kişi; ek kutular bazı karelerde eşleşmeyi engelledi |
| 4 | Bir yüz iki kişiye atanmıyor | Tam | Yapısal + otomatik; canlıda ihlal görülmedi |
| 5 | Belirsizde yanlış eşleştirme yok | **Sınırlı** | Canlı görüntü 2'de `ambiguous`, zorlanmadı. Eksik: gerçek iki-kişi örtüşmesi (Bölüm B) |
| 6 | Yüz kaybolunca takip sürüyor | **Sınırlı (güçlendi)** | Canlıda baş çevirme + yüz kapatma başarılı bildirildi. Eksik: **A6 arkasını dönme**, sayısal kayıt, çoklu kişi |
| 7 | Oturum/Track ID değişimi güvenli | **Sınırlı (güçlendi)** | Canlıda kadrajdan çıkış → yeni ID + `FACE*` başarılı bildirildi; süreklilik olayları kaydedildi. Eksik: gerçek ID switch/devralma, ID switch oranı (`OT-4B-003`) |
| 8 | Çoklu kişi/yüz senaryoları otomatik testlerle | **Sınırlı** | Sentetik montaj; gerçek çoklu-kişi yok; T10 yapılamadı |
| 9 | Gerçek model entegrasyonu | Tam | Canlı dâhil |
| 10 | OpenCV iş parçacığı + performans kaydı | Tam | Canlı sayısal `keep`/`restore` eksik (kriter dışı; `OT-5B-004/008` açık) |
| 11 | Aşama 1–5A testleri bozulmuyor | Tam | Son çalıştırma 1404/0 başarısız; bu güncellemede kod değişmedi |
| 12 | Manuel test planı | Tam | Şablon kısmen dolu (§13.3) |
| 13 | Yapılamayan çoklu-kişi testleri açık kayıt | Tam | `OT-5B-002`, `OT-4B-001/003`, `KL-4B-009/010`, `FP-4B-001`, `FP-4A-001` açık |
| 14 | Rapor/belgeler | Tam | Güncellendi |

**Ayrı değerlendirilen riskler (çözülmedi):** (a) ek insan kutuları nedeniyle yüzün `ambiguous` kalması — gözlendi (`OT-5B-009`); (b) tek kişi için birden fazla Track ID — gözlendi, kaynak ayrıştırılmadı (`OT-5B-012`); (c) Track ID sürekliliği ≠ çalışan kimliği — tasarımla ayrık, Aşama 6/7 sözleşmesi; (d) `trusted=False` eşleşmeler kimlik kanıtı sayılmamalı (1. koşuda ana izin 149 eşleşmesi güvensizdi); (e) gerçek çoklu insan takibi **doğrulanmadı** (Bölüm B).

### 19.6 5B'yi kapatmadan önce yapılması gerekenler (güncel)
1. **Veri paylaşımı:** A8 `keep` ve `restore` konsol özetleri (uçtan uca FPS, YOLO/ByteTrack/YuNet/ilişkilendirme süreleri), A9 çıkış kodları (0/0/0/130) ve kamera ışığı bilgisi → `OT-5B-004`, `OT-5B-011`.
2. **Tek kişiyle kalanlar:** A2 (başarı etiketi), A4 (uzaklaşma), **A6 (arkasını dönme)**, A10 (telefon/poster sahte yüz) → `OT-5B-001`, `OT-5B-005`.
3. **İkinci/üçüncü kişi bulununca (Bölüm B):** en az B1, B3/B4, B6, B7 (ID değişimi ile yanlış yüz bağlantısı **ayrı** sayılsın), mümkünse B8 → `OT-5B-002`, `OT-5B-006`.
4. **Ek insan kutuları tanısı:** ham YOLO çıktısı ile takip çıktısının aynı karelerde karşılaştırılması → `OT-5B-012` (yeni geliştirme gerektirmez; 4B'nin T0–T7 planı).
5. **Kararlar:** `--opencv-threads` varsayılanı (`OT-5B-008`); eşik ayarı için etiketli çoklu-kişi verisi (`OT-5B-003/007`); 5C kapsamı. **`COMPLETED` yalnızca kullanıcı onayıyla.**

### 19.7 Bu güncellemenin dosya etkisi
Değişen dosyalar yalnızca belge/yapılandırma: `docs/face_association.md`, `docs/face_detection.md`, `docs/project_plan.md`, `docs/architecture.md`, `README.md`, `configs/project_stages.yaml`, `reports/face_association/STAGE_5B_REPORT.md`, `reports/face_detection/STAGE_5A_REPORT.md`. `src/`, `scripts/`, `tests/`, `models/`, `data/`, `requirements*.txt` dosya özetleri (SHA-256) görev öncesi/sonrası **aynıdır**.
