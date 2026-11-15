# Yüz–Kişi İlişkilendirme ve Çoklu İnsan Takibi Entegrasyonu (Aşama 5B)

> **Durum: `READY_FOR_MANUAL_VERIFICATION` (2026-11-15) — `COMPLETED` DEĞİL.** Kod, otomatik testler (mock + gerçek model), kayıtlı-sentetik video doğrulamaları ve ölçümler tamam.
> **Gerçek MacBook kamerasında yalnızca tek kişilik (Bölüm A) manuel kontroller yapıldı; altı kontrol kullanıcı tarafından BAŞARILI BİLDİRİLDİ (§13.4). Tam konsol çıktısı, sayısal ölçüm veya ekran görüntüsü
> bulunmayan ayrıntılar kaydedilmemiştir; bu bildirimler bağımsız, etiketli doğruluk ölçümü DEĞİLDİR.** Bölüm B (2–3 kişi) **yapılmadı** (kullanıcının yanında ikinci kişi yok), **gerçek çoklu-kişi videosu yoktur**,
> ilişkilendirme/ID switch **doğruluğu gerçek veride ölçülmedi.** Canlı ekran görüntülerinde tek gerçek kişi için **ek insan kutuları ve yüzün `ambiguous` kalması** gözlendi; kök neden ayrıştırılmadı, sorun **çözülmüş değildir** (§13.5).
> Aşama 5 genel durumu `IN_PROGRESS`'tir. Aşama 4B'nin açık sınırlamaları (KL-4B-009/010, FP-4B-001, FP-4A-001, OT-4B-001/003) ve Aşama 5A'nın açık görevleri **aynen geçerlidir; hiçbiri çözülmüş sayılmaz.**
> Sonuç raporu: [reports/face_association/STAGE_5B_REPORT.md](../reports/face_association/STAGE_5B_REPORT.md) · Öncül belgeler: [person_detection.md](person_detection.md), [person_tracking.md](person_tracking.md), [face_detection.md](face_detection.md)

## 1. Amaç ve kapsam

Aşama 4A (YOLO11n insan tespiti), 4B (ByteTrack takibi) ve 5A (YuNet yüz tespiti) bileşenlerini **tek bir kamera/video işleme hattında birleştirmek** ve tespit edilen yüzleri, **güvenilir
GEOMETRİK kanıt varsa**, doğru kişinin **geçici Track ID**'sine bağlamaktır. Kamerada birden fazla kişi varken her kişi takip edilir, görünür yüzü tespit edilir ve yüz, belirsizlik yoksa ilgili ID ile
eşleştirilir; belirsizse **eşleştirilmez** ("yanlış kişiyi eşleştirmek yerine bilinmiyor bırak").

**Bu aşamada YOKTUR (bilinçli):** gerçek çalışan kimliği, isim, personel numarası, yüz tanıma, embedding/biyometrik şablon, çalışan kaydı, KKD (baret/yelek) denetimi, model eğitimi, ByteTrack
eşiklerinin veya YOLO ayarlarının değiştirilmesi, görüntü kaydı, ağ kullanımı. `track_id` kimliğe dönüştürülmez; yüz tespit skoru kimlik doğruluğu olarak yorumlanmaz; yüz–kişi bağı
**canlı insan doğrulaması değildir** (poster/fotoğraf/ekrandaki yüz de bir kişi kutusunun kafa bölgesine denk gelirse bağlanabilir, §12).

## 2. Mimari ve dosya yapısı

```
KAMERA / VİDEO ──► FramePacket (frame_index, timestamp, frame)
                        │   (her model kare başına TAM BİR KEZ çalışır; aynı kare, aynı pikseller)
                        ├──► PersonDetector (YOLO11n)  ──► DetectionResult ──► PersonTracker (ByteTrack adaptörü) ──► TrackingResult
                        └──► FaceDetector (YuNet)      ──────────────────────────────────────────────────────────► FaceDetectionResult
                                                                        │                                                  │
                                                                        └──────────────► FacePersonAssociator ◄────────────┘
                                                                              (matching: durumsuz geometri | continuity: ID güveni)
                                                                                           │
                                                                                 FaceAssociationResult ──► draw_face_associations ──► pencere
```

Yeni paket `src/isg_vision/face_association/` (mevcut `integration` paketi Aşama 11'e ayrılmıştır ve dokunulmamıştır; ``integration`` bu paketi ileride tüketebilir). Sorumluluklar ayrıdır:

| Dosya | Sorumluluk |
|---|---|
| `config.py`, `errors.py` | Doğrulanmış eşikler (`AssociationConfig`), hata aileleri |
| `geometry.py` | Saf kutu geometrisi ve (yüz, kişi) çifti için geometrik skor (§3) |
| `matching.py` | **Tek karelik, durumsuz** karar: karşılıklı en iyi + belirsizlik payı (§3.3). Geçmiş KULLANMAZ |
| `continuity.py` | İz bazında ID switch/devralma/örtüşme/oturum izleme: `epoch` + `trusted` (§5). Eşleşme KURMAZ |
| `associator.py` | Girdi doğrulama (aynı kare/oturum), eşleştirme + süreklilik birleştirme, sonuç oluşturma, sayaçlar |
| `results.py` | `AssociatedPerson`, `AssociatedFace`, `TrackContinuity`, `FaceAssociationResult` (+ oluşturma anında denetlenen değişmezler) |
| `pipeline.py` | `run_face_association`: okuma → YOLO → ByteTrack → YuNet → ilişkilendirme → çizim; ölçümler, çıkış nedenleri |
| `visualization.py` | Kopya üzerinde çizim: kişi kutusu + ID, yüz kutusu + bağlı ID, sayılar, süreler |
| `cli.py`, `threads.py` | Komut satırı; isteğe bağlı `--opencv-threads` (varsayılan: dokunma, §11) |
| `scenes.py`, `evaluation.py`, `benchmark.py` | **Yalnızca test/ölçüm:** sentetik montaj sahneleri + ground-truth değerlendirme + performans ölçümü |
| `scripts/run_face_person_association.py` | Canlı/kayıtlı uygulama (ince başlatıcı) |
| `scripts/benchmark_face_person_association.py`, `scripts/evaluate_face_association_scenarios.py` | Performans/iş parçacığı ölçümü; senaryo doğruluğu + eşik duyarlılığı |

**Yeniden kullanılanlar (değiştirilmedi):** `camera.capture.FrameSource`, `camera.display.PreviewWindow` (Q/Esc/kırmızı düğme/Ctrl+C, macOS kapatma düzeltmesi), `camera.fps.FpsMeter`, `human_detection.pipeline.LatencyStats`,
`tracking.visualization.color_for_track`, 4B/5A CLI ayrıştırıcıları (`tracking.cli.build_parser` yeniden kullanılır; 4A/4B/5A betikleri ve seçenekleri aynıdır, testle denetlenir).

**Neden `run_tracking(..., on_result=...)` kancası kullanılmadı?** Kanca `(paket, tespit, takip)` verir ama çizim `draw_person_tracks`'e sabit bağlıdır ve özet yüz/ilişkilendirme sürelerini taşımaz; 4B'nin
`run_detection` → `run_tracking` kararıyla aynı gerekçeyle kardeş bir döngü yazıldı. Kancanın yine de yüz tespiti + ilişkilendirmeyi çalıştırabildiği ve bu döngüyle **birebir aynı eşleşmeleri** verdiği testle
gösterilmiştir (`test_run_tracking_on_result_hook_gives_the_same_associations`).

## 3. Eşleştirme algoritması

### 3.1 Neden yalnızca "yüz merkezi kutu içinde mi?" yetmez
Örtüşen kişi kutularında bir yüz merkezi birden fazla kutunun içinde olabilir (gerçek örnek: `zidane.jpg`'de iki kişi kutusunun ortak alanı küçük kutunun %72'si); ayrıca bir kişi birden fazla yüz adayına
(fotoğraf, ikinci yüz), bir yüz birden fazla kişi adayına sahip olabilir. Bu yüzden yöntem **içerme + konum + boyut + rakip analizi + karşılıklılık + belirsizlik payı** birleşimidir.

Değerlendirilen alternatifler:

| Yöntem | Sonuç |
|---|---|
| Yüz merkezi ∈ kişi kutusu | **Yetersiz:** örtüşmede çoklu aday, konum bilgisi yok. Sert eleme olarak "içerme" kullanılır, tek başına karar vermez |
| IoU (yüz, kişi) | Yüz kutusu kişi kutusundan çok küçük: IoU ≈ 0,01–0,03, ayırt edici değil. **Kullanılmadı** (bunun yerine yüz alanının kişi içinde kalan oranı) |
| Macar (global en iyi) atama | Toplam skoru maksimize ederken **zayıf/belirsiz çifti de zorla eşleştirir** (tahmin). Güvenlik ilkesine aykırı: **kullanılmadı** |
| Çok kareli oylama / "yapışkan" eşleşme | ID switch'te eski eşleşmeyi yanlış kişiye taşır. **Kullanılmadı:** eşleşme her karede yalnızca o karenin geometrisinden kurulur (§5) |
| Karşılıklı en iyi + belirsizlik payı + sert elemeler | **Seçildi** (aşağıda) |

### 3.2 Geometrik uygunluk (`geometry.py`)
`F` yüz kutusu, `P` kişi kutusu (`Pw, Ph` genişlik/yükseklik). **Sert elemeler** (biri bile sağlanmazsa çift aday DEĞİLDİR):

1. `içerme = alan(F ∩ P) / alan(F) ≥ min_containment` (0,8)
2. `cy = (F.merkez_y − P.y1) / Ph ≤ max_center_y_ratio` (0,55) — yüz kişinin ÜST bölgesinde
3. `min_face_height_ratio (0,03) ≤ F.yükseklik / Ph ≤ max_face_height_ratio (0,8)` — makul yüz/kişi boyutu

Adaylar için skor, dört terimin ağırlıklı ortalamasıdır (hepsi 0..1): `c` (içerme, ağırlık 0,30), `v` (dikey konum: `cy ≤ 0,30` → 1, `max_center_y_ratio`'da 0; 0,35), `h` (yatay merkezlilik:
`|F.cx − P.cx| / Pw ≤ 0,20` → 1, 0,5'te 0; 0,20), `z` (boyut tipikliği: yüz/kişi yükseklik oranı 0,07–0,5 → 1; 0,15). Kadraj kenarındaki (`at_border`) yüzün skoru `border_factor` (0,9) ile çarpılır
(N5-5). **Skor yüz tespit güvenini İÇERMEZ** (testle denetlenir) ve kimlik doğruluğu, canlılık veya olasılık DEĞİLDİR; yalnızca "bu yüz kutusu bu kutunun kafa bölgesine geometrik olarak ne kadar uyuyor" ölçüsüdür.
Gerekçeler: ayakta tam boy kişide yüz merkezi ≈ 0,05–0,15, masa başı/dizüstü kamerasında yarım boy kişide ≈ 0,15–0,35; baş kutunun yatay ortasında olur; kutu taşan yüz o kişinin değildir.
**Ağırlıklar ve plato sınırları gerçek kare ölçümlerinden optimize EDİLMEMİŞ**, anatomik/kamera geometrisi gerekçeli başlangıç değerleridir (eşik duyarlılığı: §9.3).

### 3.3 Karar kuralı (`matching.py`): karşılıklı en iyi + belirsizlik payı
Bir çift `(f, p)` ancak **şunların hepsi** sağlanırsa eşleşir:

* a. `p`, `f` için en yüksek skorlu kişi adayıdır **ve** ikinci aday varsa fark `≥ ambiguity_margin` (0,15);
* b. `f`, `p` için en yüksek skorlu yüz adayıdır **ve** ikinci aday varsa fark `≥ ambiguity_margin`;
* c. skor `≥ min_score` (0,5).

(a)+(b) **tek-birebirliği garanti eder** (bir yüzün en iyi kişisi tektir, bir kişinin en iyi yüzü tektir). Eşit skor fark 0 verir → **her zaman belirsiz** (girdi sırasına göre sessizce biri seçilmez; testle
denetlenir). Eşleşmeyenler: kalan (başka eşleşmeyle alınmamış) adayı varsa **`ambiguous`** (tahmini eşleşme YAPILMAZ), yoksa kişi için **`no_face`**, yüz için **`unmatched_face`**.

**Bilinçli olarak yapılmayan:** "eleme yoluyla atama" (yüz `f`'nin iki adayından biri başka yüzle güvenle eşleşti diye `f`'yi diğerine atamak bir tahmindir; örtüşmede yanlış kişiye yol açabilir). Maliyeti: bazı
gerçek eşleşmeler `ambiguous` kalır (ölçüldü, §9.3).

### 3.4 Belirsizlik politikası (özet)
| Durum | Sonuç |
|---|---|
| Yüz hiçbir kişinin sert elemelerini geçmiyor | Yüz `unmatched_face` (Track ID yazılmaz) |
| Kişinin hiçbir aday yüzü yok | Kişi `no_face` (takipten ÇIKARILMAZ) |
| İki kişi kutusu aynı yüz için yakın skorda (yinelenen/iç içe kutu, aynı boyda kafa hizası) | Yüz ve kişiler `ambiguous` (`contested`); eşleşme yok |
| Bir kişinin iki yüzü yakın skorda | Kişi ve yüzler `ambiguous` |
| Bir kişinin iki yüzünden biri belirgin üstün (≥ 0,15) | Üstün eşleşir; diğer yüz `unmatched_face` (`candidates_taken`) |
| Tek aday var ama skor `min_score` altında | `ambiguous` (`low_score`) |
| Yüz ve takip sonuçları aynı kareye ait değil / tutarsız | `invalid_input`: **hiçbir** eşleştirme yapılmaz, süreklilik güncellenmez |

## 4. Güvenlik ilkeleri ve doğrulanma biçimleri
| İlke | Nasıl sağlanır / doğrulanır |
|---|---|
| Bir yüz aynı karede en fazla bir kişiye, bir kişi en fazla bir yüze | Karar kuralı (a)+(b); `FaceAssociationResult` oluşturulurken **değişmez denetimi** (ihlal → `AssociationResultError`); 600 rastgele sahnede özellik testi |
| Belirsizde tahmini eşleştirme yok | Eşit skor = belirsiz (giriş sırasından bağımsız, testli); duplicate-kutu, iki-yüz, düşük-skor testleri |
| Yüz kaybı takipten çıkarmaz | Yüzsüz kişi `no_face` olarak sonuçta kalır; gerçek modelde T7/T8 testi (iz ID'si sabit) |
| Poster/ekran yüzü çalışan kanıtı değildir | Çıktıda ve ekranda uyarı (`LINK = geometry only (not identity, not liveness)`); yüz skoru kimlik olarak kullanılmaz; `possibly_static` bayrağı (yargı değil); sınır §12 |
| `track_id` kimliğe dönüşmez | Sonuçlarda isim/numara/embedding alanı yoktur; `(session_id, track_id)` yalnızca takip anahtarıdır |
| Yeni Track ID önceki kişinin eşleşmesini devralmaz | Eşleşme geçmişi taşınmaz; yeni iz `epoch=1`, güvensiz başlar; gerçek ByteTrack ID değişimlerinde testli |
| Görüntü kaydı/ağ yok | CLI testleri: dosya yazılmadığı, `imwrite`/`VideoWriter`/`socket.connect` çağrılmadığı doğrulanır |

## 5. ID switch, ID devralma ve süreklilik (`continuity.py`)

**Temel tasarım kararı:** eşleşme **her karede yalnızca o karenin geometrisinden** kurulur ve geçmişten HİÇBİR eşleşme taşınmaz. Bu yüzden "eski kişinin yüzü yeni ID'ye yapışması" mümkün değildir. Ancak sonraki aşamalar
(kimlik tanıma, çok kareli kanıt) bir Track ID altında kareler boyunca kanıt biriktirebilir; ID kişi değiştirmişse bu birikim yanlış kişiye ait olur. Bu yüzden her iz için **süreklilik dönemi (`epoch`)** ve
**güven (`trusted`)** üretilir: kanıt yalnızca aynı `(session_id, track_id, epoch)` içinde ve `trusted` karelerde birikmelidir.

**Yeni epoch** (`epoch` artar, `frames_in_epoch=1`, eşleşme geçmişi sıfırlanır) şu olaylarda:

| Olay (`BREAK_*`) | Koşul | Neden |
|---|---|---|
| `new_track` | `(oturum, iz)` ilk kez görüldü | Yeni numara önceki kişinin kimliğini/eşleşmesini devralmaz |
| `session_reset` | `session_id` değişti veya `reset_reason` var | Yeni kaynak/oturum: tüm geçmiş silinir |
| `jump` | Ardışık karelerde aynı ID kutusu `jump_iou` (0,3) altında örtüştü | ID devri / anlık yer değiştirme şüphesi |
| `gap` | İz `max_gap_frames` (3) kareden uzun görünmeyip döndü, **veya** döndüğünde son kutuyla IoU < 0,3 | Kaybolanın yerine giren başkası ID'yi devralmış olabilir (KL-4B-001) |
| `overlap_end` | İz başka bir izle örtüşüyordu (ortak alan / küçük kutu ≥ `overlap_guard` 0,25) ve 10 kare boyunca örtüşmedi | Örtüşme sırasında ID değişmiş olabilir. **Örtüşmeden ÖNCEKİ kanıt epoch'u korur** (o zaman kimlik doğruydu) |

**`trusted`:** epoch'ta ≥ `min_trusted_frames` (3) kare **ve** şu anda örtüşme/sakinleşme yok **ve** bu karede epoch yenilenmedi. Yeni iz her zaman güvensizle başlar (güvenli varsayılan). Eşleşme var ama
`trusted=False` ise ekranda `ID n FACE*` görünür ve `AssociatedPerson.trusted_match` `False`'tur.

**Hareketsizlik bayrağı (`possibly_static`):** kutu 150 kare boyunca IoU ≥ 0,95 ile değişmedi → poster/yorgan benzeri kalıcı yanlış pozitif **adayı** (N5-3). Yargı, silme veya eşleşme engeli DEĞİLDİR;
masa başındaki gerçek kişi de bayraklanabilir.

**Davranış tablosu** (testle doğrulanan; 4B'nin gerçek ByteTrack davranışı değişmedi):

| Senaryo | ByteTrack ne yapar (4B) | 5B ne yapar |
|---|---|---|
| İki kişi ID değiş tokuş eder | ID switch (KL-4B-001/009) | Yüz o karedeki kutuya bağlanır (eski yüz taşınmaz); iki iz `jump` ile yeni epoch, güvensiz |
| Kaybolan kişinin konumuna yeni biri girer | Eski ID'yi devralabilir | Uzun boşluk → `gap`, yeni epoch, güvensiz |
| Kişi kadrajdan çıkıp döner | Çoğu kez yeni ID (KL-4B-003/010) | Yeni ID: `new_track`, `epoch=1`, geçmiş yok, güvensiz |
| İki kişi örtüşür/yer değiştirir | Örtüşmede ID değişebilir | Örtüşen izler güvensiz; örtüşme bitince `overlap_end` ile yeni epoch |
| Yüz birkaç kare görünmez | — | Kişi `no_face`, takip ve epoch bozulmaz; `frames_since_match` artar |
| Oturum/kaynak değişir | Takipçi sıfırlanır, ID'ler 1'den başlar | Tüm geçmiş silinir (`session_reset`) |

**ID switch ile yüz–kişi eşleştirmesinin yanlış çalışması AYRI raporlanır:** `SceneEvaluation.id_report` (ByteTrack'in ID davranışı) ve `wrong_pairs` (yüzün yanlış ize bağlanması) ayrı sayılardır; ID switch,
ilişkilendirmenin yanlış olduğu anlamına gelmez (testle gösterilmiştir).

## 6. Veri yapıları (`results.py`)
* **`AssociatedPerson`** — bir aktif iz: `session_id, track_id, frame_index, timestamp, person_bbox, person_confidence`, `status` (`matched` / `no_face` / `ambiguous` / `invalid_input`), `reason`, eşleşirse `face_index, face_bbox,
  face_confidence, face_at_border, score, margin`, `candidate_face_indices`, `continuity` (`TrackContinuity`), `trusted_match`.
* **`AssociatedFace`** — bir yüz: `face_index, face_bbox, face_confidence, at_border`, `status` (`matched` / `unmatched_face` / `ambiguous` / `invalid_input`), `reason`, eşleşirse `session_id, track_id, score`, `candidate_track_ids`.
* **`TrackContinuity`** — `epoch, frames_in_epoch, trusted, untrusted_reasons, breaks, overlap_fraction, overlapping_track_ids, matched_streak, frames_since_match, stationary_frames, possibly_static`.
* **`FaceAssociationResult`** — `persons, faces, image_size, frame_index, timestamp, session_id, association_ms, input_valid, invalid_reason, breaks` + sayılar: `person_count`, `face_count`, `matched_count`,
  `trusted_match_count`, `no_face_count`, `ambiguous_person_count`, `ambiguous_face_count`, `unmatched_face_count`, `possibly_static_count`; `pairs()`, `person()`, `face()`.
* Yüz ve kişi sonuçları aynı `frame_index`, `timestamp` (ikisinde de varsa) ve `image_size`'a ait olmalıdır; değilse `invalid_input` (hiçbir eşleştirme). Tüm yapılar `frozen`'dır; kimlik alanı içermez.
* Aşama 6/7 için: `(session_id, track_id, epoch)` kanıt anahtarı, `trusted_match`, `face_bbox` (kırpma için) hazırdır; **gerçek kimlik bilgisi eklenmemiştir.**

## 7. Komut satırı
```bash
.venv/bin/python scripts/run_face_person_association.py --source camera --camera-index 0 --device mps
.venv/bin/python scripts/run_face_person_association.py --source video --video-path /yol/video.mp4 --device mps
.venv/bin/python scripts/run_face_person_association.py --source camera --device mps --no-display --max-frames 300
.venv/bin/python scripts/run_face_person_association.py --source camera --device mps --opencv-threads restore   # YuNet yavaşlamasını deney (§11)
.venv/bin/python scripts/run_face_person_association.py --help
```
Kaynak/YOLO/ByteTrack seçenekleri `run_person_tracking.py` ile aynıdır; yüz: `--face-model --face-conf --face-nms --face-max-faces --face-detect-width`; ilişkilendirme: `--min-containment --min-score --ambiguity-margin
--overlap-guard`; çalışma: `--no-display --max-frames`, `--opencv-threads keep|restore|N`. Çıkış: **Q / Esc / pencereyi kapatma / Ctrl+C** (çıkış kodları 0/0/0/130; hata 1; hatalı argüman 2). Sıra: YOLO → takipçi →
YuNet yüklenir (hiçbiri başarısızsa kamera açılmaz) → `--opencv-threads` → kaynak → ısınma → pencere → döngü. **Görüntü kaydedilmez, ağ kullanılmaz, paket kurulmaz.**

Pencerede: insan kutusu + `ID n` (+ durum: `FACE` / `FACE*` / `NO FACE` / `FACE ?` / `STATIC?`), yüz kutusu (köşe ayraçlı) + `FACE>ID n` (yalnızca eşleşen yüzde ID yazılır; eşleşmeyen `FACE unmatched` amber, belirsiz `FACE ambiguous`
macenta), üst şeritte `PERSONS / FACES / MATCHED (trusted) / NO FACE / AMBIGUOUS / UNMATCHED FACES`, `FPS`, `YOLO (cihaz)`, `BYTETRACK`, `YUNET`, `ASSOC` süreleri; alt şeritte daima `LINK = geometry only (not identity, not liveness)`.

## 8. Otomatik testler
Ayrıntılı sayılar ve kanıt türü ayrımı §8.1'de (sonuç raporuyla aynıdır). Mock, gerçek model, GUI ve gerçek kamera **ayrı** tutulur; hiçbir mock test gerçek kamera, doğruluk veya canlı performans kanıtı değildir.

**T1–T10 senaryo eşlemesi** (bağımsız ByteTrack = Aşama 4B'den mevcut, değiştirilmemiş gerçek-ByteTrack testleri; 5B entegrasyon = bu aşamanın yeni testleri):

| T | Senaryo | Bağımsız ByteTrack testi (4B, mevcut) | 5B birim/mock | 5B gerçek model (sentetik montaj) | Gerçek kamera (kullanıcı) |
|---|---|---|---|---|---|
| T1 | Tek kişi | `SinglePersonTest` | `test_single_person_with_head_face_is_matched` | `test_t1_single_person` | **A1–A2 (bekliyor)** |
| T2 | İki kişi yan yana | `MultiPersonTest` | `test_two_people_side_by_side_get_their_own_faces…` | `test_t2_two_people_side_by_side…` | B1 (bekliyor) |
| T3 | Üç kişi | `MultiPersonTest` (3 ve 30 kişi) | `test_three_people_three_faces` | `test_t3_three_people_at_once` | B6 (bekliyor) |
| T4 | Yer değiştirme | `SwapTest` | `test_id_switch_between_two_people_never_carries…` | `test_t4_two_people_swap_places…` | B3 (bekliyor) |
| T5 | Önünden geçme | `OcclusionTest` | `OverlapTest`, `test_overlapping_people_keep_per_frame_geometry_but_lose_trust` | `test_t5_person_passing_in_front…` | B4–B5 (bekliyor) |
| T6 | Kadrajdan çıkıp girme | `ExitAndReenterTest` | `GapTest`, `test_new_track_id_for_a_returning_person_starts_clean` | `test_t6_person_leaves_and_returns`, `test_id_changes_are_reported_separately…` | A7 (bekliyor) |
| T7 | Arkasını dönme | — (takipçi yüzü bilmez) | `test_face_loss_for_a_few_frames_keeps_the_person_tracked…` | `test_t7_person_keeps_being_tracked…` (**yüz maskeleme TAKLİDİ**) | A6 (bekliyor) |
| T8 | Yüz görünmüyor | — | `test_person_without_a_face_is_reported_as_no_face…` | `test_t8_person_without_any_visible_face…` (taklit) | A5 (bekliyor) |
| T9 | Poster/telefon/fotoğraf | `FalsePositiveTest` (statik FP, KL-4B-002/008) | iki-yüz/düşük-skor/belirsizlik testleri | `RealDecoyTest` (sahte yüz montajı; **gerçek poster/telefon DEĞİL**) | A10 (bekliyor) |
| T10 | Yüksek kamera açısı | — | — | **YAPILAMAZ** (montaj açı simüle etmez) | B8 (bekliyor) |

### 8.1 Test sayıları (ölçüldü: `unittest discover -s tests`, 2026-11-15)

| | Sayı |
|---|---|
| Başlangıç (geliştirme öncesi) | **1145 test, OK, 10 atlandı** (hepsi isteğe bağlı gerçek-pencere GUI testi) |
| **Son — varsayılan** | **1404 test, `OK (skipped=10)`** → **1394 çalıştı ve geçti, 0 başarısız, 0 hata, 10 atlandı** (aynı 10 GUI testi) |
| Yeni testler | **259** (= 1404 − 1145) |
| `test_no_side_effects` | +15 modül adı; **hiçbir kontrol gevşetilmedi** (onaylı model listesi aynı) |

| Tür | Dosya (test) | Toplam |
|---|---|---|
| **Mock/birim** (sahte takip/yüz/kamera/pencere; model yok) | `config` 16 (+ içe aktarma yan etkisi), `geometry` 20, `matching` 21 (600 + 200 + 300 rastgele sahnelik özellik testleri), `results` 19, `continuity` 28, `associator` 31, `visualization` 18, `pipeline` 21, `cli` 29, `benchmark` 15 | **218** |
| **Sahne/değerlendirme** (Ultralytics örnek fotoğrafları; model yok) | `scenes` 18 | **18** |
| **Gerçek model** (gerçek YOLO11n + ByteTrack + YuNet) | `real_model` 23 (1'i MPS değişmez testi; bu makinede çalıştı) | **23** |

Atlanan 10 test (değişmedi): Aşama 3'ten 3, 4A'dan 2, 4B'den 2, 5A'dan 3 — hepsi `ISG_GUI_TESTS=1` + macOS grafik oturumu gerektiren gerçek-pencere testleri. **5B bu GUI testlerini yeniden çalıştırmadı** (kamera/pencere kodu
değiştirilmedi; 5B'nin kendi pencere yolu sahte pencereyle sınanır). Koşullu atlamalar (ağırlık/örnek görüntü yoksa) bu makinede tetiklenmedi. Hiçbir test gerçek kamera, doğruluk veya canlı performans kanıtı değildir.

## 9. Gerçek model doğrulamaları *[Gerçek model; kamerasız]*

### 9.1 Gerçek fotoğraflar (`zidane.jpg`, `bus.jpg`; tekrarlanan kare; sonuç görüntüde GÖZLE doğrulandı)
| Görüntü | Gerçek YOLO + ByteTrack + YuNet çıktısı | İlişkilendirme sonucu |
|---|---|---|
| `zidane.jpg` (1280×720) | 2 kişi kutusu (Ancelotti `[748,42,1148,711]`, Zidane `[149,203,1126,715]`; **ortak alan küçük kutunun %72'si**), 3 yüz (0,94 / 0,90 / bulanık arka plan 0,61) | **2 eşleşme (yüzler doğru kişilere: Ancelotti ↔ sağdaki kutu, Zidane ↔ soldaki geniş kutu, skor ≥ 0,99), 1 `unmatched_face` (sağ kenardaki bulanık arka plan yüzü; hiçbir kutunun içinde değil, Track ID yazılmaz).** İki kişi büyük ölçüde örtüştüğü için ikisi de `trusted=False` |
| `bus.jpg` (810×1080) | 4 kişi kutusu, 2 yüz | **2 eşleşme (skor ≥ 0,99), 2 `no_face`** (sırtı dönük/kısmi kişiler takipten çıkarılmadı), 0 eşleşmeyen yüz |

### 9.2 Sentetik montaj senaryoları (gerçek YOLO + gerçek ByteTrack + gerçek YuNet, CPU; `scripts/evaluate_face_association_scenarios.py`)
**KANIT SINIRI:** bu bir MONTAJDIR (tek fotoğraftan kırpılmış üç gerçek kişi; piksel-aynı kareler; kaydırma hareketi; "arkasını dönme/yüz yok" yüz maskelemeyle TAKLİT; "poster/fotoğraf" yüz/kişi kırpıntısı). Gerçek farklı kişilerin hareket ettiği
video ile eşdeğer DEĞİLDİR; gerçek kamerada/gerçek çoklu-kişi sahnesinde doğruluk, ID switch oranı veya IDF1/HOTA/MOTA kanıtı DEĞİLDİR. Sayılar `ultralytics==8.4.210`, SHA-256'sı sabit ağırlıklar ve CPU ile deterministiktir.
**ID switch (ByteTrack) ve yanlış yüz–kişi eşleşmesi AYRI sütunlardır.**

| Senaryo (T) | Kare | Doğru çift | **Yanlış çift** | Görünür yüzü olan (kişi,kare) | Doğru bağlanan | Yüz tespit edilemedi | Belirsiz | **ID switch** (ByteTrack) | Süreklilik kesintileri |
|---|---|---|---|---|---|---|---|---|---|
| `single_person` (T1) | 40 | 40 | **0** | 40 | 40 | 0 | 0 | 0 | new_track 1 |
| `side_by_side` (T2) | 40 | 80 | **0** | 80 | 80 | 0 | 0 | 0 | new_track 2 |
| `three_people` (T3) | 40 | 120 | **0** | 120 | 120 | 0 | 0 | 0 | new_track 3 |
| `swap_places` (T4) | 60 | 110 | **0** | 110 | 110 | 0 | 0 | 0 | new_track 2, overlap_end 3, gap 1 |
| `pass_in_front` (T5) | 60 | 107 | **0** | 107 | 107 | 0 | 0 | **1** (B: ID 2 → 3; tamamen örtülünce kayboldu) | new_track 3, overlap_end 1 |
| `leave_and_return` (T6) | 80 | 129 | **0** | 129 | 129 | 0 | 0 | **1** (A: ID 1 → 4; kadrajdan çıkıp döndü, KL-4B-003/010) | new_track 3, jump 7 |
| `turn_away` (T7, taklit) | 60 | 100 | **0** | 100 | 100 | 0 | 0 | 0 (kişi izlendi; 20 karede yüz yok → `no_face`) | new_track 2 |
| `no_visible_face` (T8, taklit) | 40 | 40 | **0** | 40 | 40 | 0 | 0 | 0 (A 40/40 karede `no_face`) | new_track 2 |
| `photo_decoys` (T9) | 40 | 40 | **0** | 40 | 40 | 0 | 0 | 0 | new_track 3 |
| `photo_held_up` (T9b) | 40 | 0 | **0** | 40 | **0** | 0 | **40** | 0 | new_track 3 |

Okuma: **hiçbir senaryoda gerçek bir yüz yanlış kişiye bağlanmadı (0 yanlış çift).** ID switch'ler (T5, T6) ByteTrack'in bilinen davranışıdır (4B, açık sınırlama): yeni ID `new_track` ile başlar, **eski kişinin eşleşmesi/geçmişi yeni ID'ye taşınmaz** ve ilk karede güvenilir sayılmaz (testli).
`swap_places`'te örtüşmede ByteTrack ID değiştirmedi (bu **ByteTrack'in her durumda böyle davranacağı anlamına gelmez**; KL-4B-001/009). Örtüşme sırasındaki eşleşmeler doğru olsa da `trusted=False` işaretlendi.
**Yüz–poster davranışı (T9):** poster/fotoğraf kırpıntıları YOLO'da kişi gibi izlendi ve yüzleri kendi izlerine eşleşti (80 "sahte-yüz çifti"); gerçek kişi B'ye hiçbiri bağlanmadı (FP-4B-001'in yüz tarafı: **geometri bunu ayıramaz**). **T9b:** fotoğrafı göğsünde tutan B'yi YOLO iki kutuya böldü; B'nin gerçek yüzü iki üst-gövde kutusu arasında
çekişmeli olduğundan **40/40 karede `ambiguous` (eşleştirilmedi)**, sahte yüz ise alt-gövde parçasına **40/40 karede bağlandı (`trusted=False`: örtüşen parça)**. Yani bu durumda sistem gerçek yüzü bağlamadı ve sahte yüzü güvenilir saymadı ama sahte yüz ekranda `FACE*` ile bir ize bağlı görünür — bilinen sınır (§12.1).

### 9.3 Eşik duyarlılığı (aynı takip/yüz sonuçları, ilişkilendirme yeniden çalıştırıldı; tüm senaryolar toplamı; 806 görünür-yüz (kişi,kare))
| `min_containment` | `ambiguity_margin` | Doğru | Yanlış | Sahte yüz gerçek kişide | Doğru bağlanan | Belirsiz | Kaçan |
|---|---|---|---|---|---|---|---|
| 0,6 / 0,8 | 0,05 / 0,15 / 0,30 | 766 | **0** | 40 | 766 | 40 | 0 |
| 0,95 | 0,05 / 0,15 / 0,30 | 765 | **0** | 40 | 765 | 40 | 1 |

**Dürüst yorum:** bu sahneler eşikleri ayırt edecek kadar zor DEĞİLDİR (örtüşen-yakın-baş durumları azdır); tablo "eşiklere duyarsız" der, "eşikler optimaldir" DEMEZ. Eşik davranışı birim testlerinde sınanır: kafa hizası yakın iki kişi (skor farkı ≈ 0,05) 0,15 payında `ambiguous`,
pay düşürülünce eşleşir (`test_margin_threshold_decides_between_matched_and_ambiguous`); çocuk–yetişkin (fark ≈ 0,2+) eşleşir. Varsayılanlar ayarlanmadı.

### 9.4 Kayıtlı video ve CLI uçtan uca
3 kişilik montaj mp4'e yazılıp `FrameSource` video modunda gerçek hatla işlendi: 24/24 kare, 3 farklı iz, `invalid_frames=0`, eşleşme ≥ %90 (mp4 sıkıştırması sonrası); `cli.main` penceresiz çıkış kodu 0 ve **çalışma dizinine hiçbir dosya yazılmadı**; `scripts/run_face_person_association.py` alt süreci çıkış kodu 0.
**Kare başına model çağrısı:** gerçek `PersonDetector.detect`/`YOLO.predict`, gerçek `FaceDetector.detect`/`cv2.FaceDetectorYN.detect` ve ByteTrack `update` **tam N kez** çağrıldı (N kare) — testle doğrulandı.

## 10. Performans ve gecikme *[Ölçüldü; kayıtlı sentetik montaj videosu — GERÇEK KAMERA ÖLÇÜMÜ DEĞİLDİR]*
Makine: MacBook Pro **Apple M4**, 10 çekirdek, macOS 26.5.2, Python 3.11.17, OpenCV 5.0.0. Girdi: 1280×720, 240 kare, 3 kişi/3 yüz, penceresiz, her varyant **yeni alt süreçte**, 3 tekrar (varyant sırası dönüşümlü), medyanlar. Ham çıktı: `results/face_association/benchmark_5B_full.json` (Git dışı).
Kayıtlı videodan ölçülen FPS **canlı kamera FPS'i DEĞİLDİR** (kamera pozlaması/sürücü tamponu/okuma bekleme yok; pencere çizimi ve `imshow` maliyeti yok: 5A'da `show` ≈ 15 ms ölçülmüştü).

| YOLO cihazı | `--opencv-threads` | FPS (uçtan uca; 3 tekrar) | YOLO ms | ByteTrack ms | YuNet ms | İlişkilendirme ms | Toplam hesaplama ms |
|---|---|---|---|---|---|---|---|
| mps | keep (varsayılan) | **22,1** (22,8 / 22,1 / 21,8) | 13,3 | 0,27 | **30,7** | 0,095 | 44,5 |
| mps | restore | **43,1** (44,1 / 43,1 / 42,1) | 10,9 | 0,29 | **11,0** | 0,099 | 22,6 |
| cpu | keep | **19,5** (19,5 / 19,5 / 19,5) | 19,5 | 0,31 | **30,6** | 0,097 | 50,5 |
| cpu | restore | **32,1** (32,2 / 32,1 / 32,1) | 18,7 | 0,31 | **11,1** | 0,098 | 30,2 |

* **Kare başına süreler ayrı ayrı:** YOLO (cihazda), ByteTrack (CPU, ≈ 0,3 ms), YuNet (**her zaman CPU**), ilişkilendirme (CPU, ≈ 0,1 ms; ihmal edilebilir). YOLO MPS ↔ CPU ayrımı yukarıdadır; YuNet süresi YOLO cihazından bağımsızdır (≈ 30,6–30,7 ↔ 11,0–11,1).
* **İlişkilendirme ölçeklenmesi** (saf, 1280×720 ölçeğinde sentetik kutular; geçerli girdi): 1 kişi/yüz 0,02 ms · 3 → 0,06 · 10 → 0,22 · 30 → 1,1 · 60 → 3,8 ms (≈ O(n·m)); gerçek sahnelerde (≤ 10 kişi) YOLO/YuNet'in ≈ %1'inden azdır.
* **Kare→sonuç gecikmesi (kayıtlı video):** `FaceAssociationRunSummary.frame_to_result` döngüde ölçülür ve CLI özetinde yazılır; **canlı kamera için ölçülmedi** (backlog B3-3/B3-4; OT-4B-004, OT-5A-004 açık).
* 4B (`0,2–0,5 ms` ByteTrack) ve 5A (`≈ 11 ms` YuNet, `88 FPS` tek başına) sonuçları 5B ölçümü olarak sunulmamıştır; 5A'nın 11 ms'si yalnızca `ultralytics` içe aktarılmamışken (veya iş parçacığı geri yüklenince) geçerlidir.

## 11. OpenCV ↔ Ultralytics iş parçacığı sorunu: araştırma, ölçüm ve karar *[Ölçüldü]*

**Neden:** `ultralytics/utils/__init__.py:186` içe aktarma sırasında `cv2.setNumThreads(0)` çağırır (yorum: PyTorch DataLoader ile uyumsuzluk). Bu, süreç genelinde OpenCV paralel bölgelerini kapatır; OpenCV DNN ile çalışan YuNet ≈ 2,8× yavaşlar. YOLO ve YuNet aynı süreçte
çalıştığı için etki entegre hatta doğrudan görünür (§10: 30,6 ms ↔ 11,0 ms).

**Deney 1 — aynı süreçte üç aşama** (`--worker probe`; zidane 1280×720; 50 örnek; YOLO MPS; her satır yeni süreç):

| `cv2.setNumThreads(N)` (3. aşama) | 1) `ultralytics` ÖNCESİ: `getNumThreads` / YuNet | 2) `ultralytics` SONRASI: `getNumThreads` / YuNet / YOLO | 3) `setNumThreads(N)` SONRASI: `getNumThreads` / YuNet / YOLO |
|---|---|---|---|
| 0 | 10 / 10,9 ms | **1** / **31,3** ms / 8,3 ms | 1 / **29,7** ms / 7,2 ms |
| 1 | 10 / 11,8 | 1 / **30,9** / 8,0 | 10 (!) / **30,7** / 9,1 |
| 2 | 10 / 13,2 | 1 / **30,7** / 7,3 | 10 / **11,8** / 8,1 |
| 4 | 10 / 10,8 | 1 / **30,3** / 7,0 | 10 / **11,4** / 9,0 |
| 8 | 10 / 10,7 | 1 / **30,2** / 8,7 | 10 / **10,7** / 8,8 |
| −1 (OpenCV varsayılanı) | 10 / 10,9 | 1 / **30,3** / 7,0 | 10 / **11,4** / 7,7 |

Bulgular: (1) `ultralytics` içe aktarımı sonrası `getNumThreads()` 10 → **1**, YuNet ≈ 11 → **≈ 30 ms**. (2) `setNumThreads(0)` ve `setNumThreads(1)` **yavaş kalır**; **`N ≥ 2` veya `−1` ≈ 11 ms'ye döndürür** (bu OpenCV/macOS yapısında N ≥ 2 arasında fark yok).
(3) **`cv2.getNumThreads()` güvenilir bir gösterge DEĞİLDİR:** `setNumThreads(1)` sonrası 10 gösterir ama süre yavaştır; karar için süre ölçümü esastır. (4) YOLO süresi bu tek-görüntü sondasında ±1–2 ms gürültüyle değişir; ayara bağlı sistematik bir değişim görülmedi.

**Deney 2 — entegre hatta (`--opencv-threads keep` ↔ `restore`; §10 tablosu).** FPS: MPS **22,1 → 43,1**, CPU **19,5 → 32,1**. **YOLO süresi gerilemedi** (MPS 13,3 → 10,9; CPU 19,5 → 18,7; 3/3 tekrarda; nedeni izole edilmedi, olasılıkla ön işleme),
ByteTrack aynı (0,27–0,31 ms). **Davranış değişmedi:** 12 koşunun (2 cihaz × 2 ayar × 3 tekrar) tamamında her karedeki `(Track ID, eşleşen yüz, durum)` imzası **aynı** (`381c690fb021af21`): iş parçacığı ayarı tespit, takip veya eşleşme sonuçlarını değiştirmedi.

**Yan etki değerlendirmesi:** ayar süreç geneldir ve yalnızca `ultralytics` içe aktarımı **sonrasında** uygulandığında anlamlıdır (CLI bunu yükleme sonrasına koyar; testli). Etkilenebilecekler: OpenCV'nin paralel `resize`/`cvtColor`/DNN çağrıları (YOLO ön işleme, YuNet, çizim) — ölçülen etkiler yukarıdadır.
**Uyarı:** `ultralytics`'in `0` ayarı eğitim DataLoader'ı içindir; **Aşama 8/10 eğitim süreçlerinde bu bayrak kullanılmamalıdır** (5B yalnızca çıkarım/gösterim). Torch'un kendi iş parçacığı havuzu (YOLO CPU) bu ayardan bağımsızdır (ölçülmedi: ayrı bir etkileşim denemesi yapılmadı).

**KARAR (kullanıcı onayı beklenir):**
* **Varsayılan DEĞİŞTİRİLMEDİ.** `run_face_person_association.py` varsayılan olarak `cv2.setNumThreads` çağırmaz (`--opencv-threads keep`); `getNumThreads()==1` ise açık bir uyarı ve önerilen komut yazdırır. 4A/4B/5A kodları ve varsayılanları değişmedi (testle denetlenir: yalnızca 5B ayrıştırıcısında `--opencv-threads` vardır).
* **Öneri (ölçüm kanıtlı):** canlı kullanımda `--opencv-threads restore` kullanın: aynı kayıtlı videoda FPS ≈ 1,6–2,0×, YuNet ≈ 2,8× hızlı, YOLO/ByteTrack gerilemedi, sonuç imzası aynı. Varsayılan yapma kararı kullanıcıdadır (`OT-5B-004`); gerçek kamerada A8 testiyle (keep ↔ restore) teyit edilmelidir.
* **Alternatif (seçilmedi):** modül içinde otomatik `setNumThreads(-1)` — sessiz süreç geneli yan etki olurdu; `ultralytics` yeniden içe aktarılırsa/başka bir modül ayarı yeniden değiştirirse bozulur ve eğitim aşamalarıyla çakışabilir.

## 12. Bilinen sınırlamalar (açık; çözülmüş sayılmaz)

1. **Yüz–kişi bağı canlı insan doğrulaması değildir.** Poster, basılı fotoğraf, telefon/ekrandaki yüz bir kişi kutusunun kafa bölgesine denk gelirse bağlanır. Gözlenen (sentetik montaj, `RealDecoyTest`): poster/fotoğraf
   kırpıntıları YOLO'da kişi gibi izlenebilir (FP-4B-001/KL-4B-008) ve içlerindeki yüz o izle eşleşir; B kişisi bir fotoğrafı göğsünde tutarken YOLO kişiyi **iki kutuya bölebilir** ve sahte yüz bu parçalardan birine bağlanabilir
   (örtüşen parçalar `trusted=False` olduğundan kanıt biriktirmeye uygun sayılmaz; gerçek poster/telefon ile test EDİLMEDİ). `possibly_static` yalnızca kalıcı hareketsiz izleri bayraklar.
2. **Geometri sınırı:** kafa hizası birbirine yakın örtüşen iki kişi veya yinelenen/iç içe kutular (KL-4B-008) gerçek eşleşmeleri `ambiguous` bırakır (kasıtlı; maliyet ölçüldü, §9.3). Eşik değerleri gerçek veriyle ayarlanmadı.
3. **ID switch/devralma 4B'nin sınırlamasıdır ve çözülmedi** (KL-4B-001/003/009/010): ByteTrack görünüm kullanmaz. 5B bunu önlemez; yalnızca yanlış taşınmayı engeller (yüz her karede yeniden kurulur) ve şüpheli durumları `epoch`/`trusted` ile işaretler.
   Süreklilik eşikleri (zıplama IoU 0,3, boşluk 3 kare, örtüşme 0,25, sakinleşme 10 kare) gerçek veriyle ayarlanmadı; muhafazakârdır: hızlı hareket veya kadraj kenarında kırpılan kutu **yanlış alarm** (gereksiz `jump`) üretebilir
   (sentetik `leave_and_return` senaryosunda kenarda `jump` kesintileri gözlendi); bunun bedeli yalnızca kaybedilen güven karesidir, yanlış eşleşme değildir.
4. **Doğruluk gerçek veride ölçülmedi.** Elimizde etiketli çoklu-kişi videosu, gerçek poster/telefon testi ve kaydedilmiş gerçek kamera verisi (video/ham çıktı) yoktur; gerçek kamera kanıtı, kullanıcının nitel bildirimi ile paylaştığı iki ekran görüntüsü ve iki iz özetinden ibarettir (§13.4–13.6). Sentetik montaj (aynı üç kırpıntı, piksel-aynı kareler, kaydırma hareketi, yüz
   maskeleme taklidi) algoritma mantığını sınar; gerçek çoklu-kişi videosuyla eşdeğer DEĞİLDİR. IDF1/HOTA/MOTA, ID switch oranı ve ilişkilendirme doğruluğu **gerçek veride ölçülmedi** (OT-4B-003 açık kalır).
5. **Yüksek kamera açısı (≈ 2 m)** yüz görünürlüğü ve eşleştirme geometrisi (yüz merkezi dikey oranı) üzerindeki etkisi **bilinmiyor** (yukarıdan bakışta yüz kutusu kişi kutusunun ortasına yaklaşabilir; `max_center_y_ratio` 0,55 bu durumda yeterli olmayabilir).
6. **Performans** kayıtlı sentetik videodan ölçülmüştür (§10). Canlıda yalnızca iki ekran görüntüsündeki anlık overlay değerleri vardır (pencereli ≈ 16,6–16,7 FPS; §13.5); `keep`/`restore` konsol özetleri **paylaşılmadı**, kare→sonuç gecikmesi ve penceresiz canlı FPS **ölçülmedi**. YuNet yalnızca CPU'dadır (KL-5A-007); ilişkilendirme CPU'da ≈ 0,1 ms'dir.
7. **`--opencv-threads` süreç geneli bir ayardır;** varsayılan DEĞİŞTİRİLMEDİ (`keep`). Gerekçeli öneri §11'dedir; kullanıcı onayı beklenir.
8. **4B/5A sınırlamaları aynen geçerlidir:** FP-4A-001 (yorgan), FP-4B-001 (poster), KL-5A-001…010, OT-5A-001…008 (OT-5A-006 bu aşamada araştırıldı; varsayılan kararı kullanıcıdadır), kullanılmayan `mediapipe` (OT-5A-007).
9. Yüz tespiti `conf 0,6`'da bulanık arka plan yüzlerini kabul edebilir (KL-5A-006); bunlar `unmatched_face` olarak raporlanır (zidane.jpg'de gözlendi), kimseye bağlanmaz.
10. **Ek/çakışan insan kutuları gerçek kamerada tek gerçek kişide gözlendi (kullanıcının ekran görüntüleri; §13.5):** el bölgesinde ek bir `NO FACE` kutusu; gövde ve diz bölgesinde örtüşen birden fazla kutu ve bunun sonucu olarak yüzün `ambiguous` kalıp eşleştirilmemesi. Ham YOLO çıktısı ile aynı karelerde karşılaştırılmadı: kaynak (YOLO yanlış pozitifi/parçalanması mı, ByteTrack izi mi) **ayrıştırılmadı**; yanlış pozitifin kök nedeni belirlenmiş değildir. Sıklığı ölçülmedi. 4B'deki benzer gözlemle (OBS-4B-001, OT-4B-005) aynı ailedir; çözülmedi (`OT-5B-009`, `OT-5B-012`).
11. **Tek gerçek kişi için birden fazla Track ID** iz özetlerinde görüldü (1. koşu 7, 2. koşu 2 farklı ID; §13.6). Hangi ID'nin gerçek kişiye, hangisinin ek kutuya/ID değişimine karşılık geldiği doğrulanmadı; ID numaraları ayrıca ID switch sayısı olarak yorumlanmamalıdır.
12. **`trusted=False` eşleşmeler kimlik kanıtı değildir:** 1. koşuda ana izin 448 eşleşmesinden yalnızca 299'u güvenilirdi (149 eşleşme güvensiz). Sonraki aşamalar yalnızca `trusted_match` karelerinden ve aynı `(session_id, track_id, epoch)` içinden kanıt biriktirmelidir; `track_id`/`epoch` çalışan kimliği değildir.

## 13. Manuel MacBook kamera test planı (kullanıcı; Claude ortamında kamera izni yoktur)

**Kural:** *Gözlem yapılmadan hiçbir kutu doldurulmaz; bildirilmeyen test yapılmamış sayılır; Bölüm B kullanıcı gerçekleştirmeden başarılı sayılmaz.* Aşağıdaki "Beklenen" sütunları hedeftir, sonuç DEĞİLDİR.
Kayıt şablonu (§13.3) **yalnızca kullanıcının bildirdiği alanlarla** doldurulmuştur; bildirilmeyen alanlar `bildirilmedi` yazılmış veya boş bırakılmıştır (tam konsol çıktısı, sayı ve ekran görüntüsü kullanıcı tarafından üretilmeyen ayrıntılar eklenmemiştir).

Hazırlık (ağ kullanmaz; yalnızca dosya doğrular):
```bash
.venv/bin/python scripts/download_person_model.py --verify-only
.venv/bin/python scripts/download_face_model.py --verify-only
.venv/bin/python scripts/run_face_person_association.py --help
```
Kamera iznini Terminal/VS Code'a verin. Her koşuda ortamı (ışık, mesafe, arka plan, kaç kişi, kamera yüksekliği) ve konsol özetini kaydedin. **Yanlış yüz bağlantısı** = yüz etiketindeki `FACE>ID n` ile o yüzün gerçekte ait olduğu kişinin
kutusundaki `ID` aynı değil; **ID değişimi** = aynı kişinin `ID` numarasının değişmesi. Her ikisini **ayrı** sayın.

### 13.1 Bölüm A — tek kişiyle hemen yapılabilecekler
Ana komut: `.venv/bin/python scripts/run_face_person_association.py --source camera --camera-index 0 --device mps`

| # | Test | Yapılacak | Beklenen (hedef) | Kontrol listesi |
|---|---|---|---|---|
| A1 | İnsan + yüz kutuları birlikte | ≈ 0,5–1,5 m, kameraya bakın, 30 sn | `PERSONS: 1 FACES: 1 MATCHED: 1`; kişi kutusunda `ID n FACE`, yüz kutusunda `FACE>ID n` (aynı n, aynı renk) | Kutular doğru yerde mi? Yüz etiketindeki ID = kişi ID'si mi? İlk 3 karede `FACE*` (yıldız) sonra yıldızsız mı? |
| A2 | Track ID–yüz ilişkisi | 60 sn serbest hareket (otur, kalk, sağa-sola kay) | ID sabit kalır; yüz hep aynı ID'ye bağlı | ID numarası değişti mi (kaç kez)? Yüz hiç `unmatched`/`ambiguous` oldu mu? |
| A3 | Yüzü çevirme | Başı sağa/sola 15°, 30°, 45°, 60°, profil | Yüz bulunduğu sürece `FACE>ID n`; bulunmayınca `NO FACE`; **kişi kutusu ve ID kalır** | Hangi açıda `NO FACE`? ID korundu mu? Yüz geri gelince aynı ID mi? |
| A4 | Uzaklaşma | 1, 2, 3, 4 m (mümkünse) | Yüz kaybolunca `NO FACE`, ID korunur | Yüzün kaybolduğu mesafe? Kişi kutusu hâlâ var mı? |
| A5 | Yüzü kapatma | El / maske / kâğıtla yüzü kapatın | `NO FACE`, ID korunur; kapatınca yanlış yüz bağlanmaz | Kapatırken başka bir (sahte) yüz eşleşti mi? |
| A6 | Arkasını dönme | 5–10 sn arkanızı dönün, sonra dönün | Kişi izlenmeye devam eder (`NO FACE`); dönünce yüz yeniden bağlanır | Dönerken ID değişti mi? Geri dönünce aynı ID mi? |
| A7 | Kadrajdan çıkıp geri gelme | Kenardan çıkın, 2 sn / 5 sn / 10 sn sonra aynı kenardan girin | Büyük olasılıkla **yeni ID** (KL-4B-003/010); yeni ID `FACE*` ile başlar, eski ID'nin yüzü devralınmaz | Yeni ID mi aldınız? Hangi sürede? Yeni ID'de ilk kareler yıldızlı mı? |
| A8 | Performans | `--no-display --max-frames 300` ile `--opencv-threads keep` ve `restore` (aşağıdaki komutlar); sonra pencereli koşu | Konsol özeti: uçtan uca FPS, YOLO/ByteTrack/YuNet/ilişkilendirme süreleri; `restore` YuNet'i belirgin hızlandırır (kayıtlı videoda ölçüldü) | Her iki koşunun özetini kaydedin; pencereli/penceresiz fark; ID değişimi/yanlış bağ davranışında fark var mı? |
| A9 | Çıkış yöntemleri | `Q`, `Esc`, kırmızı düğme, terminalde `Ctrl+C` (ayrı koşular) | Pencere kapanır, kamera ışığı söner; çıkış kodu 0/0/0/130 | `echo $?`; kamera serbest kaldı mı? |
| A10 | Sahte yüz (opsiyonel, tek kişiyle) | Telefon ekranında bir yüz fotoğrafı tutun (önce tek başına, sonra göğsünüzün önünde) | **Beklenti: yüz tespit edilir; bağlanıp bağlanmadığı ölçülür.** Bu bir tasarım sınırıdır (§12) | Fotoğraf yüzü tespit edildi mi? Hangi etiketle (`FACE>ID`/`unmatched`/`ambiguous`)? Kişiyi iki kutuya böldü mü? |

A8 komutları:
```bash
.venv/bin/python scripts/run_face_person_association.py --source camera --camera-index 0 --device mps --no-display --max-frames 300
.venv/bin/python scripts/run_face_person_association.py --source camera --camera-index 0 --device mps --no-display --max-frames 300 --opencv-threads restore
.venv/bin/python scripts/run_face_person_association.py --source camera --camera-index 0 --device mps --opencv-threads restore
```

### 13.2 Bölüm B — daha sonra iki veya üç kişiyle (**kullanıcı yapmadan başarılı sayılmaz**)
Komut: `.venv/bin/python scripts/run_face_person_association.py --source camera --camera-index 0 --device mps --opencv-threads restore` (isteğe bağlı: aynı testleri `keep` ile de tekrarlayın).

| # | Test | Yapılacak | Beklenen (hedef) | Kontrol listesi |
|---|---|---|---|---|
| B1 | Yan yana | İki kişi yan yana, kameraya bakar, 30 sn | `PERSONS: 2 FACES: 2 MATCHED: 2`; her yüz kendi kişisinin ID'siyle | Yanlış yüz bağlantısı? ID değişimi? |
| B2 | Birlikte yürüme | İki kişi kameraya doğru/paralel yürür | Eşleşmeler doğru; örtüşme anlarında `FACE*`/`FACE ?` olabilir (güvenli) | Yanlış bağ var mı? Kaç kare belirsiz? |
| B3 | Yer değiştirme | İki kişi yan yana yürüyüp yer değiştirir | ID değişebilir (KL-4B-009); **yüz yeni kutudaki kişiye bağlanır, eski ID'nin yüzü yanlış kişiye yapışmaz**; ID değişimi sonrası `FACE*` | ID değişimi sayısı? Değişimden hemen sonra yüz doğru kişide mi? |
| B4 | Birbirinin önünden geçme | Biri diğerinin önünden geçer | Örtüşmede `FACE*`/`FACE ?`/`NO FACE`; geçtikten sonra doğru eşleşme | Örtüşmede yanlış bağlandı mı? Sonrasında ID değişti mi? |
| B5 | Kısmi/tam örtüşme | Biri diğerini kısmen, sonra tamamen kapatır | Kapanan kişi `NO FACE`/kaybolur; geri gelince yeni/aynı ID; yanlış yüz bağlanmaz | Kapanan yüz başkasına bağlandı mı? |
| B6 | Üç kişi aynı karede | Üç kişi, 30 sn | `PERSONS: 3 FACES: 3 MATCHED: 3` | Her yüz doğru ID'de mi? Kaç ID değişimi? |
| B7 | ID değişimi ve yanlış eşleştirme kontrolü | B3–B6 sırasında her ID değişiminde ve örtüşmede yüz bağlantısına bakın | Yanlış bağlantı sayısı **0** hedeflenir; ID değişimi ayrıca kaydedilir | ID değişimi sayısı ile yanlış yüz bağlantısı sayısını **ayrı** yazın |
| B8 | Yüksek kamera açısı (≈ 2 m) | Kamerayı ≈ 2 m yüksekliğe koyun, 1–3 kişi | Yüzler görünür mü ve eşleşiyor mu **bilinmiyor** (§12.5) | Yüz kutusu kişi kutusunda nerede? `NO FACE`/`unmatched` sıklığı? |

### 13.3 Gözlem kayıt şablonu (yalnızca kullanıcının bildirdiği alanlar dolu; boş = bildirilmedi/yapılmadı)

Her koşu için bir satır (gözlemden SONRA doldurulur). **"Beklenenle uyum" sütunundaki "Kullanıcı: başarılı" kullanıcı bildirimidir; bağımsız ölçüm değildir.** Komut: aşağıdaki A-satırlarının hepsi `--device mps --opencv-threads restore` ile verilen komutla (§13.4) yapılan koşulardır; test tarih/saati, ortam ve süreler bildirilmedi.

| Test | Tarih/saat | Komut (+ bayraklar) | Ortam (ışık/mesafe/yükseklik/arka plan) | Kişi sayısı | Süre | Gözlem (serbest metin) | ID değişimi sayısı | Yanlış yüz bağlantısı sayısı | Belirsiz/`unmatched`/`NO FACE` gözlemi | Beklenenle uyum (Evet/Hayır/Kısmen) | Konsol özeti (yapıştır) | Not |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| A1 | bildirilmedi | `run_face_person_association.py --source camera --camera-index 0 --device mps --opencv-threads restore` | bildirilmedi | 1 (Bölüm A) | bildirilmedi | İnsan kutusu, yüz kutusu ve Track ID birlikte gösterildi | bildirilmedi | bildirilmedi | İlk ekran görüntülerinde ek kutular ve `ambiguous` görüldü (§13.5) | Kullanıcı: başarılı. (Hedef `PERSONS: 1` ekran görüntülerinde karşılanmadı: 2 ve 3) | paylaşılmadı | Ham sayılar §13.5 |
| A2 | bildirilmedi | — | bildirilmedi | 1 (Bölüm A) | bildirilmedi | Ayrı kullanıcı kontrolü/başarı etiketi YOK; ham sayaç gözlemi §13.6 (1. koşu ID 1: 457 aktif/448 eşleşme; 2. koşu ID 1: 719/645) | bildirilmedi | bildirilmedi | — | Başarı etiketi yok (A2 başarılı SAYILMADI) | iz özetleri paylaşıldı (§13.6) | |
| A3 | bildirilmedi | `run_face_person_association.py --source camera --camera-index 0 --device mps --opencv-threads restore` | bildirilmedi | 1 (Bölüm A) | bildirilmedi | Baş çevirme: yüz kaybolduğunda insan takibi devam etti | bildirilmedi | bildirilmedi | — | Kullanıcı: başarılı (KISMİ kapsam: açı değerleri 15–60°/profil, yukarı-aşağı bakma ve eğme bildirilmedi) | paylaşılmadı | |
| A4 | | | | | | | | | | | | |
| A5 | bildirilmedi | `run_face_person_association.py --source camera --camera-index 0 --device mps --opencv-threads restore` | bildirilmedi | 1 (Bölüm A) | bildirilmedi | Yüz bulunmadığında `NO FACE` gösterildi | bildirilmedi | bildirilmedi | `NO FACE` | Kullanıcı: başarılı | paylaşılmadı | Kapatma sırasında sahte yüz eşleşip eşleşmediği bildirilmedi |
| A6 | | | | | | | | | | | | |
| A7 | bildirilmedi | `run_face_person_association.py --source camera --camera-index 0 --device mps --opencv-threads restore` | bildirilmedi | 1 (Bölüm A) | bildirilmedi | Kadrajdan çıkış; geri girişte yeni ID ve `FACE*` davranışı kontrol edildi | bildirilmedi (sayı/süre yok) | bildirilmedi | — | Kullanıcı: başarılı | paylaşılmadı | Çıkış süreleri (2/5/10 sn) bildirilmedi |
| A8 (keep) | bildirilmedi | `--opencv-threads keep` (bayrak/komut tam metni bildirilmedi) | bildirilmedi | 1 (Bölüm A) | bildirilmedi | `keep` denendi; kontrol başarılı bildirildi | — | — | — | Kullanıcı: başarılı (YALNIZCA kontrol; sayısal karşılaştırma YOK) | **paylaşılmadı** | FPS/süre özeti yok |
| A8 (restore) | bildirilmedi | `run_face_person_association.py --source camera --camera-index 0 --device mps --opencv-threads restore` | bildirilmedi | 1 (Bölüm A) | bildirilmedi | `restore` ile sistem gerçek kamerada başarıyla çalıştı | — | — | — | Kullanıcı: başarılı (YALNIZCA kontrol; sayısal karşılaştırma YOK) | **paylaşılmadı** | İki ekran görüntüsünün hangi ayarla alındığı yazılmadı (§13.5) |
| A9 (Q/Esc/düğme/Ctrl+C) | bildirilmedi | — | bildirilmedi | 1 (Bölüm A) | bildirilmedi | Q, Esc, kırmızı kapatma düğmesi ve Ctrl+C davranışları başarılı bildirildi | — | — | — | Kullanıcı: başarılı | paylaşılmadı | **Çıkış kodları (0/0/0/130) ve kamera ışığı bildirilmedi**; 4B betiği çıkış testi (OT-4B-001) yerine geçmez |
| A10 | | | | | | | | | | | | |
| B1 | | | | | | | | | | | | |
| B2 | | | | | | | | | | | | |
| B3 | | | | | | | | | | | | |
| B4 | | | | | | | | | | | | |
| B5 | | | | | | | | | | | | |
| B6 | | | | | | | | | | | | |
| B7 | | | | | | | | | | | | |
| B8 | | | | | | | | | | | | |

Boş bırakılanlar: **A4, A6, A10 ve B1–B8** (bildirilmedi/yapılmadı). Kalan sonuçlar gelince yaml'daki `OT-5B-001` (Bölüm A) ve `OT-5B-002` (Bölüm B) güncellenir; 4B'nin `OT-4B-001/003` ve 5A'nın `OT-5A-001/004/005` kayıtlarına AYRICA işlenir (geriye dönük başarılı işaretlenmez).

### 13.4 Kullanıcı tarafından BAŞARILI BİLDİRİLEN manuel kontroller (Bölüm A; tek kişi)
**Kaynak/tür:** kullanıcının bildirimi (kayıt tarihi 2026-11-15; testlerin tarih/saati ve süreleri bildirilmedi). **Nitel kullanıcı kontrolleridir; tam konsol çıktısı, sayısal ölçüm ve ekran görüntüsü olmayan ayrıntılar KAYDEDİLMEMİŞTİR ve bağımsız, etiketli doğruluk ölçümü olarak yorumlanmamalıdır.** Gerçek MacBook kamerası, tek gerçek kişi.
Komut: `.venv/bin/python scripts/run_face_person_association.py --source camera --camera-index 0 --device mps --opencv-threads restore`

| # | Kullanıcı kontrolü | Plan eşlemesi | Kullanıcı sonucu | Kaydedilmeyen/eksik |
|---|---|---|---|---|
| 1 | Birleşik görüntü | A1 | İnsan kutusu, yüz kutusu ve Track ID birlikte gösterildi — **başarılı** | Ek insan kutuları da görüldü (§13.5); "tam bir kişi kutusu" hedefi ekran görüntülerinde karşılanmadı |
| 2 | Baş çevirme | A3 (kısmi) | Yüz kaybolduğunda insan takibi devam etti — **başarılı** | Açı değerleri, yukarı/aşağı bakma, eğme; `NO FACE` süreleri |
| 3 | Kadrajdan çıkış | A7 | Geri girişte yeni ID ve `FACE*` davranışı kontrol edildi — **başarılı** | Süreler (2/5/10 sn), ID değişim sayısı |
| 4 | Yüz kapatma | A5 | Yüz bulunmadığında `NO FACE` gösterildi — **başarılı** | Kapatma sırasında sahte yüz eşleşmesi olup olmadığı |
| 5 | Performans karşılaştırması | A8 | `keep` ve `restore` denendi; kontrol **başarılı** bildirildi | **Her iki koşunun sayısal FPS/süre özeti paylaşılmadı** |
| 6 | Pencere ve çıkış | A9 | Q, Esc, kırmızı kapatma düğmesi, Ctrl+C — **başarılı** | **Çıkış kodları paylaşılmadı**; kamera ışığı |

**Bildirilmeyenler (başarılı SAYILMAZ):** A2 (ID–yüz ilişkisinin 60 sn kararlılığı; yalnızca ham sayaç gözlemi var, §13.6), A4 (uzaklaşma), **A6 (arkasını dönme; "baş çevirme" A3'tür, A6 değildir)**, A10 (sahte yüz/telefon/poster) ve tüm Bölüm B.

### 13.5 İlk canlı ekran görüntülerindeki bulgular: ek insan kutuları ve belirsiz eşleşme *[Kullanıcı; ekran görüntüsü okuması]*
Tek gerçek kişi bulunmasına rağmen fazla insan kutusu oluştu. Değerler ekrandaki overlay'in (kayan ortalama) anlık okumasıdır; konsol özeti DEĞİLDİR. **İki ekran görüntüsünün hangi `--opencv-threads` ayarıyla alındığı ve aynı koşuya ait olup olmadıkları yazılmamıştır** (komut `restore` ile verilmiştir; YuNet ≈ 14 ms, kayıtlı videoda ölçülen `restore` ≈ 11 ms / `keep` ≈ 30 ms ile `restore` ile UYUMLU bir okumadır, ancak ayar ekranda görünmez).

| | Ekran görüntüsü 1 | Ekran görüntüsü 2 |
|---|---|---|
| `PERSONS` / `FACES` | 2 / 1 | 3 / 1 |
| `MATCHED` | 1 (trusted 1) | **0** (trusted 0) |
| `NO FACE` / `AMBIGUOUS` | 1 / — | 1 / **2** |
| FPS (overlay) | 16,7 | 16,6 |
| YOLO (MPS) / ByteTrack / YuNet / ASSOC (ms) | 15 / 0,3 / 14 / 0,09 | 15 / 0,4 / 14 / 0,10 |
| Gözlem | Yüz `Track ID 27` ile eşleşti. Kaldırılmış el bölgesinde `ID 43 NO FACE` şeklinde **ek bir insan kutusu** | İnsan vücudu ve öndeki diz bölgesinde **birden fazla örtüşen kişi kutusu**; yüz `FACE ambiguous` işaretlendi ve **zorla eşleştirilmedi** |

**Yorum (sınırlı):**
* Ek kişi izleri ve belirsiz yüz eşleşmesi bulunduğu gösterilmiştir. Ek kutuların kaynağı **ham YOLO çıktısıyla aynı karelerde karşılaştırılmadı**; yanlış pozitifin (YOLO tespiti mi, ByteTrack izi mi, parçalanma mı) kök nedeni **belirlenmiş değildir** (`OT-5B-012`; 4B `OT-4B-005` araştırma görevi ailesi).
* Görüntü 2, tasarlanan güvenli davranışın (belirsizde eşleştirmeme) gerçek kamerada **gözlendiğini** destekler; aynı zamanda ek kutuların gerçek yüzü kullanılamaz hâle getirebildiğini (`MATCHED: 0`) gösterir: bu bir **kullanım riskidir** (`OT-5B-009`), yanlış eşleştirme değildir. Sıklığı ölçülmedi.
* Farklı ekran görüntülerinde farklı ID numaralarının (27, 43) görülmesi **ID switch sayısı DEĞİLDİR**; ByteTrack onaylanmamış geçici izler de numara tüketir (numaralar bitişik olmak zorunda değildir). Ayrıca bu numaralar (27, 43) aşağıdaki iz özetlerindeki ID'lerle (en çok 20) uyuşmaz: ekran görüntüleri ile iz özetleri **aynı koşuya ait olmayabilir.**
* Pencereli ≈ 16,6–16,7 FPS, 4B'de pencereli MPS için bildirilen 16,3 FPS ile aynı ailedir (OBS-4B-002: pencere/okuma/çizim maliyeti tam ayrıştırılmadı). Kayıtlı penceresiz videodaki 43,1 FPS (`restore`, MPS; §10) ile **karşılaştırılamaz** (canlı kamera + pencere vs. kayıtlı video, penceresiz).

### 13.6 Canlı koşu iz özetleri *[Kullanıcı; konsol "İz özeti" çıktısı]*
Bunlar takip ve ilişkilendirme **sayaçlarıdır**; gerçek kimlik tanıma doğruluğu, precision/recall, IDF1 veya HOTA DEĞİLDİR. Koşuların ortamı, hareketi ve test içeriği birebir karşılaştırılabilir olarak doğrulanmamıştır; koşu başına işlenen toplam kare ve gerçek kişi sayısı 1. koşu için ayrıca bildirilmemiştir.

**1. koşu**

| Track ID | Aktif kare | Yüz eşleşmesi | Güvenilir eşleşme | Epoch |
|---|---:|---:|---:|---:|
| 1 | 457 | 448 | 299 | 7 |
| 2 | 8 | 0 | 0 | 2 |
| 3 | 3 | 0 | 0 | 1 |
| 5 | 71 | 0 | 0 | 4 |
| 14 | 5 | 3 | 0 | 2 |
| 15 | 2 | 0 | 0 | 1 |
| 20 | 8 | 4 | 0 | 1 |

Süreklilik olayları: `gap: 5`, `new_track: 7`, `overlap_end: 6`. **7 farklı Track ID.** Tutarlılık denetimi (yalnızca iç tutarlılık): `new_track` = 7 = ID sayısı ✔; Σ(epoch − 1) = 11 = `gap` 5 + `overlap_end` 6 ✔.
Ana iz ID 1: yüz eşleşmesi/aktif ≈ %98,0 (448/457); güvenilir eşleşme/eşleşme ≈ %66,7 (299/448) — **149 eşleşme güvensizdir**. Diğer izlerde toplam 7 eşleşen kare var (ID 14: 3, ID 20: 4), hiçbiri güvenilir değil; ID 5 71 kare aktif ve hiç yüz eşleşmedi. Bu izlerin hangi fiziksel bölgeden (ek kutu, ID değişimi, gerçek kişi) kaynaklandığı **kesinleştirilmedi**; ID 1'in güvensiz kareleri ile `overlap_end: 6` olayları ek kutularla örtüşmeyle UYUMLUDUR ancak kare bazında neden kaydı olmadığı için bu bir **varsayımdır.**

**2. koşu**

| Track ID | Kare aralığı | Aktif kare | Yüz eşleşmesi | Güvenilir eşleşme | Epoch |
|---|---|---:|---:|---:|---:|
| 1 | 0–739 | 719 | 645 | 643 | 2 |
| 6 | 389–389 | 1 | 0 | 0 | 1 |

740 karelik aralık; ID 1 719 karede aktif (aralıkta 21 kare aktif değil), 645 karede yüz eşleşmiş, 643'ü güvenilir; ID 6 yalnızca bir kare. Sayaç oranları: ana iz yüz eşleşmesi/aktif ≈ %89,7; güvenilir eşleşme/aktif ≈ %89,4; güvenilir/eşleşen ≈ %99,7. Bu koşunun süreklilik olayı sayıları (`gap`/`overlap_end`/`jump`) paylaşılmadı; ID 1'in `epoch 2` değeri bir süreklilik kesintisi olduğunu gösterir, nedeni kaydedilmedi. Eşleşmeyen ~74 karenin nedeni (baş çevirme/yüz kapatma testleri olabilir) bildirilmedi.

**1. ve 2. koşu arasındaki fark (yalnızca gözlem):** farklı Track ID sayısı 7 → 2; ana izin güvenilir/eşleşen oranı ≈ %66,7 → ≈ %99,7; ana izin epoch sayısı 7 → 2. Bu, 2. koşuda daha kararlı bir gözleme **işaret eder**; ancak iki koşunun ortamı, hareketi ve içeriği (ek kutuya yol açan koşullar dâhil) karşılaştırılabilir olarak doğrulanmadığından **algoritmanın veya ayarın iyileştiği iddia EDİLMEZ**, nedeni (ortam, hareket, `keep`/`restore`, şans) ayrıştırılmadı.

### 13.7 `keep` ↔ `restore`: kanıt durumu
| Var olan kanıt | Eksik |
|---|---|
| Kayıtlı sentetik videoda (§10–11): `restore` FPS MPS 22,1 → 43,1, CPU 19,5 → 32,1; sonuç imzası aynı *[Ölçüldü; canlı DEĞİL]* | **Canlı kamerada `keep` ve `restore` için aynı ortamda alınmış sayısal FPS/süre özetleri paylaşılmadı** |
| `restore` komutuyla gerçek kamerada sistem başarıyla çalıştı (kullanıcı bildirimi, §13.4 #5) | `keep`/`restore` canlı karşılaştırmasının sayıları; pencereli/penceresiz fark; kare→sonuç gecikmesi |
| İki ekran görüntüsünde YuNet ≈ 14 ms (`restore` ile uyumlu okuma; ayar ekranda yazmıyor) | O ekran görüntülerinin ayarı doğrulanmadı |

Karar: `--opencv-threads` **varsayılanı değiştirilmedi** (`keep`); `OT-5B-004` (canlı performans) ve `OT-5B-008` (varsayılan kararı) **kullanıcı onayına açık kalır.** Kayıtlı video ölçümleri canlı ölçüm gibi sunulmaz.

### 13.8 Bölüm A/B'de hâlâ eksik olanlar
* **Bölüm A:** A2 (başarı etiketi), A3'ün açı/eğim ayrıntıları, **A4**, **A6**, **A10**, A5/A7 sayısal ayrıntıları, A8 sayısal `keep`/`restore` özetleri, A9 çıkış kodları (0/0/0/130) ve kamera ışığı. → `OT-5B-001` (kısmen yapıldı), `OT-5B-004`, `OT-5B-011`.
* **Bölüm B (tamamı):** B1–B8 yapılmadı (ikinci kişi yok). → `OT-5B-002`. Gerçek çoklu insan takibi doğrulanmış DEĞİLDİR.
* **Ek tanı (araştırma):** ek insan kutularının kaynağı (ham YOLO ↔ ByteTrack aynı karelerde) → `OT-5B-012`.


## 14. Kabul kriterleri ve kanıt düzeyi (5B) — canlı kamera bildirimi sonrası güncelleme

Sınıflar: **Tam kanıtlı** · **Sınırlı kanıtlı** (eksik kısım yazılı) · **Karşılanmadı**. Önceki 14 kriter esas alındı; gerçek kamera kanıtı **yalnızca tek kişi, kullanıcı bildirimi (nitel)** düzeyindedir ve gerçek çoklu-kişi testi yapılmadan hiçbir çoklu-kişi kriteri tam sayılmaz. **Sonuç: 14 kriterin 9'u tam, 5'i sınırlı kanıtlıdır; karşılanmayan yoktur. Sınıflar DEĞİŞMEDİ; 3, 5, 6, 7 için gerçek tek-kişi kanıtı eklendi ve eksikler güncellendi.**

| # | Kriter | Önceki | **Güncel** | Kanıt / eksik kısım |
|---|---|---|---|---|
| 1 | YOLO11n, ByteTrack ve YuNet aynı süreçte çalışıyor | Tam | **Tam kanıtlı** | Gerçek model testleri, CLI/betik, benchmark + **gerçek MacBook kamerasında çalıştı** (kullanıcı; ekran görüntüleri YOLO/ByteTrack/YuNet sürelerini birlikte gösteriyor) |
| 2 | Aynı karede insan ve yüz kutuları üretiliyor | Tam | **Tam kanıtlı** | Gerçek fotoğraflar, montaj + **canlı ekran görüntüleri** (insan kutusu + yüz kutusu + ID) |
| 3 | Güvenilir yüz–kişi eşleşmeleri doğru | Sınırlı | **Sınırlı kanıtlı** | Eklenen: canlıda tek kişide yüz `Track ID 27` ile eşleşti, birleşik görüntü kullanıcı tarafından başarılı bildirildi. **Eksik:** bağımsız/etiketli doğruluk yok; gerçek çoklu-kişi yok; canlı koşuda ek kutular nedeniyle bazı karelerde yüz eşleşmedi (§13.5) |
| 4 | Tek yüz aynı karede iki kişiye atanmıyor | Tam | **Tam kanıtlı** | Yapısal garanti + değişmez denetimi + özellik testleri + gerçek model; canlı ekran görüntülerinde de ihlal görülmedi (kanıt asıl olarak yapısal/otomatiktir) |
| 5 | Belirsiz durumlarda yanlış eşleştirme yapılmıyor | Sınırlı | **Sınırlı kanıtlı** | Eklenen: canlı ekran görüntüsü 2'de `AMBIGUOUS: 2` ve yüz zorla eşleştirilmedi. **Eksik:** gerçek iki-kişi örtüşmesinde (Bölüm B) denenmedi; T9b'de sahte yüz parçaya bağlandı (`trusted=False`) |
| 6 | Yüz kaybolduğunda insan takibi devam ediyor | Sınırlı | **Sınırlı kanıtlı (güçlendi)** | Eklenen: gerçek kamerada **baş çevirme ve yüz kapatma** kullanıcı tarafından başarılı bildirildi (`NO FACE`, takip sürdü). **Eksik:** **A6 (arkasını dönme) bildirilmedi**, sayısal kayıt/ekran görüntüsü yok, çoklu kişi yok; bu yüzden tam sayılmadı |
| 7 | Oturum ve Track ID değişimi güvenli yönetiliyor | Sınırlı | **Sınırlı kanıtlı (güçlendi)** | Eklenen: kadrajdan çıkış → yeni ID ve `FACE*` (güvensiz başlangıç) kullanıcı tarafından başarılı bildirildi; 1. koşuda süreklilik olayları (`gap 5`, `overlap_end 6`, `new_track 7`) kaydedildi. **Eksik:** gerçek ID switch/devralma (yer değiştirme, örtüşme) doğrulanmadı; ID switch oranı ölçülmedi (`OT-4B-003`); hangi olayın hangi fiziksel nedene ait olduğu bilinmiyor |
| 8 | Çoklu insan/yüz senaryoları otomatik testlerle doğrulanıyor | Sınırlı | **Sınırlı kanıtlı** | T1–T9b sentetik montaj + gerçek model. **Eksik:** gerçek çoklu-kişi videosu/kamerası yok; Bölüm B yapılmadı; T10 yapılamadı. (Canlı kanıt bu kriteri yükseltmez: otomatik test kriteri) |
| 9 | Gerçek model entegrasyonu çalışıyor | Tam | **Tam kanıtlı** | §9 + gerçek kamera |
| 10 | OpenCV iş parçacığı ve performans etkisi araştırılıp kaydediliyor | Tam | **Tam kanıtlı** | §10–11 (araştırma + kayıt). Canlı `keep`/`restore` sayısal karşılaştırması **yok** (kriterin parçası değil; `OT-5B-004/008` açık); varsayılan kararı kullanıcıda |
| 11 | Aşama 1–5A testleri bozulmuyor | Tam | **Tam kanıtlı** | Son çalıştırma: 1404 test, 0 başarısız. Bu belge güncellemesinde kod/test değişmedi (dosya özetleriyle doğrulandı) |
| 12 | Gerçek kamera için manuel test planı hazır | Tam | **Tam kanıtlı** | §13 (plan + kayıt şablonu; şablon artık kısmen doludur) |
| 13 | Yapılamayan çoklu-kişi testleri açık kayıt olarak korunuyor | Tam | **Tam kanıtlı** | `OT-5B-002`, `OT-4B-001`, `OT-4B-003`, `KL-4B-009/010`, `FP-4B-001`, `FP-4A-001` açık ve değişmedi |
| 14 | Teknik rapor ve belgeler hazır | Tam | **Tam kanıtlı** | Bu belge, STAGE_5B raporu, yaml, README/plan güncellendi |

### 14.1 Ayrı değerlendirilen riskler (hiçbiri çözülmüş değildir)
| Risk | Canlı gözlem | Kanıt/durum | Takip |
|---|---|---|---|
| **Ek insan kutuları nedeniyle yüzün `ambiguous` kalması** | Ekran görüntüsü 2: `PERSONS: 3`, `FACES: 1`, `MATCHED: 0`, `AMBIGUOUS: 2` | **Gözlendi**, tek örnek ekran görüntüsü; sıklığı ölçülmedi. Tasarım gereği güvenli (zorla eşleşme yok) ama gerçek yüz kullanılamaz kalıyor | `OT-5B-009` (açık), `OT-5B-012` (kök neden tanısı) |
| **Tek gerçek kişi için birden fazla Track ID** | İz özetleri: 1. koşu 7 ID, 2. koşu 2 ID; ekran görüntülerinde ek `NO FACE` kutuları | **Gözlendi**; kaynak (ek kutu/ID değişimi/parçalanma) **ayrıştırılmadı**; iki koşu karşılaştırılabilir doğrulanmadı | `OT-5B-012`; 4B `OT-4B-003` (ID switch oranı) açık |
| **Track ID sürekliliği ile çalışan kimliğinin karıştırılması** | — (kimlik alanı yok) | Tasarım: `track_id`/`epoch` kimlik DEĞİL; yeni ID geçmiş taşımaz. Canlı A7'de yeni ID `FACE*` ile başladı (kullanıcı bildirimi) | Aşama 6/7 sözleşmesi (§15) |
| **`trusted=False` sonuçlarının güvenilir kimlik kanıtı sayılması** | 1. koşu ID 1: 448 eşleşmenin 149'u güvensiz | Veri yapısı `trusted_match` sunar; sonraki aşamalarda uygulanması denetlenmedi (henüz tüketici yok) | §15: yalnızca `trusted_match` + aynı `epoch` |
| **Gerçek çoklu insan takibinin doğrulanmaması** | Bölüm B yapılmadı | **Doğrulanmadı**; yalnızca sentetik montaj | `OT-5B-002`, `OT-4B-001/003` |

### 14.2 Kapanışa hazırlık değerlendirmesi (karar kullanıcıya aittir; hiçbir aşama kapatılmadı)
* **5A:** teknik olarak hazır; tek kişilik canlı yüz tespiti (kamera açıldı, pencere çalıştı, gerçek yüz tespit edildi, yüz kutusu gösterildi) kullanıcı tarafından doğrulandı (nitel). Eksik: çoklu yüz canlı testi, mesafe/açı/kısmi/ışık/yanlış pozitif testleri, canlı FPS/gecikme, yüksek açı, doğruluk ölçümü, `mediapipe` kararı. Durum `READY_FOR_MANUAL_VERIFICATION`.
* **5B:** tek kişilik temel davranışlar (birleşik görüntü, baş çevirme/takip sürekliliği, kadrajdan çıkış/yeni ID, yüz kapatma/`NO FACE`, çıkış yolları, `restore` ile çalışma) kullanıcı tarafından başarılı bildirildi; **ancak sayısal kayıt eksik, A4/A6/A10 bildirilmedi, Bölüm B yapılmadı ve ek insan kutuları/ambiguous durumu çözülmedi.** 4B'nin kapanış emsali (kullanıcı kararıyla, açık sınırlamalarla kapsam içi prototip) vardır; 5B'nin kapatılıp kapatılmayacağı ve hangi eksiklerin kabul edileceği kullanıcı kararıdır. **Bu belgeye göre `COMPLETED` ilan edilmez.**
* **Kapatmadan önce önerilen minimum testler:** (1) A8 `keep` ve `restore` konsol özetleri + A9 çıkış kodları paylaşılsın; (2) tek kişiyle A2, A4, A6, A10; (3) ikinci kişi bulunduğunda en az B1, B3/B4, B6, B7 (ID değişimi ile yanlış yüz bağlantısı **ayrı** sayılsın); (4) ek insan kutularının kaynağı için ham YOLO ↔ takip karşılaştırması (`OT-5B-012`); (5) `--opencv-threads` varsayılanı kararı.

## 15. Aşama 5C ve sonraki aşamalara aktarılacaklar
* **Aşama 6 (çalışan kayıt)/7 (kimlik tanıma):** kanıt anahtarı `(session_id, track_id, epoch)` ve `trusted_match`; yalnızca `trusted` karelerdeki yüz kırpıntıları (`face_bbox`) bir izin kanıtına eklenmeli; `epoch` değişince veya iz güvensizken eski birikim bu ize ait SAYILMAMALI; `possibly_static` izler ve `at_border`/düşük güvenli yüzler kimlik kanıtına
  alınmamalı; poster/ekran yüzü (liveness yok) tanıma aşamasında ayrıca ele alınmalı. **Embedding/biyometrik veri 5B'de yoktur.**
* **Aşama 11 (kimlik + İSG birleştirme):** `integration` bu paketin çıktılarını tüketir (kişi↔yüz ve kişi↔ekipman ilişkilendirmesi aynı "karşılıklı en iyi + belirsizlik payı" ilkesiyle genişletilebilir); "Belirsiz" kararı için `ambiguous`/`no_face` durumları hazırdır.
* **5C (kapsamı tanımsız):** olası içerik — eşik ayarı için küçük izinli etiketli çoklu-kişi değerlendirme kümesi (OT-4B-003/OT-5A-002), yüksek açı geometrisi (§12.5), kasıtlı örtüşme/duplicate-kutu birleştirme (KL-4B-008), yüz tespit–insan tespit paralelleştirme (YuNet'i ayrı iş parçacığında çalıştırma; ölçülmedi), `--opencv-threads` varsayılan kararı.
* **Açık görevler:** `configs/project_stages.yaml` → `stage5b_open_items` (OT-5B-001…008) ve `stage5b_known_limitations` (KL-5B-001…008).
