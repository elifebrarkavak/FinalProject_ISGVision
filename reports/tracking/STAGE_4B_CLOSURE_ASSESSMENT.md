# Aşama 4B — Kapanış Değerlendirmesi

**Tarih:** 2026-11-13 · **Durum:** `COMPLETED` önerisi **kullanıcı tarafından onaylandı (2026-11-13) ve uygulandı** (kapsam içi prototip, açık sınırlamalarla). Uygulanan kayıtlar: `configs/project_stages.yaml`, `README.md`, `docs/project_plan.md`, `docs/architecture.md`, `docs/person_tracking.md`, `reports/tracking/STAGE_4B_REPORT.md` (§23, §29) ve gerekçeli düzeltme olarak `docs/person_detection.md` satır 4. **Güncel ve bağlayıcı kayıt: `STAGE_4B_REPORT.md` §23 ve §29.**
Bu dosya, onaydan önce hazırlanan değerlendirmedir (aşağıdaki §2–§6 onay öncesi metindir; §7'deki "uygulanmadı" ifadesi onayla aşılmıştır). Commit'lenmemiştir. Kod, test, eşik ve algoritma değişmedi; Aşama 5'e geçilmedi; commit/push yapılmadı.
**Onay sonrası ince düzeltme:** kullanıcı örtüşme (Manuel Test 4) ve kadrajdan çıkıp-girme (Manuel Test 5) için ID değişimi bildirdi; yer değiştirme (Manuel Test 3) **ayrıca bildirilmedi** — aşağıdaki "Test 3/4/5" ifadeleri bu ayrımla okunmalıdır.

---

## 1. Dayanak

* **Esas alınan ölçüt:** ilk Aşama 4B görev tanımının §21 "Aşama 4B kabul kriterleri" listesi (**28 madde**) ve oradaki durum kuralı (önceki oturum: "ByteTrack ile çoklu insan takibi"). Rapor §23'ün 1–28. satırları bu listeyle aynı sırada ve aynı metindedir.
* **29. satır özgün listede yoktur.** "Canlı yanlış pozitif gözlemi kayıtlı ve çözülmüş gösterilmemiş", poster gözleminden sonra Claude'un eklediği bir belgeleme maddesidir; kapanış koşulu değildir.
* **Özgün durum kuralı (özet):** otomatik ve kayıtlı video testleri başarılı, gerçek MacBook kamera testleri yapılmamışsa `READY_FOR_MANUAL_VERIFICATION`; gerçek kamera doğrulaması olmadan `COMPLETED` denmez; **IP kamera kapanış koşulu değildir.**
* **Özgün manuel test planında poster yoktur.** Manuel Test 7 yorgandır; poster, canlı oturumda sonradan çıkan bir gözlemdir.
* **Kullanıcı talimatı (2026-11-13):** poster testi kapanışta zorunlu engel yapılmayacak; başlangıç kabul kriterleri esas alınacak; gerekirse açık sınırlama ve araştırma görevi olarak taşınacak.

## 2. Kullanıcı güncellemesi (2026-11-13) — kayıt

1. **Poster erişilemiyor.** Test ortamı değişti; poster kameranın arka planında değil. Poster için planlanan aynı koşullardaki **T1 ve T2 yapılamaz → YAPILMADI; tamamlanmış veya başarılı değildir.** Yeni ortamda 4A betiğiyle yapılan kısmi T1 (telefondaki fotoğraf) poster karşılaştırması sayılmaz (rapor §26).
2. **FP-4B-001 (poster yanlış pozitifi) doğrulanmış gözlem olarak korunur** (kullanıcının ekran görüntüsü; ekran görüntüsünün kendisi depoda yoktur ve Claude bu oturumda görmemiştir, kayıt rapor §24'e dayanır). **Çözülmedi;** aynı koşullarda yeniden doğrulanamıyor; kaynağı (YOLO mu ByteTrack mı) ayrıştırılmadı.
3. **Canlı gözlem [Kullanıcı]:** gerçek kamerada çoklu insan takibi çalışıyor; **örtüşme** ve **kadrajdan çıkıp geri girme** durumlarında **ID değişimleri** gözlendi. Bu gözlem **nitel**dir: kişi sayısı, kaç değişim, hangi kenar/süre, cihaz bildirilmedi; sayı veya oran **uydurulmadı.** Manuel Test 3/4/5 için doğru ifade: "gözlem yapıldı, ayrıntılı kayıt yok" — başarılı/başarısız etiketi verilmedi.
   Gözlem, önceden belgelenmiş sınırlamalarla **uyumludur**: KL-4B-001 (ID konuma yapışır, devralma), KL-4B-003 (aynı kenardan dönen yeni ID alır), KL-4B-007 (idealize sentetik sonuçlar gerçeği temsil etmez). Yeni bir sınırlama değil, bu sınırlamaların ilk gerçek-kamera teyididir.

## 3. Bağımsız yeniden doğrulama (bugün, Claude) *[Ölçüldü]*

| Kontrol | Sonuç |
|---|---|
| `unittest discover -s tests` | **883 test, OK, 7 atlandı** (32,5 sn); 287 4B takip testi + **596 önceki** (sayım ayrıca doğrulandı) |
| Gerçek pencere testleri (`ISG_GUI_TESTS=1`, `RealWindowTrackingLoopTest`) | **2/2 OK**, kırmızı düğme tıklaması dahil (sahte kareler; kamera açılmadı) |
| `scripts/check_environment.py` | **BAŞARILI** (0 uyarı, 0 kritik hata) |
| `scripts/run_stage2c_checks.py` | **12/12 geçti** (T6.8 ham veri + model dahil). Betik izlenen `stage2c_integrity_checks.json` dosyasının yalnızca `date` alanını değiştirdi; **`git checkout` ile geri alındı** |
| Model SHA-256 | `0ebbc80d…ee1` — rapordaki değerle aynı |
| `lap` | `requirements.txt:18` ve `requirements-lock.txt:18` `lap==0.5.14`; `pip check` temiz |
| Eski kaynaklar (`c51ae14` → `HEAD`) | `human_detection/`, `camera/`, 4A/3 betikleri, `data/`, `models/`: **fark yok**. Tek fark `reports/detection/STAGE_4A_REPORT.md` **+17/−0** (§24 ek gözlem; 4A sonuçları değişmedi) |
| Eski testler | `tests/` altında yalnızca **ekleme** (3423 +, 0 −); `test_no_side_effects.py` +9 satır (yeni modül adları), gevşetme yok |
| Git | HEAD `b90fadf`; çalışma ağacı temizdi (bu dosyadan önce); commit/push yok |

## 4. Özgün 28 kriter — karar ve kanıt

✔ = karşılandı · ✔* = kriter metni karşılandı, **canlı kamera kanıtı sınırlı** (sütundaki nota bakın). Karşılanmayan (✖) özgün kriter **yok**; yapılmayan alt unsurlar §5'tedir.

| # | Kriter | Karar | Kanıt |
|---|---|---|---|
| 1 | 4A kaynak kodları korunuyor | ✔ | Bkz. §3: kaynak diff'i boş. **Not:** rapor §23 satır 1'deki "raporlar boş" ifadesi 4A raporundaki +17 satırlık ek gözlem nedeniyle düzeltilmeli |
| 2 | ByteTrack mevcut Ultralytics ile çalışıyor | ✔ | Rapor §3 (8.4.210); `test_person_tracking_bytetrack_real.py` (59) + `test_person_tracking_real_model.py` (20) bugün geçti; canlı ByteTrack süresi 0,2–0,5 ms [Kullanıcı] |
| 3 | Bağımlılıklar kontrollü | ✔ | `lap==0.5.14` iki dosyada, `pip check` temiz; `scipy` eklenmedi; wheel SHA-256/lisans kaydı rapor §4 |
| 4 | Ağsız takip | ✔ | `test_real_bytetrack_loads_offline_without_network_or_pip`, `test_models_and_tracker_run_offline_without_network_or_pip`, `test_prepare_offline_environment_blocks_autoinstall_and_network_probe` bugün geçti |
| 5 | YOLO kare başına bir kez | ✔ | `test_yolo_model_runs_exactly_once_per_frame_with_real_detector_class`, `test_detector_and_tracker_are_called_exactly_once_per_frame_in_order`; gerçek model 45 kare = 45 çağrı (rapor §6) |
| 6 | DetectionResult → ByteTrack | ✔ | adaptör testleri (51) + gerçek ByteTrack (59) |
| 7 | Her aktif iz için geçici Track ID | ✔ | Rapor §7; canlı ekran görüntüsünde `ID n` etiketleri [Kullanıcı] |
| 8 | Aynı karede benzersiz ID | ✔ | `test_duplicate_track_id_in_one_frame_is_an_error`, `test_three_independent_unique_tracks`, `test_thirty_people_all_get_unique_ids`; ihlal çalışma anında hata verir, canlı koşularda hata bildirilmedi |
| 9 | Tek insan takibi | ✔* | Sentetik: ID 1 / 40 kare, switch 0; gerçek model kesiti 30/30 kare aynı ID (§14). **Canlıda ayrı tek kişi testi yok** (Manuel Test 1 doldurulmadı) |
| 10 | Çoklu insan takibi | ✔ | 3 ve 30 kişi benzersiz ID; gerçek model 2 kişi 50/50 kare. **Canlı:** 2 gerçek kişi, penceresiz CPU ve MPS'te 300'er karede 2 ID (300/300 ve 300/299 kare) ve "çoklu takip çalışıyor" [Kullanıcı]. 3 kişilik canlı test bildirilmedi |
| 11 | Kısa örtüşme davranışı değerlendiriliyor | ✔ | §15-D (örtülen 8 kare `LOST`, sonra aynı ID; NMS birleşmesi); **canlı: örtüşmede ID değişimi gözlendi** [Kullanıcı] → KL-4B-001. Kriter "değerlendiriliyor" der, kusursuz takip istemez |
| 12 | Kadraja giriş/çıkış yönetiliyor | ✔ | §15-E (tampon, `LostTrack`, aynı kenardan dönüşte yeni ID); **canlı: çıkıp-girişte ID değişimi gözlendi** [Kullanıcı] → KL-4B-003 |
| 13 | Kayıp/aktif iz ayrımı | ✔ | `LostTrack` ayrı; `ACTIVE`/`LOST` ayrı sayılır, kayıp iz çizilmez (§10, §15) |
| 14 | Tracker reset doğru | ✔ | `test_explicit_reset_clears_everything_and_restarts_ids`, `test_image_size_change_resets_real_tracker`, `test_source_change_resets_state_and_flags_reason`, `test_tracker_is_reset_once_before_the_first_frame` |
| 15 | İnsan olmayan kareler | ✔ | §15-F (50 boş kare, hata yok); `test_empty_detections_still_update_backend_exactly_once` |
| 16 | Eşik davranışları analiz edildi | ✔ | Rapor §8–9 (A–F, gerçek YOLO, 300 kare); varsayılanlar değişmedi |
| 17 | Yorgan yanlış pozitifi riski değerlendirildi | ✔* | §18 (kontrollü dizi: tek kare iz almaz, kalıcı statik sürekli ID alır). **Canlı poster gözlemi bu öngörüyle tutarlıdır** (FP-4B-001). Gerçek yorgan testi yapılmadı; risk **çözülmedi** |
| 18 | Bounding box + Track ID görüntüleniyor | ✔ | `test_person_tracking_visualization.py` (21); canlı ekran görüntüsünde kutular ve ID'ler [Kullanıcı] |
| 19 | FPS ve takip gecikmesi ölçülüyor | ✔* | Kayıtlı videoda tam ölçüm (§16); canlıda FPS, YOLO, ByteTrack süreleri [Kullanıcı]. **Canlı kare→sonuç gecikmesi bildirilmedi** |
| 20 | Kamera/video kaynakları destekleniyor | ✔ | Video: gerçek koşu CPU/MPS (`exit=0`); kamera: kullanıcının üç canlı koşusu |
| 21 | Q/Esc/kırmızı düğme/Ctrl+C korunuyor | ✔* | `test_q_exit_closes_everything`, `test_esc_exit`, `test_window_close_button_exit`, `test_ctrl_c_during_loop_exits_130_and_releases_everything` + gerçek Cocoa kırmızı düğme testi (bugün 2/2). **Gerçek kamerada yalnızca Q doğrulandı** [Kullanıcı] |
| 22 | Gerçek ByteTrack entegrasyonu | ✔ | 62 + 20 test bugün geçti |
| 23 | Önceki 596 test korunuyor | ✔ | Bkz. §3 (596 + 287 = 883) |
| 24 | Aşama 2 bütünlük kontrolleri | ✔ | 12/12 ve `check_environment.py` BAŞARILI (bugün) |
| 25 | Ham veri ve model değişmemiş | ✔ | T6.8 geçti; model SHA-256 aynı; `data/`, `models/` diff boş |
| 26 | Teknik rapor | ✔ | `STAGE_4B_REPORT.md`, `docs/person_tracking.md` |
| 27 | Eksik manuel testler açıkça belirtilmiş | ✔ | Rapor §21, docs §17 durum tabloları |
| 28 | Aşama 5'e geçilmemiş | ✔ | Başlatılmadı |

**Özet:** 28/28 karşılandı (24 tam, 4 ✔*: #9, #17, #19, #21). *Mock/sentetik kanıt ile gerçek-kamera kanıtı ayrı tutuldu; sentetik sonuçlardan doğruluk çıkarılmadı.*

## 5. Yapılmayan / karşılanmayan unsurlar ve kapanışa etkisi

| Unsur | Durum | Kapanışı engeller mi? | Gerekçe | Taşınacağı yer |
|---|---|---|---|---|
| Poster T1/T2 (aynı koşullarda) | **Yapılamıyor** (ortam değişti) | Hayır | Özgün listede yok; kullanıcı kararı | FP-4B-001 açık sınırlama + OT-4B-005 araştırma görevi |
| Poster/çakışan kutu kaynağı (YOLO mu ByteTrack mı) | **Ayrıştırılmadı** | Hayır | Araştırma sorusu; T0–T7 planı başka bir sahnedeki yanlış pozitif için de geçerli | OT-4B-005 |
| FP-4A-001 (yorgan) ve FP-4B-001 (poster) çözümü | **Çözülmedi** | Hayır | 4A da bu sınıf sınırlamayla `COMPLETED` oldu; ByteTrack kalıcı statik yanlış pozitifi elemez (KL-4B-002/008); eşik denemesi kabul ölçütüyle doğrulanmadan uygulanmaz (§27) | Araştırma görevi / sonraki aşamalar |
| Gerçek yorgan canlı testi (Manuel Test 7) | **Yapılmadı** | Hayır | Kriter 17 "risk değerlendirildi" der; poster aynı riskin canlı örneği | OT-4B-002 (FP-4A-001 ile birlikte) |
| Üç kişilik canlı test (Manuel Test 2) | **Bildirilmedi / belirsiz** | Hayır | Kriter 10, ≥ 2 kişiyle karşılandı; 3/30 kişi otomatik testli. Kullanıcı kaç kişi olduğunu teyit edebilir | OT-4B-001 kalanı |
| Yüksek açı ≈ 2 m (Manuel Test 6) | **Yapılmadı** | Hayır | Kabul listesinde yok; yalnızca manuel planda. Etkisi bilinmiyor | OT-4B-001 kalanı |
| Esc / kırmızı düğme / Ctrl+C — gerçek kamerada | **Bildirilmedi** | Hayır (≈ 2 dk'lık kontrol önerilir) | Otomatik + gerçek Cocoa testiyle karşılandı; pencere kodu Aşama 3'ten yeniden kullanılıyor | OT-4B-001 kalanı |
| Test 3/4/5 sayısal ayrıntı (kaç ID değişimi, hangi hız/kenar) | **Nitel gözlem var, ölçüm yok** | Hayır | Özgün §16: veri yoksa "ölçülmedi" | OT-4B-003 |
| IDF1 / HOTA / MOTA, ID switch oranı | **Ölçülmedi** | Hayır | Etiketli veri yok; özgün §16 ölçülmemiş yazılmasını ister, %85 iddiası yok | OT-4B-003 |
| Canlı kare→sonuç gecikmesi, pencereli CPU, ham kamera tavanı | **Bildirilmedi** | Hayır | Aşama 3 backlog B3-3/B3-4 ile ilişkili, ertelenmiş | OT-4B-004 |
| Pencereli 16,3 ↔ penceresiz 29,9 FPS nedeni | **Açık** | Hayır | Performans yorumu; takibin 0,5 ms'si bunu açıklamaz | OBS-4B-002 |
| Gerçek IP kamera | **Yapılmadı** | Hayır | Özgün metin açıkça kapanış koşulu yapmaz | OT-4A-003 |

## 6. Kapanış değerlendirmesi

**Öneri:** Aşama 4B, **kapsam içi prototip olarak `COMPLETED`** işaretlenmeye uygundur; iki koşulla:

1. Kapanış metni **"ID sürekliliği doğrulandı" demez.** Doğru ifade: *ID sürekliliği gerçek kamerada nitel olarak gözlendi; örtüşme ve kadrajdan çıkıp-girişte ID değişimleri görüldü ve belgelenmiş sınırlamalarla uyumludur; sayısal doğruluk (ID switch oranı, IDF1/HOTA) ölçülmedi.*
2. Açık sınırlamalar ve araştırma görevleri kayıtta açık kalır (§5, §7).

**Gerekçe:**

* **28/28 özgün kriter karşılandı;** karşılanmayan kriter yok. Kanıtlar türlerine göre ayrıdır (§4).
* **Özgün kapı kuralı sağlandı.** Kural, gerçek kamera doğrulaması olmadan `COMPLETED` denmemesini ister. Artık gerçek kameradan elde edilen kanıt var: üç canlı koşu (FPS ve süreler), iki gerçek kişide 300'er karede kararlı ID, çoklu takibin çalıştığı bildirimi, örtüşme ve çıkış-giriş davranışının gözlenmesi, Q ile temiz çıkış.
* **Gözlenen ID değişimleri sürpriz değil.** ByteTrack görünüm kullanmaz; bu tasarım sınırı raporda başından beri yazılıdır (KL-4B-001/003/007). Kabul kriterleri sıfır ID değişimi istemez; %85 başarı iddiası da yapılmaz.
* **Emsal:** 4A, FP-4A-001 açık sınırlamasıyla `COMPLETED` oldu; poster aynı sınıfın bir örneğidir ve kullanıcı kararıyla kapanış engeli değildir.
* **Yapılmayan testler başarılı gösterilmedi;** bunlar §5'te açık borç olarak kalır.

**Karar verirken bilinmesi gerekenler (çekinceler):**

* ID değişimlerinin **sıklığı bilinmiyor.** Sık ise Aşama 5 (yüz–kişi ilişkilendirme) için gerçek bir risktir; rapor §22 tek kare yerine çok kareli doğrulamayı öneriyor. 4B kapanışını engellemez, **Aşama 5 tasarım girdisidir.**
* Poster kaynağı bilinmediği için, Aşama 5'te posterdeki yüzün yanlış pozitif bir kişi izine bağlanma riski vardır (rapor §27: poster de yüz içerir). Aşama 5'te "iz hareketsiz ve kalıcı" bayrağı düşünülebilir; silme filtresi olarak değil.
* Üç kişi ve Esc/kırmızı düğme/Ctrl+C gerçek kamera teyidi (≈ 10 dk) **zorunlu değil,** ama kapanış kaydını sağlamlaştırır.

**Önerimi değiştirecek durum:** Siz özgün Manuel Test 2, 6 veya 8'in gerçek kamerada yapılmasını kapanış şartı sayarsanız `READY_FOR_MANUAL_VERIFICATION` korunur; kalan iş yalnızca o testlerdir.

## 7. Onayınızla yapılacak kayıt değişiklikleri (**şu an uygulanmadı**)

* `configs/project_stages.yaml`: `stage4_substages.4B` → `completed`; Aşama 4 genel durumu → `completed` (yaml notu: 4, yalnızca 4B `COMPLETED` olunca `COMPLETED` olur).
* `OT-4B-001`, `OT-4B-004`: `partially_performed` → kalan alt görevler açıkça listelenerek "kapanışı engellemeyen kalan kontrol". `OT-4B-002`: FP-4A-001 ile birlikte ertelenmiş. `OT-4B-003`: açık, Aşama 12/değerlendirme planına bağlı. `OT-4B-005`: **araştırma görevi, kapanışı engellemez; poster aynı koşullarda yeniden doğrulanamıyor.**
* `FP-4B-001`: `open_known_limitation` kalır, "ortam değişti, aynı koşullarda yeniden doğrulanamıyor" notu eklenir. `OBS-4B-001` `open_investigation` kalır. Yeni kayıt: 2026-11-13 kullanıcı gözlemi (örtüşme ve çıkış-girişte ID değişimi, nitel).
* `STAGE_4B_REPORT.md`: başlık durumu, §23 satır 1 ifadesi düzeltmesi, satır 29'un "sonradan eklenen belgeleme maddesi" notu, yeni §29 kapanış kaydı; `docs/person_tracking.md`, `README.md`, `docs/project_plan.md` tutarlılığı.
* Hiçbir değişiklik commit'lenmez; commit kararı sizindir.
