# Aşama 3 — Canlı Kamera Altyapısı: Sonuç Raporu

**Aşama durumu: `IN_PROGRESS` → otomatik kontroller tamam, gerçek kamera doğrulaması bekleniyor (`READY_FOR_MANUAL_VERIFICATION`).**
Aşama 3'ün kapanış kararı kullanıcıya aittir. Aşama 4 başlatılmamıştır. Hiçbir commit/push yapılmamıştır.

---

## A. Devralınan proje durumu

| Kontrol | Sonuç (gerçek dosyalarda doğrulandı) |
|---|---|
| Çalışma dizini / depo | `/Users/ebrmac/Desktop/bitirme`, Git deposu, dal `main`, başlangıçta temiz; `origin` = GitHub deposu; son commit `9b98b78 Complete Stage 2 dataset validation` |
| Python / OpenCV / NumPy | Python 3.11.17, OpenCV 5.0.0 (AVFoundation + FFmpeg + Cocoa derlenmiş), NumPy 2.4.6 |
| Plan belgeleri | `docs/project_plan.md` ve `configs/project_stages.yaml` 12 aşamalı planla uyumlu; Aşama 1–2 tamamlandı |
| Aşama 2 rakamları | Manifest 1000 satır = **990 kullanılabilir (794 train + 196 validation)** + 10 dışlanan; `STAGE_2_CLOSURE_REPORT.md` ile tutarlı |
| Başlangıç testleri | **170/170 başarılı** (rapordaki rakamla aynı) |
| Aşama 2 bütünlük denetimi | **12/12 geçti** (`scripts/run_stage2c_checks.py`; rapor dosyasının SHA-256 özeti çalıştırma öncesi/sonrası aynı) |
| Mevcut kamera paketi | `src/isg_vision/camera/__init__.py` yalnızca bir docstring içeren boş iskeletti; silinen/yeniden yazılan mevcut kod yok |
| Paket kurulumu | `isg_vision` `.venv`'e kurulu değil; mevcut betikler `sys.path.insert(0, src)` kullanıyor — aynı düzen izlendi |

Fark/uyarı: rapordaki rakamlarla gerçek dosyalar arasında **fark bulunmadı**.

## B. Gerçekleştirilen geliştirmeler

Geliştirme, istendiği gibi üç grupta yapıldı; her gruptan sonra testler çalıştırıldı ve geçmeden sonraki gruba geçilmedi.

* **Grup 1 — Temel kare okuma** (40 test): `SourceConfig` (doğrulama), `FramePacket`, `FrameSource` (kamera + video ortak okuma, açma/kapatma, kare numaralama). Sonuç: 210/210.
* **Grup 2 — Görüntüleme, FPS, çözünürlük** (+55 test): istenen/gerçek çözünürlük ayrımı, `FpsMeter` (kayan pencere), overlay, `PreviewWindow`, video oynatma hız denetimi (`run_preview`). Sonuç: 265/265.
* **Grup 3 — Hata yönetimi, CLI, testler, doküman** (+43 test): sınırlı yeniden deneme, bağlantı kopması ayrımı, video erken bitiş tespiti, CLI + çıkış kodları, import yan etkisizliği testleri, dokümantasyon. Sonuç: **308/308**.

Özellikler: varsayılan/seçilebilir kamera indeksi · macOS'ta AVFoundation backend'i · MP4/MOV/AVI okuma · kaynak FPS ≠ ölçülen FPS · 1280×720 varsayılan istek ve gerçek çözünürlüğün geri okunması (fark uyarısı) · Q/Esc/pencere kapatma/Ctrl+C ile çıkış · `--no-display`, `--max-frames` · kamera/video/pencere hatalarında Türkçe, nedeni ve çözümü söyleyen mesajlar.

## C. Oluşturulan / değiştirilen dosyalar

Yeni dosyalar:

| Dosya | Görev |
|---|---|
| `src/isg_vision/camera/errors.py` | Hata sınıfları (yapılandırma, açma, video dosyası, okuma, bağlantı kopması, pencere) |
| `src/isg_vision/camera/frame.py` | `FramePacket`: kare, `frame_index`, `timestamp` (monotonic), `wall_time`, `source`, `source_kind`, gerçek `width/height` |
| `src/isg_vision/camera/config.py` | `SourceConfig` + doğrulama (indeks, çözünürlük, video yolu); `cv2` import etmez |
| `src/isg_vision/camera/capture.py` | `FrameSource`: kamera/video için ortak okuma, yeniden deneme, kaynak serbest bırakma |
| `src/isg_vision/camera/fps.py` | `FpsMeter` (bağımsız; `time.monotonic`, kayan pencere) |
| `src/isg_vision/camera/display.py` | `draw_overlay` (kopya üzerine), `PreviewWindow`, tuş/ASCII yardımcıları |
| `src/isg_vision/camera/preview.py` | `run_preview`: kaynak → FPS → overlay → pencere döngüsü, video hız denetimi |
| `src/isg_vision/camera/cli.py` | Argüman ayrıştırma, çıkış kodları, özet çıktısı |
| `scripts/run_camera.py` | `cli.main()` için ince başlatıcı |
| `docs/camera_module.md` | Teknik doküman (mimari, komutlar, izinler, hata tablosu, gelecek bağlantısı) |
| `tests/test_camera_{capture,fps,display,errors,cli}.py`, `tests/camera_test_support.py` | 138 yeni test + sahte `VideoCapture`/pencere/saat yardımcıları |
| `reports/camera/STAGE_3_REPORT.md` | Bu rapor |

Değiştirilen mevcut dosyalar (küçük, ekleme ağırlıklı): `src/isg_vision/camera/__init__.py` (dışa açılan adlar), `README.md` (durum, §13.2, §14–15), `docs/architecture.md` (camera satırı/durum), `docs/project_plan.md` ve `configs/project_stages.yaml` (yalnızca Aşama 3 durumu: `IN_PROGRESS`), `src/isg_vision/utils/paths.py` (`REQUIRED_FILES`'a 2 dosya), `tests/test_no_side_effects.py` (kamera alt modülleri import listesine eklendi). Aşama 2 kodu, `data/`, `models/`, `reports/dataset_audit/` değişmedi.

## D. Kamera mimarisi

```
kamera / video ─► FrameSource ─► FramePacket akışı ─► (Aşama 4+: tespit/takip/yüz/KKD)
                      │                │
                      │                └─► run_preview ─► FpsMeter + draw_overlay ─► PreviewWindow ─► Q/Esc/kapat
                      └─ open / read / frames / close (idempotent)
```

* `FrameSource` yalnızca kare üretir; ekran/FPS/yapay zekâ bilmez. Kamera ve video aynı sınıf ve aynı `FramePacket` ile çalışır.
* `FpsMeter` bağımsızdır (zaman damgası alır). `draw_overlay` girdi karesini değiştirmez.
* Modüller import edildiğinde kamera/pencere açılmaz (testle doğrulandı); kamera paketi `torch/ultralytics/mediapipe/onnxruntime` yüklemez (testle doğrulandı).
* Gelecek modüller yalnızca `source.frames()` akışını tüketir; kamera kodu yeniden yazılmaz.

## E. Gerçek kamera testleri

**Gerçek kamera testi tamamlanamadı.** Yapılanlar ve sonuçlar:

| Deneme | Sonuç |
|---|---|
| `scripts/run_camera.py --source camera --camera-index 0 --no-display --max-frames 60` (Claude'un Bash ortamı) | `OpenCV: not authorized to capture video (status 0)` → `[HATA] Kamera 0 açılamadı ...`, çıkış kodu 1 |
| Aynı komut, uygulamanın Terminal panelinde | Aynı hata; macOS izin penceresi görünmedi |

Bu, ana işlevi doğrulamasa da **hata yolunun gerçek ortamda doğru çalıştığını** gösterdi (anlaşılır mesaj, kod 1, kaynak serbest, çökme yok). İzin ben veremem ve aşmaya çalışmadım. Sistemde iki kamera görünüyor (MacBook Pro Kamerası ve iPhone Süreklilik Kamerası); hangi indeksin hangisine karşılık geldiği doğrulanamadı.
**Doğrulanmayan:** gerçek kamerada açılış, görüntü, FPS değeri, gerçek çözünürlük, Q/Esc ile çıkış, kamera ışığının sönmesi. Komutlar bölüm J'dedir.

## F. Video dosyası testleri (gerçekten çalıştırıldı)

| Test | Sonuç |
|---|---|
| Sentetik MP4/MOV/AVI (geçici dosya) okuma | 15–100 kare, doğru sırada (`frame_index` 0..N-1, içerik sırası doğrulandı), gerçek çözünürlük doğru; kaynak dosyanın SHA-256'sı değişmedi |
| **Gerçek pencere** + 640×480, 25 FPS, 100 karelik hareketli video (Terminal panelinde) | Pencere açıldı, 100 kare **4,0 sn**'de oynatıldı, ölçülen ortalama **24,9 FPS** (kaynak 25,0), video sonunda **kod 0** ile temiz çıkış |
| Gerçek Cocoa penceresi tanısı | Gösterilince `WND_PROP_VISIBLE=1.0`; pencere yok edilince `getWindowProperty` `cv2.error` fırlatıyor ve `closed_by_user` bunu `True` olarak yorumluyor |
| Gerçek `SIGINT` (Ctrl+C) | Çalışan süreç 1 sn'de durduruldu: kod **130**, özet yazıldı, traceback yok |
| Olmayan dosya / klasör / boş dosya / rastgele bayt | `VideoFileError` + anlaşılır mesaj, kod 1 |
| Kesik AVI (%60) | 56/100 kare okundu, `premature_end=True`, uyarı, çökme yok |
| Kesik MP4/MOV | OpenCV açamıyor → `VideoFileError` (çökme yok) |

Pencerenin ekranda görsel olarak göründüğünü ben görüntüleyemedim; yukarıdaki sonuç programatik doğrulamadır.

## G. Otomatik testler

| Grup | Test |
|---|---|
| Aşama 1–2 (mevcut, değişmedi; `test_no_side_effects` listesine kamera modülleri eklendi) | 170 |
| `test_camera_capture` | 40 |
| `test_camera_fps` | 22 |
| `test_camera_display` | 33 |
| `test_camera_errors` | 16 |
| `test_camera_cli` | 27 |
| **Toplam** | **308 — hepsi başarılı** (`unittest discover -s tests` ve kökten `unittest discover` ile; ~8 sn) |

Testler gerçek kamera açmaz, izin istemez, ağ/model kullanmaz; `data/` ve `models/` altına dokunmaz. Aşama 2 bütünlük denetimi 12/12, `check_environment.py` başarılı. Başarısız test yok. (Geliştirme sırasında birkaç kendi test hatamı düzelttim — bkz. bölüm I.)

## H. Performans gözlemleri

* **Ölçülen:** 640×480 / 25 FPS sentetik video, pencereli oynatma: 100 kare, 4,0 sn, ortalama **24,9 FPS** (kaynak 25,0 FPS) — oynatma hızı kaynağa uygun, hızlandırılmadı.
* Penceresiz okuma hızı (640×480 mp4v sentetik): ~7 900 kare/sn — yalnızca dosya okuma/çözme hızıdır; kamera veya model performansı **değildir**.
* **Ölçülmeyen:** gerçek kamera FPS'i ve çözünürlüğü (izin yok). İleride model eklendiğinde FPS düşecektir; bu rakamlar onu temsil etmez.

## I. Karşılaşılan sorunlar

1. **Kamera izni yok** (hem Bash ortamında hem Terminal panelinde): macOS izin penceresi çıkmadı → gerçek kamera testi yapılamadı. Önerilen çözüm: Terminal.app veya VS Code terminalinden çalıştırmak (bölüm J).
2. Kendi testlerimdeki hatalar: sahte kamera kareleri bitince `FrameReadError` beklenen davranıştı (testler düzeltildi); `FakeFactory` `KeyboardInterrupt`'ı tanımıyordu; `time.sleep` varsayılan argümanı import anında bağlandığı için mock'lanamıyordu → `FrameSource` `sleep`'i çağrı anında çözecek şekilde değiştirildi; gecikme tolerans aralığı ve bir regex düzeltildi.
3. Cocoa'da pencere yok edilince `getWindowProperty` istisna fırlatıyor → `closed_by_user` bunu "kapandı" sayacak şekilde yazılmıştı; gerçek pencerede doğrulandı.
4. `run_stage2c_checks.py` girdi ve çıktı için aynı dizini kullanıyor (`--out-dir` yalnızca yönlendirme değil); depo içinde çalıştırıldı, rapor dosyasının SHA-256'sı değişmedi, `git status` `data/ reports/ models/` için temiz.
5. Belge tutarlılık denetimi (`T2D.2`) "Aşama 3 + eğitim/YOLO" ifadelerini olumsuzlama olmadan engelliyor; yeni metinler buna uygun yazıldı, denetim geçti.

## J. Benim çalıştırmam gereken komutlar

İzin penceresini gösterebilen **Terminal.app** (veya VS Code'un entegre terminali) kullanın. İlk çalıştırmada macOS "kameraya erişmek istiyor" derse **İzin ver**; çıkmazsa Sistem Ayarları → Gizlilik ve Güvenlik → Kamera'dan o uygulamaya izin verip uygulamayı yeniden başlatın.

```bash
cd /Users/ebrmac/Desktop/bitirme

# 1) Penceresiz: izin + gerçek kare + FPS + gerçek çözünürlük (60 kare)
.venv/bin/python scripts/run_camera.py --source camera --camera-index 0 --no-display --max-frames 60

# 2) Canlı pencere; Q, Esc veya pencere kapatma ile çıkın (çıkış kodunu da görün)
.venv/bin/python scripts/run_camera.py --source camera; echo "exit=$?"

# 3) Başka kamera indeksi / çözünürlük
.venv/bin/python scripts/run_camera.py --source camera --camera-index 1
.venv/bin/python scripts/run_camera.py --source camera --width 640 --height 480

# 4) Kendi videonuzla
.venv/bin/python scripts/run_camera.py --source video --video-path /yol/video.mp4

# 5) Tüm otomatik testler
.venv/bin/python -m unittest discover -s tests
```

Kontrol listesi: kamera açılıyor mu · görüntü + `FPS`/`RESOLUTION` görünüyor mu · konsoldaki `gerçek çözünürlük` doğru mu · Q / Esc / pencere kapatma çalışıyor mu · çıkışta kamera ışığı sönüyor mu · `Bitiş nedeni` satırı ve çıkış kodu 0 mı. (Sonuçları bana yazarsanız raporu güncelleyebilirim.)

## K. Aşama 3 tamamlanma durumu (kabul kriterleri)

`[x]` = doğrulandı, `[~]` = kod yazıldı ve test edildi ama gerçek kamerada doğrulanmadı.

- [x] Mevcut proje yapısı korundu
- [x] Kamera modülü oluşturuldu
- [~] OpenCV ile kamera erişimi sağlandı — kod + mock testleri tamam; **gerçek donanım doğrulanmadı (izin)**
- [~] Canlı görüntü penceresi — gerçek pencere + video ile doğrulandı; canlı kamerayla doğrulanmadı
- [x] Video dosyasından kare okuma desteği
- [x] FPS ölçümü
- [x] Kaynak FPS ile ölçülen FPS ayrıldı
- [x] Çözünürlük ayarları eklendi (mock ve gerçek video ile test)
- [~] Gerçek çözünürlük okunabiliyor — video gerçek doğrulandı; kamera bekliyor
- [~] Q/Esc ile çıkış — tuş mantığı ve döngü birim testli; gerçek tuş basışı manuel. Ctrl+C gerçek süreçle doğrulandı
- [x] Kaynaklar güvenli biçimde serbest bırakılıyor (idempotent `close`, `finally`, testli)
- [x] Kamera/video hataları yönetiliyor
- [x] Kareler gelecekteki modüllere aktarılabilir (`FramePacket`)
- [x] Komut satırı desteği
- [x] Otomatik kamera testleri yazıldı (138)
- [x] Önceki testler bozulmadı (170/170; 12/12 bütünlük)
- [x] Dokümantasyon güncellendi
- [x] Ham veri setleri değiştirilmedi
- [x] Model eğitimi yapılmadı
- [x] Aşama 4'e geçilmedi

**Karar önerisi:** Durum `READY_FOR_MANUAL_VERIFICATION`. Bölüm J'deki gerçek kamera/Q-Esc kontrolleri başarılı olursa aşamayı kapatma kararı sizindir.

**Plan belgesindeki ek madde (karar sizin):** `docs/project_plan.md` Aşama 3 için "teknik işlev testleri (FPS, gecikme, yeniden bağlanma, kare kaybı)" der; sizin görev listenizde bunlar yoktu. Kapsam dışı bırakıldığı için **otomatik yeniden bağlanma uygulanmadı** (bağlantı koparsa `SourceDisconnectedError` ile durur), kare kaybı/gecikme için ayrı ölçüm aracı yazılmadı (yalnızca `FramePacket.timestamp` altyapısı var). Bunları Aşama 3'e eklemek isteyip istemediğinizi belirtin.

## L. Aşama 4'e hazırlık

Aşama 4 çoklu insan tespiti ve takibi (ByteTrack/BoT-SORT) olacaktır. Hazırlık durumu: kare akışı `FrameSource.frames()` → `FramePacket` olarak hazır; `frame_index` ve `timestamp` takipçi için gereken sıra/zaman bilgisini veriyor; aynı akış kayıtlı videolarla tekrarlanabilir değerlendirme sağlıyor; overlay genişletilebilir. **Aşama 4 başlatılmamıştır**; başlamadan önce Aşama 3 kapanış kararınız ve kamera doğrulaması beklenmektedir.
