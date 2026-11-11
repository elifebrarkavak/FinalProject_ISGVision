# Aşama 3 — Canlı Kamera Altyapısı: Sonuç Raporu

**Aşama durumu: `COMPLETED` — 2026-11-11, kullanıcının açık onayıyla resmen kapatıldı.**
Kapanış koşulları: (1) gerçek kamera doğrulamaları kullanıcı tarafından yapıldı (bölüm E.4, H.1), (2) kapsamı ertelenen görevler açık backlog olarak belgelendi (bölüm K.2). Aşama 4 başlatılmamıştır.
Bu kapanışta commit/push yapılmamıştır (Aşama 3'ün ilk geliştirmesi kullanıcının kendi commit'i `f12db14` içindedir; sonraki düzeltme ve kapanış değişiklikleri commit edilmemiştir).

**Rapor sürümleri:** (1) gerçek kamera olmadan (`READY_FOR_MANUAL_VERIFICATION`); (2) kullanıcının ilk gerçek kamera sonuçlarıyla; (3) macOS kırmızı pencere kapatma düzeltmesi (bölüm M); (4) **nihai kapanış**: kalan manuel doğrulamalar, FPS tanı testi, durum `COMPLETED`.
Gerçek kamera rakamları **kullanıcı beyanıdır**; Claude'un ortamında kamera izni olmadığından ben yeniden üretemedim. Kendi çalıştırdığım doğrulamalar (otomatik testler, gerçek pencere/video denemeleri) ayrıca belirtilmiştir.

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

Güncelleme notu: bu rapor güncellenirken `HEAD` = `f12db14 cheapter3 is started` idi (Aşama 3 dosyalarını kullanıcı commit etmiş; çalışma ağacı temizdi). Güncellemeden önce ve sonra testler yeniden çalıştırıldı (bölüm G); `data/`, `models/`, `reports/dataset_audit/` ve kaynak kod değiştirilmedi.

## B. Gerçekleştirilen geliştirmeler

Geliştirme, istendiği gibi üç grupta yapıldı; her gruptan sonra testler çalıştırıldı ve geçmeden sonraki gruba geçilmedi.

* **Grup 1 — Temel kare okuma** (40 test): `SourceConfig` (doğrulama), `FramePacket`, `FrameSource` (kamera + video ortak okuma, açma/kapatma, kare numaralama). Sonuç: 210/210.
* **Grup 2 — Görüntüleme, FPS, çözünürlük** (+55 test): istenen/gerçek çözünürlük ayrımı, `FpsMeter` (kayan pencere), overlay, `PreviewWindow`, video oynatma hız denetimi (`run_preview`). Sonuç: 265/265.
* **Grup 3 — Hata yönetimi, CLI, testler, doküman** (+43 test): sınırlı yeniden deneme, bağlantı kopması ayrımı, video erken bitiş tespiti, CLI + çıkış kodları, import yan etkisizliği testleri, dokümantasyon. Sonuç: **308/308**.

Özellikler: varsayılan/seçilebilir kamera indeksi · macOS'ta AVFoundation backend'i · MP4/MOV/AVI okuma · kaynak FPS ≠ ölçülen FPS · 1280×720 varsayılan istek ve gerçek çözünürlüğün geri okunması (fark uyarısı) · Q/Esc/pencere kapatma/Ctrl+C ile çıkış · `--no-display`, `--max-frames` · kamera/video/pencere hatalarında Türkçe, nedeni ve çözümü söyleyen mesajlar.

## C. Oluşturulan / değiştirilen dosyalar

Yeni dosyalar (ilk geliştirme):

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

**Sürümlere göre değişen dosyalar**

| Sürüm | Değişen / eklenen dosyalar |
|---|---|
| İlk geliştirme | Aşağıdaki "Yeni dosyalar" ve "Değiştirilen mevcut dosyalar" |
| 2. (ilk gerçek kamera sonuçları) | Yalnızca belge: `reports/camera/STAGE_3_REPORT.md`, `README.md`, `docs/camera_module.md`, `docs/architecture.md` |
| 3. (pencere kapatma düzeltmesi, bölüm M) | Yeni: `src/isg_vision/camera/macos_window.py`, `tests/test_camera_window_close.py` (32 test). Değişen: `src/isg_vision/camera/display.py` (`PreviewWindow.open` + `close_button_enabled`), `src/isg_vision/camera/cli.py` (uyarı + başlangıç ipucu), `tests/test_no_side_effects.py` ve `tests/test_camera_cli.py` (yalnızca modül listesine `macos_window` eklendi), belgeler |
| 4. (nihai kapanış) | **Yalnızca belge/yapılandırma:** `reports/camera/STAGE_3_REPORT.md`, `README.md`, `docs/camera_module.md`, `docs/architecture.md`, `docs/project_plan.md`, `configs/project_stages.yaml`. Kaynak kod ve testler **değiştirilmedi** (FPS ölçümü doğru göründüğü için kod değişikliği yapılmadı) |

Her sürümde Aşama 1–2 kodu, `data/`, `models/` ve `reports/dataset_audit/` değişmedi.

Değiştirilen mevcut dosyalar (ilk geliştirmede; küçük, ekleme ağırlıklı): `src/isg_vision/camera/__init__.py` (dışa açılan adlar), `README.md` (durum, §13.2, §14–15), `docs/architecture.md` (camera satırı/durum), `docs/project_plan.md` ve `configs/project_stages.yaml` (yalnızca Aşama 3 durumu: ilk geliştirmede `IN_PROGRESS` yapılmıştı; nihai kapanışta `COMPLETED` oldu, sürüm tablosuna bakın), `src/isg_vision/utils/paths.py` (`REQUIRED_FILES`'a 2 dosya), `tests/test_no_side_effects.py` (kamera alt modülleri import listesine eklendi). Aşama 2 kodu, `data/`, `models/`, `reports/dataset_audit/` değişmedi.

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

**Gerçek kamera ana testi yapıldı ve çalıştı; ancak testi ben değil kullanıcı çalıştırdı.** Claude'un ortamında (Bash ve Terminal paneli) macOS kamera izni yoktur; ben kamera açamadım ve aşağıdaki rakamları yeniden üretmedim. Rakamlar kullanıcının MacBook Pro M4 üzerindeki oturum özetinden alınmıştır.

### E.1 Kullanıcının bildirdiği gerçek kamera oturumu

| Ölçüt | Bildirilen değer |
|---|---|
| Kaynak | `camera:0` |
| Gerçek çözünürlük | 1280×720 (varsayılan istek 1280×720 ile aynı) |
| İşlenen kare | 529 |
| Çalışma süresi | 17,9 sn |
| Ölçülen ortalama FPS | 29,5 |
| Kaynağın bildirdiği FPS | 15,0 |
| Çıkış nedeni | `kullanıcı çıkışı (Q/Esc)` |
| Canlı görüntü | Açıldı ve görüntülendi |

Beyanın iç tutarlılığını kodla karşılaştırarak denetledim (yeni ölçüm değildir):
* `FpsMeter.average_fps = (kare − 1) / süre` olduğundan (528 / 17,9 ≈ **29,50**) bildirilen 29,5 ile uyumludur.
* "kullanıcı çıkışı (Q/Esc)" metni `cli.py`'de `REASON_USER_QUIT` için yazılan metindir. Pencere kapatma `pencere kapatıldı` yazardı; yani bu oturum pencere kapatmayla bitmemiştir.
* "Kaynağın bildirdiği FPS" = `cap.get(cv2.CAP_PROP_FPS)` değeridir (`capture.py`); ölçülen FPS ise `read()` çağrılarının zaman damgalarından hesaplanır. İki değer **birbirinden bağımsız ölçülür** (bölüm H.1).

**Bildirimde yer almadığı için doğrulanmış sayılmayanlar (nihai kapanışta da bildirilmedi):** `camera:0`'ın hangi fiziksel kamera olduğu (MacBook yerleşik kamerası / iPhone Süreklilik Kamerası), pencerede `FPS`/`RESOLUTION` satırlarının okunurluğu, istenen–gerçek çözünürlük uyarısının çıkmaması, çıkışta kamera ışığının sönmesi. Bunlar tamamlanmış doğrulama olarak işaretlenmemiştir.

### E.2 Çıkış yolları — hangisi gerçekten doğrulandı?

| Çıkış yolu | Durum | Dayanak |
|---|---|---|
| **Q tuşu** | **Doğrulandı** (gerçek kamera, kullanıcı beyanı) | `exit=0` (E.4) |
| **Esc tuşu** | **Doğrulandı** (gerçek kamera, kullanıcı beyanı) | `exit=0` (E.4) |
| **Pencere kapatma (kırmızı ✕ tıklaması)** | **Doğrulandı** (gerçek kamera, gerçek fare tıklaması, kullanıcı beyanı) | `Bitiş nedeni: pencere kapatıldı`, `exit=0`. Düğme OpenCV'de devre dışıydı; bölüm M'deki düzeltmeden sonra çalışıyor. Ayrıca gerçek Cocoa penceresinde taklit tıklamayla otomatik doğrulandı |
| Ctrl+C | Doğrulandı — gerçek süreç, **video** kaynağı, çıkış kodu 130 (bölüm F, M.5); gerçek kamerada ayrıca denenmedi | Bölüm F |
| Birim testleri (sahte tuş/pencere) | Q, q, Esc ve pencere-kapandı durumları ayrı testlerle kapsanır | `test_camera_display.py`, `test_camera_cli.py`, `test_camera_window_close.py` — gerçek tuş basışı değildir |

### E.3 İlk denemeler (Claude ortamı, izin yok) — tarihçe

| Deneme | Sonuç |
|---|---|
| `scripts/run_camera.py --source camera --camera-index 0 --no-display --max-frames 60` (Claude'un Bash ortamı) | `OpenCV: not authorized to capture video (status 0)` → `[HATA] Kamera 0 açılamadı ...`, çıkış kodu 1 |
| Aynı komut, uygulamanın Terminal panelinde | Aynı hata; macOS izin penceresi görünmedi |

Bu denemeler ana işlevi doğrulamasa da **kamera erişilemezken hata yolunun doğru çalıştığını** gösterdi (anlaşılır mesaj, kod 1, kaynak serbest, çökme yok). İzin ben veremem ve aşmaya çalışmadım.

### E.4 Kalan manuel doğrulamalar — kullanıcı beyanı (nihai kapanış)

Kullanıcı bu doğrulamaları MacBook Pro M4 üzerinde, `camera:0` ile yapıp bildirmiştir (ben yeniden üretemedim):

| Doğrulama | Bildirilen sonuç |
|---|---|
| Kamera açılışı | `camera:0` ile başarıyla açıldı |
| Canlı görüntü | 1280×720 çözünürlükte gösterildi |
| Q ile çıkış | Başarılı, `exit=0` |
| Esc ile çıkış | Başarılı, `exit=0` |
| Kırmızı pencere kapatma düğmesi | Başarılı; çıktı: `Bitiş nedeni: pencere kapatıldı`, `exit=0` |
| FPS tanı testi | Bölüm H.1 |

## F. Video dosyası testleri (gerçekten çalıştırıldı)

| Test | Sonuç |
|---|---|
| Sentetik MP4/MOV/AVI (geçici dosya) okuma | 15–100 kare, doğru sırada (`frame_index` 0..N-1, içerik sırası doğrulandı), gerçek çözünürlük doğru; kaynak dosyanın SHA-256'sı değişmedi |
| **Gerçek pencere** + 640×480, 25 FPS, 100 karelik hareketli video (Terminal panelinde) | Pencere açıldı, 100 kare **4,0 sn**'de oynatıldı, ölçülen ortalama **24,9 FPS** (kaynak 25,0), video sonunda **kod 0** ile temiz çıkış |
| Gerçek Cocoa penceresi tanısı (ilk sürüm) | Gösterilince `WND_PROP_VISIBLE=1.0`; pencere **programatik `destroyWindow` ile** yok edilince `getWindowProperty` `cv2.error` fırlatıyor ve `closed_by_user` bunu `True` yorumluyor. **Sınır:** bu deneme kırmızı düğmenin yolunu denemiyordu; düğmenin devre dışı olduğunu bu yüzden yakalayamadı (bölüm M.3) |
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
| `test_camera_window_close` (pencere kapatma düzeltmesi) | 32 (29 varsayılan + 3 isteğe bağlı gerçek-pencere testi) |
| **Toplam (ilk geliştirme sonu)** | **308 — hepsi başarılı** (`unittest discover -s tests` ve kökten `unittest discover` ile; ~8 sn) |
| **Toplam (pencere kapatma düzeltmesi sonrası)** | **340 test: 337 çalıştı + 3 atlandı (gerçek pencere testleri), hepsi başarılı**; önceki 308 test aynen geçiyor (bölüm M.5) |
| **Nihai kapanış çalıştırması** | **340 test tanımlı: 337 başarılı, 3 isteğe bağlı GUI testi atlandı, başarısız 0** (bölüm N) |

Testler gerçek kamera açmaz, izin istemez, ağ/model kullanmaz; `data/` ve `models/` altına dokunmaz. Aşama 2 bütünlük denetimi 12/12, `check_environment.py` başarılı. Başarısız test yok.

**Gerçek kamera sonuçları işlendikten sonra yeniden çalıştırma (bu güncellemede, gerçekten çalıştırıldı):** `unittest discover -s tests` → **308/308 OK** (belge düzenlemesinden önce ve sonra ikişer kez; kökten `unittest discover` de 308/308) · `scripts/run_stage2c_checks.py` → **12/12 geçti** (T2D.2 belge tutarlılık denetimi dahil) · `scripts/check_environment.py` → BAŞARILI (0 uyarı, 0 kritik hata). Çalıştırmalardan sonra `git status` yalnızca bu güncellemenin dört belge dosyasını gösterir. Bu testler gerçek kamerayı sınamaz; gerçek kamera sonuçları bölüm E.1'deki kullanıcı beyanıdır. (Geliştirme sırasında birkaç kendi test hatamı düzelttim — bkz. bölüm I.)

## H. Performans gözlemleri

* **Ölçülen:** 640×480 / 25 FPS sentetik video, pencereli oynatma: 100 kare, 4,0 sn, ortalama **24,9 FPS** (kaynak 25,0 FPS) — oynatma hızı kaynağa uygun, hızlandırılmadı.
* Penceresiz okuma hızı (640×480 mp4v sentetik): ~7 900 kare/sn — yalnızca dosya okuma/çözme hızıdır; kamera veya model performansı **değildir**.
* **Gerçek kamera (kullanıcı beyanı, `camera:0`, 1280×720, pencereli):** 529 kare / 17,9 sn → ölçülen ortalama **29,5 FPS**; kaynağın bildirdiği FPS **15,0**. Bu, **uygulamanın `read()` ile kare alma hızıdır**. Nihai kapanışta bildirilen FPS tanı testi (H.1) ~30 FPS okuma hızını güçlü biçimde destekler; kaynağın bildirdiği 15,0 değeri gerçek performans olarak **kabul edilmemiştir**.
* Rakamlar yapay zekâ modeli **olmadan** ölçülmüştür. İleride model eklendiğinde FPS düşecektir; bu rakamlar onu temsil etmez.

### H.1 Kaynak FPS (15,0) ≠ ölçülen FPS (~30): FPS tanı testi ve sonuç

**Durum: büyük ölçüde açıklandı; ileri doğrulama ertelendi (B3-4). Kod değişikliği gerekmedi.**

Kullanıcı, gerçek kamerada (`camera:0`) bölüm J'deki ham `cv2.VideoCapture` tanı komutunu çalıştırdı (kullanıcı beyanı; uygulamanın `FpsMeter`'ini kullanmaz):

| Ölçüt | Bildirilen değer |
|---|---|
| `CAP_PROP_FPS` (bildirilen) | 15,0 |
| Okunan kare | 300 |
| Süre | 9,97 sn |
| Ölçülen ortalama FPS | 29,98 |
| Birebir aynı ardışık kare | 0 |
| Kareler arası medyan | 33,3 ms (≈ 30,0 FPS) |
| Kareler arası en büyük | 41,6 ms |

İç tutarlılık: (300 − 1) / 9,97 ≈ 29,99; medyan 33,3 ms ≈ 30 FPS; en büyük aralık 41,6 ms bir-iki karelik dalgalanmadır; 300 karelik sınırın ~10 sn'de dolması beklenen sonuçtur.

**Hipotezlerin durumu**

| # | Hipotez | Sonuç |
|---|---|---|
| H1 | `CAP_PROP_FPS` gerçek akış hızını yansıtmıyor; kamera ~30 FPS veriyor | **Güçlü biçimde destekleniyor.** Medyan aralık 33,3 ms ve ardışık kopya kare yok; bildirilen 15,0 gözlenen okuma hızıyla uyuşmuyor |
| H2 | `read()` aynı kareyi tekrar tekrar döndürüyor; gerçek hız ~15 FPS | **Gözlenmedi** (299 ardışık karşılaştırmada birebir aynı kare 0). Sınır aşağıda |
| H3 | `FpsMeter` / zaman damgası hatası | **Bulgu yok.** Bağımsız ham okuma (29,98 FPS) ile uygulama oturumundaki ölçüm (29,5 FPS) tutarlı; ayrı oturumlar olduğu için birebir kıyas değildir; küçük fark (~%1,6) ayrı oturumlarla ve pencere/overlay maliyetiyle açıklanabilir (bu açıklama doğrulanmadı). 25 FPS videoda 24,9 FPS sonucu ve birim testleri de hata göstermedi |

**Teknik sınırlamalar (korunmuştur)**
* "Birebir aynı kare sayısı = 0", **yalnızca tam kopya kareleri eler**. Bütün karelerin bağımsız pozlamalar olduğunu kesin olarak kanıtlamaz (ör. sürücü/kamera tarafında kare enterpolasyonu ya da küçük farklarla tekrar eden kareler bu testle ayırt edilemez).
* Tek kamera, tek oturum, 300 kare, model olmadan, ışık/ortam koşulu kaydedilmeden ölçüldü; bir bağımsız zaman referansı (ör. harici zamanlayıcı) kullanılmadı.
* `FpsMeter` hâlâ *tamamlanan `read()` çağrılarını* sayar, *benzersiz kare* sayısını değil.

**Sonuç ve kural:** Kameranın sürücüsünün bildirdiği `CAP_PROP_FPS` **gerçek performans olarak kabul edilmez**; kamera için performans ve zaman hesaplarında **ölçülen FPS ve `FramePacket.timestamp`** kullanılır (video dosyalarında kaynak FPS dosyadan gelir ve oynatma hızını belirler). Mevcut ölçülen FPS hesabı doğru göründüğünden **kodda değişiklik yapılmadı**; yalnızca performans yorumu ve dokümantasyon güncellendi (`docs/camera_module.md` bölüm 6, 9, 13). İleri doğrulama B3-4 olarak ertelendi.

## I. Karşılaşılan sorunlar

1. **Kamera izni yok** (hem Bash ortamında hem Terminal panelinde): macOS izin penceresi çıkmadı → gerçek kamera testini Claude yapamadı. Kullanıcı testleri izinli bir terminalde kendisi çalıştırdı ve sonuçları bildirdi (bölüm E.1, E.4). **Çözüldü.**
1a. **Kaynak FPS ≠ ölçülen FPS** (15,0 / ~30) — FPS tanı testiyle büyük ölçüde açıklandı: kaynak FPS güvenilmez, kamera ~30 FPS okuma hızı veriyor (bölüm H.1). İleri doğrulama B3-4 olarak ertelendi.
1b. **Q ve Esc ayrı ayrı doğrulanmamıştı** → kullanıcı ayrı ayrı doğruladı, `exit=0` (bölüm E.2). **Çözüldü.**
1c. **macOS'ta kırmızı pencere kapatma düğmesi çalışmıyordu** — kök neden: OpenCV pencereyi `Closable` stil biti olmadan açıyor (düğme devre dışı). **Düzeltildi** ve kullanıcı gerçek fare tıklamasıyla doğruladı (bölüm M, E.4).
2. Kendi testlerimdeki hatalar: sahte kamera kareleri bitince `FrameReadError` beklenen davranıştı (testler düzeltildi); `FakeFactory` `KeyboardInterrupt`'ı tanımıyordu; `time.sleep` varsayılan argümanı import anında bağlandığı için mock'lanamıyordu → `FrameSource` `sleep`'i çağrı anında çözecek şekilde değiştirildi; gecikme tolerans aralığı ve bir regex düzeltildi.
3. Cocoa'da pencere yok edilince `getWindowProperty` istisna fırlatıyor → `closed_by_user` bunu "kapandı" sayacak şekilde yazılmıştı; gerçek pencerede doğrulandı.
4. `run_stage2c_checks.py` girdi ve çıktı için aynı dizini kullanıyor (`--out-dir` yalnızca yönlendirme değil); depo içinde çalıştırıldı, rapor dosyasının SHA-256'sı değişmedi, `git status` `data/ reports/ models/` için temiz.
5. Belge tutarlılık denetimi (`T2D.2`) "Aşama 3 + eğitim/YOLO" ifadelerini olumsuzlama olmadan engelliyor; yeni metinler buna uygun yazıldı, denetim geçti.


## J. Manuel doğrulama komutları (kullanıcı tarafından çalıştırıldı; yinelemek için)

Aşağıdaki komutlar kullanıcı tarafından MacBook Pro M4 üzerinde çalıştırılmış ve sonuçları bölüm E.4 ve H.1'de kayıtlıdır. Yeniden doğrulamak isterseniz izin penceresini gösterebilen **Terminal.app** (veya VS Code'un entegre terminali) kullanın.

**1) Pencereli canlı kamera** — Q, Esc veya kırmızı kapatma düğmesiyle çıkın; çıkış kodunu görün:

```bash
cd /Users/ebrmac/Desktop/bitirme && .venv/bin/python scripts/run_camera.py --source camera; echo "exit=$?"
```

Beklenen: `Bitiş nedeni: kullanıcı çıkışı (Q/Esc)` veya `pencere kapatıldı`, `exit=0`; başlangıçta `[UYARI] ... kapatma düğmesi etkinleştirilemedi` satırı çıkmamalı.

**2) Kaynak FPS / ölçülen FPS tanısı** — kodu değiştirmez, projeye dosya eklemez, diske görüntü yazmaz; kamerayı ~10 sn okur:

```bash
cd /Users/ebrmac/Desktop/bitirme && .venv/bin/python - 0 <<'EOF'
import sys, time, statistics
import cv2
import numpy as np

arg = sys.argv[1]
src = int(arg) if arg.isdigit() else arg
cap = cv2.VideoCapture(src, cv2.CAP_AVFOUNDATION) if isinstance(src, int) else cv2.VideoCapture(src)
if not cap.isOpened():
    raise SystemExit("Kaynak acilamadi")
print("CAP_PROP_FPS (bildirilen):", cap.get(cv2.CAP_PROP_FPS))
times, dup, prev, n = [], 0, None, 0
t_end = time.monotonic() + 10.0
while time.monotonic() < t_end and n < 300:
    ok, frame = cap.read()
    if not ok:
        break
    times.append(time.monotonic())
    if prev is not None and np.array_equal(prev, frame):
        dup += 1
    prev, n = frame, n + 1
cap.release()
gaps = [b - a for a, b in zip(times, times[1:])]
print("kare:", n, "| sure: %.2f sn" % (times[-1] - times[0]), "| ortalama FPS: %.2f" % ((n - 1) / (times[-1] - times[0])))
print("onceki kareyle birebir ayni kare:", dup)
print("kareler arasi ms -> medyan %.1f | max %.1f" % (statistics.median(gaps) * 1000, max(gaps) * 1000))
EOF
```

**3) Penceresiz kısa deneme ve kendi videonuzla:**

```bash
cd /Users/ebrmac/Desktop/bitirme && .venv/bin/python scripts/run_camera.py --source camera --camera-index 0 --no-display --max-frames 60
cd /Users/ebrmac/Desktop/bitirme && .venv/bin/python scripts/run_camera.py --source video --video-path /yol/video.mp4; echo "exit=$?"
```

**4) Otomatik testler:**

```bash
cd /Users/ebrmac/Desktop/bitirme && .venv/bin/python -m unittest discover -s tests
cd /Users/ebrmac/Desktop/bitirme && ISG_GUI_TESTS=1 .venv/bin/python -m unittest discover -s tests -p "test_camera_window_close.py" -v
```

İkinci komut (isteğe bağlı) **gerçek bir Cocoa penceresi açar**; yalnızca macOS'ta ve grafik oturumu olan bir terminalde çalıştırın.

## K. Aşama 3 tamamlanma durumu (kabul kriterleri) — nihai kontrol

`[x]` = doğrulandı. "Kullanıcı beyanı" = ben yeniden üretemedim; kullanıcının MacBook'ta bildirdiği gerçek sonuçtur. Hiçbir madde varsayılarak işaretlenmemiştir; kanıt sütunu kaynağı gösterir.

| # | Kabul kriteri | Durum | Kanıt |
|---|---|---|---|
| 1 | Mevcut proje yapısı korundu | [x] | Yeni kök klasör yok; Aşama 1–2 dosyaları değişmedi (`git status`, bölüm C) |
| 2 | Kamera modülü oluşturuldu | [x] | `src/isg_vision/camera/` (bölüm C, D) |
| 3 | OpenCV ile kamera erişimi sağlandı | [x] | `camera:0` açıldı *(kullanıcı beyanı, E.1/E.4)* |
| 4 | Canlı görüntü penceresi çalışıyor | [x] | 1280×720 canlı görüntü *(kullanıcı beyanı)*; gerçek pencere + video ile de doğrulandı (F) |
| 5 | Video dosyasından kare okuma | [x] | MP4/MOV/AVI, gerçek dosyalarla (F) |
| 6 | FPS ölçümü | [x] | `FpsMeter` testli (22 test); gerçek kamerada 29,5 FPS *(beyan)*; bağımsız ham ölçüm 29,98 FPS *(beyan, H.1)* |
| 7 | Kaynak FPS ile ölçülen FPS ayrıldı | [x] | Kodda ayrı alanlar; gerçek kamerada 15,0 ↔ ~30 farkı gözlendi ve yorumlandı (H.1) |
| 8 | Çözünürlük ayarları | [x] | Mock ve gerçek video testleri (`test_camera_capture.py`, `test_camera_cli.py`) |
| 9 | Gerçek çözünürlük okunabiliyor | [x] | Gerçek kamerada 1280×720 *(beyan)*; videoda doğrulandı |
| 10 | Q ile çıkış | [x] | Gerçek kamera, `exit=0` *(beyan, E.2)* |
| 11 | Esc ile çıkış | [x] | Gerçek kamera, `exit=0` *(beyan, E.2)* |
| 12 | Pencere kapatma düğmesiyle çıkış | [x] | Hata bulundu/düzeltildi (M); gerçek kamera, `pencere kapatıldı`, `exit=0` *(beyan)*; taklit tıklama + gerçek Cocoa testleri (kendi çalıştırmam) |
| 13 | Ctrl+C ile çıkış | [x] | Gerçek süreç + video, kod 130 (F, M.5). Gerçek kamerada ayrıca denenmedi |
| 14 | Kaynaklar güvenli serbest bırakılıyor | [x] | İdempotent `close`, `finally`; her çıkış yolunda `release_calls == 1` testleri. Gerçek kamerada ışığın sönmesi bildirilmedi (doğrulanmış sayılmadı) |
| 15 | Kamera/video hataları yönetiliyor | [x] | 16 hata testi; izin yokken gerçek hata yolu çalıştı (E.3) |
| 16 | Kareler gelecekteki modüllere aktarılabilir | [x] | `FramePacket` + `FrameSource.frames()` (D) |
| 17 | Komut satırı desteği | [x] | `scripts/run_camera.py`, 27 CLI testi, gerçek süreç denemeleri |
| 18 | Otomatik kamera testleri | [x] | 138 + 32 = 170 kamera testi; toplam 340 (337 çalıştı, 3 isteğe bağlı GUI testi atlandı) |
| 19 | Önceki testler bozulmadı | [x] | Aşama 1–2'nin 170 testi geçiyor; Aşama 2 bütünlük 12/12 (N) |
| 20 | Dokümantasyon güncellendi | [x] | `README.md`, `docs/camera_module.md`, `docs/architecture.md`, `docs/project_plan.md`, `configs/project_stages.yaml`, bu rapor |
| 21 | Ham veri setleri değiştirilmedi | [x] | `git status`: `data/`, `models/`, `reports/dataset_audit/` değişmedi |
| 22 | Model eğitimi / YOLO / takip / yüz yapılmadı | [x] | Kamera paketi `torch/ultralytics/mediapipe/onnxruntime` yüklemez (test) |
| 23 | Aşama 4'e geçilmedi | [x] | Aşama 4 dosyası/kodu yok; `configs/project_stages.yaml`'da 4 = `planned` |
| 24 | Kapsamı ertelenen görevler belgelendi | [x] | B3-1…B3-5, K.2, `docs/project_plan.md`, `configs/project_stages.yaml` |

**Karar: Aşama 3 `COMPLETED`.** Kullanıcı kapanışı, bölüm E.4/H.1'deki gerçek kamera doğrulamaları yapılmış ve ertelenen görevler belgelenmiş olması koşuluyla açıkça onaylamıştır. Kabul kriterlerinin hiçbirinde açık sorun yoktur. **Kapanışın kapsamı** şudur: yeniden bağlanma, kare kaybı, gecikme, ileri FPS doğrulaması ve fiziksel bağlantı kopması **yapılmamıştır**; bunlar tamamlanmış sayılmaz.

**Kapanışta bilinen, kabul edilmiş sınırlar:** (a) `camera:0`'ın fiziksel kamerası ve çıkışta kamera ışığı bildirilmedi; (b) kamera FPS bulgusu tek oturumdur ve "kopya kare yok" tam bağımsız pozlamayı kanıtlamaz (H.1); (c) pencere kapatma düzeltmesi yalnızca macOS arm64'te (M4) ve OpenCV 5.0.0 ile denendi; (d) Ctrl+C gerçek kamerada ayrıca denenmedi.

### K.2 Ertelenen geliştirmeler (açık backlog) — Aşama 3 kapanışında yapılmadı

**Bunlar tamamlanmış sayılmaz.** Kullanıcı görev listesinde yoktu; Aşama 3 kapanışında kullanıcı onayıyla ertelendi. Planlanan aşama, ilgili aşamanın başında kullanıcı kararıyla yeniden gözden geçirilir.

| ID | Madde | Durum | Planlanan değerlendirme aşaması | Gerekçe / not |
|---|---|---|---|---|
| B3-1 | Otomatik yeniden bağlanma (sınırlı sayıda deneme + bekleme) | **Yapılmadı** | **Aşama 12** (canlı sistem dayanıklılığı) | Bugün bağlantı koparsa `SourceDisconnectedError` ile kontrollü durur. Canlı demo sırasında ihtiyaç doğarsa daha erken kapsama alınabilir |
| B3-2 | Kare kaybı ölçümü | **Yapılmadı** | **Aşama 4**, sonra **Aşama 12** | Takip sürekliliği kare kaybına duyarlıdır; ilk ölçüm takipçi akışa bağlanınca, nihai ölçüm uçtan uca testlerde. Altyapı: `frame_index`, `timestamp` |
| B3-3 | Gecikme (yakalama → sonuç) ölçümü | **Yapılmadı** | **Aşama 4**, sonra **Aşama 12** | Anlamlı gecikme model döngüdeyken ölçülür; nihai ölçüm Aşama 12'deki canlı kamera sistem testlerindedir (plan: "FPS, gecikme, alarm/saat"). Altyapı: `timestamp` |
| B3-4 | Gerçek kamera FPS'inin ileri doğrulaması: model döngüdeyken, uzun süreli, bağımsız zaman referanslı, benzersiz kare sayımı | **Yapılmadı** | **Aşama 4**, sonra **Aşama 12** | Aşama 3'te yalnızca modelsiz, tek oturumluk tanı yapıldı (H.1). Gerekirse `FpsMeter`'e benzersiz kare sayımı eklenebilir |
| B3-5 | Fiziksel bağlantı kopması testi (kamera/iPhone bağlantısını kesme) | **Yapılmadı** | **Aşama 12**, B3-1 ile birlikte | Mantık yalnızca sahte kamerayla test edildi. İstenirse mevcut "kontrollü durma" davranışı daha önce de denenebilir |

`docs/project_plan.md` Aşama 3 satırı ve "Değerlendirme türlerinin aşamalara dağılımı" tablosu bu ertelemeyi yansıtacak şekilde güncellendi; makine tarafından okunur sürüm `configs/project_stages.yaml` içindeki `stage3_deferred_backlog` anahtarıdır.

## L. Aşama 4'e hazırlık

Aşama 4 çoklu insan tespiti ve takibi (ByteTrack/BoT-SORT) olacaktır. Hazırlık durumu: kare akışı `FrameSource.frames()` → `FramePacket` olarak hazırdır ve gerçek kamerada 1280×720'de çalışmaktadır *(kullanıcı beyanı)*. `frame_index` ve `timestamp` takipçi için gereken sıra/zaman bilgisini verir; aynı akış kayıtlı videolarla tekrarlanabilir değerlendirme sağlar.

Notlar: (1) kamera için kaynak FPS'e değil **ölçülen FPS ve `timestamp`**'e güvenin (H.1); (2) B3-2, B3-3 ve B3-4 Aşama 4'te model döngüdeyken ilk kez ölçülmek üzere backlog'dadır (K.2); (3) Aşama 4'te kamera kodu yeniden yazılmaz, yalnızca kare akışı tüketilir. **Aşama 4 başlatılmamıştır**; başlatma kararı kullanıcıya aittir.

## M. macOS pencere kapatma hatası (kırmızı düğme): hata, neden, düzeltme, testler

### M.1 Hata

Kullanıcı bildirimi: `ISG Vision - Camera` penceresinin sol üstündeki kırmızı kapatma düğmesine tıklayınca uygulama düzgün kapanmıyor; kamera 1280×720'de çalışıyor (529 kare, ~29,5 FPS). Kullanıcının ekranda gördüğü tam belirtiyi ben gözlemlemedim; aşağıdaki neden **ölçülen** davranışla bu bildirimle uyumludur.

### M.2 Araştırma (kod okuma + gerçek Cocoa penceresinde ölçüm)

Kod: `PreviewWindow.closed_by_user`, `getWindowProperty(WND_PROP_VISIBLE) < 1` veya `cv2.error` durumunda `True` döndürüyor; `run_preview` her karede `show()` sonrası bunu denetliyor. İlk varsayımım "kapanış algılanmıyor"du; bunu **varsaymadan** sınadım. Kırmızı düğmeye tıklamayı, `NSWindow`'un standart kapatma düğmesine `performClick:` göndererek (gerçek düğme eylemi) ve `ctypes` ile Objective-C çalışma zamanını kullanarak gerçek pencerede taklit eden tanı betikleri yazdım (proje dışı, scratchpad). İzin/erişilebilirlik ayarı gerekmedi.

| Deney (gerçek OpenCV 5.0.0 Cocoa penceresi) | Sonuç |
|---|---|
| A: `WINDOW_AUTOSIZE` penceresinde kırmızı düğmeye tıklama, 64 kare boyunca izleme | Pencere **kapanmadı**: `VISIBLE=1.0`, `NSWindow.isVisible=True`. Tıklama anında düğme **`isEnabled=False`** |
| B: `AUTOSIZE`, `NORMAL`, `GUI_NORMAL` pencereleri | Hepsinde `styleMask=0xd` (Titled\|Miniaturizable\|Resizable), **`Closable` (0x2) yok**; kapatma düğmesi `isEnabled=False`, küçültme ve yakınlaştırma düğmeleri etkin; tıklama etkisiz |
| C: `styleMask`'a `Closable` eklenince (0xd → 0xf) | Kapatma düğmesi `isEnabled=True`; tıklama pencereyi kapattı ve `getWindowProperty(WND_PROP_VISIBLE)` **aynı adımda `0.0`** döndü; `destroyWindow` hatasız; `releasedWhenClosed` varsayılan ve `NO` iki durumda da, 3 tekrarda çökme yok |
| `NSWindow`, `namedWindow` sonrasında `imshow`'dan önce de var mı? | Evet: `NSApp.windows` içinde var; düzeltme `open()` içinde uygulanabilir |

### M.3 Kesin neden

OpenCV 5.0.0'ın Cocoa arka ucu pencereyi `Closable` stil biti olmadan oluşturuyor; bu yüzden **kırmızı kapatma düğmesi devre dışı** ve tıklamak hiçbir şey yapmıyor. Sorun "kapanmanın algılanmaması" **değildi**: pencere hiç kapanmadığından `getWindowProperty` doğru olarak `1.0` döndürüyordu; `Closable` eklenince aynı algılama mantığı (`closed_by_user`) kapanışı hemen yakaladı. İlk sürümdeki gerçek-pencere tanı denemem yalnızca programatik `destroyWindow` ile kapanışı denediği için düğmenin devre dışı olduğunu yakalayamamıştı. (OpenCV'nin kaynak kodunu okumadım; neden, ölçülen pencere özelliklerine dayanır.)

### M.4 Yapılan düzeltme

* **Yeni** `src/isg_vision/camera/macos_window.py`: `enable_close_button(title)`, pencere oluşturulduktan sonra `NSWindow.setStyleMask:` ile `Closable` bitini ekler ve sonucu yeniden okuyarak doğrular. `ctypes` + Objective-C çalışma zamanı kullanır (`pyobjc` yok). macOS dışında, ana iş parçacığı dışında, pencere bulunamazsa veya herhangi bir hata olursa `False` döndürür ve **asla istisna fırlatmaz**; `NSApplication` yoksa yeni oluşturmaz. Import sırasında hiçbir kütüphane yüklenmez (testle doğrulandı).
* `display.py`: `PreviewWindow.open()` pencere oluşturulunca (yalnızca macOS'ta) `enable_close_button` çağırır; `close_button_enabled` özelliği (`None`/`True`/`False`) eklendi. `closed_by_user`, `show`, `close` mantığı **değişmedi**.
* `cli.py`: macOS'ta düğme etkinleştirilemezse `[UYARI]` yazar (pencere yine Q/Esc/Ctrl+C ile kapanır); başlangıç ipucuna "pencerenin kapatma düğmesine tıklayın" eklendi.
* `preview.py`, `capture.py` ve kamera mimarisi **değişmedi**. Kaynak/pencere serbest bırakma yolu (`finally`, `with`) aynı.

### M.5 Test sonuçları (gerçekten çalıştırıldı)

| Kontrol | Sonuç |
|---|---|
| Önceki 308 test | **Aynen geçti** (yalnızca `test_no_side_effects.py` ve `test_camera_cli.py` içindeki modül listelerine `macos_window` eklendi) |
| Yeni `tests/test_camera_window_close.py` | **32 test**: bit mantığı (1), `enable_close_button` sahte çalışma zamanıyla (11: ekleme, zaten etkin, yalnızca adlı pencere, pencere yok, etkisiz set, macOS dışı, ana iş parçacığı dışı, çalışma zamanı hataları, kütüphane yüklenemedi, varsayılan platform, import yan etkisizliği), `PreviewWindow` düğme davranışı (4), pencere kapanış algılama (8: `getWindowProperty` `0.0`, `-1.0`, `cv2.error`, `destroyWindow` hatası, ilk karede kapanma, görünür pencere devam eder, Q/q/Esc korunur, video sonu korunur; her birinde kaynak serbest bırakıldı), CLI (5: kapanışta çıkış kodu 0 + "pencere kapatıldı" + kamera serbest, uyarı koşulları, ipucu), gerçek Cocoa (3, isteğe bağlı) |
| Tam paket (`unittest discover -s tests` ve kökten) | **340 test: 337 çalıştı, 3 atlandı (gerçek pencere), OK** |
| Gerçek Cocoa testleri (`ISG_GUI_TESTS=1`, Terminal panelinde) | **32/32 OK, atlanan yok**: ham OpenCV penceresinde `Closable` yok ve düğme devre dışı (kök neden testi); düzeltilmiş `PreviewWindow`'da düğme etkin, taklit tıklama pencereyi kapatıyor ve `closed_by_user` algılıyor; tam döngü `window_closed` ile bitiyor ve kamera serbest bırakılıyor |
| Gerçek CLI + gerçek pencere + gerçek video (taklit tıklama, proje dışı betik) | 1,5 sn'de tıklama: `Bitiş nedeni: pencere kapatıldı`, 35 kare, **çıkış kodu 0**, toplam 1,56 sn. 0,3 sn'de tıklama: 6 kare, **kod 0**. Tıklamasız kontrol: video sonuna kadar (100 kare, 4,0 sn, 24,9 FPS), **kod 0** |
| Ctrl+C (gerçek pencere + video) | Kontrol çalıştırması sırasında gelen Ctrl+C: `Ctrl+C ile durduruldu`, **kod 130** (daha önce de doğrulanmıştı) |
| Aşama 2 bütünlük denetimi | **12/12 geçti** (T2D.2 dahil) |
| `check_environment.py` | BAŞARILI (0 uyarı, 0 kritik hata) |
| `git status` | Yalnızca kamera modülü/testleri ve belgeler; `data/`, `models/`, `reports/dataset_audit/`, Aşama 1–2 kodu ve `configs/` değişmedi |

### M.6 Doğrulama durumu ve sınırlar (nihai kapanış sonrası)

* **Gerçek fare tıklaması:** Düzeltme sırasında ben tıklamayı `NSButton.performClick:` ile taklit edebildim; fareyle tıklayamadım. Kullanıcı sonradan gerçek kamerada kırmızı düğmeyle çıkışı doğruladı: `Bitiş nedeni: pencere kapatıldı`, `exit=0` (E.4). **Bu madde kapandı.**
* **Q/Esc:** Bu düzeltmede gerçek tuş basışını ben denemedim; kullanıcı Q ve Esc ile çıkışı ayrı ayrı doğruladı (`exit=0`, E.2).
* **Kırılganlık:** Düzeltme OpenCV'nin iç pencere stilini Objective-C ile değiştirir. Başka bir OpenCV/macOS sürümünde bu çalışmazsa sessizce devre dışı kalır, CLI uyarı yazar ve Q/Esc/Ctrl+C çıkışları etkilenmez.
* **x86_64 macOS:** Yalnızca arm64'te (M4) denendi.

## N. Nihai kapanış doğrulaması (bu kapanışta gerçekten çalıştırıldı)

Aşağıdaki sonuçlar bu kapanışta, belge/yapılandırma değişikliklerinden **sonra** yeniden çalıştırılarak elde edilmiştir; önceki sonuçlar varsayılmamıştır.

| Kontrol | Komut | Gerçek sonuç |
|---|---|---|
| Tüm otomatik testler | `.venv/bin/python -m unittest discover -s tests` | **340 test tanımlı: 337 başarılı, 3 atlandı (isteğe bağlı gerçek-pencere testleri), başarısız 0, hata 0** — `OK (skipped=3)` |
| Aynı, kök dizinden | `.venv/bin/python -m unittest discover` | **340 test, `OK (skipped=3)`** |
| Gerçek Cocoa pencere testleri (isteğe bağlı) | `ISG_GUI_TESTS=1 ... -p test_camera_window_close.py` (Terminal panelinde) | **32/32 OK, atlanan 0** (3 gerçek-pencere testi dahil) |
| Aşama 2 bütünlük denetimi | `scripts/run_stage2c_checks.py` | **12/12 geçti** (T2D.2 belge/aşama numarası tutarlılığı dahil; Aşama 3 = `completed` yapılandırmasıyla) |
| Ortam kontrolü | `scripts/check_environment.py` | BAŞARILI (0 uyarı, 0 kritik hata) |
| `configs/project_stages.yaml` | YAML okuma | 1, 2, 3 = `completed`; 4–12 = `planned`; `stage3_deferred_backlog` B3-1…B3-5 |
| Değişen dosyalar (`git status`) | — | Yalnızca kamera modülü/testleri ve belgeler/yapılandırma; `data/`, `models/`, `reports/dataset_audit/`, Aşama 1–2 kodu **değişmedi**; commit/push yapılmadı |

Önceki beklenen durum (340 tanımlı, 337 başarılı, 3 atlandı; 12/12) bu çalıştırmada **gerçek sonuçla aynı çıktı**.

**Bu kapanışta yapılmayanlar:** yeniden bağlanma, kare kaybı ölçümü, gecikme ölçümü, ileri FPS doğrulaması, fiziksel bağlantı kopması testi (K.2); gerçek kamerada ışığın sönmesi ve fiziksel kamera kimliğinin doğrulanması (bildirilmedi); Aşama 4.
