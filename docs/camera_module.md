# Kamera Modülü (Aşama 3 — Canlı kamera altyapısı)

> **Durum: `COMPLETED` (2026-11-11, kullanıcı onayıyla).** Otomatik testler geçmektedir; gerçek kamera doğrulamaları (kamera açılışı, 1280×720 görüntü,
> Q, Esc ve kırmızı pencere kapatma düğmesiyle çıkış) kullanıcı tarafından MacBook Pro M4 üzerinde yapılmıştır (bkz. [Gerçek kamera doğrulaması](#9-gerçek-kamera-doğrulaması)).
> Yeniden bağlanma, kare kaybı, gecikme ölçümü, ileri FPS doğrulaması ve fiziksel bağlantı kopması testi **ertelenmiştir, yapılmamıştır** (bkz. [Bilinen sınırlamalar](#13-bilinen-sınırlamalar)).
> Aşama 3'te YOLO eğitimi, insan tespiti, yüz tespiti/tanıma, takip ve veri birleştirme **yapılmaz**; modül hiçbir yapay zekâ
> modeline bağımlı değildir. Aşama sırası: [project_plan.md](project_plan.md).

## 1. Amaç

MacBook kamerasından veya kayıtlı bir video dosyasından kare okuyan, kareleri düzenli bir veri yapısıyla (`FramePacket`)
sunan, FPS ölçen ve durum bilgisini OpenCV penceresinde gösteren modüler bir altyapı. Sonraki aşamalardaki modüller
(insan tespiti ve takip, yüz tespiti, kimlik tanıma, KKD analizi) **aynı kare akışını** kullanır; kamera kodunun yeniden
yazılması gerekmez.

## 2. Dosya yapısı

```
src/isg_vision/camera/
├── __init__.py   # dışa açılan adlar: FrameSource, SourceConfig, FramePacket, hata sınıfları
├── errors.py     # hata sınıfları (yapılandırma / kaynak / okuma / bağlantı kopması / pencere)
├── frame.py      # FramePacket: kare + metadata
├── config.py     # SourceConfig: doğrulanmış kaynak yapılandırması (cv2 import etmez)
├── capture.py    # FrameSource: cv2.VideoCapture üzerinde kamera + video için ortak okuma
├── fps.py        # FpsMeter: kayan pencere FPS ölçümü (bağımsız, OpenCV'ye bağımlı değil)
├── macos_window.py  # macOS: OpenCV penceresinin kırmızı kapatma düğmesini etkinleştirir (ctypes + Objective-C)
├── display.py    # overlay çizimi + PreviewWindow (OpenCV penceresi)
├── preview.py    # run_preview: kaynak -> FPS -> overlay -> pencere döngüsü
└── cli.py        # komut satırı arayüzü (main)
scripts/run_camera.py          # cli.main() için ince başlatıcı
tests/test_camera_*.py         # otomatik testler (+ tests/camera_test_support.py yardımcıları)
```

## 3. Mimari

```
                 ┌──────────────────────────────┐
 kamera / video ─►  FrameSource (capture.py)     │──► FramePacket akışı ──► [Aşama 4+: tespit, takip, yüz, KKD ...]
                 │  open / read / frames / close │            │
                 └──────────────────────────────┘            ▼
                                                    run_preview (preview.py)
                                         FpsMeter ◄──┤   │
                                       (fps.py)      │   ▼ draw_overlay (kopya üzerine çizer)
                                                     └─► PreviewWindow ──► Q / Esc / pencere kapatma
```

* `FrameSource` yalnızca kare üretir; ekran, FPS veya yapay zekâ bilmez.
* `FpsMeter` zaman damgası alan bağımsız bir sınıftır; kamerasız da kullanılır/test edilir.
* `draw_overlay` girdi karesini değiştirmez, kopyası üzerine çizer.
* **Hiçbir modül import edildiğinde kamera açmaz veya pencere oluşturmaz.** Kaynak yalnızca `FrameSource.open()` (veya `with`)
  ile açılır; pencere ilk `show()`/`open()` ile oluşur. (Doğrulayan test: `tests/test_camera_cli.py::ImportSideEffectsTest`,
  `tests/test_no_side_effects.py`.)
* Kamera paketi `torch`, `ultralytics`, `mediapipe` gibi kütüphaneleri yüklemez (testle doğrulanır).

### FramePacket

| Alan | Anlam |
|---|---|
| `frame` | OpenCV BGR görüntüsü (`uint8`, `(yükseklik, genişlik, 3)`) |
| `frame_index` | Başarıyla okunan karenin 0'dan başlayan, **ardışık** sıra numarası (başarısız okumalar numara tüketmez) |
| `timestamp` | Okuma anı, `time.monotonic()` saniyesi (süre/gecikme/FPS için) |
| `wall_time` | Okuma anı, Unix zamanı (ileride olay kaydında tarih/saat için) |
| `source` | `"camera:0"` veya `"video:/yol/klip.mp4"` |
| `source_kind` | `"camera"` / `"video"` |
| `width`, `height` | **Gerçek** kare boyutu (özellik; `frame.shape`'ten) |

`frame` kopyalanmadan paylaşılır. Üzerine çizim yapacak modüller önce `packet.frame.copy()` almalıdır.

### Hata sınıfları (`errors.py`)

| Sınıf | Ne zaman |
|---|---|
| `SourceConfigError` (`ValueError`) | Geçersiz indeks, çözünürlük, yol biçimi, çelişkili ayar — kaynak açılmadan önce |
| `CameraOpenError` | Kamera açılamadı (izin yok / başka uygulama kullanıyor / indeks yok) |
| `VideoFileError` | Dosya yok, klasör, boş, okunamıyor, açılamıyor veya hiç kare içermiyor |
| `FrameReadError` | Kamera açıldı ama ardışık okumalar başarısız (ilk kare hiç gelmedi) |
| `SourceDisconnectedError` (`FrameReadError`) | Akış başladıktan sonra bağlantı koptu |
| `DisplayError` | OpenCV penceresi oluşturulamadı / kullanılamadı |

`FrameSource` kamera okumasında geçici başarısızlıkları sınırlı sayıda yeniden dener (`max_read_failures=30`, aralık
`0.05 sn`); art arda hata sayısı aşılırsa hata fırlatır — **sonsuz döngü oluşmaz**. Başarılı okuma sayacı sıfırlar.

## 4. Komutlar

Tümü proje kökünde, sanal ortamla çalıştırılır:

```bash
# Varsayılan kamera (indeks 0, 1280x720 istenir)
.venv/bin/python scripts/run_camera.py --source camera

# Kamera indeksi seçme (iPhone Süreklilik Kamerası da bir indeks alabilir; gerekirse 1, 2 ... deneyin)
.venv/bin/python scripts/run_camera.py --source camera --camera-index 1

# Çözünürlük belirleme (--width ve --height birlikte verilir)
.venv/bin/python scripts/run_camera.py --source camera --width 640 --height 480

# Video dosyası (MP4, MOV, AVI — codec desteklenirse)
.venv/bin/python scripts/run_camera.py --source video --video-path /yol/video.mp4

# Pencere açmadan 100 kare okuyup özet yazdırma (otomasyon / SSH için)
.venv/bin/python scripts/run_camera.py --source camera --no-display --max-frames 100

.venv/bin/python scripts/run_camera.py --help
```

| Argüman | Açıklama |
|---|---|
| `--source {camera,video}` | Varsayılan `camera` |
| `--camera-index N` | Varsayılan 0; yalnızca kamera |
| `--width PX --height PX` | İstenen çözünürlük (varsayılan 1280x720); yalnızca kamera, birlikte verilmeli |
| `--video-path DOSYA` | `--source video` için zorunlu |
| `--backend {auto,avfoundation,any}` | Varsayılan `auto` (macOS'ta AVFoundation); yalnızca kamera |
| `--no-display` | Pencere açmaz; video için oynatma beklemesi yapılmaz |
| `--max-frames N` | N kare işlenince durur |

**Çıkış kodları:** `0` normal bitiş (Q/Esc, pencere kapatma, video sonu, `--max-frames`) · `1` kamera/video/pencere hatası ·
`2` hatalı argüman · `130` `Ctrl+C`.

**Çıkış tuşları:** pencerede `Q` veya `Esc`; pencereyi kapatma düğmesi; terminalde `Ctrl+C`. Her durumda kamera ve pencere
serbest bırakılır (`finally`).

## 5. Pencerede gösterilenler

Üst şerit: `SOURCE` (CAMERA 0 / VIDEO dosya_adı), `FPS` (ölçülen), `RESOLUTION` (gerçek kare boyutu), `SOURCE FPS`
(kaynağın bildirdiği), `STATUS` (`LIVE` veya `PLAYING 12/300`), gerekirse `NOTE` (istenen çözünürlük uygulanamadıysa).
Alt şerit: `Press Q or ESC to exit`. Metinler ASCII'dir (OpenCV yazı tipleri `ç ş ğ ı ö ü` çizemez). İnsan kutusu, isim, baret/yelek
gibi yapay zekâ çıktıları bu aşamada **gösterilmez**.

## 6. FPS: kaynak FPS ve ölçülen FPS

* **Kaynak FPS** — kameranın/video dosyasının *bildirdiği* kare hızı (`FrameSource.source_fps`; bilinmiyorsa `None`).
* **Ölçülen FPS** — uygulamanın gerçekten saniyede işlediği kare sayısı (`FpsMeter`). `time.monotonic` ile, son 30 karenin kayan
  penceresi üzerinden hesaplanır (tek kare aralığına göre dalgalanmaz). Oturum özeti ortalama FPS'i (`average_fps`) da yazar.

**Kameralarda kaynak FPS'e güvenmeyin.** Gerçek MacBook kamerasında sürücü `CAP_PROP_FPS = 15,0` bildirirken ölçülen hız ~30 FPS çıkmıştır
(tanı oturumu: 300 kare / 9,97 sn = 29,98 FPS, medyan kare aralığı 33,3 ms, en büyük aralık 41,6 ms, birebir aynı ardışık kare 0; ayrıca uygulamanın kendi
oturumunda 529 kare / 17,9 sn = 29,5 FPS). Bu nedenle kamera için **kaynak FPS yalnızca bilgidir; performans/zaman hesaplarında ölçülen FPS ve
`FramePacket.timestamp` kullanılır.** Video dosyalarında kaynak FPS dosyadan gelir ve oynatma hızını belirler. Ölçülen FPS hesabında (`FpsMeter`) hata
görülmemiştir (bağımsız ham okuma 29,98 FPS ile uygulama oturumundaki 29,5 FPS tutarlıdır; iki ölçüm ayrı oturumlardadır). Sınır: "birebir aynı kare = 0"
yalnızca tam kopya kareleri eler; her karenin bağımsız bir pozlama olduğunu kesin olarak kanıtlamaz. Bu bulgu tek bir tanı oturumuna dayanır; ileri doğrulama
ertelenmiştir (B3-4).

Video oynatma pencereliyken kaynağın FPS değerine göre **hızlandırmadan** yapılır (kaynak FPS bilinmiyorsa 30). İşlem yavaşsa kare atlanmaz,
oynatma yavaşlar ve ölçülen FPS bunu olduğu gibi gösterir. `--no-display` modunda bekleme yoktur; ölçülen FPS yalnızca okuma hızıdır.

## 7. Çözünürlük

`SourceConfig.camera(width=1280, height=720)` bir **istektir**. Açılışta istenen değer ayarlanır, sonra gerçekte uygulanan değer
okunur; ilk kareden sonra gerçek değer kare boyutundan (`FramePacket.width/height`) alınır. İstenen ile gerçek farklıysa:
konsolda `[UYARI]`, pencerede `NOTE:` satırı gösterilir. `width=None, height=None` cihazın varsayılanını kullanır. Geçerli aralık
16–7680 piksel. Video dosyaları kendi çözünürlüğünde okunur (`--width/--height` video ile kullanılamaz).

## 8. macOS kamera izinleri

macOS, kameraya erişen uygulamanın (programı çalıştıran **uygulama**: Terminal, iTerm, VS Code, Claude masaüstü uygulaması vb.)
izin almasını ister.

1. Programı ilk kez çalıştırdığınızda macOS "… kameraya erişmek istiyor" penceresini gösterirse **Tamam/İzin ver** deyin.
2. Pencere çıkmadıysa veya izni reddettiyseniz: **Sistem Ayarları → Gizlilik ve Güvenlik → Kamera** bölümünde ilgili uygulamayı açın,
   sonra o uygulamayı tamamen kapatıp yeniden başlatın.
3. Belirti: `OpenCV: not authorized to capture video (status 0)` ve `[HATA] Kamera 0 açılamadı ...`.
4. Bu proje izin ayarlarını **değiştirmez ve aşmaya çalışmaz**; izni yalnızca siz verebilirsiniz. (İsterseniz bir uygulamanın kayıtlı
   kamera kararını sıfırlamak için `tccutil reset Camera` komutunu kendiniz çalıştırabilirsiniz; bu bir sistem ayarı değişikliğidir.)

Not: Bu geliştirme sırasında Claude masaüstü uygulamasının Terminal panelinde kamera izni isteği görünmedi ve erişim reddedildi
(`status 0`). Kamera testini izin penceresi gösterebilen **Terminal.app** veya **VS Code'un entegre terminali** üzerinden çalıştırmanız önerilir.

## 9. Gerçek kamera doğrulaması

Aşağıdaki sonuçlar **kullanıcı tarafından** MacBook Pro M4 üzerinde, izinli bir terminalde yapılmış ve bildirilmiştir (Claude'un ortamında kamera izni yoktur; bu rakamlar yeniden üretilmemiştir).

| Doğrulama | Sonuç (kullanıcı beyanı) |
|---|---|
| Kamera `camera:0` ile açıldı, canlı görüntü gösterildi | Başarılı; gerçek çözünürlük 1280×720 |
| Uygulama oturumu | 529 kare, 17,9 sn, ölçülen ortalama 29,5 FPS, kaynak FPS (bildirilen) 15,0 |
| Q ile çıkış | Başarılı, `exit=0` |
| Esc ile çıkış | Başarılı, `exit=0` |
| Kırmızı pencere kapatma düğmesi | Başarılı: `Bitiş nedeni: pencere kapatıldı`, `exit=0` |
| FPS tanı testi (ham `cv2.VideoCapture`, 10 sn) | `CAP_PROP_FPS` 15,0; 300 kare / 9,97 sn = 29,98 FPS; birebir aynı ardışık kare 0; kare aralığı medyan 33,3 ms, en büyük 41,6 ms |

Bildirimde yer almadığı için **doğrulanmış sayılmayanlar:** `camera:0`'ın hangi fiziksel kamera olduğu (yerleşik / iPhone Süreklilik Kamerası), çıkışta kamera ışığının sönmesi,
istenen–gerçek çözünürlük uyarısının çıkmaması.

### 9.1 macOS: kırmızı pencere kapatma düğmesi çalışmıyordu (düzeltildi ve kullanıcı tarafından doğrulandı)

* **Hata:** `ISG Vision - Camera` penceresinin kırmızı düğmesine tıklamak pencereyi kapatmıyordu.
* **Kök neden (gerçek Cocoa penceresinde ölçüldü, OpenCV 5.0.0 / macOS arm64):** OpenCV pencereyi `NSWindow.styleMask = 0xd` (Titled | Miniaturizable |
  Resizable) ile oluşturur; `Closable` biti (`0x2`) **yok**. Bu yüzden kırmızı düğme devre dışıdır (`NSButton.isEnabled = NO`) ve tıklama hiçbir şey
  yapmaz. `WINDOW_AUTOSIZE`, `WINDOW_NORMAL` ve `WINDOW_GUI_NORMAL` için aynı sonuç alındı. Sorun, kapanmanın *algılanmaması* değildi:
  pencere hiç kapanmadığı için `cv2.getWindowProperty(WND_PROP_VISIBLE)` doğru olarak `1.0` döndürüyordu.
* **Düzeltme:** `macos_window.enable_close_button` pencere oluşturulduktan hemen sonra `Closable` bitini ekler (`PreviewWindow.open()` çağırır).
  Bundan sonra düğme etkindir; tıklayınca pencere kapanır, `getWindowProperty` `0.0` döndürür ve mevcut `closed_by_user` mantığı döngüyü sonlandırır:
  kamera/video serbest bırakılır, pencere yok edilir, çıkış kodu 0 ve `Bitiş nedeni: pencere kapatıldı`.
* **Güvenlik/yedek davranış:** macOS dışında, ana iş parçacığı dışında veya herhangi bir hata olursa hiçbir şey yapılmaz (`False`); bu durumda
  CLI `[UYARI]` yazar ve pencere Q/Esc/Ctrl+C ile kapatılabilir. `pyobjc` gerekmez.
* **Doğrulama:** gerçek Cocoa penceresinde tıklama `NSButton.performClick:` ile taklit edilerek otomatik doğrulandı; kullanıcı düzeltmeden sonra gerçek fare tıklamasıyla da
  doğruladı (yukarıdaki tablo). Ayrıntı: `reports/camera/STAGE_3_REPORT.md`, bölüm M.

Doğrulamayı yinelemek için Terminal.app'te:

```bash
cd /Users/ebrmac/Desktop/bitirme
.venv/bin/python scripts/run_camera.py --source camera --camera-index 0 --no-display --max-frames 60   # penceresiz; izin ve FPS
.venv/bin/python scripts/run_camera.py --source camera; echo "exit=$?"                                  # pencere; Q / Esc / kırmızı düğme ile çıkın
```

Kontrol listesi: kamera açılıyor mu · görüntü geliyor mu · `FPS` ve `RESOLUTION` gösteriliyor mu · gerçek çözünürlük konsolda doğru raporlanıyor mu ·
`Q`/`Esc`/pencere kapatma ile çıkılıyor mu · `Kaynak açıldı` / `Bitiş nedeni` satırları ve çıkış kodu 0 mı · çıkıştan sonra kamera ışığı sönüyor mu.

## 10. Yaygın hatalar ve çözümleri

| Belirti | Olası neden | Çözüm |
|---|---|---|
| `Kamera 0 açılamadı ... izni` | macOS kamera izni yok | Bölüm 8 |
| Aynı hata, izin var | Kamerayı başka uygulama kullanıyor (FaceTime, Zoom, Photo Booth, tarayıcı) | O uygulamayı kapatın |
| Aynı hata | Bu indekste kamera yok / indeks farklı | `--camera-index 1`, `2` deneyin; iPhone Süreklilik Kamerası indeks alabilir |
| `Kamera açıldı ancak hiç kare okunamadı` | İzin yarım kalmış veya cihaz hazır değil | Uygulamayı yeniden başlatın, kamerayı başka uygulama kullanıyor mu bakın |
| `Kamera bağlantısı kesildi` | Kamera/iPhone bağlantısı koptu | Bağlantıyı kontrol edip yeniden çalıştırın (otomatik yeniden bağlanma yoktur) |
| `[UYARI] İstenen çözünürlük ... uygulanamadı` | Cihaz bu çözünürlüğü desteklemiyor | Uygulanan değeri kullanın veya `--width/--height` değiştirin |
| `Video dosyası bulunamadı / bir dosya değil / boş` | Yol hatalı | Yolu kontrol edin |
| `Video dosyası açılamadı` | Bozuk dosya veya desteklenmeyen codec | Dosyayı MP4 (H.264) olarak yeniden dışa aktarın |
| `[UYARI] Video beklenenden erken bitti` | Dosya kesik/bozuk (kare sayısı bildirilenden çok az) | Dosyayı kontrol edin; çökme olmaz, kontrollü biter |
| Kırmızı kapatma düğmesine tıklayınca pencere kapanmıyor (macOS) | OpenCV pencereyi kapatma düğmesi devre dışı oluşturur (bölüm 9.1) | Güncel sürümde otomatik düzeltilir; `[UYARI] ... kapatma düğmesi etkinleştirilemedi` görürseniz Q/Esc/Ctrl+C kullanın |
| `OpenCV görüntü penceresi oluşturulamadı` | Grafik oturumu yok (ör. SSH) | Terminal.app/VS Code'da çalıştırın veya `--no-display` kullanın |

## 11. Otomatik testler

```bash
.venv/bin/python -m unittest discover -s tests            # tüm testler (Aşama 1, 2 ve 3)
.venv/bin/python -m unittest tests.test_camera_capture tests.test_camera_fps \
    tests.test_camera_display tests.test_camera_errors tests.test_camera_cli \
    tests.test_camera_window_close                                              # yalnızca kamera testleri
ISG_GUI_TESTS=1 .venv/bin/python -m unittest discover -s tests -p "test_camera_window_close.py" -v
# ^ isteğe bağlı: GERÇEK Cocoa penceresi açar (macOS + grafik oturumu gerekir); varsayılan çalıştırmada atlanır
```

Testler gerçek kamera açmaz, izin istemez, ağ kullanmaz, model indirmez, `data/` ve `models/` altına dokunmaz. `cv2.VideoCapture`
yerine sahte kaynak (`tests/camera_test_support.py`), video testlerinde geçici sentetik MP4/MOV/AVI dosyaları kullanılır; OpenCV GUI
çağrıları taklit edilir (gerçek pencere açılmaz).

## 12. Gelecekteki modüllerin bağlanması

Sonraki aşamalarda (ör. Aşama 4'te insan tespiti ve takip) kamera kodu değiştirilmez; yalnızca kare akışı tüketilir (sözde kod):

```python
from isg_vision.camera import FrameSource, SourceConfig

with FrameSource(SourceConfig.camera(0)) as source:        # veya SourceConfig.video("klip.mp4")
    for packet in source.frames():                         # FramePacket
        tracks = my_module.process(packet.frame, packet.timestamp)   # Aşama 4+: eklenecek modül
        canvas = packet.frame.copy()                       # çizimden önce kopya
        ...                                                # kutu/isim çizimi, penceresi: PreviewWindow
```

* Her modül kareyi `FramePacket` olarak alır; `frame_index` ve `timestamp` ile takip/olay kaydı eşlenir.
* Aynı akış canlı kamera ve kayıtlı video için aynıdır; kaydedilmiş videolarla tekrarlanabilir değerlendirme yapılabilir.
* Overlay'e yeni satırlar eklemek için `display.OverlayInfo`/`draw_overlay` genişletilir; pencere ve döngü yeniden yazılmaz.

## 13. Bilinen sınırlamalar

* OpenCV, "izin yok", "kamera yok" ve "başka uygulama kullanıyor" durumlarını ayırt edemez; hata mesajı olası nedenleri listeler.
* Kamera bağlantısı koparsa uygulama **durur** (`SourceDisconnectedError`); otomatik yeniden bağlanma **uygulanmamıştır** (B3-1).
* Kare kaybı ve gecikme için `FramePacket.timestamp` ve `frame_index` altyapısı vardır, ancak ayrı bir ölçüm aracı **yazılmamıştır** (B3-2, B3-3).
* `FpsMeter` tamamlanan `read()` çağrılarını sayar; kaç **benzersiz** kare geldiğini ayırt etmez. Gerçek kamerada kaynağın bildirdiği FPS (15,0) ile ölçülen FPS (~30)
  farklı çıkmıştır; tanı testi (birebir aynı kare 0, medyan aralık 33,3 ms) ~30 FPS okuma hızını güçlü biçimde destekler, ancak tek oturumdur ve "aynı kare yok" tam bağımsız
  pozlamayı kanıtlamaz (bölüm 6, B3-4). Kamera için kaynak FPS performans ölçüsü olarak kullanılmaz.
* Video dosyasında "beklenenden erken bitti" tespiti, OpenCV'nin bildirdiği kare sayısına (yaklaşık olabilir) ve %2 toleransa dayanır.
* Pencerenin kullanıcı tarafından kapatılması Cocoa'da `getWindowProperty` ile algılanır; kırmızı düğme OpenCV'de varsayılan olarak devre dışı olduğundan
  `macos_window` ile etkinleştirilir (bölüm 9.1). Düzeltme gerçek pencerede taklit tıklamayla, kullanıcı tarafından gerçek fare tıklamasıyla doğrulandı. Yalnızca macOS arm64'te (M4) denendi.
* `macos_window` OpenCV'nin iç pencere stilini Objective-C çalışma zamanıyla değiştirir; OpenCV/macOS sürümleri arasında davranış değişirse hata durumunda
  sessizce devre dışı kalır (CLI uyarır).
* Görüntü ve yüzler varsayılan olarak diske kaydedilmez; video kaydetme özelliği yoktur.

### Ertelenen geliştirme maddeleri (backlog) — Aşama 3 kapanışında yapılmadı

Bu maddeler **tamamlanmış sayılmaz**. Aşama 3, kullanıcı onayıyla bunlar yapılmadan `COMPLETED` olarak kapatılmıştır. Planlanan değerlendirme aşamaları
kullanıcı kararına tabidir ve ilgili aşamanın başında gözden geçirilir.

| ID | Madde | Planlanan aşama | Not |
|---|---|---|---|
| B3-1 | Otomatik yeniden bağlanma | 12 | Bugün bağlantı koparsa kontrollü durur |
| B3-2 | Kare kaybı ölçümü | 4, sonra 12 | `frame_index` / `timestamp` altyapısı hazır |
| B3-3 | Gecikme ölçümü | 4, sonra 12 | Model döngüdeyken anlamlıdır |
| B3-4 | Gerçek kamera FPS'inin ileri doğrulaması | 4, sonra 12 | Uzun süreli, bağımsız zaman referanslı, model döngüdeyken |
| B3-5 | Fiziksel bağlantı kopması testi | 12 | B3-1 ile birlikte |

Tablonun tam hâli: `docs/project_plan.md` ("Aşama 3'te ertelenen maddeler") ve `reports/camera/STAGE_3_REPORT.md` (bölüm K.2).
