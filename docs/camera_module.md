# Kamera Modülü (Aşama 3 — Canlı kamera altyapısı)

> **Durum: `IN_PROGRESS`.** Otomatik testler geçmektedir. Gerçek kamera donanım doğrulaması macOS kamera izni gerektirdiği
> için kullanıcı tarafından yapılmalıdır (`READY_FOR_MANUAL_VERIFICATION`; bkz. [Gerçek kamera doğrulaması](#9-gerçek-kamera-doğrulaması)).
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

Gerçek kamera/Q-Esc doğrulaması manuel yapılır. Terminal.app'te:

```bash
cd /Users/ebrmac/Desktop/bitirme
.venv/bin/python scripts/run_camera.py --source camera --camera-index 0 --no-display --max-frames 60   # penceresiz; izin ve FPS
.venv/bin/python scripts/run_camera.py --source camera                                                  # pencere; Q / Esc ile çıkın
```

Kontrol listesi: kamera açılıyor mu · görüntü geliyor mu · `FPS` ve `RESOLUTION` gösteriliyor mu · gerçek çözünürlük konsolda doğru
raporlanıyor mu · `Q`/`Esc`/pencere kapatma ile çıkılıyor mu · çıktıdaki `Kaynak açıldı` / `Bitiş nedeni` satırları ve çıkış kodu 0 mı ·
çıkıştan sonra kamera ışığı sönüyor mu (kaynak serbest bırakıldı).

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
| `OpenCV görüntü penceresi oluşturulamadı` | Grafik oturumu yok (ör. SSH) | Terminal.app/VS Code'da çalıştırın veya `--no-display` kullanın |

## 11. Otomatik testler

```bash
.venv/bin/python -m unittest discover -s tests            # tüm testler (Aşama 1, 2 ve 3)
.venv/bin/python -m unittest tests.test_camera_capture tests.test_camera_fps \
    tests.test_camera_display tests.test_camera_errors tests.test_camera_cli   # yalnızca kamera testleri
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
* Kamera bağlantısı koparsa uygulama **durur** (`SourceDisconnectedError`); otomatik yeniden bağlanma uygulanmamıştır.
* Kare kaybı ve gecikme için `FramePacket.timestamp` altyapısı vardır, ancak ayrı bir ölçüm aracı yazılmamıştır.
* Video dosyasında "beklenenden erken bitti" tespiti, OpenCV'nin bildirdiği kare sayısına (yaklaşık olabilir) ve %2 toleransa dayanır.
* Pencerenin kullanıcı tarafından kapatılması Cocoa'da `getWindowProperty` ile algılanır; gerçek tıklamayla doğrulama manueldir.
* Görüntü ve yüzler varsayılan olarak diske kaydedilmez; video kaydetme özelliği yoktur.
