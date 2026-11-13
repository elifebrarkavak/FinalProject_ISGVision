# Planlanan Mimari

> **Durum: Aşama 3 tamamlandı (`COMPLETED`); Aşama 4A tamamlandı (`COMPLETED`, kapsam içi prototip); Aşama 4B tamamlandı (`COMPLETED`, 2026-11-13, kapsam içi prototip, açık sınırlamalarla); Aşama 4 genel `COMPLETED`; Aşama 5 başlatılmadı.** `camera` (bkz. [camera_module.md](camera_module.md)), `human_detection` (bkz. [person_detection.md](person_detection.md)) ve `tracking` (bkz. [person_tracking.md](person_tracking.md)) modülleri uygulanmıştır;
> diğer modüllerin hiçbiri henüz uygulanmamıştır, yalnızca boş paket iskeletleri vardır (Aşama 2'nin veri denetim kodu `dataset_audit` altındadır). Bu doküman planlanan görev dağılımını ve ilişkileri açıklar; her
> modül, aşağıdaki tabloda belirtilen aşamada geliştirilip güncellenecektir. Aşama sırası: [project_plan.md](project_plan.md).

## Modüller (`src/isg_vision/`)

| Modül | Planlanan görev | Aşama | Durum |
|---|---|---|---|
| `camera` | Kamera/video kareleri okuma (`FrameSource`, `FramePacket`), FPS, canlı pencere, kamera hata yönetimi. Yapay zekâ modüllerinden bağımsızdır. | 3 | Uygulandı (`COMPLETED`; gerçek kamera doğrulandı. Yeniden bağlanma, kare kaybı ve gecikme ölçümü ertelendi: backlog B3-1…B3-5) |
| `human_detection` | Kare başına birden fazla insan kutusu tespiti (YOLO, önceden eğitilmiş, yalnızca `person`); canlı görüntüleme, FPS/çıkarım gecikmesi ölçümü, CPU/MPS karşılaştırma ve Precision/Recall/mAP değerlendirme altyapısı. `FramePacket` tüketir; kameranın içini bilmez. | 4A | Uygulandı (`COMPLETED`; MacBook kamerasında kullanıcı tarafından doğrulandı. Açık: yorgan yanlış pozitifi FP-4A-001 (`conf 0.35` testinde de görüldü, çözülmedi), bağımsız doğruluk ölçümü, gerçek IP kamera `DEFERRED_TO_SYSTEM_INTEGRATION`) |
| `tracking` | Tespit edilen her kişiye **geçici, oturum kapsamlı** takip ID'si atama (Ultralytics ByteTrack; BoT-SORT kullanılmadı). `human_detection` çıktısını (`DetectionResult`) tüketir; YOLO'yu yeniden çalıştırmaz. Track ID kimlik değildir. | 4B | Uygulandı (`COMPLETED`, 2026-11-13; 28 kabul kriterinin 24'ü tam, 4'ü sınırlı kanıtlı; gerçek MacBook kamera testleri kısmen yapıldı. Açık: kalıcı yanlış pozitif iz alır (canlıda basılı fotoğraf/poster de Track ID aldı: FP-4B-001, çözülmedi, kaynağı ayrıştırılmadı), ID'ler konuma yapışır; örtüşmede ID değişimi ve kadrajdan çıkıp-girişte yeni ID gerçek kamerada gözlendi (KL-4B-009/010); ID switch oranı ve IDF1/HOTA/MOTA ölçülmedi; bkz. `stage4b_*` yaml bölümleri). Aşama 5 notu: Track ID çalışan kimliği değildir, ID değişimlerinde yanlış çalışan eşleştirmesi önlenmelidir |
| `face_detection` | Karedeki tüm görünür yüzleri tespit etme (MediaPipe veya SCRFD). | 5 | Planlandı |
| `face_recognition` | Yüz embedding çıkarma ve kayıtlı (en fazla 10) çalışanlarla karşılaştırma; güvenilmezse "Bilinmeyen". Biyometrik veri bu modülle sınırlı tutulur. | 6 (kayıt), 7 (tanıma) | Planlandı |
| `ppe_detection` | Baret ve yelek tespiti; bağımsız geliştirilebilir ve eğitilebilir. | 8 (ilk model), 10 (geliştirilmiş) | Planlandı |
| `integration` | Modül çıktılarını birleştirir: yüz↔kişi ve ekipman↔kişi eşleştirme, "Belirsiz" kararı, uyarı üretimi. | 11 | Planlandı |
| `storage` | Çalışan kayıtları ve tarih/saat damgalı temel olay kayıtları. | 6, 12 | Planlandı |
| `utils` | Yol tanımları (`paths.py`) ve cihaz seçimi (`device.py`: MPS > CUDA > CPU). | 1 | Yazıldı |

## Planlanan veri akışı

```
camera ──► human_detection ──► tracking ──┐
   │                                       ├──► integration ──► canlı ekran
   ├──► face_detection ──► face_recognition┤        │
   │                                       │        └──► storage (olay kaydı)
   └──► ppe_detection ─────────────────────┘
```

- `integration` tek birleştirme noktasıdır; diğer modüller birbirini doğrudan çağırmaz.
- Her modül kişi **listesi** üzerinde çalışacak şekilde tasarlanır (çoklu kişi başlangıçtan itibaren hedeftir).
- Dosya yolları `isg_vision.utils.paths` üzerinden proje köküne göre çözülür; kullanıcıya özel yol yazılmaz.
- Kimlik tanıma ve biyometrik veri (`data/employees/`) Git dışındadır ve yalnızca `face_recognition` tarafından kullanılacaktır.

## Kapsam dışı (ikinci etap)

Sensör entegrasyonu, MQTT, PIR/mesafe sensörü, gece güvenliği, insan-hayvan ayrımı ve edge AI optimizasyonları
için şu anda mimari veya kod bulunmamaktadır.
