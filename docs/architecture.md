# Planlanan Mimari

> **Durum: Aşama 2 sonu.** Aşağıdaki modüllerin hiçbiri henüz uygulanmamıştır. Yalnızca boş paket iskeletleri vardır
> (Aşama 2'nin veri denetim kodu `dataset_audit` altındadır). Bu doküman planlanan görev dağılımını ve ilişkileri açıklar; her
> modül, aşağıdaki tabloda belirtilen aşamada geliştirilip güncellenecektir. Aşama sırası: [project_plan.md](project_plan.md).

## Modüller (`src/isg_vision/`)

| Modül | Planlanan görev | Aşama | Durum |
|---|---|---|---|
| `camera` | Kamera/video kareleri okuma, FPS, kamera hata yönetimi. Yapay zekâ modüllerinden bağımsızdır. | 3 | Planlandı |
| `human_detection` | Kare başına birden fazla insan kutusu tespiti (YOLO, önceden eğitilmiş). | 4 | Planlandı |
| `tracking` | Tespit edilen her kişiye geçici takip ID'si atama (ByteTrack / BoT-SORT). | 4 | Planlandı |
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
