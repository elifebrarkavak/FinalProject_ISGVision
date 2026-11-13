# Proje Geliştirme Planı (12 Aşama) — Tek Referans

_Bu sıra proje boyunca tek referanstır. Makine tarafından okunabilir sürüm: `configs/project_stages.yaml`. Diğer belgelerdeki aşama
numaraları bu tabloya göre yazılır ve otomatik olarak denetlenir (`scripts/run_stage2c_checks.py` T2D.2)._

| Aşama | Başlık | Durum | Kapsam (özet) | Bu aşamada yapılmayacaklar |
|---|---|---|---|---|
| 1 | Proje ortamının hazırlanması | Tamamlandı | Python 3.11 ortamı, bağımlılıklar, klasör iskeleti, testler, Git kuralları | — |
| 2 | Veri setlerinin analizi ve doğrulanması | `COMPLETED` | Envanter, etiket ve sınıf doğrulama, sızıntı analizi, lisans, kullanım politikası, bölme manifesti, değerlendirme planı, son görsel inceleme | Model eğitimi, veri birleştirme, ham veri değişikliği |
| 3 | Canlı kamera altyapısı | `COMPLETED` | OpenCV ile kamera erişimi, video akışı, FPS hesaplama, kamera hatalarının yönetimi, çoklu kişi analizi için görüntü akışı altyapısı. **Teknik işlev testleri:** FPS ölçümü ve hata yönetimi tamamlandı; gecikme ölçümü, yeniden bağlanma ve kare kaybı ölçümü kapanışta **ertelendi** (backlog B3-1…B3-5, aşağıda) | Yok: YOLO eğitimi, veri seti birleştirme, model doğruluğu değerlendirmesi |
| 4 | Çoklu insan tespiti ve takibi | `COMPLETED` (4A `COMPLETED`; 4B `COMPLETED`, 2026-11-13; kapsam içi prototip, açık sınırlamalarla) | **4A:** kare başına çoklu insan tespiti (hazır YOLO, yalnızca `person`, kutu + güven + kişi sayısı, FPS/gecikme ölçümü). **4B:** geçici takip ID'leri (Ultralytics ByteTrack; YOLO kare başına tek çağrı) | KKD modeli eğitimi. 4A'da ayrıca: takip/Track ID, yüz tespiti, kimlik tanıma |
| 5 | Çoklu yüz tespiti | Planlandı | Karedeki tüm görünür yüzlerin tespiti ve kişi izleriyle ilişkilendirilmesi | Kimlik tanıma |
| 6 | Çalışan kayıt sistemi | Planlandı | En fazla 10 çalışanın KVKK uyumlu kaydı (`data/employees/`, Git dışı) | — |
| 7 | Çoklu çalışan kimlik tanıma | Planlandı | ArcFace tabanlı tanıma; güvenilmeyen eşleşme "Bilinmeyen kişi" | — |
| 8 | İlk YOLO İSG modelinin eğitilmesi | Planlandı | **Eğitim hazırlığı:** bölme manifestinden türetilmiş eğitim verisi, manifest tabanlı sınıf dönüşümü (12 → 4 sınıf), görsel inceleme kapısı. Construction-PPE ile ilk baret/yelek modeli; val ile geliştirme ölçümü; dondurulmuş bağımsız saha testiyle ilk bilimsel değerlendirme | Veri seti birleştirme |
| 9 | Veri setlerinin birleştirilmesi | Planlandı | Yalnızca lisans/hak doğrulanan kaynaklarla, kaynak grubu kurallarıyla (G1–G6) birleşik veri ve birleşik bölme | Doğrulanmamış kaynağın onaylanması |
| 10 | Geliştirilmiş YOLO İSG modelinin eğitilmesi | Planlandı | Birleşik veriyle eğitim; aynı dondurulmuş saha testiyle Aşama 8 modeliyle karşılaştırma | — |
| 11 | Kimlik tanıma ve İSG kontrolünün birleştirilmesi | Planlandı | Kişi–baret/yelek ilişkilendirmesi, kişi düzeyi İSG kararı, kaydedilmiş video oturumlarında değerlendirme | — |
| 12 | Canlı arayüz, sistem testleri ve raporlama | Planlandı | Canlı sonuç ekranı, olay kayıtları, uçtan uca canlı kamera sistem testleri (FPS, gecikme, alarm/saat), nihai rapor | — |

## Değerlendirme türlerinin aşamalara dağılımı

| Değerlendirme | Aşama |
|---|---|
| Kamera teknik işlev testleri (FPS, gecikme, hata yönetimi) | 3 (FPS ölçümü ve hata yönetimi tamamlandı); gecikme, kare kaybı ve yeniden bağlanma: 4 ve 12 (backlog) |
| İnsan tespiti/takibi ölçümleri | 4 (4A: hız/gecikme ölçüldü, canlı MacBook kamerasında kullanıcı tarafından da; Precision/Recall/mAP altyapısı hazır, etiketli bağımsız küme olmadığı için doğruluk **ölçülmedi** (açık görev OT-4A-002); 4B: takip hızı kayıtlı videoda ölçüldü; takip **doğruluğu** (ID switch oranı, IDF1/HOTA/MOTA — hiçbir değer üretilmedi) ve canlı kare→sonuç gecikmesi **ölçülmedi** (OT-4B-003, OT-4B-004); canlı FPS ve çoklu takip kullanıcı tarafından kısmen gözlendi, örtüşmede/çıkış-girişte ID değişimleri nitel gözlendi)) |
| Yüz tespiti / kimlik tanıma ölçümleri | 5 / 7 |
| KKD modeli geliştirme ölçümü (val) ve bağımsız saha testi | 8 (ilk model), 10 (geliştirilmiş model) |
| Kişi düzeyi İSG uyumu (kaydedilmiş video) | 11 |
| Canlı kamera uçtan uca sistem testleri | 12 |

Model doğruluğuna ilişkin bilimsel değerlendirmeler, ilgili modeller eğitildikten sonra yapılır. Aşama 3'te yalnızca kamera
altyapısının teknik işlevi test edilir.

## Aşama 3'te ertelenen maddeler (backlog)

Aşama 3, kullanıcı onayıyla (2026-11-11) aşağıdaki maddeler **yapılmadan** `COMPLETED` olarak kapatılmıştır. Bu maddeler tamamlanmış sayılmaz; ilgili aşamanın
başında kapsama alınıp alınmayacağı kullanıcı kararıyla yeniden gözden geçirilir. Ayrıntı: `reports/camera/STAGE_3_REPORT.md` (bölüm K.2). Makine tarafından okunur
sürüm: `configs/project_stages.yaml` (`stage3_deferred_backlog`).

| ID | Madde | Planlanan değerlendirme aşaması | Gerekçe |
|---|---|---|---|
| B3-1 | Otomatik yeniden bağlanma | 12 | Canlı sistemin dayanıklılığı uçtan uca sistem testlerinin konusudur. Bugün bağlantı koparsa uygulama kontrollü biçimde durur (`SourceDisconnectedError`) |
| B3-2 | Kare kaybı ölçümü | 4, sonra 12 | Takip sürekliliği kare kaybına duyarlıdır; ilk ölçüm takipçi akışa bağlanınca, nihai ölçüm uçtan uca testlerde |
| B3-3 | Gecikme (yakalama → sonuç) ölçümü | 4, sonra 12 | Anlamlı gecikme, model döngüdeyken ölçülür; nihai ölçüm Aşama 12'deki canlı kamera sistem testlerindedir |
| B3-4 | Gerçek kamera FPS'inin ileri doğrulaması | 4, sonra 12 | Aşama 3'te yalnızca modelsiz FPS ölçüldü; model döngüdeyken, uzun süreli ve bağımsız zaman referansıyla doğrulanmalı |
| B3-5 | Fiziksel bağlantı kopması testi | 12 | B3-1 ile birlikte; mevcut "kontrollü durma" davranışı istenirse daha önce de denenebilir |

## Aşama 4 alt aşamaları

Aşama 4 iki alt aşamada geliştirilir (kullanıcı kararı, 2026-11-12). Aşama 4A `COMPLETED`'dır; 4B `COMPLETED`'dır (2026-11-13, kullanıcı onayıyla; önceki durum `READY_FOR_MANUAL_VERIFICATION`). Kural: Aşama 4 genel durumu **4B de tamamlanana kadar `COMPLETED` olmaz**; 4B tamamlandığı için Aşama 4 genel durumu `COMPLETED`'dır. Aşama 5 başlatılmamıştır (`planned`). Makine tarafından okunur sürüm: `configs/project_stages.yaml` (`stage4_substages`).

| Alt aşama | Başlık | Durum | Not |
|---|---|---|---|
| 4A | Çoklu insan tespiti | `COMPLETED` (2026-11-12) | **Yalnızca kapsam içi prototip geliştirme ve doğrulama.** Kod, otomatik testler, gerçek model doğrulamaları ve kullanıcının gerçek MacBook kamerasıyla yaptığı manuel testler (üç kişi, boş ortam, penceresiz CPU/MPS 30,0 FPS, ≈2 m ön test, Q/Esc/kırmızı düğme/Ctrl+C) tamam. OT-4A-001 `conf 0.35` karşılaştırma testi kapanıştan sonra kullanıcı tarafından yapıldı (`TEST_PERFORMED`): yorgan 0,35'te de kısa süreliğine insan sanıldı, sorun çözülmedi, gerçek insan tespitinin başarısı ölçülmedi, varsayılan `conf` 0,25. **Açık kalanlar** (tamamlanmış sayılmaz): OT-4A-002 bağımsız doğruluk değerlendirmesi (Precision/Recall/mAP **ölçülmedi**, %85 hedefinin karşılandığı iddia edilmez); OT-4A-003 gerçek IP kamera doğrulaması `DEFERRED_TO_SYSTEM_INTEGRATION`; OT-4A-004 / OBS-4A-001 en yüksek 7 tespit gözlemi; bilinen sınırlama FP-4A-001 (yorgan yanlış pozitifi; açık). Ayrıntı: `docs/person_detection.md`, `reports/detection/STAGE_4A_REPORT.md` |
| 4B | Çoklu insan takibi | `COMPLETED` (2026-11-13; önceki: `READY_FOR_MANUAL_VERIFICATION`, 2026-11-12) | **Yalnızca kapsam içi prototip geliştirme ve doğrulama.** İlk görev tanımındaki **28 kabul kriterinin 28'i karşılandı: 24 tam kanıtlı, 4 sınırlı kanıtlı** (#9 tek insan, #17 yorgan riski, #19 FPS/gecikme, #21 çıkış yöntemleri; canlı kamera kanıtı kısmi, tam doğrulanmış sayılmaz). Ultralytics **ByteTrack** ile geçici, oturum kapsamlı Track ID (kimlik DEĞİL); YOLO kare başına tek çağrı; `lap==0.5.14` kontrollü kuruldu. 883 otomatik test (287 yeni: 203 mock/birim, 62 gerçek ByteTrack, 20 gerçek YOLO + ByteTrack, 2 isteğe bağlı GUI); ByteTrack ≈ 0,4 ms/kare (kayıtlı sentetik video). **Açık kalanlar (kapanışı engellemez, tamamlanmış sayılmaz):** gerçek MacBook kamerası manuel testlerinin kalanı (OT-4B-001, OT-4B-002, OT-4B-004; kullanıcı pencereli MPS 16,3 FPS, penceresiz CPU/MPS 29,9 FPS, Q çıkışı ve 2026-11-13'te çoklu takibin çalıştığını bildirdi; üç kişi belirsiz, yüksek açı, yorgan, Esc/kırmızı düğme/Ctrl+C yok); **örtüşmede ID değişimi ve kadrajdan çıkıp-girişte yeni ID gerçek kamerada gözlendi (KL-4B-009/010; nitel, oran ölçülmedi)**; canlıda **poster yanlış pozitifi FP-4B-001 açık ve çözülmedi, test ortamı değiştiği için aynı koşullarda T1/T2 yeniden doğrulanamıyor, kaynağı ayrıştırılmadı, eşik değiştirilmedi** (OT-4B-005 **araştırma görevi**); takip doğruluğu (ID switch oranı, IDF1/HOTA/MOTA) **ölçülmedi** (OT-4B-003). Gerçek IP kamera 4B kapanış koşulu değildir (OT-4A-003). **Aşama 5 notu:** geçici Track ID kayıtlı çalışan kimliği değildir; ID değişimlerinde yanlış çalışan eşleştirmesi önlenmelidir (`docs/person_tracking.md` §18). Ayrıntı: `docs/person_tracking.md`, `reports/tracking/STAGE_4B_REPORT.md` |
