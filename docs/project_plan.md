# Proje Geliştirme Planı (12 Aşama) — Tek Referans

_Bu sıra proje boyunca tek referanstır. Makine tarafından okunabilir sürüm: `configs/project_stages.yaml`. Diğer belgelerdeki aşama
numaraları bu tabloya göre yazılır ve otomatik olarak denetlenir (`scripts/run_stage2c_checks.py` T2D.2)._

| Aşama | Başlık | Durum | Kapsam (özet) | Bu aşamada yapılmayacaklar |
|---|---|---|---|---|
| 1 | Proje ortamının hazırlanması | Tamamlandı | Python 3.11 ortamı, bağımlılıklar, klasör iskeleti, testler, Git kuralları | — |
| 2 | Veri setlerinin analizi ve doğrulanması | `COMPLETED` | Envanter, etiket ve sınıf doğrulama, sızıntı analizi, lisans, kullanım politikası, bölme manifesti, değerlendirme planı, son görsel inceleme | Model eğitimi, veri birleştirme, ham veri değişikliği |
| 3 | Canlı kamera altyapısı | `IN_PROGRESS` | OpenCV ile kamera erişimi, video akışı, FPS hesaplama, kamera hatalarının yönetimi, çoklu kişi analizi için görüntü akışı altyapısı. **Kamera altyapısının teknik işlev testleri** (FPS, gecikme, yeniden bağlanma, kare kaybı) | Yok: YOLO eğitimi, veri seti birleştirme, model doğruluğu değerlendirmesi |
| 4 | Çoklu insan tespiti ve takibi | Planlandı | Kare başına çoklu insan tespiti, takip ID'leri (ByteTrack / BoT-SORT) | KKD modeli eğitimi |
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
| Kamera teknik işlev testleri (FPS, gecikme, hata yönetimi) | 3 (ve sonraki aşamalarda tekrar) |
| İnsan tespiti/takibi ölçümleri | 4 |
| Yüz tespiti / kimlik tanıma ölçümleri | 5 / 7 |
| KKD modeli geliştirme ölçümü (val) ve bağımsız saha testi | 8 (ilk model), 10 (geliştirilmiş model) |
| Kişi düzeyi İSG uyumu (kaydedilmiş video) | 11 |
| Canlı kamera uçtan uca sistem testleri | 12 |

Model doğruluğuna ilişkin bilimsel değerlendirmeler, ilgili modeller eğitildikten sonra yapılır. Aşama 3'te yalnızca kamera
altyapısının teknik işlevi test edilir.
