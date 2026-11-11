# Veri Seti Kaynak, Lisans ve Atıf Kayıtları

_Oluşturulma: 2026-11-11 (Aşama 2B); Aşama 2C ile yeniden doğrulandı ve güncellendi (2026-11-11). Bu dosya yalnızca DOĞRULANABİLEN bilgileri içerir; doğrulanamayanlar açıkça belirtilmiştir. Yazar, yayın tarihi veya sürüm UYDURULMAMIŞTIR. Hukuki görüş değildir; bitirme raporuna yazmadan önce resmi sayfalar yeniden kontrol edilmelidir. Kanıt ayrıntısı: `reports/dataset_audit/license_verification_report.md`, `configs/review/license_evidence.yaml`._

## 1. Ultralytics Construction-PPE — ONAYLI KAYNAK (eğitim için kullanılabilir)

| Alan | Değer | Doğrulama |
|---|---|---|
| Veri seti adı | Ultralytics Construction-PPE | resmi doküman |
| Yayıncı | Ultralytics team | resmi doküman |
| Resmî kaynak bağlantısı | https://docs.ultralytics.com/datasets/detect/construction-ppe/ | erişildi (HTTP 200) |
| Doğrudan ZIP | https://github.com/ultralytics/assets/releases/download/v0.0.0/construction-ppe.zip | kullanıcı bu ZIP'i indirdi |
| Lisans | CC BY 4.0 (https://creativecommons.org/licenses/by/4.0/) | resmi sayfa + arşivdeki LICENSE.txt (tam yasal metin) |
| Kullanılan sürüm | Sürüm numarası YOK; yayıncı tarihi: 'November 2026' (README/doküman) | yerel README + resmi sayfa |
| Erişim tarihi | 2026-11-11 | — |
| Yerel ZIP SHA-256 | `e5817aa9c73f33e525d76ca6bc6af4e4eac9dcea9bcfef1cfd1a71c09fbc1232` (647666037 bayt) | kendi hesabımız |
| İçerik | 1000 görüntü (800 train / 200 val; ayrı test yok), 12 sınıf, 14863 kutu | yerel sayım = yayımlanan (manifest.jsonl ile SHA-256 eşleşmesi) |
| Kaynak notu | README: 'human-collected and generated images' (insan toplu ve üretilmiş görüntüler) | yerel README |
| Platform sayfası | https://platform.ultralytics.com/ultralytics/datasets/construction-ppe | yerel README + resmi doküman (2C) |
| Sınıf tanımı notu | 'no_*' sınıfları görünür vücut bölgesini işaretler; örtülü/kadraj dışı bölge tek başına eksik KKD kanıtı değildir | resmi doküman + yerel README (2C) |

**Aşama 2C yeniden doğrulama (2026-11-11):**

| Kontrol | Sonuç |
|---|---|
| Resmî doküman sayfası | HTTP 200. Lisans "CC BY 4.0" (creativecommons.org/licenses/by/4.0/). Atıf metni ve BibTeX aşağıdakiyle birebir aynı. "1,000 images (800 train, 200 validation)", "12 classes", "14,863 reviewed bounding boxes", "no separate test split". Sürüm numarası yok; tarih "November 2026" |
| Yerel arşiv | `LICENSE.txt` = CC BY 4.0 tam yasal metni; `README.md` aynı lisans ve atıf talimatını içeriyor |
| Yerel ZIP | SHA-256 yeniden hesaplandı: `e5817aa9…fbc1232`, 647666037 bayt. Kayıtla aynı |
| İçerik | 1000/1000 görüntü yayıncı `manifest.jsonl` ile eşleşiyor (ad, bölüm, SHA-256, anotasyon sayısı). Tüm ham dosyalar özgün ZIP ile boyut ve CRC32 olarak eşleşiyor (`stage2c_integrity_checks.json`) |
| Tutarsızlık notu | Platform kartında "812 classes" görünüyor; doküman metni, `data.yaml` ve yerel sayım 12 sınıf diyor. Kart değeri hatalı kabul edildi ve atıfta kullanılmadı |

Sürüm bilgisi: yayıncı sürüm numarası vermiyor. Raporda sürüm olarak **"November 2026 sürümü, ZIP SHA-256 e5817aa9…fbc1232, erişim 2026-11-11"** yazılmalıdır.

**Zorunlu atıf (yayıncının önerdiği metin):**

> Ultralytics Construction-PPE Dataset, Ultralytics team, November 2026, CC BY 4.0.

Yayıncı sayfasının önerdiği BibTeX (resmi sayfadan alındı; kullanmadan önce sayfadan yeniden kopyalayın):

```bibtex
@dataset{Ultralytics_Construction_PPE_2026,
  author = "{Ultralytics team}",
  title = {Ultralytics Construction-PPE Dataset},
  month = {November},
  year = {2026},
  license = {CC-BY-4.0},
  url = {https://docs.ultralytics.com/datasets/detect/construction-ppe},
  publisher = {Ultralytics}
}
```

**CC BY 4.0 koşulları:** uygun atıf (yaratıcı adı, telif/lisans bildirimi, lisans bağlantısı, materyal bağlantısı), değişiklik yapıldıysa belirtme (CC resmi sayfası). Bu projede yapılan değişiklikler aşağıdaki kayıt alanına işlenmelidir.

**Açıkta kalan:** üretilmiş (generated) görüntülerin ayrıca üçüncü taraf hakkı taşıyıp taşımadığı belgelenmemiştir.

### Değişiklik kayıtları (akademik raporda 'indicate if changes were made' için)

| Tarih | Değişiklik | Kapsam | Uygulayan |
|---|---|---|---|
| 2026-11-11 | Yok — ham veri değiştirilmedi; yalnızca okunarak analiz edildi (Aşama 2/2B) | — | — |
| 2026-11-11 | Aşama 2C: kullanım kararları **manifestte** kaydedildi, dosyalar değişmedi. (1) `val/source-9e10b040…` görüntüsü aynı taban karenin train varyantlarıyla aynı kaynak birimi olduğu için **train'e** atandı. (2) `train/ultralytics-ppe-nov2026-690` görünür baretsiz baş etiketsiz olduğu için yeniden etiketlenene kadar **kullanım dışı**. (3) 12 sınıftan yalnızca helmet/no_helmet/vest/no_vest hedef; diğer 8 sınıf yok sayılacak (eşleme: `configs/dataset_usage_policy.yaml`) | `reports/dataset_audit/construction_ppe_split_manifest.csv` | proje (model destekli inceleme) |
|  | (Gelecek: türetilmiş eğitim klasörü ve manifest tabanlı sınıf dönüşümü — Aşama 8 eğitim hazırlığı; birleşik veri — Aşama 9; yeniden boyutlandırma/augmentation ve yeniden etiketleme burada belgelenmeli) |  |  |

### Bitirme raporu için hazır atıf metinleri (Construction-PPE)

Kaynakça girdisi (APA benzeri):

> Ultralytics team. (2026). *Ultralytics Construction-PPE Dataset* [Veri seti]. Ultralytics. https://docs.ultralytics.com/datasets/detect/construction-ppe. Lisans: CC BY 4.0 (https://creativecommons.org/licenses/by/4.0/). Erişim: 11 Kasım 2026.

Yöntem/veri bölümüne konacak Türkçe metin:

> Bu çalışmada Ultralytics ekibi tarafından Kasım 2026'da yayımlanan Ultralytics Construction-PPE veri seti (Ultralytics Construction-PPE Dataset, Ultralytics team, November 2026, CC BY 4.0; https://docs.ultralytics.com/datasets/detect/construction-ppe) kullanılmıştır. Veri seti Creative Commons Atıf 4.0 Uluslararası (CC BY 4.0, https://creativecommons.org/licenses/by/4.0/) lisansı ile dağıtılmaktadır. Yapılan değişiklikler: özgün 12 sınıftan yalnızca baret (helmet), baretsiz (no_helmet), yelek (vest) ve yeleksiz (no_vest) sınıfları kullanılmış; aynı kaynaktan türetilmiş görüntülerin farklı bölümlere düşmemesi için bir doğrulama görüntüsü eğitim bölümüne alınmış; görünür bir hedef nesnesi etiketsiz olan bir görüntü yeniden etiketlenene kadar kullanım dışı bırakılmıştır. Özgün görüntü ve etiket dosyaları değiştirilmemiştir.

Yayıncının önerdiği kısa atıf (şekil/tablo altı, sunum vb.):

> Ultralytics Construction-PPE Dataset, Ultralytics team, November 2026, CC BY 4.0.

English (if the report or paper is in English):

> We use the Ultralytics Construction-PPE Dataset (Ultralytics team, November 2026, CC BY 4.0; https://docs.ultralytics.com/datasets/detect/construction-ppe), licensed under CC BY 4.0 (https://creativecommons.org/licenses/by/4.0/). Changes: only the helmet, no_helmet, vest and no_vest classes were used; one validation image was moved to the training split to keep generated variants of the same source frame in one split; one image with an unlabeled visible target was excluded pending re-annotation. Original files were not modified.

Rapora veri setinden görüntü konulursa görüntünün altına kısa atıf ve "değiştirildi: kutular eklendi/kırpıldı" notu yazılmalıdır.

### Gelecekteki değişikliklerin kayıt prosedürü (tüm veri setleri)

Aşama numaraları projenin 12 aşamalı planına göredir (`docs/project_plan.md`). Aşama 3 yalnızca kamera altyapısıdır; veri türetme veya eğitim içermez.

1. **Ham veri değiştirilmez.** Her değişiklik türetilmiş bir çıktıda veya manifest/politika dosyasında yapılır.
2. Her değişiklik için yukarıdaki "Değişiklik kayıtları" tablosuna bir satır eklenir: tarih, değişiklik, etkilenen dosya/manifest, uygulayan.
3. Kullanım kararı değişiyorsa `configs/dataset_usage_policy.yaml` dosyasında `policy_version` artırılır. Görüntü bazlı karar `image_overrides` ile, gerekçe `configs/review/stage2c_decisions.yaml` (veya yeni aşamanın karar dosyası) ile kaydedilir.
4. Ardından şu betikler sırayla çalıştırılır ve tüm testler yeniden koşulur:
   - `python scripts/apply_usage_policy.py`
   - `python scripts/generate_stage2b_reports.py`
   - `python scripts/run_stage2c_checks.py`
   - `python -m unittest discover -s tests`
5. Türetilmiş eğitim klasörü üretildiğinde (Aşama 8: ilk YOLO İSG modelinin eğitim hazırlığı; birleşik veri için Aşama 9) şunlar bu tabloya yazılır:
   - kaynak manifestin SHA-256 değeri
   - üretim betiği ve commit kimliği
   - sınıf dönüşüm tablosu
   - yeniden boyutlandırma ve augmentation ayarları
6. Yeni sürüm veri seti (ör. yayıncı güncellemesi) ayrı bir klasöre çıkarılır. ZIP SHA-256, erişim tarihi ve fark özeti bu dosyaya eklenir. Eski sürümle karıştırılmaz.
7. Beklemedeki bir kaynak (Mendeley/Roboflow) onaylanacaksa önce lisans/hak kanıtı `configs/review/license_evidence.yaml` dosyasına `official_page` türünde eklenir. Ancak ondan sonra `license_status`/`rights_status` ve `source_usage` değiştirilir. `validate_policy`, onaylı kapsamdaki bir kaynağın lisans durumu `verified_official` ve hak durumu `verified_publisher_original` değilse hata verir.

## 2. Mendeley Data — Dataset of Personal Protective Equipment (PPE), v6 — `rights_unverified` (ONAYLI KÜMEDE DEĞİL)

| Alan | Değer | Doğrulama |
|---|---|---|
| Yazarlar | Mei-Ling Huang, Ying Cheng | resmi sayfa + API |
| Kurum | National Chin-Yi University of Technology | resmi API |
| Resmî bağlantı | https://data.mendeley.com/datasets/zkzghjvpn2/6 | erişildi (HTTP 200) |
| DOI / sürüm / yayım | 10.17632/zkzghjvpn2.6 / v6 / 28 Temmuz 2025 | resmi sayfa + API |
| Veri seti lisansı | CC BY 4.0 | resmi sayfa + API (doğrulandı) |
| Yerel ZIP | 20250731-PPE2286y.zip, SHA-256 `7a22e5cbc0327971f56358e14f1fce88fec1d6ab30cfac642de0746202ae1f0d` | resmi API'deki SHA-256 ile BİREBİR aynı |
| Üst kaynaklar | 'GitHub, Kaggle, Roboflow ve gerçek fotoğraflar' (kaynak listesi/bağlantı/lisans YOK) | DOĞRULANAMADI |

Önerilen künye (yalnızca veri seti düzeyi; üst kaynak hakları ayrı sorudur): Huang, M.-L.; Cheng, Y. Dataset of Personal Protective Equipment (PPE), Mendeley Data, v6, 2025. doi:10.17632/zkzghjvpn2.6 (CC BY 4.0).

**Aşama 2C durumu (F02):** `external_verification_pending_source_excluded`. Harici hak doğrulaması hâlâ bekliyor. Kaynak onaylı kapsamdan çıkarıldığı için kullanım riski kontrol altında; hak sorunu **çözülmedi**. Ham veri ve analiz kayıtları yeniden değerlendirme için korunuyor.

**Belirsizlikler:** Görseller 'GitHub, Kaggle, Roboflow ve gerçek fotoğraflar'dan derlenmiş (sayfa: yalnızca genel kaynak adları; kaynak listesi, bağlantı ve üst kaynak lisansları YOK).; Üst kaynakların lisansları ve görsel bazında hak sahipliği DOĞRULANAMADI; veri setinin CC BY 4.0 olması, içindeki üçüncü taraf görseller için otomatik hak doğrulaması sağlamaz (Mendeley sayfası da bunu belirtiyor).; Görsel incelemede (bu aşamada) veri setindeki bazı görüntülerin ticari fotoğraf/ürün resimleri (filigranlı stok fotoğraf, ürün fotoğrafı) olduğu görüldü: kaynak haklarının ayrıca değerlendirilmesi gerekir.

## 3. Roboflow Universe — Construction Site Safety v27 — `pending_license_verification` (ONAYLI KÜMEDE DEĞİL)

| Alan | Değer | Doğrulama |
|---|---|---|
| Resmî bağlantı | https://universe.roboflow.com/roboflow-universe-projects/construction-site-safety/dataset/27 | HTTP 403 (erişilemedi; erişim kontrolü aşılmadı) |
| Yayıncı/çalışma alanı | roboflow-universe-projects (yerel data.yaml); 'Provided by a Roboflow user' (README) | yalnızca yerel |
| Sürüm | v27; 10 Ocak 2023 dışa aktarım (yerel README.roboflow.txt) | yalnızca yerel |
| Lisans iddiası | CC BY 4.0 (yerel data.yaml ve README.dataset.txt) | BAĞIMSIZ DOĞRULANAMADI |
| Yerel ZIP SHA-256 | `b2aa413f5e3a3691178df94ba0b8000c2bd671791506f945244908311c478adf` | kendi hesabımız |
| Kaynaklar | Üst Roboflow projeleri + iki YouTube videosu (README) | lisansları bilinmiyor (sayfalar 403) |

Roboflow Şartları (roboflow.com/terms, resmî sayfa; kısmen özetle okundu): yükleyici veri seti lisansını seçer; seçilmezse varsayılan CC BY 4.0 olduğu belirtilir; indirilen veri setlerinin izinsiz yeniden barındırılamayacağı yazar. Bu, bu veri setinin lisansını DOĞRULAMAZ.

**Aşama 2C durumu (F01):** `external_verification_pending_source_excluded`. Lisans hâlâ bağımsız doğrulanamadı ve yeni lisans iddiası oluşturulmadı. Kaynak onaylı kapsamdan çıkarıldığı için kullanım riski kontrol altında; lisans sorunu **çözülmedi**. Ham veri ve analiz kayıtları korunuyor.

**Bu veri seti için atıf metni YAZILMAMIŞTIR** (lisans doğrulanana kadar). Doğrulandıktan sonra eklenecek alan:

- Künye: _doğrulama bekliyor_
- Doğrulayan/tarih: _—_

