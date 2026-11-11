# Yakın Kopya Doğrulama Raporu (Aşama 2 eksik giderme)

_Üretim tarihi 2026-11-11. Veriler: `duplicate_verification.csv`, `duplicate_verification_summary.json`, `duplicate_method_validation.json`, `configs/review/duplicate_visual_checks.yaml`. Hiçbir görüntü silinmedi/değiştirilmedi; bölünmeler korunuyor._

## 1. Önceki yöntemin değerlendirmesi
Önceki aşamada 64-bit dHash (≤5 bit) kullanıldı: 0 birebir (SHA-256) ve 83 yakın kopya çifti bulundu. Bu yöntem kırpma/dönme/anizotropik ölçekleme gibi augmentation'lara dayanıklı değildir; ayrıca yalnızca hash mesafesine dayanıyordu. Yeni yöntem aday üretimini ve kararı ayırır:

1. **Aday üretimi (üçü birlikte):** (a) dHash + çok pencereli pHash (tam/%90/%80 kırpma, yatay çevirme); (b) ORB tanımlayıcıları üzerinde bit-örnekleme LSH oylaması (kırpma/dönme/çevirmeye dayanıklı, 16 tablo × 20 bit, kova ≤12, ≥15 oy); (c) aynı kaynak adı (`.rf.` öncesi).
2. **Doğrulama:** her aday çifte ORB (≤1500 öznitelik) + RANSAC ile benzerlik ve TAM AFİN dönüşüm (Roboflow 640×640 *stretch* anizotropisini yakalamak için), inlier sayısı/oranı, yayılım ve hizalanmış alanda sıfır-ortalamalı NCC. Saf siyah (cutout) bölgeler ORB'den maskelenir.
3. **Karar:** `same_image_derived`, `same_scene_different_frame`, `partial_region_match` (kolaj: yalnızca bir bölge ortak), `uncertain`, `not_duplicate`, `exact_duplicate` (SHA-256). Hash mesafesi tek başına karar vermez.

## 2. Yöntem doğrulama deneyi

90 gerçek görüntüden (üç veri setinden 30'ar) üretilen bozulmuş kopyalar (Roboflow benzeri) ve rastgele çiftlerle. Eşikler: {'min_inliers': 25, 'min_inlier_ratio': 0.3, 'uncertain_inliers': 12, 'min_ncc': 0.8, 'candidate_hash_bits': 10}.

| Bozulma | Hash aday yakalama | LSH aday yakalama (6085 dikkat dağıtıcı) | ORB geometrik eşleşme | Karar: same_image_derived | Inlier medyan | NCC medyan |
|---|---|---|---|---|---|---|
| identity_resize50 | 1.0 | 0.989 | 0.989 | 0.989 | 410 | 0.998 |
| jpeg_q30 | 1.0 | 1.0 | 1.0 | 1.0 | 1183 | 1.0 |
| brightness+25 | 0.989 | 0.989 | 1.0 | 1.0 | 1275 | 0.996 |
| brightness-25 | 1.0 | 1.0 | 1.0 | 1.0 | 1415 | 1.0 |
| blur | 1.0 | 1.0 | 1.0 | 1.0 | 830 | 0.999 |
| flip | 1.0 | 1.0 | 1.0 | 1.0 | 1500 | 1.0 |
| crop20 | 0.489 | 0.989 | 1.0 | 1.0 | 934 | 1.0 |
| rot12 | 0.111 | 1.0 | 1.0 | 1.0 | 945 | 1.0 |
| stretch640 | 1.0 | 0.489 | 1.0 | 1.0 | 275 | 0.999 |
| combo | 0.0 | 1.0 | 1.0 | 0.756 | 673 | 0.869 |

**Negatifler (rastgele 400 çift):** hash yanlış adayı 0; geometrik eşleşen çift 2 (inlier medyan 3, P99 8, maks 110). Bu iki çift görsel olarak incelendi: biri (`reflectivevest1529` ↔ `reflectivevest-398`) iki Mendeley kolajında ortak bisikletçi karesi (gerçek ortak kaynak), diğeri (`construction-684` ↔ `image_941`, Roboflow kolajları) ortak bölge görsel olarak KESİNLEŞTİRİLEMEDİ. Dolayısıyla gerçek yanlış pozitif oranı en çok 1/400 (%0.25) civarındadır; tüm rastgele çiftlerin tam negatif olduğu garanti değildir (veri setlerinde gerçek tekrarlar var).

**Seçilen eşikler ve dayanak:** rastgele çiftlerin %99'unda inlier ≤8; 12-24 inlier 'uncertain', ≥25 eşleşme. Hizalanmış NCC≥0.95 ve inlier≥12 ise aynı görüntü (rastgele hizalamada görülmez).

**Bilinen sınırlar:** (i) `combo` bozulmasında (kırpma+dönme+parlaklık+bulanıklık+çevirme) `same_image_derived` oranı yalnızca %75.6; geri kalanı `same_scene_different_frame` etiketi alır (her ikisi de eşleşme sayılır). (ii) Aday üretimi tüm olası çiftleri taramaz; ≥15 LSH oyu veya hash ≤10 bit gerekir — bu nedenle aşağıdaki oranlar ALT SINIRDIR.

## 3. Gerçek augmentation üzerinde aday üretici geri çağırması (kaynak adı = sahte gerçek)

Roboflow'da aynı kaynak adını taşıyan aynı-bölüm çiftlerinden 400 tanesi örneklendi: geometrik eşleşme 101/400; dHash≤5 0, pHash-çok-pencere≤10 0, LSH oyu≥15 13. **Yorum:** Roboflow *train* görüntüleri 2×2 kolajdır (görsel doğrulandı); aynı kaynak adını taşıyan iki görüntü genellikle yalnızca bir kolaj karesini paylaşır, bu yüzden hash/LSH yakalaması düşüktür ve geometrik eşleşme çoğunlukla bölgesel (`partial_region_match`/`same_scene_different_frame`). Kaynak adı tek başına 'aynı görüntü' kanıtı değildir.

**Aynı kaynak adı ≠ kopya.** Önceki aşamada bölümler arasında aynı kaynak adını taşıyan 8 Roboflow grubu 'olası sızıntı' sayılmıştı. Geometrik ve görsel doğrulama: yalnızca 1 grupta (`Movie-on…mov-13`) kanıtlı eşleşme var; 4 grupta görsel olarak ortak içerik YOK (kolaj karesi adı ilk kareden gelir); 1 grupta yalnızca geometrik (görsel bakılmadı) eşleşme yok; 2 grupta (`IMG_3103_mp4-17`, `IMG_0871_mp4-23`) aynı kişi/videodan olma ihtimali var ama yöntemle KESİNLEŞTİRİLEMEDİ. Dolayısıyla Roboflow dosya adı öneki güvenilir bir kopya anahtarı değildir.

## 4. Sonuçlar

Doğrulanan aday çift: **4621** (6085 görüntü). Karar dağılımı: {'not_duplicate': 500, 'same_image_derived': 630, 'partial_region_match': 273, 'same_scene_different_frame': 3152, 'uncertain': 66}.

| İlişki | same_image_derived | same_scene_different_frame | partial_region_match | uncertain | not_duplicate |
|---|---|---|---|---|---|
| cross_dataset | 4 | 10 | 0 | 3 | 79 |
| within_dataset_cross_split | 160 | 570 | 16 | 5 | 79 |
| within_dataset_same_split | 466 | 2572 | 257 | 58 | 342 |

### 4.1 Bölümler arası maruziyet (en az bir geometrik eşleşmesi olan görüntüler; ALT SINIR)

| Veri seti/bölüm | Eşleşmeli görüntü | Bölüm boyutu | Oran |
|---|---|---|---|
| construction_ppe/train | 2 | 800 | %0.2 |
| construction_ppe/val | 1 | 200 | %0.5 |
| mendeley_ppe/train | 77 | 1829 | %4.2 |
| mendeley_ppe/val | 43 | 457 | %9.4 |
| site_safety/test | 70 | 82 | %85.4 |
| site_safety/train | 522 | 2603 | %20.1 |
| site_safety/val | 90 | 114 | %78.9 |

**Yüksek öncelikli risk:** `site_safety` val ve test görüntülerinin büyük çoğunluğu başka bir bölümdeki (çoğunlukla train) görüntüyle aynı videodan/aynı fotoğraftan türemiştir. Mevcut Roboflow val/test metrikleri bu nedenle iyimser olur. Mendeley val'de de anlamlı bir pay vardır (rastgele %20 bölme).

### 4.2 Öncelikli çiftler

| Çift | Veri seti A | Dosya A | Veri seti B | Dosya B | Ölçülen benzerlikler | Otomatik karar | Görsel doğrulama | Sızıntı riski |
|---|---|---|---|---|---|---|---|---|
| DV0079 | mendeley_ppe/train | `00596_jpg.rf.e056867eae13eca16cc7c…` | site_safety/val | `00596_jpg.rf.d030c5d98b937d080d75d…` | dHash 3, pHash 0, LSH 6/2, ORB inlier 48 (oran 0.8), NCC 0.999, anizotropi 0.56 | same_image_derived | confirmed_same_source_image: Mendeley train ile Roboflow val aynı fotoğraf; Roboflow sürümü 640x640 'stretch' (anizotropi 0.56) | HIGH (birleştirmede doğrulama/test sızıntısı) |
| DV0751 | mendeley_ppe/val | `02912_jpg.rf.d68699bf2aa81dc0c86ab…` | site_safety/val | `ppe_1228_jpg.rf.29b714c415ea74df31…` | dHash 0, pHash 0, LSH 9/1, ORB inlier 150 (oran 0.773), NCC 0.998, anizotropi 0.664 | same_image_derived | confirmed_same_source_image: Mendeley val ile Roboflow val aynı fotoğraf (kırmızı baretli işçi, tablet); Roboflow sürümü gerilmiş | HIGH (birleştirmede doğrulama/test sızıntısı) |
| DV0081 | mendeley_ppe/train | `00624_jpg.rf.6d31c7a4e92fe52fa01a7…` | site_safety/val | `ppe_0665_jpg.rf.1dad479f7f54b2a712…` | dHash 2, pHash 0, LSH 2/3, ORB inlier 24 (oran 0.545), NCC 0.997, anizotropi 0.476 | same_image_derived | confirmed_same_source_image: Mendeley train ile Roboflow val aynı grup fotoğrafı; Roboflow sürümü gerilmiş | HIGH (birleştirmede doğrulama/test sızıntısı) |
| DV0428 | mendeley_ppe/train | `02646_jpg.rf.4b5bddb3448f6298cb316…` | site_safety/val | `02646_jpg.rf.5c93ba95bdc03808bcf87…` | dHash 0, pHash 0, LSH 12/4, ORB inlier 249 (oran 0.85), NCC 0.999, anizotropi 0.715 | same_image_derived | confirmed_same_source_image: Mendeley train ile Roboflow val aynı fotoğraf (kürekli işçi); Roboflow sürümü gerilmiş | HIGH (birleştirmede doğrulama/test sızıntısı) |
| DV4609 | site_safety/val | `Movie-on-10-31-22-at-10_08-AM_mov-…` | site_safety/test | `Movie-on-10-31-22-at-10_08-AM_mov-…` | dHash 4, pHash 4, LSH 150/1, ORB inlier 752 (oran 0.983), NCC 0.922, anizotropi 1.0 | same_image_derived | confirmed_same_scene_different_frame: Roboflow val/test: aynı videodan ardışık kareler (aynı oda, farklı poz); aynı görüntü değil | HIGH (train/val/test sızıntısı) |
| DV4612 | site_safety/val | `youtube-194_jpg.rf.601da70a10e87c3…` | site_safety/test | `youtube-198_jpg.rf.e89faeb9765c6bd…` | dHash 7, pHash 7, LSH 34/1, ORB inlier 227 (oran 0.492), NCC 0.922, anizotropi 1.03 | same_image_derived | confirmed_same_scene_different_frame: Roboflow val/test: aynı tünel/ekskavatör videosundan neredeyse özdeş ardışık kareler | HIGH (train/val/test sızıntısı) |

- **Mendeley–Roboflow 4 çift: dördü de AYNI KAYNAK FOTOĞRAF** (Roboflow sürümü 640×640 gerilmiş). Mendeley sayfası verinin Roboflow'dan da derlendiğini söylüyor; ancak Mendeley ZIP'inin kendi README'si de bir Roboflow projesinden dışa aktarım olduğunu gösteriyor. İlk yöntem benzerlik dönüşümüyle bu çiftleri 'uncertain/not_duplicate' saymıştı; tam afin dönüşümü eklenince düzeldi.
- **val↔test 2 çift:** aynı videodan ardışık kareler (poz farklı): aynı görüntü değil ama aynı sahne; val/test bağımsızlığını bozar.
- Veri setleri arası eşleşme adayı toplam 17 çift (karar: {'same_scene_different_frame': 10, 'same_image_derived': 4, 'uncertain': 3}); 4'ü yan yana incelenip aynı-kaynak fotoğraf olarak doğrulandı; kalan çiftler aynı kaynak/sahne ADAYI olarak CSV'dedir (görsel doğrulanmadı).

### 4.3 Görsel doğrulama örneklemi

Otomatik kararı 'eşleşme' olan 26 çift (6 öncelikli + rastgele 12 Roboflow + 9 Mendeley bölümler-arası; bir Roboflow çifti ortak) yan yana açılıp incelendi: **26/26 çift görsel olarak aynı kaynağa/sahneye ait** (dağılım: {'confirmed_same_source_image': 8, 'confirmed_same_scene_different_frame': 7, 'confirmed_shared_collage_tile': 10, 'confirmed_same_photo_session': 1}). Bu, bölümler arası geometrik eşleşmelerde tahmini kesinliğin yüksek olduğunu gösterir (Wilson %95 alt sınır ≈ %87); yine de örneklem küçüktür ve 'uncertain' ile `not_duplicate` kararları görsel doğrulanmadı.

## 5. Augmentation izleri

Siyah kare (cutout) içeren görüntü oranı (sezgisel): Roboflow train %11.6, val %1.8, test %1.2. Augmentation yalnızca eğitim bölümüne uygulanmış görünüyor. Yerel README `Resize 640x640 (Stretch)` ve flip/crop/rotation/shear/brightness/exposure/blur listeler; cutout ve 2×2 kolaj ise README'de YAZMIYOR ancak görüntülerde görülüyor: README ile gerçek içerik uyuşmuyor.

## 6. Karar ve öneriler (uygulanmadı)

- Veri setleri birleştirilmeden ve eğitim yapılmadan önce bölünmeler **grup bazlı** (aynı video/sahne/fotoğraf bir arada) yeniden kurulmalı; mevcut Roboflow val/test'e güvenilmemeli.
- Mendeley `val`'deki train ile eşleşen görüntüler (aşağı yukarı %9) değerlendirme dışı bırakılmalı veya gruplanmalı.
- Veri setleri arası aynı-kaynak çiftleri (en az 4) birleşik sette tek örneğe indirilmeli.
- Yeni, daha geniş kapsamlı aday üretimi gerekirse (≥%99 geri çağırma) LSH oy eşiği düşürülüp ORB doğrulaması genişletilebilir; maliyet ≈ dakikalar.
