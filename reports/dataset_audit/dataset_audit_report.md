# Aşama 2 — Veri Seti Denetim Raporu

_Otomatik üretilmiştir (`scripts/generate_dataset_report.py`), tarih: 2026-11-11. Sayılar `reports/dataset_audit/` altındaki betik çıktılarından okunur. "Görsel ön gözlem" ifadeleri kullanıcı tarafından DOĞRULANMAMIŞTIR._

## 1. Analizin amacı
Bu rapor, üç açık erişimli İSG (KKD) veri setinin ileride baret ve yelek (helmet / no_helmet / vest / no_vest) tespiti için YOLO eğitiminde güvenilir biçimde kullanılıp kullanılamayacağını değerlendirir. Yalnızca analiz yapılmıştır: veri setleri birleştirilmemiş, yeniden bölünmemiş, etiketler değiştirilmemiş, hiçbir dosya silinmemiş ve model eğitilmemiştir.

## 2. Kullanılan veri setleri
| Veri seti | Proje içi konum | Kaynak ZIP (indirilenler) | ZIP boyutu | ZIP SHA-256 (ilk 16) |
|---|---|---|---|---|
| Ultralytics Construction-PPE | `data/raw/construction_ppe/` | construction-ppe.zip | 617.7 MB | `e5817aa9c73f33e5…` |
| Mendeley Data - Dataset of Personal Protective Equipment (PPE), v6 | `data/raw/mendeley_ppe/` | 20250731-PPE2286y.zip | 225.1 MB | `7a22e5cbc0327971…` |
| Roboflow Universe - Construction Site Safety v27 (YOLOv8) | `data/raw/site_safety/` | Construction Site Safety.v27-yolov8.yolov8.zip | 149.9 MB | `b2aa413f5e3a3691…` |

ZIP'ler `~/Downloads` içinde olduğu gibi korunmuştur; içerik `scripts/extract_datasets.py` ile güvenli biçimde (üzerine yazmadan, zip-slip denetimiyle, dosyalar salt-okunur) `data/raw/` altına çıkarılmıştır. Tek üst düzey klasör (`construction-ppe/`, `20250731-ppe2286y/`) sıyrılmıştır; bu yüzden orijinal `data.yaml` içindeki göreli `path` değerleri burada olduğu gibi çözümlenmez (aşağıya bakın).

- `mendeley_ppe` arşivi, veri dışında bir Python betiği (`01_*.py`, rastgele %20'yi `valid/` klasörüne taşıyan bir bölme betiği) içerir. Bu betik **çalıştırılmamıştır**; yalnızca okunmuştur ve verinin rastgele bölündüğünün işaretidir.

## 3. Veri setlerinin sürümleri
| Veri seti | Sürüm | Yayıncı/sağlayıcı | DOI | Kaynak |
|---|---|---|---|---|
| construction_ppe | Sürüm numarası yok; 'November 2026' olarak yayımlanmış (yerel README) | Ultralytics team | — | https://docs.ultralytics.com/datasets/detect/construction-ppe/ |
| mendeley_ppe | v6, yayım tarihi 28 Temmuz 2025 | Mei-Ling Huang, Ying Cheng (National Chin-Yi University of Technology) | 10.17632/zkzghjvpn2.6 | https://data.mendeley.com/datasets/zkzghjvpn2/6 |
| site_safety | v27, Roboflow'dan 10 Ocak 2023'te dışa aktarılmış (yerel README.roboflow.txt) | Roboflow Universe, workspace roboflow-universe-projects (veriyi bir Roboflow kullanıcısı sağlamış) | — | https://universe.roboflow.com/roboflow-universe-projects/construction-site-safety/dataset/27 |

## 4. Görüntü ve etiket sayıları
Sayılar gerçek dosyalardan sayılmıştır; "yayımlanan" sütunu yerel README/kaynak sayfasındaki değerdir.

| Veri seti | Bölüm | Görüntü | Etiket dosyası | Kutu | Sınıf | Boş etiket | Eksik etiket | Yetim etiket | Bozuk görüntü | Görüntü MB |
|---|---|---|---|---|---|---|---|---|---|---|
| construction_ppe | train | 800 | 800 | 11902 | 12 | 0 | 0 | 0 | 0 | 492.0 |
| construction_ppe | val | 200 | 200 | 2961 | 12 | 0 | 0 | 0 | 0 | 125.1 |
| construction_ppe | ALL | 1000 | 1000 | 14863 | 12 | 0 | 0 | 0 | 0 | 617.1 |
| mendeley_ppe | train | 1829 | 1829 | 4862 | 4 | 0 | 0 | 0 | 0 | 181.1 |
| mendeley_ppe | val | 457 | 457 | 1176 | 4 | 0 | 0 | 0 | 0 | 46.9 |
| mendeley_ppe | ALL | 2286 | 2286 | 6038 | 4 | 0 | 0 | 0 | 0 | 228.0 |
| site_safety | train | 2603 | 2603 | 37378 | 10 | 5 | 0 | 0 | 0 | 136.0 |
| site_safety | val | 114 | 114 | 697 | 10 | 10 | 0 | 0 | 0 | 6.2 |
| site_safety | test | 82 | 82 | 760 | 10 | 8 | 0 | 0 | 0 | 4.5 |
| site_safety | ALL | 2799 | 2799 | 38835 | 10 | 23 | 0 | 0 | 0 | 146.6 |

- **construction_ppe**: görüntü gerçek 1000 / yayımlanan 1000 (fark +0); kutu gerçek 14863 / yayımlanan 14863. (Yerel README.md ve docs.ultralytics.com sayfası (ayrı test bölümü yok))
- **mendeley_ppe**: görüntü gerçek 2286 / yayımlanan 2286 (fark +0). (Mendeley sayfası: 2,286 görüntü, 640x640 piksel)
- **site_safety**: görüntü gerçek 2799 / yayımlanan 2801 (fark -2). (Yerel README.roboflow.txt: 'The dataset includes 2801 images')

**Çözünürlük bulguları.** `site_safety` tüm görüntüleri 640×640'tır (Roboflow ön işlemi: stretch). `construction_ppe` görüntüleri 1264x848 ile 2048x2048 arasındadır. `mendeley_ppe` için kaynak sayfa 640×640 belirtse de gerçekte 905 farklı çözünürlük vardır (204x136 … 5712x4284); yalnızca 472 görüntü 640x640 boyutundadır. Bu fark kaynak sayfanın betimlemesiyle gerçek içerik arasındaki tutarsızlıktır.

**YAML dosyaları.**

| Veri seti | YAML | names biçimi | nc | names içindeki sınıf | Sınıf uyarıları | Yabancı mutlak path | YAML yolları bu konumda |
|---|---|---|---|---|---|---|---|
| construction_ppe | `data.yaml` | sözlük | yok | 12 | — | hayır | train=çözülmüyor, val=çözülmüyor, test=— |
| mendeley_ppe | `data.yaml` | liste | 4 | 4 | — | evet | train=çözülmüyor, val=çözülmüyor, test=— |
| site_safety | `data.yaml` | liste | 10 | 10 | — | hayır | train=çözülmüyor, val=çözülmüyor, test=çözülmüyor |

`construction_ppe/data.yaml` `nc` alanı içermez (sınıf sayısı `names` sözlüğünden 12 olarak okunmuştur). `mendeley_ppe/data.yaml` Windows mutlak yolu (`C:/Users/...`) içerir; `site_safety/data.yaml` Roboflow'un bilinen `../train/images` göreli yollarını kullanır. Hiçbir YAML olduğu gibi eğitimde kullanılamaz; Aşama 8+ eğitimde yeni, yerel yollarla bir YAML oluşturulacaktır (bu aşamada yazılmadı).

**Sınıf tanımları (YAML'dan okunan sıra, değiştirilmedi):**

| Veri seti | ID:ad |
|---|---|
| construction_ppe | 0:helmet, 1:gloves, 2:vest, 3:boots, 4:goggles, 5:Person, 6:no_helmet, 7:no_goggle, 8:no_gloves, 9:no_boots, 10:forklift, 11:no_vest |
| mendeley_ppe | 0:Helmet, 1:NoHelmet, 2:NoVest, 3:Vest |
| site_safety | 0:Hardhat, 1:Mask, 2:NO-Hardhat, 3:NO-Mask, 4:NO-Safety Vest, 5:Person, 6:Safety Cone, 7:Safety Vest, 8:machinery, 9:vehicle |

## 5. Sınıf dağılımları
Grafikler: `reports/dataset_audit/charts/class_distribution_<veri_seti>.png`. Ayrıntılı tablo: `class_distribution.csv`.

**construction_ppe** (toplam 14863 kutu)

| ID | Sınıf | Kutu | Görüntü | Pay | train | val | test |
|---|---|---|---|---|---|---|---|
| 0 | helmet | 1576 | 759 | %10.6 | 1282 | 294 | — |
| 1 | gloves | 1929 | 682 | %12.98 | 1548 | 381 | — |
| 2 | vest | 1816 | 801 | %12.22 | 1476 | 340 | — |
| 3 | boots | 2484 | 731 | %16.71 | 2003 | 481 | — |
| 4 | goggles | 751 | 539 | %5.05 | 612 | 139 | — |
| 5 | Person | 2443 | 991 | %16.44 | 1962 | 481 | — |
| 6 | no_helmet | 813 | 428 | %5.47 | 634 | 179 | — |
| 7 | no_goggle | 347 | 283 | %2.33 | 265 | 82 | — |
| 8 | no_gloves | 1042 | 434 | %7.01 | 839 | 203 | — |
| 9 | no_boots | 747 | 315 | %5.03 | 569 | 178 | — |
| 10 | forklift | 348 | 247 | %2.34 | 275 | 73 | — |
| 11 | no_vest | 567 | 384 | %3.81 | 437 | 130 | — |

**mendeley_ppe** (toplam 6038 kutu)

| ID | Sınıf | Kutu | Görüntü | Pay | train | val | test |
|---|---|---|---|---|---|---|---|
| 0 | Helmet | 2421 | 1390 | %40.1 | 1970 | 451 | — |
| 1 | NoHelmet | 1174 | 622 | %19.44 | 960 | 214 | — |
| 2 | NoVest | 1058 | 661 | %17.52 | 836 | 222 | — |
| 3 | Vest | 1385 | 809 | %22.94 | 1096 | 289 | — |

**site_safety** (toplam 38835 kutu)

| ID | Sınıf | Kutu | Görüntü | Pay | train | val | test |
|---|---|---|---|---|---|---|---|
| 0 | Hardhat | 3551 | 1394 | %9.14 | 3362 | 79 | 110 |
| 1 | Mask | 1792 | 1139 | %4.61 | 1743 | 21 | 28 |
| 2 | NO-Hardhat | 2428 | 1453 | %6.25 | 2318 | 69 | 41 |
| 3 | NO-Mask | 3362 | 1577 | %8.66 | 3209 | 74 | 79 |
| 4 | NO-Safety Vest | 4153 | 1921 | %10.69 | 3957 | 106 | 90 |
| 5 | Person | 10031 | 2649 | %25.83 | 9691 | 166 | 174 |
| 6 | Safety Cone | 3306 | 647 | %8.51 | 3170 | 44 | 92 |
| 7 | Safety Vest | 3258 | 1363 | %8.39 | 3156 | 41 | 61 |
| 8 | machinery | 5337 | 2129 | %13.74 | 5238 | 55 | 44 |
| 9 | vehicle | 1617 | 772 | %4.16 | 1534 | 42 | 41 |

**Hedef sınıf gruplarına göre kutu sayıları (aday eşlemeye göre; eşleme kesin değildir):**

| Veri seti | helmet | no_helmet | helmet/no_helmet | vest | no_vest | vest/no_vest |
|---|---|---|---|---|---|---|
| construction_ppe | 1576 | 813 | 1.94 | 1816 | 567 | 3.20 |
| mendeley_ppe | 2421 | 1174 | 2.06 | 1385 | 1058 | 1.31 |
| site_safety | 3551 | 2428 | 1.46 | 3258 | 4153 | 0.78 |

**Dengesizlik değerlendirmesi.**
- `construction_ppe`: en çok kutu `boots` (2484), en az `no_goggle` (347); oran 7.2:1.
- `mendeley_ppe`: en çok kutu `Helmet` (2421), en az `NoVest` (1058); oran 2.3:1.
- `site_safety`: en çok kutu `Person` (10031), en az `vehicle` (1617); oran 6.2:1.
- Hedef sınıf çiftlerinde (yukarıdaki tablo) ihlal ve uyum sınıfları arasında oran farkı vardır. Bu tür bir dengesizlik eğitilen modelin baskın sınıfa (ör. baretli) yönelmesine ve nadir sınıfın (baretsiz/yeleksiz) recall değerinin düşmesine yol açabilir. Bu aşamada oversampling, undersampling veya augmentation UYGULANMAMIŞTIR; yalnızca belgelenmiştir.
- Küçük bölümler: `site_safety/val` (114 görüntü), `site_safety/test` (82 görüntü). Bu bölümlerde bazı sınıfların kutu sayısı çok düşüktür; sınıf bazlı doğrulama/test metrikleri istatistiksel olarak güvenilmez olabilir.

## 6. Görüntü–etiket eşleşme sonuçları
| Veri seti | Görüntü | Etiket dosyası | Etiketsiz görüntü | Görüntüsüz etiket | Boş etiket | Okunamayan etiket | Bozuk görüntü | EXIF döndürmeli |
|---|---|---|---|---|---|---|---|---|
| construction_ppe | 1000 | 1000 | 0 | 0 | 0 | 0 | 0 | 0 |
| mendeley_ppe | 2286 | 2286 | 0 | 0 | 0 | 0 | 0 | 0 |
| site_safety | 2799 | 2799 | 0 | 0 | 23 | 0 | 0 | 0 |

Eşleşme aynı dosya adı (uzantısız) üzerinden yapılmıştır. Boş veya eksik etiketler hata sayılmamıştır (kasıtlı negatif örnek olabilir); yalnızca raporlanmıştır.

## 7. Bounding box doğrulama sonuçları
Her satır `class_id x_center y_center width height` (5 sütun) biçiminde beklenir; segmentation/OBB benzeri ek sütunlar detection etiketi olarak işlenmez. Toleranslar: normalize aralık ve görüntü sınırı için 1e-4, çok küçük kutu için 2 piksel, aynı sınıfta kopya kutu için IoU≥0.95, çelişkili sınıf örtüşmesi için IoU≥0.3.

| Kontrol (sorun kodu) | construction_ppe | mendeley_ppe | site_safety |
|---|---|---|---|
| UNSUPPORTED_FORMAT | 0 | 0 | 0 |
| BAD_COLUMN_COUNT | 0 | 0 | 0 |
| NON_NUMERIC | 0 | 0 | 0 |
| CLASS_ID_INVALID | 0 | 0 | 0 |
| CLASS_ID_UNDEFINED | 0 | 0 | 0 |
| COORD_OUT_OF_RANGE | 0 | 0 | 0 |
| BOX_NONPOSITIVE_SIZE | 0 | 0 | 0 |
| BOX_OUTSIDE_IMAGE | 0 | 0 | 0 |
| LABEL_UNREADABLE | 0 | 0 | 0 |
| IMAGE_UNREADABLE | 0 | 0 | 0 |

Yukarıdaki kontrollerde bulunan sorun sayısı 0 ise ilgili hata türü TESPİT EDİLMEMİŞTİR. Sonuç, doğrulayıcıdan bağımsız yazılmış TEK SEFERLİK bir numpy kontrolüyle (betik olarak saklanmadı) çapraz denetlenmiştir: sütun sayıları, sınıf ID'leri ve kutu sınırları uyumlu bulunmuştur.

**Matematiksel geçerlilik, semantik doğruluğu garanti etmez**: kutunun geçerli olması doğru nesneyi gösterdiği anlamına gelmez (bkz. bölüm 11-12).

## 8. Tespit edilen etiket sorunları
| Sorun kodu | Düzey | construction_ppe | mendeley_ppe | site_safety |
|---|---|---|---|---|
| BOX_VERY_SMALL | uyarı | 0 | 0 | 21 |
| CONFLICTING_OVERLAP | uyarı | 0 | 0 | 3 |
| DUPLICATE_BOX | uyarı | 0 | 0 | 4 |
| LABEL_EMPTY | bilgi | 0 | 0 | 23 |
| NEAR_DUPLICATE_BOX | uyarı | 0 | 0 | 1 |

Ayrıntı: `annotation_issues.csv` (dosya yolu, satır numarası, açıklama). Hiçbir dosya düzeltilmemiş veya silinmemiştir.

- `BOX_VERY_SMALL` örnekleri: `ka_01181_png_jpg.rf.154ee4ef254eabd62e316be50470c578.txt` satır 18 (1.5x5.0 piksel); `ka_01181_png_jpg.rf.154ee4ef254eabd62e316be50470c578.txt` satır 22 (1.5x3.5 piksel); `ka_01181_png_jpg.rf.154ee4ef254eabd62e316be50470c578.txt` satır 23 (1.0x4.5 piksel)
- `DUPLICATE_BOX` örnekleri: `construction-855-_jpg.rf.723d19d48345a0cd8e3ab92a56c270e6.txt` satır 5 (Satır 4 ile birebir aynı kutu); `mask-wearing-1632932085584_png_jpg.rf.960ad3dacd717d22e329aa5c790b1a1f.txt` satır 12 (Satır 11 ile birebir aynı kutu); `n457047_jpg.rf.9fff0be5b2f8c422e37edabbe610a5b9.txt` satır 5 (Satır 4 ile birebir aynı kutu)
- `NEAR_DUPLICATE_BOX` örnekleri: `Mask2_mov-44_jpg.rf.d5f53eda27d49519afadac22d897e455.txt` satır 5 (Satır 4 ile IoU>=0.95 (aynı sınıf))
- `CONFLICTING_OVERLAP` örnekleri: `IMG_5024_mp4-2_jpg.rf.77ca6ff11b6d3b968a249280ea18f1ef.txt` satır 17 (Sınıf 7 (satır 17) ile sınıf 4 (satır 16) aynı bölgede, IoU=0.31); `construction-5-_mp4-75_jpg.rf.4df4bcaa924d84cd4d28992b0c3a5069.txt` satır 20 (Sınıf 7 (satır 20) ile sınıf 4 (satır 19) aynı bölgede, IoU=0.42); `construction-897-_jpg.rf.c7afbfac58183e3c490473a60000b144.txt` satır 9 (Sınıf 7 (satır 9) ile sınıf 4 (satır 16) aynı bölgede, IoU=0.45)

**Görsel ön gözlemler (kod ile doğrulanamaz, KULLANICI DOĞRULAMASI GEREKİR):**
- `construction_ppe`: `helmet` kutuları baret nesnesini, `no_helmet` kutuları baretsiz baş bölgesini, `vest`/`no_vest` kutuları gövde bölgesini sarıyor görünüyor. `no_helmet` kutuları helmet kutularına göre belirgin şekilde küçük/bulanık örnekler içeriyor.
- `mendeley_ppe`: `Helmet` ve `NoHelmet` kutuları baş+baret / baş-yüz bölgesini birlikte kapsıyor görünüyor (baret nesnesine sıkı değil); `NoVest` örneklerinin birçoğu inşaat dışı sıradan kıyafetli insanlar; bazı yüzler piksellenmiş (mozaik).
- `site_safety`: Çok sayıda çok küçük, bulanık veya görüntü kenarında kırpılmış kutu var (Roboflow augmentation/stretch etkisi). Birkaç örnekte `NO-Hardhat` kutusu içinde baret ve `NO-Safety Vest` kutusunda yelek görünüyor, `Safety Vest` kutularında yelek olmayan (beyaz/koyu giysi) örnekler var: **olası etiket gürültüsü**.

## 9. Tekrarlanan görüntüler
Toplam 6085 görüntü SHA-256 ve 64-bit dHash ile işlendi (okunamayan: 0).

- **Birebir aynı (SHA-256) dosya çifti: 0.** (Veri seti içi, bölümler arası ve veri setleri arası.)
- **yakın kopya adayı (dHash ≤ 5/64): 54 çift, 79 görüntü.**
    - `cross_dataset:mendeley_ppe|site_safety`: 4
    - `within_dataset_cross_split:test|val`: 2
    - `within_dataset_cross_split:train|val`: 10
    - `within_dataset_same_split:train|train`: 32
    - `within_dataset_same_split:val|val`: 6
- **yatay çevrilmiş yakın kopya adayı (dHash ≤ 5/64): 29 çift, 53 görüntü.**
    - `within_dataset_cross_split:train|val`: 7
    - `within_dataset_same_split:train|train`: 20
    - `within_dataset_same_split:val|val`: 2
- Eşik duyarlılığı (çift sayısı, ≤ mesafe): {'<=0': 35, '<=2': 53, '<=5': 83, '<=8': 131, '<=10': 234, '<=12': 737}. Eşik büyüdükçe yanlış pozitif artar.
- Aynı **kaynak adını** paylaşan görüntü grupları (ipucu): {'cross_dataset': 8, 'within_dataset_same_split': 503, 'within_dataset_cross_split': 8}.
- `construction_ppe/manifest.jsonl` kontrolü: 1000 girdi, SHA-256 uyuşmazlığı 0, bölüm uyuşmazlığı 0, kutu sayısı uyuşmazlığı 0, yerelde eksik 0.

Veri setleri arası yakın kopya adayları (inceleme gerekir; kesin aynı görüntü değildir):

| Tür | Mesafe | Dosya A | Dosya B |
|---|---|---|---|
| near | 1 | `data/raw/mendeley_ppe/train/images/00596_jpg.rf.e056867eae13eca16cc7cb8fe55be7df.jpg` | `data/raw/site_safety/valid/images/00596_jpg.rf.d030c5d98b937d080d75db1c1b269a84.jpg` |
| near | 1 | `data/raw/mendeley_ppe/valid/images/02912_jpg.rf.d68699bf2aa81dc0c86ab630b024a50c.jpg` | `data/raw/site_safety/valid/images/ppe_1228_jpg.rf.29b714c415ea74df3160f56a603463a0.jpg` |
| near | 3 | `data/raw/mendeley_ppe/train/images/00624_jpg.rf.6d31c7a4e92fe52fa01a7537024167a4.jpg` | `data/raw/site_safety/valid/images/ppe_0665_jpg.rf.1dad479f7f54b2a7127cf18ef74ffd85.jpg` |
| near | 3 | `data/raw/mendeley_ppe/train/images/02646_jpg.rf.4b5bddb3448f6298cb316b95d256f800.jpg` | `data/raw/site_safety/valid/images/02646_jpg.rf.5c93ba95bdc03808bcf872c7218ac5ef.jpg` |

**Önemli:** Yakın kopya eşleşmeleri algısal özet (dHash) ile bulunmuştur; kesin aynı görüntü kabul EDİLMEMİŞTİR. Kırpma, döndürme veya güçlü renk değişimi içeren augmentation türevleri dHash ile kaçırılmış olabilir; `site_safety` içinde aynı kaynaktan türeyen çok sayıda görüntü grubu kaynak adı ipucuyla görülmektedir.

## 10. Veri bölünmelerinin durumu
| Veri seti | train | val | test |
|---|---|---|---|
| construction_ppe | 800 (%80.0) | 200 (%20.0) | yok |
| mendeley_ppe | 1829 (%80.0) | 457 (%20.0) | yok |
| site_safety | 2603 (%93.0) | 114 (%4.1) | 82 (%2.9) |

- **construction_ppe**: bölümler arası yakın kopya adayı 1 çift {'train|val': 1}; bölümler arası ortak kaynak adı grubu 0; aynı bölüm içinde ortak kaynak adı grubu 0.
- **mendeley_ppe**: bölümler arası yakın kopya adayı 16 çift {'train|val': 16}; bölümler arası ortak kaynak adı grubu 0; aynı bölüm içinde ortak kaynak adı grubu 0.
- **site_safety**: bölümler arası yakın kopya adayı 2 çift {'test|val': 2}; bölümler arası ortak kaynak adı grubu 8; aynı bölüm içinde ortak kaynak adı grubu 503.

- Ayrı bir **test** bölümü yalnızca `site_safety` içinde vardır (ve çok küçüktür); `construction_ppe` ve `mendeley_ppe` yalnızca train/val içerir. `construction_ppe` dokümantasyonu nihai değerlendirme için bağımsız saha görüntüsü önerir.
- `mendeley_ppe` bölmesi, arşivdeki betiğe göre rastgele %20 ayırmadır; aynı sahneden gelen görüntüler train ve val'e dağılmış olabilir (veri sızıntısı riski, yukarıdaki bölümler arası eşleşmeler).
- Üç kaynak birleştirilirse: (i) veri setleri arası yakın kopya/ortak kaynak adları nedeniyle bir kaynağın train görüntüsü diğerinin val/test görüntüsüyle örtüşebilir; (ii) `site_safety` train içindeki augmentation türevleri etkin örnek sayısını ve doğrulama güvenilirliğini etkiler. Mevcut bölmeler DEĞİŞTİRİLMEMİŞTİR.

## 11. Veri setleri arasındaki sınıf uyumu
Hedef sınıflar: 0=helmet, 1=no_helmet, 2=vest, 3=no_vest. Aşağıdaki tablo **ADAY** eşlemedir; hiçbir etiket ID'si değiştirilmemiştir.

| Veri seti | Kaynak ID:ad | Hedef | Durum | Kutu semantiği | Kanıt / risk |
|---|---|---|---|---|---|
| construction_ppe | 0:helmet | helmet | confirmed | Baret nesnesinin kendisi (sıkı kutu); kişiye göre yükseklik 0.13, en-boy h/w=0.85 (baş-baret genişten yüksek değil) | Resmi doküman + semantic_metrics.json (person_relative) + görsel inceleme — Risk: Taşınan baretler 'baret takıyor' kararını yanıltabilir; kişi-ekipman eşleştirmesinde baş bölgesiyle konum kontrolü gerekir |
| construction_ppe | 6:no_helmet | no_helmet | confirmed | Baretsiz tüm BAŞ (saç/yüz); kişiye göre yükseklik 0.17, h/w=1.10 | Resmi doküman ('visible head without a helmet') + ölçü + görsel — Risk: Başlıklı (şapka) baş etiketi kullanımı belirsiz |
| construction_ppe | 2:vest | vest | confirmed | Yelekli gövde; kişi yüksekliğinin ~%40'ı, üstten %15 aşağıda başlıyor | semantic_metrics.json + görsel (resmi doküman vest için kutu tanımı vermiyor) — Risk: Doküman vest kutusunu tanımlamıyor; kanıt ölçü+görsel |
| construction_ppe | 11:no_vest | no_vest | confirmed | Yeleksiz GÖVDE; kişiye göre yükseklik 0.40 (vest ile aynı), tüm kişi değil | Resmi doküman ('visible torso without high-visibility safety clothing') + ölçü + görsel — Risk: Oklüzyon tek başına 'yelek yok' kanıtı değil (dokümanın kendi uyarısı) |
| construction_ppe | 1:gloves | (hedef dışı) | incompatible | Eldiven | Sınıf adı/YAML — Risk: Birleştirmede dışarıda bırakılmalı |
| construction_ppe | 3:boots | (hedef dışı) | incompatible | Bot | Sınıf adı/YAML |
| construction_ppe | 4:goggles | (hedef dışı) | incompatible | Gözlük | Sınıf adı/YAML |
| construction_ppe | 5:Person | (hedef dışı) | incompatible | Tüm kişi | Sınıf adı/YAML — Risk: İnsan tespiti projede ayrı modülde (human_detection) yapılacak; kullanım kararı bekliyor |
| construction_ppe | 7:no_goggle | (hedef dışı) | incompatible | Gözlüksüz göz bölgesi | Sınıf adı/YAML |
| construction_ppe | 8:no_gloves | (hedef dışı) | incompatible | Eldivensiz el | Sınıf adı/YAML |
| construction_ppe | 9:no_boots | (hedef dışı) | incompatible | Botsuz ayak | Sınıf adı/YAML |
| construction_ppe | 10:forklift | (hedef dışı) | incompatible | Forklift | Sınıf adı/YAML |
| mendeley_ppe | 0:Helmet | helmet | incompatible | Baret + yüzün/başın tamamı (baş kutusu); baş/gövde oranı 0.53 (Construction helmet 0.33), h/w=1.19 (Construction 0.85) | semantic_metrics.json (single_head_torso_pairs) + görsel — Risk: Doğrudan helmet olarak birleştirilirse kutu tanımı tutarsız olur; dönüştürme/yeniden etiketleme kararı gerekir (UYGULANMADI) |
| mendeley_ppe | 1:NoHelmet | no_helmet | confirmed | Baretsiz tüm baş (saç/yüz); baş/gövde oranı 0.44 (Construction no_helmet 0.43), h/w=1.20 (1.10) | Nicel baş/gövde oranı + görsel — Risk: Alan kayması (inşaat dışı); 'Helmet' sınıfıyla aynı kutu tanımını paylaşıyor (baş), Construction no_helmet ile uyumlu |
| mendeley_ppe | 3:Vest | vest | confirmed | Yelekli gövde (torso); torso/baş boyutları Construction ile tutarlı (NoHelmet bazlı oran 0.44 vs 0.43) | Baş/gövde oranı + görsel — Risk: Dolaylı nicel kanıt; birkaç kutu yelek parçası veya kolaj kenarı |
| mendeley_ppe | 2:NoVest | no_vest | confirmed | Yeleksiz gövde (göğüs-bel) | Baş/gövde oranı + görsel — Risk: Alan kayması; modelin 'her sıradan kıyafet = no_vest' öğrenme riski |
| site_safety | 0:Hardhat | helmet | confirmed | Baret nesnesi (sıkı kutu), h/w=1.08 | label_review.csv (AH01-AH24) + semantic_metrics.json — Risk: Etiket gürültüsü ~%4 olası hata, ~%21 doğrulanamayan (örneklem küçük: n=24) |
| site_safety | 2:NO-Hardhat | no_helmet | incompatible | Karışık: saç/alın ŞERİDİ (h/w medyan 0.80, yaygın <0.5), yüzün üst yarısı ve bazen tüm baş; Construction no_helmet h/w=1.10 | label_review.csv (AN01-AN24) + semantic_metrics.json (aspect, person_relative) — Risk: Doğrudan no_helmet'e birleştirilmemeli; filtrelenmiş alt küme (h/w>=0.9 ve yükseklik>=24 px) insan kontrolüyle değerlendirilebilir |
| site_safety | 7:Safety Vest | vest | confirmed | Yelek/gövde bölgesi, kişiye göre yükseklik 0.49 (kısmi/yakın çekim kişi kutuları nedeniyle yayılmış) | label_review.csv (AV01-AV24) + görsel — Risk: Örneklem küçük (n=24); parça/şerit kutular |
| site_safety | 4:NO-Safety Vest | no_vest | needs_review | Yeleksiz gövde/göğüs; çok sayıda kısmi (şerit), yakın çekim (maskeli yüz+tişört) veya belirsiz bölge | label_review.csv (AX01-AX24) + görsel — Risk: Olası hata %8, belirsiz %38: eşiği aşıyor; insan incelemesi/filtreleme olmadan kullanılmamalı |
| site_safety | 1:Mask | (hedef dışı) | incompatible | Maske | Sınıf adı/YAML |
| site_safety | 3:NO-Mask | (hedef dışı) | incompatible | Maskesiz yüz | Sınıf adı/YAML |
| site_safety | 5:Person | (hedef dışı) | incompatible | Kişi (çoğu kısmi/kırpılmış: h/w medyan 1.64) | Sınıf adı/YAML + semantic_metrics.json — Risk: İnsan tespiti için kullanım kararı bekliyor |
| site_safety | 6:Safety Cone | (hedef dışı) | incompatible | Güvenlik konisi | Sınıf adı/YAML |
| site_safety | 8:machinery | (hedef dışı) | incompatible | İş makinesi | Sınıf adı/YAML |
| site_safety | 9:vehicle | (hedef dışı) | incompatible | Araç | Sınıf adı/YAML |

Tam liste: `candidate_class_mapping.csv`. Durum anlamları (eksik giderme sonrası): **confirmed** = kutu tanımı hedefle tutarlı (nicel ölçü + görsel + dokümantasyon) ve örneklemde olası hata ≤%10; **needs_review** = tanım tutarlı görünüyor ama etiket kalitesi/örneklem belirsiz; **incompatible** = kutu tanımı hedefle tutarsız veya hedef dışı sınıf (doğrudan birleştirilmemeli). Tüm satırlar `user_review=Pending`. Ayrıntı: `semantic_validation_report.md`.

**Kutu geometrisi karşılaştırması (normalize; kutu semantiği için nesnel ipucu):**

| Hedef | Veri seti | Kaynak sınıf | Kutu | Medyan genişlik | Medyan yükseklik | Medyan alan | Medyan h/w |
|---|---|---|---|---|---|---|---|
| helmet | construction_ppe | helmet | 1576 | 0.0466 | 0.0574 | 0.00273 | 1.208 |
| helmet | mendeley_ppe | Helmet | 2421 | 0.0734 | 0.1204 | 0.0089 | 1.691 |
| helmet | site_safety | Hardhat | 3551 | 0.0437 | 0.0443 | 0.00201 | 1.077 |
| no_helmet | construction_ppe | no_helmet | 813 | 0.0219 | 0.0319 | 0.00068 | 1.5 |
| no_helmet | mendeley_ppe | NoHelmet | 1174 | 0.0715 | 0.1131 | 0.00825 | 1.659 |
| no_helmet | site_safety | NO-Hardhat | 2428 | 0.1 | 0.0781 | 0.00821 | 0.806 |
| no_vest | construction_ppe | no_vest | 567 | 0.1091 | 0.1983 | 0.02499 | 1.714 |
| no_vest | mendeley_ppe | NoVest | 1058 | 0.1235 | 0.2328 | 0.02936 | 1.845 |
| no_vest | site_safety | NO-Safety Vest | 4153 | 0.1469 | 0.175 | 0.02839 | 1.296 |
| vest | construction_ppe | vest | 1816 | 0.0637 | 0.1306 | 0.00831 | 1.974 |
| vest | mendeley_ppe | Vest | 1385 | 0.1446 | 0.2641 | 0.03818 | 1.816 |
| vest | site_safety | Safety Vest | 3258 | 0.074 | 0.125 | 0.00964 | 1.6 |

_Not: Medyan alan görüntü kadrajının yakınlığından da etkilenir (yakın çekim/uzak çekim); tek başına kanıt değil, görsel inceleme için ipucudur._

**Düzeltme:** Bu medyan-alan tablosu kadraj yakınlığından etkilendiği için kutu semantiği kanıtı olarak YANILTICIDIR (önceki taslaktaki 'no_helmet helmet'ten 4 kat küçük' sonucu bu yüzden hatalıydı). Kişi kutusuna göre göreli boyutlar ve baş/gövde oranlarıyla yapılan doğru karşılaştırma `semantic_validation_report.md` içindedir: Mendeley `Helmet` baş+baret (yüz dahil) olduğundan Construction/Roboflow baret nesnesiyle UYUMSUZ; Roboflow `NO-Hardhat` heterojen.

## 12. Manuel inceleme gerektiren noktalar
Güncel durum (eksik giderme sonrası): onay listesi `manual_review_checklist.md`, kalanlar `remaining_manual_reviews.md`. İlk taslaktaki 7 başlık (kutu semantiği, sınıf karşılaştırması, Roboflow gürültüsü, kopyalar, boş etiketler, eşleme onayı, lisans) analizle büyük ölçüde sonuçlandı; kullanıcıya yalnızca gerçek karar/onay gerektiren maddeler kaldı (bkz. §16).

## 13. Lisans ve kaynak bilgileri
| Veri seti | Lisans | Doğrulama durumu | Koşullar / notlar |
|---|---|---|---|
| construction_ppe | CC BY 4.0 | Doğrulandı: yerel README.md ve resmi docs sayfası | Atıf zorunlu; değişiklik yapıldıysa belirtilmeli; paylaşım ve uyarlamaya izin verir. |
| mendeley_ppe | CC BY 4.0 | Kısmen doğrulandı: resmi Mendeley sayfası ve API 'CC BY 4.0' diyor (yerel ZIP'in SHA-256'sı resmi API ile eşleşiyor); ÜST KAYNAK lisansları doğrulanamadı. Ayrıntı: configs/review/license_evidence.yaml | Sayfa, veri setinin GitHub/Kaggle/Roboflow ve gerçek fotoğraflardan derlendiğini belirtiyor; bu üst kaynakların lisansları belirtilmemiş (doğrulanamadı). |
| site_safety | CC BY 4.0 | Yalnızca YEREL dosyalardan (data.yaml, README.dataset.txt); resmi Roboflow sayfaları HTTP 403 verdi, çevrimiçi DOĞRULANAMADI. Ayrıntı: configs/review/license_evidence.yaml | README.dataset.txt, görüntülerin başka Roboflow Universe projelerinden ve iki YouTube videosundan türetildiğini söylüyor; üst kaynakların lisansları ayrıca doğrulanmadı. |

**Atıf metinleri:**

- Ultralytics Construction-PPE Dataset, Ultralytics team, November 2026, CC BY 4.0.
- Huang, M.-L.; Cheng, Y. Dataset of Personal Protective Equipment (PPE), Mendeley Data, v6, 2025. doi:10.17632/zkzghjvpn2.6
- Construction Site Safety (v27), Roboflow Universe, roboflow-universe-projects, CC BY 4.0, https://universe.roboflow.com/roboflow-universe-projects/construction-site-safety/dataset/27

Eksik giderme sonrası ayrıntılı kanıt tablosu: `license_verification_report.md`. Özet: Construction-PPE resmi kaynaktan doğrulandı; Mendeley lisansı resmi sayfa/API'den doğrulandı ve yerel ZIP'in SHA-256'sı resmi API ile eşleşti, ancak üst kaynak lisansları doğrulanamadı; Roboflow v27 lisansı YALNIZCA yerel dosyalardan biliniyor (resmi sayfa 403).


Notlar: CC BY 4.0 atıf zorunluluğu ve değişiklik belirtme koşulu getirir. Mendeley ve Roboflow veri setleri başka kaynaklardan derlenmiştir (README'lere göre); üst kaynakların lisansları bu aşamada DOĞRULANMADI. Hukuki bir görüş değildir; bitirme raporunda atıf yapılmadan önce kaynak sayfaları yeniden kontrol edilmelidir.

## 14. Gelecek eğitim aşamaları için öneriler
(Öneridir; hiçbiri bu aşamada uygulanmamıştır.)

- Nihai değerlendirme için hiçbir eğitim verisiyle örtüşmeyen, bağımsız bir test kümesi ayrılmalı (idealde hedef ortama benzer saha görüntüleri).
- Veri setleri birleştirilmeden önce kutu semantiği kararı verilmeli: özellikle baret için 'sıkı baret kutusu' ile 'baş+baret kutusu' ayrımı.
- Birleştirilirse bölme, kaynak görüntü (ortak kaynak adı) ve yakın kopya gruplarına göre grup bazlı yapılmalı; mevcut val/test bölümleri bu kontrol olmadan güvenilir kabul edilmemeli.
- `site_safety` etiket gürültüsü örnekleme ile elle taranmalı; gürültü yüksekse bu veri setinin yalnızca bölümü kullanılmalı veya dışarıda bırakılmalı.
- Sınıf dengesizliği eğitim aşamasında (örn. sınıf ağırlıkları, hedefli örnekleme) ele alınmalı; şu an uygulanmadı.
- Her kaynak için ayrı doğrulama metriği raporlanmalı (kaynak başına mAP), böylece alan kayması görülebilir.

## 15. Veri setlerinin genel kullanılabilirlik değerlendirmesi
Aşağıdaki değerlendirme teknik denetime ve görsel ÖN gözleme dayanır; kullanıcı onayı tamamlanana kadar kesin kabul edilmemelidir.

| Veri seti | Teknik biçim | Ön değerlendirme |
|---|---|---|
| construction_ppe | Biçim sorunu yok; manifest ile birebir tutarlı | En güvenilir aday. Küçük (1000 görüntü), ayrı test yok. Baret ve yelek sınıfları dokümante edilmiş; kutu tanımı sıkı baret nesnesi. |
| mendeley_ppe | Biçim sorunu yok | Ek veri olarak yararlı ama kutu semantiği (baş+baret) diğerlerinden farklı; rastgele bölme nedeniyle train/val sızıntı riski; alan çeşitliliği yüksek, birçok inşaat dışı sahne. |
| site_safety | Küçük uyarılar (çok küçük/kopya kutu, boş etiket) | Çeşitli ama augmentation türevleri, kırpılmış kutular ve olası etiket gürültüsü var; bölümler arası ortak kaynak grupları tespit edildi. Elle inceleme olmadan eğitime katılması riskli. |

**Sonuç:** Üç veri seti biçim olarak okunabilir ve bütünlük açısından sağlamdır; ancak veri setlerinin birleştirilmesi için ön koşullar henüz karşılanmamıştır: (1) kutu semantiği ve aday sınıf eşlemesinin kullanıcı onayı, (2) olası etiket gürültüsünün incelenmesi, (3) sızıntı riskine karşı bölme stratejisi, (4) lisans doğrulamalarının tamamlanması. Bu nedenle birleştirme için **koşullu / henüz hazır değil** durumundadır.

## 16. Eksik giderme sonuçları (Aşama 2 güncellemesi)

İlk raporda çözülemeyen konular ek analizlerle yeniden ele alındı. **Önceki bulgular korunmuştur**; aşağıdaki maddeler hangi bölümlerin güncellendiğini ve yeni sonuçların kaynağını gösterir.

- **Kutu semantiği** (`semantic_validation_report.md`, `semantic_metrics.json`): kişi kutusuna göre göreli boyut ve baş/gövde oranı + 3×4 görsel karşılaştırma. Hedefe eşlenen 12 sınıfın durumu: {'confirmed': 9, 'incompatible': 2, 'needs_review': 1}. Mendeley `Helmet` baş+baret (yüz dahil) olduğu için `incompatible`; Roboflow `NO-Hardhat` heterojen olduğu için `incompatible`; Roboflow `NO-Safety Vest` `needs_review`. Önceki taslaktaki medyan-alan temelli bazı yorumlar DÜZELTİLDİ (§11 notu).
- **Roboflow şüpheli etiketler** (`roboflow_label_review_report.md`, `label_review.csv`, `roboflow_cleanup_suggestions.md`): 238 öğe → {'Needs Manual Review': 66, 'Valid Annotation': 136, 'Likely Error': 32, 'Confirmed Error': 4}. Bölüm 8'deki 'olası etiket gürültüsü' ifadesi nicelleştirildi: rastgele 96 kutuluk örneklemde olası hata ≈%5, belirsiz ≈%26.
- **Yakın kopya** (`duplicate_verification_report.md`, `duplicate_verification.csv`): çok yöntemli doğrulama (4621 aday çift). Mendeley–Roboflow 4 çiftin 4'ü AYNI kaynak fotoğraf; val↔test 2 çift aynı videodan ardışık kareler. Bölümler arası eşleşmeli görüntü oranı (alt sınır): {'mendeley_ppe/train': '%4', 'mendeley_ppe/val': '%9', 'site_safety/test': '%85', 'site_safety/train': '%20', 'site_safety/val': '%79'}. Bölüm 9-10'daki dHash tabanlı sayılar bu doğrulamayla GÜNCELLENMİŞTİR; Roboflow val/test sızıntısı yüksek öncelikli risktir.
- **Lisans** (`license_verification_report.md`): Mendeley için resmi sayfa/API lisansı doğrulandı ve yerel ZIP'in SHA-256'sı resmi API ile eşleşti; Roboflow v27 resmi sayfası HTTP 403 verdi (yerel metadata CC BY 4.0, bağımsız doğrulama yok); üst kaynak lisansları doğrulanamadı.
- **Manuel kontrol listesi** (`manual_review_checklist.md`, `remaining_manual_reviews.md`): 59 maddenin 46'si analizle çözüldü (A+B), 2 teknik olarak çözülemedi, 11 kullanıcı kararı gerektiriyor.

**Genel kullanılabilirlik değerlendirmesi (güncel, §15'in yerine geçer):** Üç veri seti biçim olarak sağlamdır. Birleştirme için engeller: (1) Mendeley `Helmet` ve Roboflow `NO-Hardhat` kutu tanımı uyumsuzluğu, (2) Roboflow val/test'in büyük ölçüde train ile aynı videodan/fotoğraftan türemesi (grup bazlı yeniden bölme gerekir), (3) Roboflow `NO-Safety Vest` etiket kalitesi, (4) lisans belirsizlikleri. Bu nedenle birleştirme **henüz hazır değil**; Construction-PPE tek başına en güvenilir ve (resmi lisansı doğrulanmış) çekirdek veri setidir.


## 17. Aşama 2B sonuçları (veri kullanım politikası) — tarihsel özet

Aşama 2 eksik giderme sonrası kalan 11 karar maddesi ve 2 lisans konusu ele alındı; sonuçlar `final_dataset_decisions.md` içindedir. **Önceki bulgular korunmuştur**. 2B'deki sayılar ve kapanış ifadesi §18 (Aşama 2C) ile GÜNCELLENMİŞTİR: 2B'de onaylı kapsam 1000 görüntü / vest 1816 kutu ve F01/F02 `pending_external_verification` idi; Construction-PPE için %70/15/15 bölme önizlemesi vardı (artık geçersiz).

- **Sınıf kararları:** Mendeley `Helmet` (E05), Roboflow `NO-Hardhat` (E10) ve `NO-Safety Vest` (E12) birleşik eğitim adaylarından dışlandı; görüntü seviyesinde karantina/yeniden etiketleme listeleri üretildi.
- **Veri sızıntısı:** kaynak-grup bazlı bölme politikası ve manifest (`source_group_manifest.csv`); Roboflow val/test bağımsız değerlendirme için kullanılamaz.

## 18. Aşama 2C sonuçları (nihai kontrol, güncel)

- **Onaylı kapsam:** yalnızca Construction-PPE. 1000 görüntünün durumu {'approved': 999, 'needs_reannotation': 1}; nihai bölme {'train': 800, 'val': 199} (`construction_ppe_split_manifest.csv`); kullanılabilir görüntülerde hedef kutular {'vest': 1815, 'helmet': 1576, 'no_helmet': 813, 'no_vest': 567}. Mendeley ve Roboflow beklemede (`pending_license_verification`, `hold_reasons`), ham verileri ve analiz kayıtları korunuyor.
- **Sızıntı kararı:** Construction-PPE'deki bölümler arası tek birim (aynı taban karenin 3 üretken varyantı) train'e alındı; yayıncı bölünmesi korundu.
- **Eksik anotasyon riski:** 63 görüntü işaretlendi; 1 görüntü görsel olarak doğrulanan eksik etiket nedeniyle `needs_reannotation`.
- **Kontrol listesi (59):** {'resolved_verified': 43, 'resolved_excluded': 13, 'external_verification_pending_source_excluded': 2, 'pending_external_verification': 1}. F01/F02: `external_verification_pending_source_excluded` (sorun çözülmedi; kaynak dışlandı).
- **Kapanış:** `STAGE_2_FINAL_REPORT.md`; bağımsız test stratejisi `docs/evaluation_plan.md`.
