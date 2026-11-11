# Aşama 2 Nihai Kapanış Raporu (Aşama 2C)

_Yapay Zekâ Destekli Çoklu Çalışan Tanıma ve İSG Denetim Sistemi. Tarih: 2026-11-11._

**Bu aşamada yapılmayanlar:** ham veri değiştirilmedi, veri setleri birleştirilmedi, model eğitilmedi, saha görüntüsü toplanmadı,
commit/push yapılmadı. Görsel kararlar model destekli, tek değerlendiricili incelemelerdir; kullanıcı doğrulaması yerine geçmez.

**Durum: `READY_FOR_USER_APPROVAL`.** Gerekçe için bkz. §11.

---

## 1. Veri setlerinin son durumu

| Veri seti | Görüntü (yerel) | Lisans doğrulaması | Hak durumu | Kullanım | Bekleme gerekçeleri (`hold_reasons`) | Ham veri |
|---|---|---|---|---|---|---|
| Ultralytics Construction-PPE | 1000 | `verified_official` | `verified_publisher_original` | **`approved_scope`** (onaylı temel kaynak) | — | Korunuyor |
| Mendeley PPE v6 | 2286 | `partially_verified` | `rights_unverified` | `pending_license_verification` (**beklemede**) | `rights_unverified`, `annotation_quality`, `class_incompatibility` | Korunuyor, silinmedi |
| Roboflow Construction Site Safety v27 | 2799 | `verified_local_only` | `rights_unverified` | `pending_license_verification` (**beklemede**) | `license_unverified`, `rights_unverified`, `label_noise`, `split_leakage`, `class_incompatibility` | Korunuyor, silinmedi |

Politika: `configs/dataset_usage_policy.yaml` (`policy_version: 2`). Validator (`validate_policy`) şunları kodla reddeder:

- beklemedeki bir kaynağın bekleme gerekçesi olmadan tanımlanması;
- beklemedeki kaynağın koşulsuz `class_map` alması;
- beklemedeki kaynakta `raw_data_retention: keep` dışında bir değer;
- lisansı/hakkı doğrulanmamış bir kaynağın `approved_scope` yapılması.

Mendeley ve Roboflow'un bütün analiz kayıtları korunuyor ve yeniden değerlendirmede kullanılabilir:

- `image_usage_decisions.csv` (`status_if_rights_verified` ön kararları)
- `quarantine_candidates.csv`
- `needs_reannotation_list.csv`
- `label_review.csv`
- `duplicate_verification.csv`
- `source_group_*.csv`
- lisans ve semantik raporlar

## 2. Onaylı veri kapsamı ve görüntü sayıları

**Onaylı kapsam yalnızca Construction-PPE'dir.** Nihai bölmenin tek kaynağı
[`construction_ppe_split_manifest.csv`](construction_ppe_split_manifest.csv) dosyasının `final_split` sütunudur. Dosyalar taşınmadı.

| | Görüntü | helmet | no_helmet | vest | no_vest |
|---|---|---|---|---|---|
| **train (nihai)** | **800** | 1282 | 635 | 1475 | 438 |
| **val (nihai)** | **199** | 294 | 178 | 340 | 129 |
| **Kullanılabilir toplam** | **999** | 1576 | 813 | 1815 | 567 |
| Kullanım dışı (`needs_reannotation`) | 1 | 0 | 0 | 1 | 0 |
| Yayıncı toplamı (1000 görüntü) | 1000 | 1576 | 813 | 1816 | 567 |

Görüntü düzeyinde sınıf varlığı:

| Bölüm | helmet | no_helmet | vest | no_vest | Hedefsiz (geçerli negatif) |
|---|---|---|---|---|---|
| train | 613 | 338 | 643 | 301 | 10 (8 forklift + 2 depo sahnesi; görsel doğrulandı) |
| val | 146 | 90 | 157 | 83 | 0 |

Orijinal bölünmeye göre iki değişiklik var:

1. `val/source-9e10b040…` görüntüsü, kaynak birimi nedeniyle **train'e** atandı (§6).
2. `train/ultralytics-ppe-nov2026-690` görüntüsü görünür bir hedef etiketsiz olduğu için **kullanım dışı** bırakıldı (§5).

Nihai val, orijinal val'in 199 görüntülük alt kümesidir. Test bölümü yoktur; test, bağımsız saha testidir (§7).

**Görev 1 doğrulamaları** (`stage2c_integrity_checks.json`):

- **T1.1:** 1000/1000 görüntü yayıncı `manifest.jsonl` dosyasıyla eşleşiyor. Ad, bölüm (800/200), SHA-256 ve anotasyon sayısı
  (etiket satırı) birebir aynı. Eksik, fazla veya uyuşmayan dosya yok.
- **T1.2:** Dört hedef sınıf, etiket dosyalarından yeniden sayıldı: helmet 1576, no_helmet 813, vest 1816, no_vest 567
  (1000 görüntü). Bu değerler 2B raporuyla aynı. Kullanılabilir 999 görüntüdeki sayılar politika özetiyle aynı.
  vest 1815 farkının nedeni, kullanım dışı 690 görüntüsündeki 1 vest kutusudur.

## 3. Kullanılabilecek hedef sınıflar

| Hedef | Construction-PPE kaynağı | Durum |
|---|---|---|
| 0 helmet | `helmet` (0) | Onaylı. Not: elde taşınan baret de `helmet` kutusudur (sınıf "baret nesnesi" demektir) |
| 1 no_helmet | `no_helmet` (6) | Onaylı |
| 2 vest | `vest` (2) | Onaylı |
| 3 no_vest | `no_vest` (11) | Onaylı |
| — | gloves, boots, goggles, Person, no_goggle, no_gloves, no_boots, forklift | Hedef dışı, yok sayılır |

Beklemedeki kaynaklarda dışlanan sınıflar (E05 Mendeley `Helmet`, E10 Roboflow `NO-Hardhat`, E12 Roboflow `NO-Safety Vest`)
dışlanmış olarak kalır. Beklemedeki kaynaklar ileride onaylansa bile bu sınıflar doğrudan kullanılmaz.

## 4. Dışlanan kaynaklar ve gerekçeleri

| Kaynak | Gerekçe | Sonuç |
|---|---|---|
| Mendeley PPE v6 | **Haklar:** veri seti lisansı CC BY 4.0 doğrulandı, ancak görseller GitHub/Kaggle/Roboflow ve stok fotoğraflardan derlenmiş (filigranlar görüldü); görsel bazında hak doğrulanamadı. **Anotasyon:** kalabalık sahnelerde eksik etiket var; `Helmet` kutusu baş+baret olduğu için hedef tanımıyla uyumsuz (E05) | Beklemede; ön karar (hak doğrulanırsa): 637 approved / 1167 needs_reannotation / 482 quarantine |
| Roboflow v27 | **Lisans:** resmi sayfa HTTP 403 verdi; lisans yalnızca yerel metadatada geçiyor; üst projeler ve YouTube kaynaklarının lisansı bilinmiyor. **Etiket:** `NO-Hardhat`/`NO-Safety Vest` gürültülü; kişilerin yalnızca %47'si eksiksiz etiketli. **Sızıntı:** val 92/114 ve test 70/82 görüntü train ile aynı kaynağı paylaşıyor | Beklemede; ön karar: 122 approved / 348 needs_reannotation / 2329 quarantine |

**Kodla doğrulandı (T6.1):**

- Mendeley ve Roboflow'dan onaylı durumda olan veya `final_split` alan görüntü sayısı **0**.
- Bölme manifestinde yalnızca `data/raw/construction_ppe/` yolları var.
- Onaylı görüntü kümesi, bölmedeki görüntü kümesiyle birebir aynı.
- 2B'deki Construction-PPE %70/15/15 önizlemesi kaldırıldı.

## 5. Etiket kalitesi değerlendirmesi

**Construction-PPE (onaylı):**

- 2B'de crop, karşılaştırma ve 4 contact sheet sayfası incelendi. Sınıflar tutarlı, kutular nesne üzerinde; yanlış sınıf görülmedi.
- **2C eksik anotasyon taraması (`CP-PERSON-ATTR-INCOMPLETE`):** toplam 2443 Person kutusu var; ≥ 60 px olanlar denetlendi.
  Bunlardan 68 kişide (63 görüntü) baş veya gövde etiketi atanamıyor (tüm Person kutularının yaklaşık %97'si sorunsuz).
  - Sezgisel neden kodları: baş kadraj dışı 19, baş etiketi var ama atanamadı 13, kenarda kesik kişi 7, küçük kişi 12,
    açıklanamayan baş 17 ve açıklanamayan gövde 9.
  - 68 kırpıntının hepsine ve 11 görüntüye tam çözünürlükte bakıldı. Büyük çoğunluk kadraj dışı veya örtülü baş/gövde, bulanık
    arka plan figürü ya da atama sezgiselinin kaçırdığı mevcut etiket.
  - **1 gerçek eksik etiket bulundu:** `ultralytics-ppe-nov2026-690`. Görünür baretsiz baş var ama `no_helmet` etiketi yok.
    Görüntü `needs_reannotation` (kural `CP-VISIBLE-TARGET-UNLABELED`), eğitim ve değerlendirme dışı.
  - Diğer 62 görüntü kullanımda kalır ama **işaretlidir** (`annotation_risk` sütunu; train 47, val 15). Açıklanamayan kod
    taşıyan 22 görüntü (train 18, val 4) Aşama 8'deki ilk YOLO eğitiminden önce insan incelemesine alınacak (§12). 2D'de bu 22 görüntü için inceleme paketi hazırlandı (§13).
  - Kodla doğrulandı (T6.6): işaretler bağımsız yeniden hesaplamayla birebir aynı, override edilen görüntü bölmede yok.
- Hedef kutusu olmayan 10 görüntünün hepsi görsel incelendi ve geçerli negatif bulundu.
- Görüntülerin bir bölümü yapay üretilmiş ("human-collected and generated"). Ad önekine göre dağılım: `source-*` 293,
  `ultralytics-ppe-nov2026-*` 707. Hangi görüntülerin üretilmiş olduğu yayıncı tarafından görüntü bazında belgelenmemiştir.

**Beklemedeki kaynaklar:** 2B bulguları geçerliliğini koruyor: Mendeley'de eksik etiket ve E05; Roboflow'da E10/E12, küçük ve
çelişkili kutular, eksik kişi etiketi. Karantina ve yeniden etiketleme adayı toplamı 4327 (Construction-690 eklendi).

## 6. Veri sızıntısı önleme stratejisi

**Construction-PPE'deki bağlantılı üç train/val görüntüsü (Görev 1.3–1.4).**
Kanıt: `stage2c_review/_construction_cross_split.jpg`, `configs/review/stage2c_decisions.yaml`.

| Görüntü | Orijinal | İçerik |
|---|---|---|
| `train/source-3978abe9…` | train | Baretli, yelekli, gözlüklü işçi |
| `train/source-efc23273…` | train | Aynı kişi ve aynı kare, yelek kaldırılmış (no_vest) |
| `val/source-9e10b040…` | val | Aynı arka plan, kamera ve poz; yüzü değiştirilmiş baretsiz ve yeleksiz kişi (no_helmet, no_vest) |

- **Bulgu:** Üç görüntü, aynı taban karenin üretken (KKD/yüz düzenlenmiş) varyantlarıdır. ORB inlier değerleri 351–726,
  NCC 0.83–0.91.
- **Risk:** Görüntü başına yüksek. Val görüntüsü arka plan, poz ve kutu konumlarını train ile paylaştığı için ölçüm iyimser
  olur. Küme düzeyinde düşük: 1/200 val görüntüsü ve 2 hedef kutu.
- **Karar:** Birim çoğunluk bölümü olan **train'e** alındı (G4). Veri kaybı yok ve val, orijinal val'in alt kümesi olarak kaldı.
- Train içindeki ikinci üretken çift (`40471b5a…`, `9ec80cef…`) sızıntı oluşturmuyor ve birlikte tutuluyor.
- **Ek tarama:** 1000 görüntünün tüm çiftleri (499.500) küçük resim NCC ile tarandı.
  - ≥ 0.85 eşiğini geçen yalnızca bilinen 2 çift var.
  - 0.80–0.85 aralığındaki adaylardan bilinen çift dışındakiler görsel incelendi; hepsi farklı sahneler.
  - Yeni ilişki bulunmadı. Bu tarama bir alt sınırdır.

**Gelecek için uygulanabilir politika (Görev 1.5):** `leakage.group_integrity_rules`

| Kural | İçerik |
|---|---|
| G1 | Bölmeden önce yeni görüntüler kopya/kaynak taramasından geçer: SHA-256, pHash/dHash, ORB tam+karo, NCC |
| G2 | Üretken varyantlar sert kenardır ve aynı birimde tutulur |
| G3 | Saha verisinde bölme birimi kayıt oturumudur (kamera + gün + konum) |
| G4 | Birimler atomiktir; çelişen birim çoğunluk bölümüne gider, eşitlikte train |
| G5 | Test oturumları eğitim/val ile hiçbir kaynak, oturum veya kamera-gün paylaşmaz |
| G6 | `unit_integrity_violations() == {}` olmadan manifest eğitimde kullanılamaz |

Kod: `final_split_assignment` ve `unit_integrity_violations` (`usage_policy.py`). Kodla doğrulandı (T6.3, T6.4):

- Her bölümler arası birim için kayıtlı bir düzeltme var.
- Nihai bölmede, kenar düzeyinde ve beklemedeki önizlemede bölünmüş birim **yok**.

Roboflow'un val/test bölümleri bağımsız test olarak kullanılmaz (kaynak zaten beklemede).

## 7. Bağımsız test planı

[`docs/evaluation_plan.md`](../../docs/evaluation_plan.md) belgesi şunları tanımlar:

- **Bölümler:** train 800 / val 199 (yalnızca model seçimi; sonuçları "dağılım içi, kısmen sentetik geliştirme ölçümü"
  olarak raporlanır) / **bağımsız saha testi**.
- **Saha testi hedefi:** ≥ 300 görüntü, ≥ 6 oturum, her sınıf için ≥ 100 örnek, iki etiketleyici, dondurulmuş sürüm.
- **Toplama:** KVKK uyumlu izin; ihlaller güvenli alanda sahnelenir.
- **Çok kişili sahneler:** nesne düzeyi + kişi düzeyi ölçüm; kişi sayısı, ölçek, örtülme ve görünüm katmanları.
- **Baret/yelek testleri:** durum karışıklığı, ihlal recall'u, zor durum alt kümeleri.
- **Ölçütler:** precision, recall, F1, mAP@0.5, mAP@0.5:0.95 ve bootstrap güven aralığı.
- **Hata raporlama:** hata taksonomisi ve `errors.csv` şeması.
- **Kamera testleri:** Aşama 3'te yalnızca kamera altyapısının teknik işlev testleri; KKD modeli değerlendirmesi Aşama 8 ve 10'da (dondurulmuş saha testi), kişi düzeyi kaydedilmiş video testi Aşama 11'de, canlı kamera sistem testleri Aşama 12'de.

**Hiçbir performans ölçümü yapılmadı ve tamamlanmış gösterilmedi.**

## 8. Lisans ve atıf durumu

| Kaynak | Durum |
|---|---|
| Construction-PPE | CC BY 4.0. 2C'de yeniden doğrulandı: resmi sayfa HTTP 200; lisans, atıf metni ve BibTeX aynı; 1000/800/200/12/14.863 tutarlı; sürüm numarası yok ("November 2026"); yerel ZIP SHA-256 `e5817aa9…fbc1232` yeniden hesaplandı ve eşleşiyor. Platform kartındaki "812 classes" ifadesi doküman ve `data.yaml` ile çelişiyor; kullanılmadı |
| F01 (Roboflow lisansı) | `external_verification_pending_source_excluded`. Yeni doğrulama yapılamadı, yeni lisans iddiası oluşturulmadı |
| F02 (Mendeley üst kaynak hakları) | `external_verification_pending_source_excluded`. Hak sorunu belgelenemedi, yeni iddia yok |

**F01/F02 durumunun anlamı:** Bu durum, lisans/hak sorununun çözüldüğü anlamına **gelmez**. İlgili kaynak onaylı kapsamdan
çıkarıldığı için kullanım riski kontrol altındadır.

- Yeni durum, kontrol listesi şemasına (`stage2b_reports.STATUSES`) eklendi ve "açık" durumlar arasında sayılıyor (`OPEN_STATUSES`).
- Politikadaki `checklist_items` alanı ile ve testlerle (T6.5) birbirine bağlı.
- Lisansı belirsiz hiçbir kaynak onaylı ilan edilmedi.
- Roboflow için atıf metni hâlâ yazılmamıştır.

**Atıf:** [`docs/dataset_attributions.md`](../../docs/dataset_attributions.md) şunları içerir:

- bitirme raporuna doğrudan konulabilecek kaynakça girdisi, Türkçe ve İngilizce yöntem metinleri ve kısa atıf;
- CC BY 4.0'ın "indicate changes" şartı için 2C değişiklik kaydı;
- gelecekteki değişikliklerin 7 adımlı kayıt prosedürü.

## 9. Tamamlanan ve açık kalan kontroller

59 maddelik kontrol listesinin durumu:

| Durum | Madde |
|---|---|
| resolved_verified | 43 |
| resolved_excluded | 13 |
| external_verification_pending_source_excluded | 2 (F01, F02) |
| pending_external_verification | 1 (G01: Aşama 2 kapanışı için kullanıcı onayı) |
| unresolved | 0 |

Aşama 2C görevlerinin durumu:

| Görev | Durum |
|---|---|
| 1. Construction-PPE son kontrol (envanter, sayımlar, bağlantılı görüntüler, sızıntı, politika, manifest, sayılar) | Tamamlandı |
| 2. `docs/evaluation_plan.md` | Tamamlandı (plan; toplama ve eğitim yapılmadı) |
| 3. Kullanım politikası doğrulaması (kodla) | Tamamlandı |
| 4. F01/F02 durumu | Tamamlandı (harici doğrulama hâlâ açık) |
| 5. Atıf ve değişiklik kaydı | Tamamlandı |
| 6. Bütünlük kontrolleri ve testler | Tamamlandı (§10) |
| 7. Bu rapor | Tamamlandı |

Açık kalanlar (onaylı kapsamı bloklamaz):

- G01: kullanıcı onayı.
- F01/F02 harici doğrulama.
- Açıklanamayan kod taşıyan 22 işaretli görüntünün insan incelemesi (Aşama 8 ön koşulu; inceleme paketi 2D'de hazırlandı, kullanıcı kararları bekleniyor).
- 690 görüntüsünün yeniden etiketlenmesi (isteğe bağlı).
- Saha test kümesinin toplanması.

## 10. Test sonuçları

Testler gerçekten çalıştırıldı (`.venv/bin/python -m unittest discover -s tests`).

| Çalıştırma | Sonuç |
|---|---|
| 2C öncesi temel durum | 125 test, **OK** |
| 2C sonrası (önceki 125 + yeni `test_stage2c_closure.py` 27 test) | **152 test, OK** (0 başarısız, 0 hata, 0 atlanan) |

Önceki testlerde yapılan iki değişiklik (gevşetme değil, yeni kararlarla uyum):

1. Onaylı kaynaktaki görüntü artık "approved **veya** politikada açıkça kayıtlı override durumu" olabilir.
2. F01/F02 beklentisi `external_verification_pending_source_excluded` oldu. "resolved_verified olamaz" kontrolü korunuyor.

**Bütünlük denetimleri** (`scripts/run_stage2c_checks.py` → `stage2c_integrity_checks.json`): **10/10 geçti**.

| No | Soru | Sonuç |
|---|---|---|
| T1.1 | 1000 görüntü envanterle eşleşiyor mu? | Evet |
| T1.2 | Hedef sınıf sayımları tutarlı mı? | Evet |
| T6.1 | Onaylı veri kapsamı doğru mu? | Evet |
| T6.2 | Kullanım politikası tutarlı mı? | Evet (validator: 0 sorun) |
| T6.3 | Train/val sızıntısı için karar tanımlı mı? | Evet |
| T6.4 | Kaynak grupları ayrılmama kuralı korunuyor mu? | Evet (0 ihlal) |
| T6.5 | Lisansı belirsiz veri yanlışlıkla onaylanıyor mu? | **Hayır** (0 görüntü) |
| T6.6 | Eksik anotasyon riski doğru işaretleniyor mu? | Evet |
| T6.7 | Önceki raporlarla çelişki var mı? | **Yok** |
| T6.8 | Ham veri korunuyor mu? | Evet |

T6.7 ayrıntısı: `final_dataset_decisions.md`, `manual_review_checklist.md`, `remaining_manual_reviews.md` ve
`dataset_audit_report.md` §17–18 yeniden üretildi; 2B sayıları "tarihsel" olarak işaretlendi. Beklemedeki kaynakların analiz
kayıtları eksiksiz.

T6.8 ayrıntısı: Construction SHA-256 değerleri yayıncı manifestiyle aynı. Üç veri setinin tüm ham dosyaları (2004 + 4576 + 5601)
özgün ZIP'lerle boyut ve CRC32 olarak eşleşiyor; eksik veya fazla dosya yok. Sahibi tarafından yazılabilir ham dosya sayısı 0.

Şeffaflık notu: T6.8 ilk çalıştırmada **başarısız** oldu. Nedeni, her ham klasördeki Aşama 1'den kalma boş `.gitkeep` yer tutucusuydu
(ZIP'te yok, veri değil). Denetim bu dosyaları açıkça listeleyip karşılaştırmadan çıkaracak şekilde düzeltildi
(`placeholders_ignored`); veri içeriğinde bir fark bulunmadı.

Ayrıca 2B üretecindeki "Testler başarılı: EVET" ölçütü önceden **koşulsuz** yazılıyordu. Artık gerçek denetim çıktısından okunuyor.

## 11. Aşama 2 kapanışa uygun mu?

| Kapanış ölçütü | Karşılandı mı | Kanıt |
|---|---|---|
| Onaylı veri kapsamı kesin olarak tanımlı | Evet | Politika v2 + `construction_ppe_split_manifest.csv` (999 görüntü: 800/199; 1 hariç) |
| Kaynak ve sınıf seçimleri tutarlı | Evet | `validate_policy` sorun yok (T6.2); 4 hedef sınıf, eşleme dosyasıyla uyumlu |
| Bilinen sızıntı riskleri kontrol altında | Evet | Construction birimi düzeltildi (T6.3/T6.4); Roboflow/Mendeley dışlı; bilinmeyen sızıntı için saha testi zorunlu |
| Lisansı doğrulanmamış kaynaklar onaylı kapsam dışında | Evet | T6.5: 0 görüntü; F01/F02 `external_verification_pending_source_excluded` |
| Bağımsız test stratejisi belgelenmiş | Evet | `docs/evaluation_plan.md` |
| Bütün testler başarılı | Evet | 152/152 test + 10/10 denetim |
| Önceki raporlardaki kritik çelişkiler giderilmiş | Evet | T6.7 |

**Sonuç: `READY_FOR_USER_APPROVAL`.** Kapanış kararı (G01) sizindir. Bu durum Aşama 3'e (canlı kamera altyapısı) geçildiği anlamına gelmez.

## 12. Sonraki aşamalara taşınması gereken kısıtlar

Aşama numaraları 12 aşamalı plana göredir (`docs/project_plan.md`). Aşama 3 (canlı kamera altyapısı) bu veri kısıtlarından
etkilenmez; aşağıdaki veri kısıtları Aşama 8 (ilk YOLO eğitimi), 9 (birleştirme) ve 10 (geliştirilmiş model) için geçerlidir.

1. **Veri kapsamı (Aşama 8):** Eğitimde yalnızca `construction_ppe_split_manifest.csv` içinde `final_split` dolu olan 999 görüntü
   kullanılır. Mendeley ve Roboflow, politika değişikliği ve yeni lisans/hak kanıtı olmadan kullanılamaz (validator engeller).
2. **Ham veri değişmez.** Eğitim klasörü ve manifest tabanlı sınıf dönüşümü Aşama 8 hazırlığında manifestten türetilir. Manifest SHA-256, sınıf dönüşümü ve augmentation ayarları
   `docs/dataset_attributions.md` değişiklik tablosuna yazılır.
3. **Aşama 8 eğitimi öncesi ön koşullar:**
   - Açıklanamayan kod taşıyan 22 işaretli görüntü (train 18, val 4) insan tarafından incelenir (2D paketi:
     `final_visual_review/`). Kararlar `review_decisions.csv` `user_decision` sütunuyla uygulanır;
     `check_visual_review.py --require-complete` geçmelidir.
   - Ardından `run_stage2c_checks.py` ve testler yeniden çalıştırılır.
4. **Değerlendirme kuralları:**
   - Val yalnızca model seçimi içindir.
   - Nihai başarı yalnızca dondurulmuş bağımsız saha testiyle iddia edilir.
   - Val sonuçları "dağılım içi, kısmen sentetik" etiketiyle ve işaretli görüntüler hariç duyarlılık analiziyle raporlanır.
5. **Alan kayması:** Kısmen üretilmiş görüntüler ve tek kaynak kullanıldığı için gerçek kamera başarısı bilinmiyor.
   Sınırlılık olarak raporlanır.
6. **Sınıf anlamı:**
   - `helmet` kutusu elde taşınan bareti de kapsar. Kişi düzeyi İSG kararında baret baş bölgesiyle ilişkilendirilmelidir.
   - `no_*` sınıfları görünür bölgeyi gösterir; örtülme tek başına eksik KKD kanıtı değildir (yayıncı notu).
7. **Sızıntı kuralları (G1–G6)** yeni her veri eklemesinde uygulanır. Video ve saha verisi oturum düzeyinde bölünür.
8. **F01/F02 açık kalır (Aşama 9 ön koşulu).** Kaynaklar ancak resmi kanıt (`license_evidence.yaml`, `official_page`) eklenip politika güncellenirse
   yeniden değerlendirilir. Yeniden değerlendirmede 2B ön kararları ve E05/E10/E12 dışlamaları başlangıç noktasıdır.
9. **Gizlilik:** Saha görüntüleri ve yüz verileri Git'e girmez. KVKK izinleri toplama başlamadan önce alınır.

## 13. Aşama 2D eki — son görsel inceleme ve plan tutarlılığı (2026-11-11)

**Durum değişmedi: `READY_FOR_USER_APPROVAL`.** 22 görüntü için model önerileri hazırlandı; kullanıcı kararları henüz verilmediği
için bu görüntüler kullanıcı tarafından doğrulanmış **sayılmaz** ve `FULLY_COMPLETED` durumuna geçilmedi.

**Görsel inceleme (22 görüntü, train 18 / val 4).** Her sayfa açılarak incelendi: orijinal görüntü, tüm kutular, şüpheli kişi ve
baş/gövde yakınlaştırması.

| Model önerisi | Sayı | İnceleme no |
|---|---|---|
| approved_candidate | 13 | R01, R03, R06, R08, R09, R10, R11, R12, R14, R16, R17, R18, R22 |
| needs_reannotation | 5 | R02, R04, R07, R15, R19 (R19 val) |
| quarantine | 1 | R05 |
| inconclusive | 3 | R13, R20, R21 (R20, R21 val) |

- Görsel olarak eksik etiket bulunan görüntüler (öneri `needs_reannotation`):
  - R02, R04: görünür baretsiz baş için `no_helmet` yok.
  - R07, R19: bulanık ama ayırt edilebilir arka plan kişisi; baş ve gövde etiketi yok.
  - R15: kabindeki operatörün gövdesi için `no_vest` yok.
- R05'te iki şüpheli etiket bir arada: kapüşonlu baş etiketsiz ve koyu montun üzerinde `vest` kutusu var.
- Belirsiz kalanlar (R13, R20, R21) aşırı bulanık figür veya cam arkasındaki giysinin yelek olup olmadığıdır.
- Geri kalan 13 görüntüde baş/gövde örtülü, kadraj dışı ya da ayırt edilemiyor; etiket beklenmez.
- Şüpheli kişinin bölgesine değen hiçbir etiket yanlış kişiye atanmış görünmüyor.

**Paket ve mekanizma:**

- Paket `final_visual_review/`: `contact_sheet.pdf` (22 sayfa, A4, 300 dpi), `pages/`, `review_decisions.csv`, `review_guide.md`.
- Yalnızca `user_decision` sütunu uygulanır (`approved` / `needs_reannotation` / `quarantine`; dışlamada not zorunlu).
- Kontrol: `check_visual_review.py`. Uygulama: `apply_usage_policy.py` (kural `USER-VISUAL-REVIEW`).
- Aşama 8 kapısı: `check_visual_review.py --require-complete`.
- Uçtan uca simülasyon (geçici 3 karar → 997 kullanılabilir görüntü, ardından geri alma → birebir aynı çıktılar) ve geçersiz CSV'nin
  reddedilmesi doğrulandı.

**Önerilere göre olası kapsam (bilgi; uygulanmadı):**

- Öneriler kabul edilirse 993–996 kullanılabilir görüntü kalır:
  - 6 görüntü dışlanır (5 `needs_reannotation` + 1 `quarantine`).
  - 3 `inconclusive` görüntü ayrıca dışlanırsa toplam 9 dışlanır.
- Bugün manifest değişmedi: 999 görüntü (train 800 / val 199).

**12 aşamalı plan.** Tek referans `docs/project_plan.md` ve `configs/project_stages.yaml`. Düzeltilen yerler:

| Dosya | Düzeltme |
|---|---|
| `evaluation_plan.md` | Türetilmiş eğitim verisi Aşama 8 hazırlığında; §9 zamanlama tablosu 3/4/5–7/8/9/10/11/12 olarak yeniden yazıldı |
| `dataset_attributions.md` | Eğitim klasörü Aşama 8, birleşik veri Aşama 9 |
| `STAGE_2_FINAL_REPORT.md` | §5, §7, §9, §11, §12 |
| `README.md` | 12 aşama listesi, teknoloji tablosunda Aşama 4/7, veri klasörlerinde Aşama 8/9 |
| `architecture.md` | Modül → aşama sütunu |
| `dataset_usage_policy.yaml`, `stage2c_decisions.yaml` | 22 görüntünün incelemesi Aşama 8 öncesi |
| Rapor üreteçleri (`report.py`, `followup_reports.py`, `stage2b_reports.py`) ve ilgili raporlar | "Aşama sonrası" yerine Aşama 8+, 9–10, 11 ve "Aşama 3 = canlı kamera altyapısı" |

Önceki analiz sonuçları değiştirilmedi. Tutarlılık kodla denetleniyor (T2D.2).

**Testler ve denetimler (2D sonrası):**

- **169 test OK** (152 + yeni `test_stage2d_review.py` 17 test; 0 başarısız, 0 atlanan).
- **12/12 bütünlük denetimi geçti** (T2D.1 paket eksiksizliği, T2D.2 aşama tutarlılığı eklendi).
- Ham veri: üç veri setinin tüm dosyaları özgün ZIP'lerle (boyut + CRC32) ve Construction SHA-256 değerleri yayıncı manifestiyle
  eşleşiyor. Yazılabilir ham dosya sayısı 0.

**Aşama 2'nin nihai onayı için kalanlar:**

1. 22 görüntü için `user_decision` girilmesi (veya güvenli dışlama kararı) ve uygulanması.
2. G01: Aşama 2 kapanışının kullanıcı onayı.
3. F01/F02 açık kalır (kaynaklar dışlı); Aşama 2 kapanışını engellemez.

---

### Ek: Bu aşamada oluşturulan/değiştirilen dosyalar

**Yeni dosyalar:**

- `docs/evaluation_plan.md`
- `configs/review/stage2c_decisions.yaml`
- `src/isg_vision/dataset_audit/stage2c.py`
- `scripts/run_stage2c_checks.py`
- `tests/test_stage2c_closure.py`
- `reports/dataset_audit/construction_ppe_split_manifest.csv`
- `reports/dataset_audit/stage2c_integrity_checks.json`
- `reports/dataset_audit/stage2c_review/` (görsel kanıtlar)
- bu rapor

**Değişen dosyalar:**

- `configs/dataset_usage_policy.yaml` (v2)
- `src/isg_vision/dataset_audit/usage_policy.py`
- `src/isg_vision/dataset_audit/stage2b_reports.py`
- `scripts/apply_usage_policy.py`
- `scripts/generate_stage2b_reports.py`
- `tests/test_stage2b_policy.py` (2 beklenti, §10)
- `docs/dataset_attributions.md`
- `README.md` (durum satırı)

**Yeniden üretilen raporlar:**

- `image_usage_decisions.csv`
- `usage_policy_summary.json`
- `quarantine_candidates.csv`
- `needs_reannotation_list.csv`
- `final_dataset_decisions.md`
- `manual_review_checklist.md`
- `remaining_manual_reviews.md`
- `followup_summary.json`
- `dataset_audit_report.md`

**Yeniden üretim sırası:**

1. `python scripts/apply_usage_policy.py`
2. `python scripts/generate_stage2b_reports.py`
3. `python scripts/generate_dataset_report.py`
4. `python scripts/run_stage2c_checks.py`
5. `python -m unittest discover -s tests`
