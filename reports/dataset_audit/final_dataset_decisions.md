# Nihai Veri Seti Kararları (Aşama 2B, Aşama 2C ile güncellendi)

_Üretim tarihi 2026-11-11 (`scripts/generate_stage2b_reports.py`). Orijinal veri/etiket/bölünme DEĞİŞTİRİLMEMİŞTİR; hiçbir filtreleme/birleştirme çalıştırılmamıştır. Görsel kararlar model destekli, tek değerlendirici incelemesidir. Politika: `configs/dataset_usage_policy.yaml`._

## 1. Sonuç

- **Bugün onaylı eğitim kapsamı yalnızca Ultralytics Construction-PPE'dir:** 1000 görüntünün durumları {'approved': 999, 'needs_reannotation': 1}; nihai bölme {'train': 800, 'val': 199} (`construction_ppe_split_manifest.csv`); kullanılabilir görüntülerdeki hedef sınıf kutuları {'vest': 1815, 'helmet': 1576, 'no_helmet': 813, 'no_vest': 567}.
- **Mendeley PPE v6 ve Roboflow Site Safety v27 onaylı kümede DEĞİLDİR** (`pending_license_verification`): üst kaynak/lisans hakları doğrulanamadı. Hak doğrulanırsa alacakları durumlar ön karar olarak hesaplandı (`status_if_rights_verified`) ama bugün hiçbiri `approved` değildir.
- Sorunlu üç sınıf (E05 Mendeley `Helmet`, E10 Roboflow `NO-Hardhat`, E12 Roboflow `NO-Safety Vest`) birleşik eğitim adaylarından DIŞLANDI; dönüşüm/yeniden boyutlandırma uygulanmadı.
- 59 maddelik kontrol listesi: {'resolved_verified': 43, 'resolved_excluded': 13, 'external_verification_pending_source_excluded': 2, 'pending_external_verification': 1}. Açık kalanlar kullanıcı/harici doğrulama bekler ve onaylı kümeyi etkilemez.

## 2. Kaynak düzeyi durum

| Kaynak | Lisans doğrulama | Hak durumu | Kullanım | Görüntü durumları (bugün) |
|---|---|---|---|---|
| construction_ppe | verified_official | verified_publisher_original | approved_scope | {'approved': 999, 'needs_reannotation': 1} |
| mendeley_ppe | partially_verified | rights_unverified | pending_license_verification | {'pending_license_verification': 2286} |
| site_safety | verified_local_only | rights_unverified | pending_license_verification | {'pending_license_verification': 2799} |

**Önemli ayrım:** bir kaynağı kullanım dışı bırakmak lisansının/hakkının doğrulandığı anlamına GELMEZ. Roboflow ve Mendeley için lisans/hak soruları AÇIK kalır (`external_verification_pending_source_excluded`); yalnızca eğitim riski dışlama ile kontrol altına alınmıştır.

## 3. Sınıf bazlı kullanım politikası

| Veri seti | Sınıf | Hedef | Bugün | Koşul (hak doğrulanırsa / notlar) |
|---|---|---|---|---|
| construction_ppe | helmet | helmet | **approved** | Grup bazlı bölme; taşınan baret de 'helmet' (konum kontrolü gerekir) |
| construction_ppe | no_helmet | no_helmet | **approved** | Grup bazlı bölme |
| construction_ppe | vest | vest | **approved** | Grup bazlı bölme |
| construction_ppe | no_vest | no_vest | **approved** | Grup bazlı bölme |
| construction_ppe | gloves, boots, goggles, no_gloves, no_boots, no_goggle, Person, forklift | — | hedef dışı (yok sayılır) | Hedef nesne değil; etiketsiz hedef bırakmaz |
| mendeley_ppe | Helmet | helmet | dışlandı (E05) | Helmet'li görüntü needs_reannotation; dönüşüm yok |
| mendeley_ppe | NoHelmet / Vest / NoVest | no_helmet / vest / no_vest | pending_license_verification | Hak doğrulanırsa: Helmet'siz ve <3 hedef kutulu görüntüler; kalabalık görüntüler karantina |
| site_safety | Hardhat / Safety Vest | helmet / vest | pending_license_verification | Hak doğrulanırsa: RF-* kurallarını geçen görüntüler |
| site_safety | NO-Hardhat | no_helmet | dışlandı (E10) | İçeren görüntü needs_reannotation |
| site_safety | NO-Safety Vest | no_vest | dışlandı (E12) | İçeren görüntü needs_reannotation |
| site_safety | Mask, NO-Mask, Person, Safety Cone, machinery, vehicle | — | hedef dışı | — |

### Hak doğrulanırsa görüntü durumları (ön karar; bugün geçerli DEĞİL)

| Veri seti | Görüntü durumları | Neden sayıları (kural kodu) |
|---|---|---|
| mendeley_ppe | {'quarantine': 482, 'needs_reannotation': 1167, 'approved': 637} | {'MD-CROWD-INCOMPLETE': 482, 'MD-HELMET-IMAGE': 1390} |
| site_safety | {'quarantine': 2329, 'needs_reannotation': 348, 'approved': 122} | {'RF-TINY-TARGET': 321, 'RF-PERSON-INCOMPLETE': 2097, 'RF-EXCLUDED-CLASS-IMAGE': 2173, 'RF-UNVERIFIED-NEGATIVE': 201, 'RF-CONFLICT-UNRESOLVED': 39, 'RF-LIKELY-ERROR': 31, 'RF-CONFLICT-OVERRIDE': 2} |

Görüntü bazlı tüm kararlar: `image_usage_decisions.csv`; karantina/yeniden etiketleme adayları (gerekçeli): `quarantine_candidates.csv`; yeniden etiketleme iş listesi: `needs_reannotation_list.csv`.

## 4. E05 / E10 / E12 kararları

### E05 — Mendeley `Helmet` → **dışla**

Kutu baş+baret (yüz dahil); baş/gövde oranı 0.53 (Construction helmet 0.33; Roboflow Hardhat baret nesnesine sıkı); h/w 1.19 vs 0.85. Otomatik geometrik dönüşüm: baret kutusunun baş kutusu içindeki konumu poz/siperlik/kamera açısına göre değiştiği ve kalibrasyon için eşleştirilmiş doğru kutu bulunmadığı için GÜVENİLİR DEĞİL; tahmini küçültme uygulanmadı. Mendeley IQR/medyan baş-gövde oranı 0.32 (sabit bir küçültme katsayısı kutu başına büyük hata üretir).

- **Yeniden etiketleme maliyeti/veri kaybı:** Helmet kutusu içeren görüntü sayısı 1390 / 2286 (%61); yalnızca Helmet satırlarını silmek bu görüntülerde etiketsiz baret bırakır, bu yüzden TÜM görüntü dışlanır (kural MD-HELMET-IMAGE). Yeniden çizilecek Helmet kutusu sayısı `needs_reannotation_list.csv` içinde.
- Karar: tahmini kutu küçültme uygulanmadı; gelecekte manuel yeniden etiketleme listesi hazır.

### E10 — Roboflow `NO-Hardhat` → **dışla**

Geometri süzgeci (h>=32 px, 0.9<=h/w<=1.8, kenara değmeyen, Hardhat ile örtüşmeyen, kişiye atanan) 2428 kutunun 354'ünü geçiriyor (%14.6). Geçen 40 kutuluk görsel örneklemde tam-baş uyumlu yalnızca ~12/40 (%30): çoğu saç/alın üst kısmı (~19/40) veya belirsiz/koyu (~9/40). Güvenilir bir tam-baş alt kümesi ayırt edilemedi. 24 örneklemde AN13 (mavi baret görünen hata) süzgeçle elendi (iyi), ancak süzgeç tam-baş uyumunu garanti etmiyor.

- Yanlış-kabul riski: süzgeçten geçen kutuların yaklaşık %70'i tam-baş tanımıyla uyumsuz (yanlış kabul), bu yüzden süzgeç güvenilir bir alt küme üretmiyor. Ölçütler (açık): `src/isg_vision/dataset_audit/box_filters.py`; örneklem: `roboflow_filter_precision_sample.csv`, `roboflow_review/filter_precision_no_helmet.jpg`.

### E12 — Roboflow `NO-Safety Vest` → **dışla**

Geometri süzgeci (h>=48 px, 0.8<=h/w<=3.0, kenara değmeyen, kişiye göre >=%25, Safety Vest/Hardhat ile örtüşmeyen) 4153 kutunun 992'sini geçiriyor. Geçen 44 kutuluk görsel örneklemde geçerli gövde ~30/44 (%68); hi-vis renkli giysi ~3, yanlış bölge ~1, belirsiz/parça ~10. Yüksek görünürlük renk ölçütü 24 denetim karosunda 2/2 olası hatayı yakalayamadı (AX09 oran 0.10, AX11 0.0). %90 kesinlik sağlanamadı.

- Kararlar (`approved` / `quarantine` / `needs_reannotation`) görüntü seviyesinde verilir: NO-Safety Vest içeren görüntü `needs_reannotation`, çelişkili/kırpık/belirsiz kutu içeren görüntü `quarantine`. Hiçbir örnek kanıt yetersizken `approved` yapılmadı.

## 5. Veri sızıntısı çözümü (D16 ve Roboflow val/test)

Kaynak grupları: eşleşme grafiğinde sert kenar (kanıtlı) 4065, yumuşak (sınırda/Likely) 67; karo düzeyi ORB ile ek 166 çift incelendi; 3750 bölme birimi (`source_group_manifest.csv`).

| Veri seti | Görüntü | Sert bağlantılı | Bölümler arası sert grupta | Bölüme göre | Veri setleri arası birimde | En büyük birim |
|---|---|---|---|---|---|---|
| construction_ppe | 1000 | 5 | 3 | {'train': 2, 'val': 1} | 0 | 3 |
| mendeley_ppe | 2286 | 288 | 163 | {'train': 117, 'val': 46} | 8 | 1986 |
| site_safety | 2799 | 2196 | 2044 | {'test': 70, 'train': 1882, 'val': 92} | 1984 | 1986 |

**Yöntem (politika):** (1) bölme birimi = sert + yumuşak bağlantılı bileşen; (2) bir birim ASLA bölünmez; (3) rastgele görüntü bazlı bölme yasak; (4) bilinmeyen/şüpheli ilişkiler yumuşak kenar olarak birlikte tutulur; (5) beklemedeki kaynaklar için oran %70/15/15, seed=42, büyük birimler önce (önizleme `image_usage_decisions.csv` `proposed_split_preview` — UYGULANMADI); (6) **2C:** onaylı Construction-PPE için yayıncı bölünmesi korunur, bölümler arası birim çoğunluk bölümüne alınır (`construction_ppe_split_manifest.csv` `final_split`). 2B'deki Construction-PPE %70/15/15 önizlemesi GEÇERSİZDİR.

**Önemli sonuçlar:**
- Roboflow'un 2799 görüntüsünün 2044'ü bölümler arası sert gruptadır (val 92/114, test 70/82). Kolaj zincirleri nedeniyle 1986 görüntülük TEK bir dev birim oluştu (veri setleri arası bağ da içeriyor): bu birim tek bölüme (train) gider; dolayısıyla Roboflow'un mevcut val/test'i bağımsız değerlendirme için KULLANILAMAZ. Bu muhafazakâr ama güvenli bir sonuçtur.
- Mendeley: 163 görüntü bölümler arası gruptadır (val 46).
- Construction-PPE: yalnızca 3 görüntü (2 train + 1 val) bölümler arası gruptadır (aynı taban karenin üretken varyantları). **2C kararı:** birim train'e alındı (val üyesi taşındı); ayrıntı `configs/review/stage2c_decisions.yaml`.
- **Açık risk:** kenarlar alt sınırdır; bağlantısı olmayan görüntü 'ilişkisiz' kanıtlı değildir. Bu nedenle bağımsız saha testi ZORUNLUDUR (§6 politika).
- **Bağımsız saha testi önerisi:** Hedef kamera/şantiye ortamından, eğitim verisiyle hiçbir kaynak bağı olmayan, oturum (kayıt) bazında gruplanmış >=300 görüntü; en az 100 baretli, 100 baretsiz, 100 yelekli, 100 yeleksiz örnek içermeli; iki bağımsız etiketleyici; ayarlama/eşik seçiminde kullanılmaz.

## 6. C01 / C03 ve D07 / D08

| Madde | Görüntü | Karar | Kanıt |
|---|---|---|---|
| C01 (SI057) | `IMG_5024_mp4-2_jpg.rf.77ca6ff1…` | quarantine | Kolajın sol alt karesi döndürülmüş, gri tonlu, yoğun kalabalık: L13-L16 ardışık NO-Safety Vest kutuları ve L17 Safety Vest kutusu aynı kişiler üzerinde üst üste biniyor; düşük çözünürlükte hangi kutunun doğru olduğu görsel olarak belirlenemedi. Belirsiz etiket sessizce kabul edilmedi. |
| C03 (SI081) | `construction-897-_jpg.rf.c7afbfac…` | needs_reannotation | Mavi panel karesinde sağ kenardaki kareli gömlekli kişi için L9 (Safety Vest) ve L16 (NO-Safety Vest) aynı bölgede (IoU 0.45): iki etiketten biri kesinlikle yanlış; kişi kareli gömlek giyiyor ve belirgin yelek görünmüyor, Safety Vest etiketi büyük olasılıkla hatalı. Görüntü yeniden etiketleme sonrası kullanılabilir. |

Görsel: `roboflow_review/c01_c03_original.jpg` (1.5x orijinal çözünürlük, tüm kutular sınıf adı ve satır no ile). Belirsiz etiketler sessizce kabul edilmedi.

| Madde | Grup | Karar | Kanıt |
|---|---|---|---|
| D07 | IMG_0871_mp4-23 | Confirmed Same Source | Tam görüntü ORB sınırdaydı (24 inlier). Karo düzeyinde ORB: train kolajının karelerinden biri test görüntüsüyle eşleşiyor (5 çiftte inlier 30/46/54/78/131, oran 0.73-0.96; en iyi çiftte hizalı NCC 0.85). Görsel: aynı kadın, aynı maske, aynı el hareketi — aynı videonun aynı karesi. |
| D08 | IMG_3103_mp4-17 | Likely Same Source | ORB 5/5 çiftte ≤6 inlier (geometrik eşleşme yok; kare/poz farklı). Görsel: train kolajındaki kulaklıklı, gözlüklü, maskeli kadın karesi val görüntüsündeki kişiyle AYNI kişi/aynı kayıt oturumu görünüyor (yüz ifadesi/poz farklı). Aynı video olduğu kesinleştirilemedi. |

Görsel: `duplicate_verification/_d07_d08.jpg`. Likely Same Source ilişkisi yumuşak kenar olarak kaydedildi; birim ileride farklı bölümlere ayrılmaz. Diğer 6 ortak-ad grubu: youtube-51: Different Source, image_55: Different Source, youtube-504: Different Source, youtube-260: Different Source, youtube-391: Different Source, Movie-on-10-31-22-at-10_08-AM_mov-13: Confirmed Same Source.

_Not:_ karo düzeyi ORB, tam görüntü ORB'sinin kaçırdığı eşleşmeleri yakaladı (9 sert kenar); bu, ilk raporlanan Roboflow val/test maruziyetini (alt sınır) daha da artırdı.

## 7. Lisans ve kullanım hakkı

| Kaynak | İddia edilen lisans | Lisans doğrulaması | Hak durumu | Kullanım |
|---|---|---|---|---|
| construction_ppe | CC BY 4.0 | verified_official | verified_publisher_original | approved_scope |
| mendeley_ppe | CC BY 4.0 | partially_verified | rights_unverified | pending_license_verification |
| site_safety | CC BY 4.0 | verified_local_only | rights_unverified | pending_license_verification |

- **Doğrulandı:** Construction-PPE (resmi sayfa + arşivdeki lisans metni); Mendeley veri seti lisansı (resmi sayfa + API) ve yerel ZIP'in SHA-256 eşleşmesi.
- **Doğrulanamadı:** Roboflow v27 (resmi sayfa ve üst projeler HTTP 403; proje ana sayfası 404; web aramasında ayni sınıf listesine sahip FARKLI bir çalışma alanı projesi CC BY 4.0 görünüyor — bu, bizim veri setinin doğrulaması DEĞİLDİR); Mendeley üst kaynak hakları.
- **Dışlandı (kullanım dışı):** Roboflow ve Mendeley onaylı kümeden; bu, lisans doğrulaması anlamına gelmez.
- Mendeley görüntülerinde çok sayıda stok-fotoğraf filigranı görüldü (alamy, depositphotos, CanStockPhoto, paixin, vcg, redocn, Fang.com, weibo): bu görseller için CC BY 4.0 etiketi hak sahipliğini göstermez. Kaynak/atıf ayrıntıları: `docs/dataset_attributions.md`.

## 8. Contact sheet incelemesi (B04) — 12/12 dosya AÇILARAK incelendi

| Veri seti | İncelenen sayfa | Bulgular |
|---|---|---|
| construction_ppe | 4 | Kutular nesneler üzerinde ve sınıflar tutarlı (helmet/no_helmet/vest/no_vest/gloves/boots/goggles/Person/forklift, no_* sınıfları görünür vücut bölgesinde). Yanlış sınıf veya belirgin yanlış konum görülmedi. Uzak/küçük kişilerin bulunduğu geniş çekimlerde (gözetim kamerası/balıkgözü) çok sayıda küçük ve iç içe kutu var. Görüntülerin önemli bölümü yapay üretilmiş görünümde (README: 'human-collected and generated images'). val 599 gibi örneklerde sağdaki kişi no_helmet ile işaretlenmiş, doğru. |
| mendeley_ppe | 4 | Etiketler EKSİK: kalabalık sahnelerde (00289, 00877, 01029, 01051, 00197, 08371) baretli/yelekli birçok kişi etiketsiz, yalnızca bazıları işaretli. Çok sayıda stok/ticari fotoğraf filigranı (alamy, depositphotos, CanStockPhoto, paixin, vcg, redocn, Fang.com, weibo) => görüntü hakları belirsiz. 00071'de elinde baret taşıyan kişi NoHelmet (geçerli). NoVest örneklerinin çoğu inşaat dışı sıradan sahneler (istasyon, çarşı, yürüyüş yolu). |
| site_safety | 4 | Train görüntüleri 2x2 kolaj, siyah kare (cutout), döndürme ve renk bozulması içeriyor; birçok görüntüde karo sınırları kutuları kesiyor. Alan dışı içerik yoğun (kitapçı, bowling, havalimanı, yakın çekim maskeli yüzler) ve bu karolarda NO-Safety Vest/NO-Hardhat kutuları var. Çok sayıda iç içe Person kutusu ve üst üste binen NO-Safety Vest kutuları. Val/test görüntüleri kolaj değil, normal tek görüntü (dağılım farkı). Yanlış sınıf örnekleri: yakın çekim yüze NO-Safety Vest, maskeli yüze Mask+Hardhat şeridi. |

**Önceki crop incelemeleriyle karşılaştırma:** Önceki crop incelemeleriyle tutarlı: Construction-PPE temiz; Mendeley tutarlı ama eksik etiketli ve haklar belirsiz; Roboflow gürültülü ve dağılım olarak train/val-test farklı. Yeni bulgular: (1) Mendeley etiket EKSİKLİĞİ, (2) Mendeley filigranlı stok görseller, (3) Roboflow train kolaj/cutout yapısı.

## 9. Kontrol listesi nihai durum özeti

| Durum | Madde |
|---|---|
| resolved_verified | 43 |
| resolved_excluded | 13 |
| external_verification_pending_source_excluded | 2 |
| pending_external_verification | 1 |
| unresolved | 0 |
| Toplam | 59 |

Ayrıntı: `manual_review_checklist.md` (59 madde, her biri için durum + eğitim riski), açık olanlar: `remaining_manual_reviews.md`.

## 10. Açık riskler ve onaylı veri kullanımına etkisi

| Risk | Onaylı (Construction-PPE) kümeyi etkiler mi? | Kontrol |
|---|---|---|
| Roboflow lisansı doğrulanamadı | Hayır | Kaynak onaylı kümeden dışlı |
| Mendeley üst kaynak hakları | Hayır | Kaynak onaylı kümeden dışlı |
| Roboflow val/test sızıntısı | Hayır | Roboflow dışlı; politika: grup bazlı bölme |
| Mendeley etiket eksikliği / filigranlı görseller | Hayır | Mendeley dışlı |
| Construction-PPE ayrı test yok, kısmen üretilmiş görüntü, taşınan baret 'helmet' | EVET (alan kayması) | Bağımsız saha testi zorunlu; baret konumu kontrolü |
| Aday üretimi alt sınır (bilinmeyen sızıntı) | Küçük (Construction'da 3 görüntü bağlantılı) | Yumuşak/bilinmeyen bağlar birlikte tutulur |

## 11. Aşama 2 kapanış ölçütleri

| Ölçüt | Karşılandı mı | Kanıt |
|---|---|---|
| 1. Açık kalan maddeler listelenmiş | EVET | 3 madde remaining_manual_reviews.md içinde |
| 2. Her maddeye karar/doğrulama ihtiyacı atanmış | EVET | durumlar: {'resolved_verified': 43, 'resolved_excluded': 13, 'external_verification_pending_source_excluded': 2, 'pending_external_verification': 1}; 'unresolved' yok |
| 3. Açık riskler onaylı veri kullanımını etkilemiyor | EVET | onaylı durumda olup hakları doğrulanmamış kaynak görüntüsü: 0 |
| 4. Uygun olmayan örnekler belirlenmiş | EVET | 4327 karantina/yeniden etiketleme adayı (quarantine_candidates.csv) |
| 5. Kullanılabilir kaynaklar açıkça tanımlı | EVET | configs/dataset_usage_policy.yaml: onaylı kapsam = construction_ppe; diğerleri pending_license_verification |
| 6. Veri sızıntısını önleme planı hazır | EVET | kaynak grup manifestosu + bölme politikası (leakage) |
| 7. Lisans ve atıf kayıtları oluşturulmuş | EVET | license_verification_report.md + docs/dataset_attributions.md |
| 8. Bütünlük denetimleri (2C) başarılı | EVET | 12/12 denetim geçti (stage2c_integrity_checks.json); birim testlerinin sonucu STAGE_2_FINAL_REPORT.md içinde |

## 12. Kapanış değerlendirmesi

Aşama 2C nihai değerlendirmesi `STAGE_2_FINAL_REPORT.md` içindedir. Roboflow lisansı (F01) ve Mendeley üst kaynak hakları (F02) `external_verification_pending_source_excluded` durumundadır: harici doğrulama hâlâ bekliyor, ancak kaynaklar onaylı kapsam dışında tutulduğu için kullanım riski kontrol altındadır (sorun ÇÖZÜLMEDİ). Kapanış kararı (G01) kullanıcıya aittir.

