# Aşama 2 Kapanış Raporu — Aşama 2F

_Tarih: 2026-11-11 — Yapay Zekâ Destekli Çoklu Çalışan Tanıma ve İSG Denetim Sistemi_

**Nihai durum: `COMPLETED`.** Kullanıcının 22/22 görsel kararı uygulanmış, kullanılabilir Construction-PPE kapsamı
**990 görüntüye** indirilmiş ve bütün kabul kontrolleri geçmiştir. Aşama 3 başlatılmamıştır.

## 1. Amaç ve kapsam

Aşama 2'nin amacı, proje veri kaynaklarını envanter, sınıf anlamı, anotasyon kalitesi, kopya/sızıntı, lisans ve kullanım hakkı
bakımından incelemek; yalnızca kanıtla desteklenen kapsam için tekrar üretilebilir bir train/validation manifesti hazırlamaktı.

Bu aşamada ham görüntü veya etiket değiştirilmedi, veri setleri birleştirilmedi, model eğitilmedi ve saha performansı ölçülmedi.

## 2. İncelenen veri setleri ve nihai kaynak kararı

| Veri seti | Yerel görüntü | Nihai durum | Gerekçe |
|---|---:|---|---|
| Ultralytics Construction-PPE | 1000 | `approved_scope` | Resmî kaynak/lisans doğrulandı; kullanıcı görsel kararları uygulandı |
| Mendeley PPE v6 | 2286 | `pending_license_verification` | Üst kaynak/görsel hakları doğrulanmadı; sınıf ve anotasyon sorunları var |
| Roboflow Construction Site Safety v27 | 2799 | `pending_license_verification` | Lisans yalnızca yerel metadata düzeyinde; hak, etiket gürültüsü ve split sızıntısı sorunları var |

Nihai onaylı veri kaynağı yalnızca **Construction-PPE**'dir. Mendeley ve Roboflow ham verileri ile analiz kayıtları korunmuş,
ancak bu iki kaynaktan onaylı veya nihai bölüme atanmış görüntü sayısı **0** olarak doğrulanmıştır.

## 3. Kullanıcının 22 nihai görsel kararı

| Karar | Sayı | İnceleme ID'leri |
|---|---:|---|
| `approved` | 13 | R01, R03, R06, R08, R09, R10, R11, R12, R14, R16, R17, R18, R22 |
| `needs_reannotation` | 4 | R02, R04, R15, R19 |
| `quarantine` | 5 | R05, R07, R13, R20, R21 |

Kararlar `final_visual_review/review_decisions.csv` dosyasına yazılmış ve
`check_visual_review.py --require-complete` ile **22/22 tamamlandı** olarak doğrulanmıştır. R07 ve R21 için nihai kullanıcı
kararı `quarantine` olup önceki model önerilerinin üzerindedir.

## 4. Nihai train/validation kapsamı

| Bölüm | Önce | Kullanıcı kararıyla dışlanan | Son |
|---|---:|---:|---:|
| Train | 800 | 6 | **794** |
| Validation | 199 | 3 | **196** |
| Toplam | 999 | 9 | **990** |

Nihai tek kaynak `construction_ppe_split_manifest.csv` dosyasındaki `final_split` alanıdır. Onaylanan 13 görüntü kendi nihai
bölümünde kalmış; dört yeniden etiketleme ve beş karantina görüntüsünün `final_split` alanı boş bırakılmıştır. Kullanıcı kararı
manifestteki `visual_review` alanından izlenebilir.

## 5. Kullanıcı kararıyla dışlanan dokuz görüntü

| ID | Orijinal bölüm | Karar | Görüntü | Gerekçe |
|---|---|---|---|---|
| R02 | train | `needs_reannotation` | `source-48e807e0a9dac5ff23128d8ebe6406ab.jpg` | Görünür baretsiz baş için etiket eksikliği şüphesi |
| R04 | train | `needs_reannotation` | `source-76c5536c698f0876c67f76c14c3c2e96.jpg` | Görünür baş için `no_helmet` etiketi eksikliği |
| R05 | train | `quarantine` | `source-8ead761447ba33e51dc962d76ef2e676.jpg` | Baş ve yelek etiketlerinin güvenilirliği yetersiz |
| R07 | train | `quarantine` | `ultralytics-ppe-nov2026-001.jpg` | Uzak kişinin başı ve gövdesi net ayırt edilemiyor |
| R13 | train | `quarantine` | `ultralytics-ppe-nov2026-516.jpg` | Aşırı bulanık arka plan figüründe KKD durumu güvenilir belirlenemiyor |
| R15 | train | `needs_reannotation` | `ultralytics-ppe-nov2026-528.jpg` | Operatör gövdesinde `no_vest` etiketi eksikliği |
| R19 | val | `needs_reannotation` | `ultralytics-ppe-nov2026-022.jpg` | Arka plan kişisinde hedef KKD etiketleri eksik |
| R20 | val | `quarantine` | `ultralytics-ppe-nov2026-023.jpg` | Uzak ve bulanık figürde KKD durumu güvenilir belirlenemiyor |
| R21 | val | `quarantine` | `ultralytics-ppe-nov2026-048.jpg` | Cam arkasındaki operatörün yelek durumu kesin doğrulanamıyor |

Tam yollar, karar notları ve model–kullanıcı karşılaştırması `final_visual_review/review_decisions.csv` içindedir.

## 6. Önceden dışlanan 690 görüntüsü

`data/raw/construction_ppe/images/train/ultralytics-ppe-nov2026-690.jpg` kullanım dışı kalmıştır:

- durum: `needs_reannotation`;
- kural: `CP-VISIBLE-TARGET-UNLABELED`;
- neden: görünür baretsiz başta `no_helmet` etiketi eksik;
- `final_split`: boş.

Böylece 1000 yayıncı görüntüsünün güncel durumları **990 approved + 5 needs_reannotation + 5 quarantine** biçimindedir.
`needs_reannotation` toplamındaki beş görüntü, dört kullanıcı kararı ile önceden dışlanan 690 görüntüsüdür.

## 7. Güncel sınıf dağılımları

Kullanılabilir 990 görüntüdeki hedef kutu sayıları:

| Bölüm | Görüntü | helmet | no_helmet | vest | no_vest |
|---|---:|---:|---:|---:|---:|
| Train | 794 | 1272 | 618 | 1456 | 428 |
| Validation | 196 | 288 | 167 | 333 | 120 |
| **Toplam** | **990** | **1560** | **785** | **1789** | **548** |

Görüntü düzeyinde sınıf varlığı: train için helmet 609, no_helmet 333, vest 637, no_vest 297; validation için helmet 143,
no_helmet 87, vest 154, no_vest 82. Train bölümündeki 10 hedefsiz görüntü daha önce görsel olarak geçerli negatif bulunmuştur.

## 8. Veri sızıntısı ve bölüm bütünlüğü

- Construction-PPE'deki aynı taban kareye bağlı üç üretken varyant tek bir atomik kaynak biriminde tutulmuş ve train'e atanmıştır.
- Validation, yayıncının orijinal validation bölümünün 196 görüntülük alt kümesidir.
- Nihai manifestte `unit_integrity_violations() == {}` sonucu alınmıştır.
- Mendeley ve Roboflow'un yayıncı bölümleri bağımsız test olarak kullanılmayacaktır.
- Gelecekteki saha testi; eğitim/validation ile kamera, gün, konum veya kayıt oturumu paylaşmayacak biçimde dondurulacaktır.

## 9. Lisans ve atıf durumu

- Construction-PPE için resmî sayfa, yerel lisans metni, CC BY 4.0 atıf bilgileri ve yerel ZIP SHA-256 kaydı doğrulanmıştır.
- Mendeley veri seti lisansı doğrulansa da üst kaynak/görsel hakları doğrulanmadığından kaynak beklemededir.
- Roboflow v27 lisansı resmî sayfadan doğrulanamadığından ve üst kaynak hakları belirsiz olduğundan kaynak beklemededir.
- F01/F02 `external_verification_pending_source_excluded` olarak açık kalır; bu durum onaylı Construction-PPE kapsamını etkilemez.
- Hazır atıf metinleri ve değişiklik kayıt yöntemi `docs/dataset_attributions.md` içindedir.

## 10. En az %85 başarı hedefi

Akademik hedef değerlendirmede en az **%85 başarı** göstermektir; ancak bu hedef henüz tek bir metriğe bağlanmamıştır.
Mevcut performans durumu **`not_measured`**'dır. Model eğitilmediği için hedefe ulaşıldığı veya ulaşılacağı iddia edilmez.

- Accuracy, precision, recall ve mAP birbirinden ayrı raporlanacaktır.
- YOLO nesne tespiti ile çalışan kimlik tanıma performansı ayrı değerlendirilecektir.
- Çoklu kişinin doğru tanınması ve KKD sonucunun doğru kişiye atanması kişi düzeyinde ayrıca ölçülecektir.
- Kısmen sentetik validation verisi nihai saha başarısı kanıtı değildir.
- Gerçek başarı, ilk YOLO eğitimi sırasında validation geliştirme ölçümleri ve ardından dondurulmuş bağımsız saha testiyle belirlenecektir.

Ayrıntılı ölçüm tanımları `docs/evaluation_plan.md` içindedir.

## 11. Test ve bütünlük sonuçları

| Kontrol | Sonuç |
|---|---|
| `check_visual_review.py --require-complete` | Başarılı — 22/22 karar, 0 bekleyen |
| `apply_usage_policy.py` | Başarılı — 990 görüntü, 794 train, 196 validation |
| `run_stage2c_checks.py` | **12/12 başarılı** |
| `python -m unittest discover -s tests -v` | **170 test, tamamı başarılı** |

Bütünlük denetimi ayrıca şunları doğrulamıştır: ham dosyalar özgün ZIP kayıtlarıyla boyut+CRC32 bakımından eşleşiyor;
Construction-PPE görüntüleri yayıncı manifestindeki ad, bölüm ve SHA-256 değerleriyle eşleşiyor; Mendeley/Roboflow'dan onaylı
görüntü yok; dokuz kullanıcı dışlaması ile 690 görüntüsü nihai bölmede yok; kaynak birimleri bölünmemiştir.

## 12. Oluşturulan veya güncellenen temel dosyalar

- `reports/dataset_audit/final_visual_review/review_decisions.csv`
- `reports/dataset_audit/construction_ppe_split_manifest.csv`
- `reports/dataset_audit/usage_policy_summary.json`
- `reports/dataset_audit/image_usage_decisions.csv`
- `reports/dataset_audit/quarantine_candidates.csv`
- `reports/dataset_audit/needs_reannotation_list.csv`
- `reports/dataset_audit/final_dataset_decisions.md`
- `reports/dataset_audit/manual_review_checklist.md`
- `reports/dataset_audit/remaining_manual_reviews.md`
- `reports/dataset_audit/dataset_audit_report.md`
- `reports/dataset_audit/STAGE_2_FINAL_REPORT.md`
- `docs/evaluation_plan.md`, `docs/project_plan.md`, `configs/project_stages.yaml`, `README.md`

## 13. Sonraki aşamalar için dikkat edilecekler

- Aşama 3 yalnızca canlı kamera altyapısı için kullanıcı talimatıyla başlatılabilir; bu kapanış onu otomatik başlatmaz.
- İlk YOLO İSG eğitimi Aşama 8'de yapılacaktır.
- Veri seti birleştirme Aşama 9'dadır; Mendeley/Roboflow hakları doğrulanmadan bu kaynaklar kullanılamaz.
- Dışlanan görüntüler ancak türetilmiş etiket kopyalarında düzeltme ve yeni görsel kontrol sonrasında yeniden değerlendirilebilir.
- Ham veri ve orijinal etiketler salt okunur kaynak olarak korunmalıdır.
- `%85` hedefi bir veri kalitesi sonucu değildir; ölçüt ve kabul eşiği deney protokolünde açıkça sabitlenmelidir.

## 14. Kapanış kararı

Kullanıcı kararları eksiksiz uygulanmış, beklenen 990/794/196 sayıları elde edilmiş, ham veri korunmuş, 12/12 bütünlük denetimi
ve 170/170 birim testi geçmiştir. Bu nedenle **Aşama 2 `COMPLETED`** olarak kapatılmıştır. Aşama 3 başlatılmamıştır.
