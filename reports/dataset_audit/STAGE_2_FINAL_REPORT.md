# Aşama 2 Nihai Sonuç Raporu — Aşama 2F

_Tarih: 2026-11-11 — Yapay Zekâ Destekli Çoklu Çalışan Tanıma ve İSG Denetim Sistemi_

**Durum: `COMPLETED`.** Bu rapor güncel, uygulanmış sonucu özetler. Ayrıntılı kanıt ve dokuz görüntünün satır bazlı gerekçeleri
[`STAGE_2_CLOSURE_REPORT.md`](STAGE_2_CLOSURE_REPORT.md) dosyasındadır. Aşama 3 başlatılmamıştır.

## 1. Nihai karar

- Onaylı tek kaynak: **Ultralytics Construction-PPE**.
- Kullanılabilir kapsam: **990 görüntü = 794 train + 196 validation**.
- Kullanıcı görsel kararları: **13 approved, 4 needs_reannotation, 5 quarantine**.
- Kullanıcı kararıyla dışlanan: 9 görüntü (6 train, 3 validation).
- Önceden dışlanan `ultralytics-ppe-nov2026-690.jpg`: hâlâ `needs_reannotation`, `final_split` boş.
- Construction-PPE yayıncı toplamındaki durum: **990 approved, 5 needs_reannotation, 5 quarantine**.
- Mendeley ve Roboflow onaylı görüntü sayısı: **0**.

Ham görüntüler ve etiketler silinmemiş, taşınmamış veya değiştirilmemiştir. Politika yalnızca rapor/manifest çıktıları üretmiştir.

## 2. Görsel kararların uygulanması

| Karar | Sayı | ID'ler |
|---|---:|---|
| `approved` | 13 | R01, R03, R06, R08, R09, R10, R11, R12, R14, R16, R17, R18, R22 |
| `needs_reannotation` | 4 | R02, R04, R15, R19 |
| `quarantine` | 5 | R05, R07, R13, R20, R21 |

`review_decisions.csv` 22/22 tamamlanmış; dışlama kararlarının tümünde gerekçe bulunmuştur. Dışlanan dokuz görüntünün
`construction_ppe_split_manifest.csv` içindeki `final_split` değeri boştur. Onaylanan 13 görüntü nihai bölümlerinde kalmıştır.

## 3. Nihai sınıf kutuları

| Bölüm | Görüntü | helmet | no_helmet | vest | no_vest |
|---|---:|---:|---:|---:|---:|
| Train | 794 | 1272 | 618 | 1456 | 428 |
| Validation | 196 | 288 | 167 | 333 | 120 |
| **Toplam** | **990** | **1560** | **785** | **1789** | **548** |

## 4. Kaynak, lisans ve sızıntı durumu

| Kaynak | Durum | Onaylı görüntü |
|---|---|---:|
| Construction-PPE | `approved_scope`; resmî kaynak/lisans doğrulandı | 990 |
| Mendeley PPE v6 | `pending_license_verification`; üst kaynak hakları ve anotasyon sorunları | 0 |
| Roboflow Site Safety v27 | `pending_license_verification`; lisans/hak, etiket ve split sızıntısı sorunları | 0 |

Construction-PPE'deki bağlı üretken varyantlar aynı atomik bölüm biriminde tutulmuştur. Nihai kaynak birimi ihlali yoktur.
Validation, orijinal validation bölümünün 196 görüntülük alt kümesidir. Lisans ve atıf metinleri
`docs/dataset_attributions.md` içinde korunmaktadır.

## 5. Başarı hedefi

Projenin akademik hedefi en az **%85 başarı**dır. Mevcut durum **`not_measured`**'dır: model eğitilmemiş ve bağımsız saha testi
oluşturulmamıştır. Accuracy, precision, recall ve mAP aynı metrik değildir; YOLO tespiti, kimlik tanıma ve çoklu kişide doğru
KKD–kişi eşleştirmesi ayrı ölçülecektir. Kısmen sentetik validation verisi tek başına gerçek saha başarısı kanıtı sayılmayacaktır.

## 6. Son doğrulamalar

| Doğrulama | Sonuç |
|---|---|
| Görsel karar kapısı | 22/22 karar, 0 bekleyen — başarılı |
| Politika uygulaması | 990 toplam, 794 train, 196 validation — başarılı |
| Bütünlük denetimi | **12/12 başarılı** |
| Birim testleri | **170/170 başarılı** |
| Ham veri denetimi | Yayıncı manifesti ve özgün ZIP kayıtlarıyla uyumlu |

Tüm kabul ölçütleri karşılandığı için Aşama 2 `COMPLETED` olarak işaretlenmiştir. Kamera çalışması, model eğitimi, yüz tanıma,
veri birleştirme, commit veya push yapılmamıştır.
