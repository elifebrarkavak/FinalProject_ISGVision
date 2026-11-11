# Roboflow (site_safety v27) Şüpheli Etiket İnceleme Raporu

_Üretim tarihi 2026-11-11. Tüm kararlar `configs/review/label_review_verdicts.yaml` içindeki görsel inceleme kayıtlarından gelir (model destekli, tek değerlendirici). Orijinal veri DEĞİŞTİRİLMEMİŞTİR._

## 1. Kapsam
`annotation_issues.csv` kayıtları (21 çok küçük kutu, 4 birebir + 1 yakın kopya kutu, 23 boş etiket), 112 çelişki adayı (`semantic_conflicts.csv`; 19'u Construction-PPE) ve 4 rol için rastgele 24'er kutuluk kalite örneklemi (96 kutu, ≥24 px) incelendi. Her öğe için dosya yolu, ilgili etiket satırları ve gerekçe `label_review.csv` içinde.

## 2. Kategorilere göre sonuç (Roboflow)

| Kaynak | Öğe | Confirmed Error | Likely Error | Valid Annotation | Needs Manual Review |
|---|---|---|---|---|---|
| audit_helmet | 24 | 0 | 1 | 18 | 5 |
| audit_no_helmet | 24 | 0 | 1 | 17 | 6 |
| audit_no_vest | 24 | 0 | 2 | 13 | 9 |
| audit_vest | 24 | 0 | 1 | 18 | 5 |
| duplicate_box | 4 | 4 | 0 | 0 | 0 |
| empty_label | 23 | 0 | 0 | 23 | 0 |
| near_duplicate_box | 1 | 0 | 1 | 0 | 0 |
| semantic_conflict:helmet:overlap | 10 | 0 | 5 | 0 | 5 |
| semantic_conflict:helmet:same_person | 22 | 0 | 0 | 19 | 3 |
| semantic_conflict:vest:overlap | 57 | 0 | 0 | 26 | 31 |
| semantic_conflict:vest:same_person | 4 | 0 | 0 | 2 | 2 |
| tiny_box | 21 | 0 | 21 | 0 | 0 |
| **TOPLAM** | 238 | 4 | 32 | 136 | 66 |

### Kategori anlamları
- **Confirmed Error:** kod ile kesin (birebir aynı etiket satırı).
- **Likely Error:** görselde açık biçimde hatalı/anlamsız ama insan teyidi önerilir (1-2 piksel kutu, baretli başta NO-Hardhat, baret boyutunun >2 katı Hardhat kutusu, yüksek görünürlüklü giysiye NO-Safety Vest vb.).
- **Valid Annotation:** meşru çoklu anotasyon (farklı kişiler, elde baret, vest kutusunun içindeki küçük arka plan kişisi, meşru negatif görüntü).
- **Needs Manual Review:** düşük çözünürlük/piksellenme/kısmi nesne nedeniyle görselden karar verilemedi.

## 3. Üç çelişkili örtüşme örneği (önceki rapordaki CONFLICTING_OVERLAP)
| Görüntü | Önceki uyarı | Yeni karar |
|---|---|---|
| `IMG_5024_mp4-2_jpg.rf.77ca6ff11b6d3b968a24…` | Sınıf 7 (satır 17) ile sınıf 4 (satır 16) aynı bölgede, IoU=0.31 | SI057: Needs Manual Review |
| `construction-5-_mp4-75_jpg.rf.4df4bcaa924d…` | Sınıf 7 (satır 20) ile sınıf 4 (satır 19) aynı bölgede, IoU=0.42 | SI079: Valid Annotation |
| `construction-897-_jpg.rf.c7afbfac58183e3c4…` | Sınıf 7 (satır 9) ile sınıf 4 (satır 16) aynı bölgede, IoU=0.45 | SI081: Needs Manual Review |

(Üç örnekte de vest/no_vest kutuları yan yana veya arka planda ayrı kişilere ait olabilir; karar tablosuna bakınız.)

## 4. Rastgele kalite örneklemi (96 kutu) ve tahmini hata oranı
| Rol | n | Geçerli | Belirsiz | Olası hata |
|---|---|---|---|---|
| helmet | 24 | 18 | 5 | 1 |
| no_helmet | 24 | 17 | 6 | 1 |
| vest | 24 | 18 | 5 | 1 |
| no_vest | 24 | 13 | 9 | 2 |

Toplam n=96: olası hata **5** (%5.2; Wilson %95 GA: %2.2–%11.6), belirsiz **25** (%26.0; GA %18.3–%35.6). **Sınırlamalar:** kutu yüksekliği <24 px olanlar örneklemden çıkarıldı (küçük kutularda hata oranı muhtemelen daha yüksek); örneklem küçük; tek değerlendirici; Roboflow train görüntülerinin çoğu 2×2 kolaj ve siyah kare (cutout) içerdiğinden kutular kolaj kenarlarında kırpılmıştır. Siyah kare içeren görüntü oranı: train %11.6, val %1.8, test %1.2 (sezgisel sayım).

## 5. Teknik sorunların değerlendirmesi
- **BOX_VERY_SMALL (21):** hepsi Likely Error — 1–2 piksel kutular, augmentation ile ince şeride kırpılmış bulanık görüntülerde; 9'u TEST görüntüsünde (`ka_01181…`), yani test metriklerini etkiler.
- **DUPLICATE_BOX (4):** Confirmed Error (birebir aynı satır). **NEAR_DUPLICATE_BOX (1):** Likely Error.
- **LABEL_EMPTY (23):** 23/23 görüntüde insan/KKD yok (kitapçı, havalimanı, kumarhane, kuş, iskele slaytı karesi): meşru negatif örnek. Ancak bunlar alan dışı (MIT indoor 'null images'), gerçek şantiye zor-negatifleri değildir.
- **Çelişki adayları (93):** çoğu meşru (farklı kişi, elde baret, küçük arka plan kişisi); Hardhat kutusu baret boyutunun çok üstünde 5 örnek Likely Error.

