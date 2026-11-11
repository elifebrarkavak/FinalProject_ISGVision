# Son Görsel İnceleme Rehberi (Aşama 2D)

_22 görüntü (train 18, val 4). Model destekli öneriler hazırlandı; bu öneriler **kullanıcı kararı değildir**. Siz `user_decision`
sütununu doldurana kadar bu 22 görüntü kullanıcı tarafından doğrulanmış sayılmaz. Ham veri ve etiket dosyaları değiştirilmedi._

## 1. Dosyalar

| Dosya | İçerik |
|---|---|
| `contact_sheet.pdf` | 22 sayfa (A4 yatay, 300 dpi), her görüntü için bir sayfa |
| `pages/RNN_<dosya>.jpg` | PDF ile aynı sayfalar, ayrı görsel dosyalar |
| `review_decisions.csv` | Görüntü başına model önerisi, gerekçe ve **boş** kullanıcı karar/not alanları |
| `review_guide.md` | Bu rehber |

Model önerilerinin kaynağı: `configs/review/stage2d_visual_review.yaml`.

## 2. Bir sayfa nasıl okunur?

| Panel | İçerik |
|---|---|
| Başlık | İnceleme no (R01–R22), dosya yolu, bölüm (train/val), çözünürlük, risk kodu, şüpheli kişi no ve kutu boyutu, şüpheli kişinin çevresindeki baş/gövde etiketlerinin **hangi kişiye atandığı** |
| A | Orijinal görüntü (kutusuz) |
| B | Tüm kutular ve sınıf adları. **Sarı kalın kutu = şüpheli kişi**; P# = kişi numarası |
| C | Şüpheli kişinin yakınlaştırması (kutulu) |
| D | Şüpheli bölgenin yakınlaştırması (kutusuz): baş sorunu varsa baş, gövde sorunu varsa gövde |
| E | D ile aynı bölge, kutulu |
| Alt satır | Model önerisi ve gerekçesi (mavi) |

Renkler:

- helmet: yeşil
- no_helmet: kırmızı
- vest: mavi
- no_vest: mor
- Person: turuncu-sarı
- diğer sınıflar (gloves, boots vb.): gri

## 3. Risk kodları

| Kod | Anlamı |
|---|---|
| `head_unexplained` | ≥ 60 px kişiye helmet/no_helmet atanmamış. Baş kadraj dışında değil ve yakında baş etiketi yok |
| `torso_unexplained` | ≥ 120 px kişiye vest/no_vest atanmamış; açıklayıcı bir neden (kenar, yakındaki etiket) yok |
| `torso_small_person` | Kişi < 120 px; gövde özniteliği güvenilir etiketlenemeyebilir |
| `head_out_of_frame_top` | Kişi üst kenara değiyor (baş kadraj dışı olabilir) |

## 4. Karar ölçütü

Yayıncı notuna göre örtülü veya kadraj dışı bir bölge tek başına eksik KKD kanıtı değildir.

| Önerilen karar | Ne zaman? | Sizin seçeceğiniz `user_decision` |
|---|---|---|
| `approved_candidate` | Baş/gövde örtülü, kadraj dışı ya da ayırt edilemeyecek kadar küçük/bulanık/karanlık; etiket beklenmez | Katılıyorsanız `approved` |
| `needs_reannotation` | Hedef bölge açıkça görünüyor ve benzer kişiler etiketliyken etiketsiz | `needs_reannotation` (eksik etiket eklenene kadar kullanım dışı) |
| `quarantine` | Birden çok etiket şüpheli; güvenilirlik yetersiz | `quarantine` |
| `inconclusive` | Görsel kanıtla karar verilemiyor | Sizin kararınız. Güvenli varsayılan `quarantine`; görüntüyü net değerlendirebiliyorsanız `approved` veya `needs_reannotation` |

## 5. Model önerilerinin özeti

| Öneri | Sayı | İnceleme no |
|---|---|---|
| approved_candidate | 13 | R01, R03, R06, R08, R09, R10, R11, R12, R14, R16, R17, R18, R22 (val: R22) |
| needs_reannotation | 5 | R02, R04, R07, R15, R19 (val: R19) |
| quarantine | 1 | R05 |
| inconclusive | 3 | R13, R20, R21 (val: R20, R21) |

**Önerilere göre onaylanabilecek görüntüler (16 görüntüye kadar):**

- 13 `approved_candidate` görüntü.
- 3 `inconclusive` görüntü, siz onaylarsanız.

**Önerilere göre dışlanması gereken görüntüler (6–9 görüntü):**

- 5 `needs_reannotation` görüntü.
- 1 `quarantine` görüntü.
- 3 `inconclusive` görüntü de güvenli varsayılan seçilirse.

"Başka kişiye ait etiket" kontrolü: şüpheli kişinin bölgesine değen baş/gövde etiketlerinin hepsi doğru kişiye atanmış
görünüyor (R08, R12, R17, R19 ve R21'de komşu etiketler P1/P2/P8'e ait). Etiketlerin yanlış kişiye kaydığı bir durum görülmedi.

## 6. Kararınızı nasıl girersiniz?

1. `contact_sheet.pdf` dosyasını açın ve her sayfayı inceleyin.
2. `review_decisions.csv` dosyasında her satır için:
   - `user_decision` sütununa şu değerlerden birini yazın: `approved`, `needs_reannotation` veya `quarantine`.
     Model önerisi değerlerini (`approved_candidate`, `inconclusive`) **yazmayın**; doğrulama bunları reddeder.
   - `user_note` sütununa kısa bir not yazın. `needs_reannotation` ve `quarantine` için **zorunludur**
     (örn. "baş görünür, no_helmet eksik").
   - Diğer sütunları değiştirmeyin.
3. Dosyayı UTF-8 CSV olarak kaydedin. Excel kullanıyorsanız "CSV UTF-8" biçimini seçin.

## 7. Doğrulama ve uygulama

Önce kararları doğrulayın ve etkisini görün:

```bash
python scripts/check_visual_review.py
```

Ardından kararları manifeste işleyin ve denetimleri çalıştırın:

```bash
python scripts/apply_usage_policy.py
```

```bash
python scripts/generate_stage2b_reports.py
```

```bash
python scripts/run_stage2c_checks.py
```

```bash
python -m unittest discover -s tests
```

- `check_visual_review.py` CSV'yi doğrular, kullanım dışı kalacak görüntüleri listeler ve model önerisinden farklı kararlarınızı
  gösterir. Hiçbir dosyayı değiştirmez.
- `apply_usage_policy.py` kararlarınızı `construction_ppe_split_manifest.csv` dosyasına işler:
  - `needs_reannotation`/`quarantine` → `final_split` boşalır, görüntü kullanım dışı kalır (kural `USER-VISUAL-REVIEW`).
  - `approved` → görüntü kullanımda kalır; `visual_review` sütununda `user=approved` görünür.
- Geçersiz değer veya gerekçesiz dışlama varsa `apply_usage_policy.py` hata verir ve hiçbir şey yazmaz.
- Ham görüntü ve etiket dosyaları hiçbir adımda değişmez. Yeniden etiketleme ileride türetilmiş bir etiket kopyası üzerinde
  yapılır (Aşama 8 hazırlığı).
- CSV yeniden üretilirse (`scripts/build_visual_review.py`) `user_decision` ve `user_note` değerleriniz korunur.

**Aşama 8 öncesi kapı:** `python scripts/check_visual_review.py --require-complete` komutu karar bekleyen satır varsa
başarısız olur. İlk YOLO eğitimine (Aşama 8) bu kapı geçilmeden başlanmamalıdır.
