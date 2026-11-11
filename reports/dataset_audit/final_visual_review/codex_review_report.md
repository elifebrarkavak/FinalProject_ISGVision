# Aşama 2E — Codex Bağımsız Görsel Doğrulama ve Kapanış Raporu

_Tarih: 2026-11-11. **Tarihsel Aşama 2E kaydıdır.** Aşağıdaki bağımsız öneriler kullanıcı kararı değildir. Aşama 2F'de kullanıcı 22/22 kararı tamamlamış; 13 `approved`, 4 `needs_reannotation`, 5 `quarantine` kararı uygulanmıştır. Güncel kapsam 990 görüntü (794 train, 196 validation) ve Aşama 2 durumu `COMPLETED`'dır. Yetkili güncel sonuç: `../STAGE_2_CLOSURE_REPORT.md`._

## A. Mevcut projenin durumu

- Proje kökü mevcut depo olan `/Users/ebrmac/Desktop/bitirme`; yeni `ISG_VISION` klasörü oluşturulmadı.
- 12 aşamalı plan `docs/project_plan.md` ve `configs/project_stages.yaml` arasında tutarlı.
- Aşama 3 yalnızca canlı kamera altyapısıdır; model eğitimi veya veri birleştirme içermez.
- İlk YOLO eğitimi Aşama 8, veri birleştirme Aşama 9, geliştirilmiş eğitim Aşama 10'dur.
- Yalnızca Construction-PPE onaylı kapsamda. Mendeley ve Roboflow ham verileri korunuyor ve `pending_license_verification` kapsamından çıkarılmadı.
- Sanal ortam Python 3.11.17. İlk doğrudan import denemesinde editable paket kaydı bulunmadığı için `ModuleNotFoundError` alındı; `.venv/bin/python -m pip install -e . --no-deps` ile mevcut paket sanal ortama yeniden bağlandı. Ardından `isg_vision`, `src/isg_vision/__init__.py` üzerinden başarıyla import edildi.
- Başlangıç git çalışma ağacı temizdi (`## main`). Otomatik commit/push yapılmadı.

## B. 22 görüntünün bağımsız incelemesi

22 sayfalık `contact_sheet.pdf` ve 22 ayrı JPG sayfasının tamamı açıldı. Her sayfada kutusuz görüntü, tüm kutular, şüpheli kişi ve kutulu/kutusuz yakınlaştırmalar incelendi. Ayrıntılı satır bazlı sonuç: `codex_independent_review.csv`.

Codex özeti:

| Codex önerisi | Sayı | İnceleme ID'leri |
|---|---:|---|
| `approved_candidate` | 13 | R01, R03, R06, R08, R09, R10, R11, R12, R14, R16, R17, R18, R22 |
| `needs_reannotation` | 5 | R02, R04, R15, R19, R21 |
| `quarantine` | 1 | R05 |
| `inconclusive` | 3 | R07, R13, R20 |

## C. Claude–Codex karşılaştırması

- Uyum: **20/22** (`agreement`).
- Farklı karar: **2/22** (`disagreement`): R07 ve R21.
- Aynı fikirde olmak insan onayı değildir; bu nedenle 22 satırın tamamı kullanıcı incelemesi gerektirir.

Farklar:

| ID | Claude | Codex | Bağımsız gerekçe |
|---|---|---|---|
| R07 | `needs_reannotation` | `inconclusive` | Uzak figür ve koyu baş/gövde olasılığı görülüyor ancak alan derinliği bulanıklığı no_helmet/no_vest sınıflarını güvenilir ayırmaya yetmiyor. İnsan kararına kadar bekletmek daha güvenli. |
| R21 | `inconclusive` | `needs_reannotation` | Baş zaten no_helmet ile etiketli. Cam/direksiyon örtmesine rağmen turuncu-kahverengi üst gövde giysisi seçiliyor ve reflektif yelek görünmüyor; no_vest eksikliği daha güçlü. |

## D. Kesinleşen sorunlar

- Güçlü eksik baş etiketi kanıtı: R02, R04 ve R19 (`no_helmet`).
- Güçlü eksik gövde etiketi kanıtı: R15 ve R21 (`no_vest`).
- Birden çok güvenilirlik sorunu: R05; baş etiketi eksikliğiyle birlikte `vest` kutusu koyu/reflectif olmayan üst giysiyi kapsıyor. Karantina önerildi.
- Görsel kanıt yetersiz: R07, R13, R20. Bunlar otomatik onaylanmamalı.
- İncelenen sayfalarda başka kişiye ait baş/gövde kutusunun şüpheli kişiye yanlış ilişkilendirildiği kesin bir örnek görülmedi.

## E. Kullanıcı onayı gerektiren görüntüler

Öncelikli hızlı inceleme sırası:

1. **Farklı değerlendirilenler:** R07 (sayfa 7), R21 (sayfa 21).
2. **Güçlü etiket hataları:** R02 (2), R04 (4), R15 (15), R19 (19); R21 (21).
3. **Karantina adayı:** R05 (5).
4. **Belirsizler:** R07 (7), R13 (13), R20 (20).
5. Kalan 13 onay adayı: R01, R03, R06, R08–R12, R14, R16–R18, R22.

Gerçek görüntü yolu ve sayfa numarası `codex_independent_review.csv` içindedir. Mevcut PDF her görüntü için orijinal, tüm kutular ve iki yakınlaştırma sunduğundan yeni PDF üretilmedi.

## F. Veri kullanım politikası ve olası sayılar

Mevcut gerçek manifest: **999** kullanılabilir görüntü = **800 train + 199 val**. Önceden doğrulanmış `ultralytics-ppe-nov2026-690.jpg` kullanım dışıdır.

22 adayın her biri `unit_size=1` olan ayrı kaynak birimidir; dolayısıyla bu görüntülerden birini dışlamak başka bir görüntünün bölümünü değiştirmez.

| Senaryo | Dışlanan train | Dışlanan val | Son train | Son val | Toplam |
|---|---:|---:|---:|---:|---:|
| Hiçbiri dışlanmaz | 0 | 0 | 800 | 199 | 999 |
| Codex'in kesin dışlama önerileri (`needs_reannotation` + `quarantine`) | 4 | 2 | 796 | 197 | 993 |
| Codex güvenli varsayılanı (belirsizler de dışarıda) | 6 | 3 | 794 | 196 | 990 |
| Claude'un kesin dışlama önerileri | 5 | 1 | 795 | 198 | 993 |
| Claude güvenli varsayılanı (belirsizler de dışarıda) | 6 | 3 | 794 | 196 | 990 |

Bu tablo Aşama 2E anındaki senaryoları gösterir. Aşama 2F'ye gelindiğinde kullanıcı güvenli 990 görüntülük senaryoyu seçmiş ve politika manifeste uygulanmıştır.

## G. Kod ve dosya değişiklikleri

Eklenen dosyalar:

- `reports/dataset_audit/final_visual_review/codex_independent_review.csv`
- `reports/dataset_audit/final_visual_review/codex_review_report.md`

Kaynak kodda hata düzeltmesi gerekmedi. Mevcut iş akışında:

- Geçersiz `user_decision` reddediliyor.
- `needs_reannotation`/`quarantine` için kullanıcı notu zorunlu.
- Model önerisi tek başına override üretmiyor.
- Kullanıcı dışlaması `final_split` alanını boşaltıyor ve sayılar yeniden hesaplanıyor.
- Kaynak birimi bütünlüğü denetleniyor.
- Geçersiz karar dosyası bütün yazma işlemlerinden önce reddediliyor.
- Ham görüntüler ve orijinal etiketler yalnızca okunuyor.
- Mevcut testler sentetik/geçici klasörlerde karar doğrulama ve tekrar çalıştırılabilirlik davranışını kapsıyor.

## H. Test sonuçları

Gerçekten çalıştırılanlar:

- `.venv/bin/python -m unittest discover -s tests -v` → **169 test, tamamı başarılı**.
- `.venv/bin/python scripts/run_stage2c_checks.py` → **12/12 denetim geçti**.
- `.venv/bin/python scripts/check_visual_review.py` → CSV geçerli; **0 kullanıcı kararı, 22 bekleyen**; normal modda başarılı.
- `.venv/bin/python scripts/check_visual_review.py --require-complete` → beklendiği gibi başarısız kapı; sebep **22 eksik kullanıcı kararı**, teknik test hatası değil.

## I. Ham veri bütünlüğü

`run_stage2c_checks.py` T6.8 doğrulaması geçti:

- Construction-PPE yayıncı manifestindeki 1000 görüntünün ad/bölüm/SHA-256 eşleşmesi doğru.
- Üç veri setinin ham dosyaları özgün ZIP kayıtlarıyla boyut ve CRC32 açısından eşleşiyor.
- Ek/eksik ham dosya bulunmadı.
- ZIP, görüntü ve etiket dosyaları değiştirilmedi veya silinmedi.

## J. Aşama 2E anındaki kapanış değerlendirmesi (tarihsel)

Aşağıdaki maddeler Aşama 2E sonunda geçerliydi ve Aşama 2F'de tamamlandı:

1. 22 satırın her biri için insan kararı girilmeli.
2. Dışlama kararlarında kısa kullanıcı notu yazılmalı.
3. `check_visual_review.py --require-complete` geçmeli.
4. Politika uygulanıp raporlar/manifest yeniden üretilmeli.
5. 12/12 bütünlük denetimi ve 169 test yeniden geçmeli.
6. Kullanıcı Aşama 2 kapanışını açıkça onaylamalı.

Model eğitimi, veri birleştirme, kamera geliştirme ve Aşama 3 çalışması yapılmadı.

## K. Aşama 2E sonunda kullanıcıya sunulan işlemler (tamamlandı)

1. Önce R07 ve R21'i PDF sayfa 7 ve 21'de kontrol edin.
2. Ardından R02, R04, R05, R15, R19 ile belirsiz R13/R20'yi gözden geçirin.
3. `review_decisions.csv` içindeki yalnızca `user_decision` ve `user_note` alanlarını doldurun. Geçerli kararlar: `approved`, `needs_reannotation`, `quarantine`.
4. Kararlar tamamlandıktan sonra doğrulama/uygulama komutlarını çalıştırın veya çalıştırmamı isteyin.
5. Son çıktıları gördükten sonra Aşama 2 kapanış kararını verin.
