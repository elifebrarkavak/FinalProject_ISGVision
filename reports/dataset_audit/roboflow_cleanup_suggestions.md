# Roboflow Etiket Temizliği — ÖNERİLEN Düzeltmeler (UYGULANMADI)

_Üretim tarihi 2026-11-11. Bu belge yalnızca gelecekteki temizlik için öneridir. Ham veri (`data/raw/site_safety/`) değiştirilmemiştir; hiçbir etiket silinmemiş veya düzeltilmemiştir. Temizlik yapılacaksa YENİ bir kopyada ve kullanıcı onayıyla uygulanmalıdır._

## Özet

| Kategori | Öğe |
|---|---|
| Confirmed Error | 4 |
| Likely Error | 32 |
| Needs Manual Review | 66 |
| Valid Annotation | 136 |

## Önerilen eylemler

| Öğe grubu | Sayı | Önerilen eylem | Gerekçe |
|---|---|---|---|
| duplicate_box (Confirmed Error) | 4 | Yinelenen satırı yeni kopyadan sil (ilkini koru) | Birebir aynı satır, eğitim kaybını iki kez ağırlıklandırır |
| near_duplicate_box (Likely Error) | 1 | İki kutudan birini kaldır (IoU≥0.95 aynı sınıf) | Aynı nesne iki kez etiketli |
| tiny_box (Likely Error) | 21 | Kutuyu yeni kopyadan çıkar veya görüntüyü eğitimden/testten hariç tut | 1-2 px kutular öğrenilemez; 9'u test setinde metrikleri bozar |
| Hardhat kutusu baret boyutunun çok üstünde (Likely Error) | 5 | Kutuyu baret sınırına küçült veya kaldır (insan onayı) | Başı/gövdeyi de kapsıyor |
| Kalite örneklemi olası hatalar | 5 | Örnek dosyaları düzelt; kalan veri için sistematik tarama (aşağıda) | Baretli başta NO-Hardhat, yelekli giysiye NO-Safety Vest, gövde bölgesinde Hardhat |
| Needs Manual Review | 66 | İnsan incelemesi; kararsızsa eğitimde `ignore`/hariç | Görselden karar verilemedi |

## Sistematik tarama önerileri (veri temizliği öncesi)

- `NO-Safety Vest` kutularında yüksek görünürlüklü renk (parlak sarı/yeşil/turuncu) oranı hesaplanıp yüksek olanlar incelenmeli (örneklemde 2/24).
- `NO-Hardhat` kutularının içinde baret rengi/şekli içerenler (ör. Hardhat sınıflı kutu ile örtüşme, renk doygunluğu) taranmalı.
- Kutu yüksekliği <12 px ve kolaj kenarına değen kutular için hariç tutma eşiği belirlenmeli; hariç tutma kararı Aşama 9–10 (birleştirme/geliştirilmiş model) eğitim stratejisine bağlıdır.
- 2×2 kolaj görüntülerde (train) kutular kolaj karosuna göre yeniden değerlendirilmeli; mümkünse kolaj olmayan görüntüler ayrı değerlendirilmeli.

## Aday listesi

`roboflow_cleanup_candidates.csv`: Confirmed Error + Likely Error öğeleri (görüntü, etiket dosyası, satır, satır metni, gerekçe).
