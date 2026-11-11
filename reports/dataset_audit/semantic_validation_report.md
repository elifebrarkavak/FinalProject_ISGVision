# Kutu Semantiği Doğrulama Raporu (Aşama 2 eksik giderme)

_Üretim tarihi 2026-11-11 (`scripts/generate_followup_reports.py`). Sayılar `semantic_metrics.json`, `label_review.csv` ve `candidate_class_mapping.yaml` dosyalarından okunur. Görsel gözlemler model destekli, tek değerlendirici incelemesidir; kullanıcı doğrulaması DEĞİLDİR._

## 1. Amaç ve yöntem
Sınıf adı aynı olsa bile kutunun neyi kapsadığı veri setleri arasında farklı olabilir. Bu nedenle üç bağımsız kanıt birleştirildi:

1. **Nicel ölçü (kişi kutusuna göre):** `Person` sınıfı olan veri setlerinde (Construction-PPE, Roboflow) her baret/yelek kutusu merkezi ve alanı yeterince içeren en küçük kişi kutusuna atanır; kutunun kişiye göre yüksekliği, genişliği ve dikey konumu ölçülür. (Medyan *görüntü alanı* kadraj yakınlığından etkilendiği için artık kullanılmıyor.)
2. **Nicel ölçü (baş/gövde oranı):** Tam bir baş kutusu ve tam bir gövde kutusu olan görüntülerde baş/gövde boyut oranı; `Person` sınıfı olmayan Mendeley dahil üç veri seti için ORTAK ölçüdür.
3. **Görsel inceleme:** Her hedef sınıf için üç veri setinden aynı çevre payıyla kırpılmış 8'er kutu yan yana (`semantic_review/compare_*.jpg`) ve önceki aşamanın 24'er kırpma sayfaları açılıp incelendi.

**Karşılaştırma örnekleme bilgisi** (kutu yüksekliği <24 px olanlar ayırt edilemediği için örneklemeden çıkarıldı):

| Rol | Veri seti | Toplam kutu | Örneklenebilir (≥24 px) | Gösterilen |
|---|---|---|---|---|
| helmet | construction_ppe | 1576 | 1253 | 8 |
| helmet | mendeley_ppe | 2421 | 2266 | 8 |
| helmet | site_safety | 3551 | 2148 | 8 |
| no_helmet | construction_ppe | 813 | 532 | 8 |
| no_helmet | mendeley_ppe | 1174 | 1122 | 8 |
| no_helmet | site_safety | 2428 | 1755 | 8 |
| vest | construction_ppe | 1816 | 1764 | 8 |
| vest | mendeley_ppe | 1385 | 1383 | 8 |
| vest | site_safety | 3258 | 2685 | 8 |
| no_vest | construction_ppe | 567 | 531 | 8 |
| no_vest | mendeley_ppe | 1058 | 1058 | 8 |
| no_vest | site_safety | 4153 | 3816 | 8 |

## 2. Nicel ölçüler
### 2.1 Kutunun kişiye göre boyutu ve konumu

| Veri seti | Rol | Kişiye atanan | Göreli yükseklik | Göreli genişlik | Kişi tepesinden uzaklık | h/w (piksel) |
|---|---|---|---|---|---|---|
| construction_ppe | helmet | 1543/1575 | 0.133 [0.112–0.165] | 0.403 [0.354–0.469] | 0.012 [0.001–0.013] | 0.853 [0.735–0.97] |
| construction_ppe | no_helmet | 795/813 | 0.174 [0.148–0.234] | 0.385 [0.324–0.474] | 0.012 [0.002–0.015] | 1.102 [0.93–1.25] |
| construction_ppe | vest | 1814/1815 | 0.4 [0.374–0.442] | 0.738 [0.642–0.847] | 0.154 [0.129–0.179] | 1.432 [1.243–1.578] |
| construction_ppe | no_vest | 565/567 | 0.404 [0.369–0.466] | 0.782 [0.678–0.883] | 0.164 [0.142–0.202] | 1.343 [1.111–1.559] |
| site_safety | helmet | 3347/3547 | 0.19 [0.125–0.328] | 0.389 [0.308–0.515] | 0.014 [0.0–0.036] | 1.077 [0.885–1.323] |
| site_safety | no_helmet | 2328/2423 | 0.24 [0.15–0.381] | 0.42 [0.297–0.654] | 0.021 [0.0–0.059] | 0.802 [0.497–1.086] |
| site_safety | vest | 3227/3256 | 0.488 [0.384–0.651] | 0.697 [0.577–0.819] | 0.174 [0.049–0.277] | 1.596 [1.156–2.077] |
| site_safety | no_vest | 4035/4150 | 0.492 [0.367–0.633] | 0.651 [0.514–0.827] | 0.25 [0.091–0.431] | 1.273 [0.765–1.888] |

Değerler: medyan [Q1–Q3]. Mendeley'de `Person` sınıfı olmadığı için yoktur. Roboflow kişi kutularının çoğu kısmi/yakın çekim olduğu için göreli yükseklikleri yayılmıştır (Q1–Q3 geniş).

### 2.2 Baş/gövde oranı (tek baş + tek gövde kutulu görüntüler)

| Veri seti | Baş rolü | Çift sayısı | baş_h / gövde_h | baş_w / gövde_w | baş h/w |
|---|---|---|---|---|---|
| construction_ppe | no_helmet | 149 | 0.431 [0.345–0.507] | 0.5 [0.431–0.59] | 1.116 [0.889–1.263] |
| construction_ppe | helmet | 312 | 0.332 [0.291–0.375] | 0.563 [0.492–0.622] | 0.807 [0.7–0.919] |
| mendeley_ppe | no_helmet | 92 | 0.439 [0.396–0.514] | 0.51 [0.478–0.554] | 1.203 [1.074–1.6] |
| mendeley_ppe | helmet | 167 | 0.526 [0.452–0.62] | 0.547 [0.486–0.639] | 1.194 [1.062–1.346] |
| site_safety | helmet | 145 | 0.432 [0.299–0.686] | 0.613 [0.495–0.734] | 1.123 [0.828–1.467] |
| site_safety | no_helmet | 167 | 0.475 [0.348–0.706] | 0.663 [0.49–0.831] | 0.594 [0.336–0.969] |

### 2.3 Rol bazında en-boy oranı ve piksel yüksekliği

| Veri seti | Rol | Kutu | h/w medyan | Yükseklik medyan (px) | Yükseklik P10 (px) |
|---|---|---|---|---|---|
| construction_ppe | helmet | 1576 | 0.853 | 78.1 | 17.5 |
| construction_ppe | no_helmet | 813 | 1.103 | 41.6 | 13.6 |
| construction_ppe | vest | 1816 | 1.432 | 172.3 | 38.7 |
| construction_ppe | no_vest | 567 | 1.343 | 328.3 | 29.9 |
| mendeley_ppe | helmet | 2421 | 1.18 | 62.4 | 27.5 |
| mendeley_ppe | no_helmet | 1174 | 1.205 | 68.3 | 29.8 |
| mendeley_ppe | vest | 1385 | 1.361 | 139.8 | 55.3 |
| mendeley_ppe | no_vest | 1058 | 1.37 | 128.0 | 62.3 |
| site_safety | helmet | 3551 | 1.077 | 28.3 | 8.0 |
| site_safety | no_helmet | 2428 | 0.806 | 50.0 | 9.1 |
| site_safety | vest | 3258 | 1.6 | 80.0 | 14.1 |
| site_safety | no_vest | 4153 | 1.296 | 112.0 | 26.0 |

## 3. Görsel inceleme sonuçları (karşılaştırma sayfaları, 8'er kutu/veri seti/sınıf)

| Hedef sınıf | Construction-PPE | Mendeley | Roboflow |
|---|---|---|---|
| helmet | Baret nesnesine SIKI (8/8) | Baş+baret, YÜZ dahil (8/8) | Çoğunlukla sıkı (6/8); 2 örnek kolaj/siyah kare etkisinde |
| no_helmet | Tüm baş: alın–çene (8/8) | Tüm baş/yüz (8/8; şapka, mozaik yüz dahil) | Heterojen: saç/alın şeridi, kısmi yüz; 2/8'de baret benzeri nesne görünüyor |
| vest | Yelekli gövde (8/8) | Yelekli gövde (8/8) | Gövde; 2/8 kısmi/şerit kutu |
| no_vest | Gövde (8/8) | Gövde (8/8; çoğu inşaat dışı) | Gövde ama 3/8'de baret/kumaş bölgesi veya şerit: gürültülü |

_Not: 8'lik karoların tek başına istatistiksel gücü yoktur; sayılar yalnızca bu sayfalardaki görsel gözlemdir. Roboflow için ayrıca 24'er kutuluk rastgele örneklem yapıldı (bkz. `roboflow_label_review_report.md`)._

## 4. Bulgular ve düzeltmeler
- **Baret kutusu yalnızca bareti mi, başı da mı kapsıyor?** Construction-PPE: yalnızca baret (kişi yüksekliğinin 0.133'ü, h/w=0.853). Roboflow: yalnızca baret (h/w=1.077). Mendeley: **başı/yüzü de kapsıyor** — baş/gövde oranı 0.526 (Construction 0.332).
- **Baretsiz sınıfı baş mı, bütün insan mı?** Hiçbir veri setinde bütün insan değil. Construction-PPE ve Mendeley: tüm baş (baş/gövde 0.431 ve 0.439 — birbirine çok yakın). Roboflow: heterojen (h/w medyan 0.806: saç/alın şeritleri ve kısmi yüzler).
- **Yeleksiz sınıfı gövde mi, bütün insan mı?** Gövde. Construction-PPE: kişi yüksekliğinin 0.404'ü (vest: 0.4). Roboflow'da kısmi/yakın çekim kutular nedeniyle yayılmış.
- **Düzeltme (önceki taslağa):** Önceki raporda 'no_helmet kutusu helmet'ten ~4 kat küçük' ve 'no_vest kutusu vest'ten ~3 kat büyük' denmişti. Bu, medyan *görüntü alanı* karşılaştırmasından geliyordu ve kadraj yakınlığından etkileniyordu. Kişiye göre ölçüde Construction-PPE'de no_helmet kutusu helmet'ten büyüktür (baş vs baret) ve no_vest ile vest aynı boydadır. Eşleme dosyası buna göre düzeltildi.
- **Taşınan baret:** Construction-PPE'de 14 görüntüde kişinin elinde taşıdığı baret `helmet` etiketli ve aynı kişinin başı `no_helmet` (çelişki incelemesi, geçerli anotasyon). `helmet` sınıfı 'takılı baret' değil 'baret nesnesi' anlamındadır; KKD kararında konum kontrolü şart.
- **Roboflow etiket kalitesi:** bkz. `roboflow_label_review_report.md`.

## 5. Çelişkili etiket aramasının sonucu
Aynı özniteliğin olumlu/olumsuz sınıfları (helmet–no_helmet, vest–no_vest) aynı bölgede (IoU≥0.1 veya merkez içeride) ya da aynı kişi kutusuna atanmış olarak aranmıştır. Aynı görüntüde farklı kişilere ait helmet + no_helmet TEK BAŞINA hata sayılmamıştır.

| Veri seti | Çelişki adayı | Valid Annotation | Likely Error | Confirmed Error | Needs Manual Review |
|---|---|---|---|---|---|
| construction_ppe | 19 | 19 | 0 | 0 | 0 |
| mendeley_ppe | 0 | 0 | 0 | 0 | 0 |
| site_safety | 93 | 47 | 5 | 0 | 41 |

## 6. Sınıf eşlemesi sonucu (hedef: 0=helmet, 1=no_helmet, 2=vest, 3=no_vest)

| Veri seti | Kaynak | Hedef | Durum | Uyum | Güvenilirlik | Kutu semantiği (ölçülen) | Görsel sonuç | Tutarsızlıklar |
|---|---|---|---|---|---|---|---|---|
| construction_ppe | 0:helmet | helmet | confirmed | Uyumlu | Yüksek | Baret nesnesinin kendisi (sıkı kutu); kişiye göre yükseklik 0.13, en-boy h/w=0.85 (baş-baret genişten yüksek değil) | 24 kırpma + karşılaştırma sayfası (8/8): kutu bareti sıkı sarıyor, yüz/boyun dahil değil | Elde TAŞINAN baret de 'helmet' (14 görüntü, çelişki incelemesinde doğrulandı): sınıf 'takılı baret' değil 'baret nesnesi' anlamında |
| construction_ppe | 6:no_helmet | no_helmet | confirmed | Uyumlu | Yüksek | Baretsiz tüm BAŞ (saç/yüz); kişiye göre yükseklik 0.17, h/w=1.10 | 8/8 karşılaştırma + 24 kırpma: kutu başı (alından çeneye) sarıyor; kişinin tamamı değil | Çok küçük/bulanık/arkadan baş örnekleri; bir örnekte koyu başlık benzeri nesne no_helmet olabilir |
| construction_ppe | 2:vest | vest | confirmed | Uyumlu | Yüksek | Yelekli gövde; kişi yüksekliğinin ~%40'ı, üstten %15 aşağıda başlıyor | 8/8 + 24 kırpma: kutu yelekli gövdeyi sarıyor | Yok (kişi dışı kutu görülmedi) |
| construction_ppe | 11:no_vest | no_vest | confirmed | Uyumlu | Yüksek | Yeleksiz GÖVDE; kişiye göre yükseklik 0.40 (vest ile aynı), tüm kişi değil | 8/8 + 24 kırpma: kutu göğüs-bel gövdesini sarıyor | Yok |
| mendeley_ppe | 0:Helmet | helmet | incompatible | Uyumsuz | Yüksek | Baret + yüzün/başın tamamı (baş kutusu); baş/gövde oranı 0.53 (Construction helmet 0.33), h/w=1.19 (Construction 0.85) | 8/8 karşılaştırma + 24 kırpma: kutu yüzü ve çeneyi de kapsıyor; baret nesnesine sıkı DEĞİL | Aynı sınıf adı farklı bölge: Construction/Roboflow 'baret nesnesi' vs Mendeley 'baş+baret' |
| mendeley_ppe | 1:NoHelmet | no_helmet | confirmed | Uyumlu | Orta | Baretsiz tüm baş (saç/yüz); baş/gövde oranı 0.44 (Construction no_helmet 0.43), h/w=1.20 (1.10) | 8/8 + 24 kırpma: baş/yüz kutusu; şapka/maske/kapüşonlular ve piksellenmiş yüzler de var | Sahneler çoğunlukla inşaat dışı sıradan insanlar; başlıklı (şapka) baş da NoHelmet |
| mendeley_ppe | 3:Vest | vest | confirmed | Uyumlu | Orta | Yelekli gövde (torso); torso/baş boyutları Construction ile tutarlı (NoHelmet bazlı oran 0.44 vs 0.43) | 8/8 + 24 kırpma: kutu yelekli gövdeyi sarıyor; yansıtıcı/yarı yansıtıcı çeşitli yelek türleri | Kişi sınıfı olmadığından kişiye göre ölçü yok; torso uyumu dolaylı (baş oranı üzerinden) |
| mendeley_ppe | 2:NoVest | no_vest | confirmed | Uyumlu | Orta | Yeleksiz gövde (göğüs-bel) | 8/8 + 24 kırpma: gövde kutuları; çoğu sıradan kıyafetli (inşaat dışı) kişi | KKD gerektiren bağlam çoğu görüntüde yok: 'yelek yok' ama yelek zorunlu bir ortam olmayabilir |
| site_safety | 0:Hardhat | helmet | confirmed | Uyumlu | Orta | Baret nesnesi (sıkı kutu), h/w=1.08 | 8/8 karşılaştırma; rastgele 24 örneklemde 18 geçerli, 5 belirsiz, 1 olası hata (gövde bölgesine kutu) | Çok küçük/bulanık kutular (p10 yükseklik 8 px), kolaj/augmentation kenarında kesik kutular; 5 kutu >=%45 kişi boyunda (çoğu yakın çekim) |
| site_safety | 2:NO-Hardhat | no_helmet | incompatible | Uyumsuz | Orta | Karışık: saç/alın ŞERİDİ (h/w medyan 0.80, yaygın <0.5), yüzün üst yarısı ve bazen tüm baş; Construction no_helmet h/w=1.10 | 8/8 karşılaştırma ve 24 örneklem: ~10/24 kutu saç/alın şeridi veya kısmi yüz; AN13'te mavi baret görünüyor (hata) | Kutu tanımı heterojen; olası etiket hatası (baretli baş); eşik altı bulanık/kenar kutular |
| site_safety | 7:Safety Vest | vest | confirmed | Uyumlu | Orta | Yelek/gövde bölgesi, kişiye göre yükseklik 0.49 (kısmi/yakın çekim kişi kutuları nedeniyle yayılmış) | 8/8 karşılaştırma; 24 örneklemde 18 geçerli, 5 belirsiz, 1 olası hata (karanlık cam bölgesi) | Kolaj/augmentation kaynaklı kısmi şerit kutular; bir örnek yerdeki yelek |
| site_safety | 4:NO-Safety Vest | no_vest | needs_review | Karar verilemedi | Düşük | Yeleksiz gövde/göğüs; çok sayıda kısmi (şerit), yakın çekim (maskeli yüz+tişört) veya belirsiz bölge | 24 örneklemde 13 geçerli, 9 belirsiz, 2 olası hata (parlak sarı-yeşil giysi 'yelek yok' etiketli); 8/8 karşılaştırmada baret/kumaş bölgeleri | Baret yakın çekimi veya yüksek görünürlüklü giysi içeren NO-Safety Vest kutuları; birçok kutu gövde dışı |

Durum sayıları (hedefe eşlenen 12 sınıf): {'confirmed': 9, 'incompatible': 2, 'needs_review': 1}. Hedef dışı 14 sınıf `incompatible` (hedef dörtlüde karşılığı yok).

**Kesin doğrulanamayanlar:** `NO-Safety Vest` (needs_review) ve Mendeley/Roboflow'daki alan kayması. `confirmed` statüsü kutu semantiğinin hedef tanımıyla tutarlı olduğunu ve örneklemde olası hatanın ≤%10 olduğunu söyler; etiketlerin tamamının doğru olduğunu DEĞİL.

## 7. Öneri (uygulanmadı)

- Hedef tanım: helmet=baret nesnesi (sıkı), no_helmet=baretsiz baş, vest/no_vest=gövde. Bu tanıma uymayan Mendeley `Helmet` (baş+baret) ve Roboflow `NO-Hardhat` (heterojen) doğrudan eşlenmemeli; kullanıcı kararı sonrası ayrı ele alınmalı.
- Baret etiketi 'takılı baret' anlamı taşımadığından KKD kararı baret kutusunun baş bölgesiyle konumsal ilişkisine dayanmalıdır (Aşama 11 entegrasyon katmanı).
