# Lisans Doğrulama Raporu (Aşama 2 eksik giderme)

_Kontrol tarihi: 2026-11-11. Hukuki görüş değildir; bitirme raporunda kullanılacak atıf/lisans ifadeleri yayıncının güncel sayfasından yeniden teyit edilmelidir. Erişim kontrolleri aşılmaya çalışılmadı: HTTP 403 veren sayfalar için yeniden/alternatif yolla erişim denenmedi._

## 1. Özet

| Veri seti | İddia edilen lisans | Doğrulama durumu | Resmî bağlantı | Kontrol tarihi |
|---|---|---|---|---|
| construction_ppe | CC BY 4.0 | Resmi kaynaktan DOĞRULANDI | https://docs.ultralytics.com/datasets/detect/construction-ppe/ | 2026-11-11 |
| mendeley_ppe | CC BY 4.0 | KISMEN doğrulandı | https://data.mendeley.com/datasets/zkzghjvpn2/6 | 2026-11-11 |
| site_safety | CC BY 4.0 | Yalnızca YEREL dosyadan (resmi doğrulama YOK) | https://universe.roboflow.com/roboflow-universe-projects/construction-site-safety/dataset/27 | 2026-11-11 |

## construction_ppe — Ultralytics Construction-PPE

- **İddia edilen lisans:** CC BY 4.0  
- **Lisans bağlantısı:** https://creativecommons.org/licenses/by/4.0/  
- **Doğrulama durumu:** Resmi kaynaktan DOĞRULANDI

**Kanıt kaynakları:**

| Tür | Kaynak | Sonuç |
|---|---|---|
| official_page | `https://docs.ultralytics.com/datasets/detect/construction-ppe/` | HTTP 200; 'released ... under Creative Commons Attribution 4.0 International (CC BY 4.0)'; önerilen atıf metni ve BibTeX var |
| local_file | `data/raw/construction_ppe/LICENSE.txt + README.md` | Tam CC BY 4.0 yasal metni ve aynı atıf talimatı arşivde mevcut |

**Bilinen kullanım koşulları:** Atıf zorunlu (yaratıcı adı, telif/lisans bildirimi, lisans bağlantısı, değişiklik belirtme); paylaşım ve uyarlamaya izin verir. Önerilen atıf: 'Ultralytics Construction-PPE Dataset, Ultralytics team, November 2026, CC BY 4.0.'

**Açıkta kalan belirsizlikler:**

- Veri 'insan toplu ve üretilmiş görüntüler' içeriyor (README); üretilmiş görüntülerin ayrıca üçüncü taraf hakkı taşıyıp taşımadığı belgelenmemiş.
- Sürüm numarası yok; 'November 2026' tarihine atıf yapılmalı.

## mendeley_ppe — Mendeley Data - Dataset of Personal Protective Equipment (PPE), v6

- **İddia edilen lisans:** CC BY 4.0  
- **Lisans bağlantısı:** http://creativecommons.org/by/4.0 (Mendeley API 'data_licence.url' alanında verildiği haliyle)  
- **Doğrulama durumu:** KISMEN doğrulandı

**Kanıt kaynakları:**

| Tür | Kaynak | Sonuç |
|---|---|---|
| official_page | `https://data.mendeley.com/datasets/zkzghjvpn2/6` | HTTP 200; 'Licence: CC BY 4.0'; Yayım 28 Temmuz 2025, Sürüm 6, DOI 10.17632/zkzghjvpn2.6; yazarlar Mei-Ling Huang, Ying Cheng (National Chin-Yi University of Technology); 2,286 görüntü 640x640 iddiası |
| official_page | `https://data.mendeley.com/public-api/datasets/zkzghjvpn2` | HTTP 200 JSON; data_licence short_name 'CC BY 4.0', full_name 'Creative Commons Attribution 4.0 International'; dosya girdisi 20250731-PPE2286y.zip, 236065015 bayt, sha256 7a22e5cbc0327971f56358e14f1fce88fec1d6ab30cfac642de0746202ae1f0d |
| local_file | `~/Downloads/20250731-PPE2286y.zip` | YEREL ZIP'in SHA-256'sı ve boyutu resmi API ile BİREBİR eşleşiyor (kendi hesabımız): indirilen dosya resmi arşivle aynı |
| local_file | `data/raw/mendeley_ppe/README.dataset.txt` | Veri seti Roboflow Universe projesi 'ying2001/20250731-ppe2286-m0ega' (v1, 26 Tem 2025) olarak dışa aktarılmış; 'License: CC BY 4.0' (yerel, bağımsız doğrulama değil) |
| official_page | `https://data.mendeley.com/faq` | Mendeley SSS: depositor veriye sahip kalır; telifli içerik (sahibi olmadığınız görüntü vb.) yüklenmemelidir; lisans alanı varsayılan CC BY 4.0'dır. Yayıncı bu veri setindeki üçüncü taraf görsellerin haklarını ayrıca teyit etmiyor |
| official_page | `https://creativecommons.org/faq/` | CC lisansları yalnızca lisans verenin sahip olduğu haklar için geçerlidir; üçüncü taraf materyal ayrıca işaretlenmeli (SSS'ten özetlenen ifade; sayfanın bir kısmı özet olarak okundu) |

**Bilinen kullanım koşulları:** CC BY 4.0: uygun atıf, lisans bağlantısı, değişiklik belirtme. Mendeley API açıklaması: 'üçüncü tarafa ait olarak tanımlanan içerik için ek izin gerekebilir'.

**Açıkta kalan belirsizlikler:**

- Görseller 'GitHub, Kaggle, Roboflow ve gerçek fotoğraflar'dan derlenmiş (sayfa: yalnızca genel kaynak adları; kaynak listesi, bağlantı ve üst kaynak lisansları YOK).
- Üst kaynakların lisansları ve görsel bazında hak sahipliği DOĞRULANAMADI; veri setinin CC BY 4.0 olması, içindeki üçüncü taraf görseller için otomatik hak doğrulaması sağlamaz (Mendeley sayfası da bunu belirtiyor).
- Görsel incelemede (bu aşamada) veri setindeki bazı görüntülerin ticari fotoğraf/ürün resimleri (filigranlı stok fotoğraf, ürün fotoğrafı) olduğu görüldü: kaynak haklarının ayrıca değerlendirilmesi gerekir.
- Mendeley sayfası 'resmi YOLO biçimi, 640x640' diyor; gerçekte 905 farklı çözünürlük var (olgu farkı).
- Roboflow proje sayfası (ying2001/20250731-ppe2286-m0ega) HTTP 403 verdi; çevrimiçi doğrulanamadı.
- Mendeley sitesinin altbilgisi site içeriği için metin/veri madenciliği ve yapay zekâ eğitimi haklarını saklı tuttuğunu söylüyor; bu site içeriği içindir, dataset lisansı CC BY 4.0'dır (açık erişim içerik için CC şartları geçerli). Hukuki yorum değildir.

## site_safety — Roboflow Universe - Construction Site Safety v27 (YOLOv8)

- **İddia edilen lisans:** CC BY 4.0  
- **Lisans bağlantısı:** https://creativecommons.org/licenses/by/4.0/  
- **Doğrulama durumu:** Yalnızca YEREL dosyadan (resmi doğrulama YOK)

**Kanıt kaynakları:**

| Tür | Kaynak | Sonuç |
|---|---|---|
| official_page | `https://universe.roboflow.com/roboflow-universe-projects/construction-site-safety/dataset/27` | HTTP 403 Forbidden (erişim kontrolü); aşılmaya çalışılmadı |
| official_page | `https://universe.roboflow.com/roboflow-universe-projects/construction-site-safety` | HTTP 403 Forbidden |
| local_file | `data/raw/site_safety/data.yaml` | 'roboflow: workspace roboflow-universe-projects, project construction-site-safety, version 27, license: CC BY 4.0' (Roboflow dışa aktarım metadatası) |
| local_file | `data/raw/site_safety/README.dataset.txt` | 'Provided by a Roboflow user. License: CC BY 4.0'; görüntüler başka Roboflow Universe projelerinden ve iki YouTube videosundan klonlanmış |
| local_file | `data/raw/site_safety/README.roboflow.txt` | 10 Ocak 2023 dışa aktarım, v27, 2801 görüntü, ön işleme ve augmentation listesi |
| official_page | `https://docs.roboflow.com/ ve https://roboflow.com/universe` | HTTP 200 ancak Universe lisans alanının tanımı veya üçüncü taraf içerik sorumluluğu hakkında metin bulunamadı |
| official_page | `https://docs.roboflow.com/datasets/create-and-upload/adding-data` | HTTP 200; 'You retain ownership over all images and videos you upload' (yükleyici sahipliği); Universe lisansını tanımlamıyor |

**Bilinen kullanım koşulları:** Yerel metadata CC BY 4.0 diyor (atıf + değişiklik belirtme varsayılır); resmi sayfadan bağımsız doğrulama YOK.

**Açıkta kalan belirsizlikler:**

- Lisans yalnızca arşivdeki Roboflow dışa aktarım metadatasından ve README'den biliniyor; yayıncının sayfası (403) ve Roboflow dokümantasyonu bu bilgiyi doğrulamadı.
- Üst projelerin sayfaları (personal-protective-equipment-combined-model, people-and-ladders, safety-vests, excavators-cwlh0, mit-indoor-scene-recognition, people-detection-general, construction-madness) HTTP 403: üst kaynak lisansları BİLİNMİYOR.
- Görüntüler iki YouTube videosundan karelerdir (README); video sahiplerinin hakları/YouTube lisansı belgelenmemiş. 'MIT indoor scenes' kaynaklı boş (null) görüntülerin kendi koşulları da doğrulanmadı.
- Görüntülerin bir kısmı filigranlı/ticari görünümlü fotoğraflar ve kolajlar (görsel inceleme); hak sahipliği doğrulanamadı.

## Atıf metinleri (bitirme raporu için)

- Ultralytics Construction-PPE Dataset, Ultralytics team, November 2026, CC BY 4.0.
- Huang, M.-L.; Cheng, Y. Dataset of Personal Protective Equipment (PPE), Mendeley Data, v6, 2025. doi:10.17632/zkzghjvpn2.6
- Construction Site Safety (v27), Roboflow Universe, roboflow-universe-projects, CC BY 4.0, https://universe.roboflow.com/roboflow-universe-projects/construction-site-safety/dataset/27

## Değerlendirme

- **Doğrulanan:** Construction-PPE (resmi sayfa + arşivdeki lisans metni); Mendeley için lisans etiketi (resmi sayfa + API) ve yerel ZIP'in resmi arşivle birebir aynı olduğu (SHA-256 eşleşmesi).
- **Doğrulanamayan:** Roboflow v27 lisansı (yalnızca yerel metadata; resmi sayfalar 403); Mendeley ve Roboflow'un ÜST KAYNAK lisansları ve görsel bazında hak sahipliği.
- **Önemli ilke:** Veri setinin CC BY 4.0 olarak yayımlanması, içindeki üçüncü taraf görsellerin haklarının otomatik olarak doğrulandığı anlamına gelmez (Mendeley'nin kendi açıklaması ek izin gerekebileceğini belirtir; CC SSS'ye göre lisans yalnızca lisans verenin sahip olduğu haklar için geçerlidir). İki veri seti de başka kaynaklardan (ve Roboflow olan YouTube videolarından) derlenmiştir; akademik kullanım için atıf yapılmalı, ticari/yeniden dağıtım senaryolarında kaynak hakları ayrıca incelenmelidir.
