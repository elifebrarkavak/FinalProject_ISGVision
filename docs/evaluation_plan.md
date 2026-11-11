# Değerlendirme Planı (Bağımsız Test Stratejisi)

_Aşama 2F, 2026-11-11; aşama numaraları 2D'de 12 aşamalı plana göre düzeltildi (tek referans: `docs/project_plan.md`). Bu belge bir **plandır**. Bu tarih itibarıyla hiçbir model eğitilmedi, saha görüntüsü toplanmadı ve test kümesi
oluşturulmadı. Bu yüzden belgede **hiçbir performans sonucu yoktur**. Tablolardaki "—" değerleri ölçülmemiş anlamına gelir._

İlgili dosyalar:

- Veri kullanım politikası: `configs/dataset_usage_policy.yaml`
- Nihai bölme manifesti: `reports/dataset_audit/construction_ppe_split_manifest.csv`
- Kaynak grupları: `reports/dataset_audit/source_group_manifest.csv`, `source_group_edges.csv`
- Aşama 2 kapanış raporu: `reports/dataset_audit/STAGE_2_FINAL_REPORT.md`

## 1. Neden bağımsız test gerekli?

| Gerçek | Sonuç |
|---|---|
| Construction-PPE'nin ayrı bir test bölümü yok (yalnızca train 800 / val 200) | Val, model seçiminde kullanılınca iyimser olur; nihai başarı iddiası val ile kanıtlanamaz |
| Görüntülerin bir bölümü yapay olarak üretilmiş (README: "human-collected and generated images"). Aynı taban kareden üretilmiş varyantlar bulundu | Gerçek kamera alanına geçerken dağılım kayması riski var; val ölçümleri gerçek saha başarısını temsil etmez |
| Kaynak ilişkisi taraması bir alt sınırdır | Bilinmeyen sızıntı ihtimali sıfırlanamaz; sızıntıdan bağımsız tek kanıt, saha testidir |
| Roboflow v27'nin val/test bölümleri train ile kaynak paylaşıyor (val 92/114, test 70/82); kaynak beklemede | Roboflow'un val/test bölümleri bağımsız test olarak **kullanılmaz** |

## 2. Train / validation / test ayrımı

| Bölüm | Kaynak | Görüntü | Kullanım | Kullanılmayacağı yer |
|---|---|---|---|---|
| **train** | Construction-PPE, `final_split = train` | 794 | Model ağırlıklarının öğrenilmesi | Ölçüm raporlama |
| **val** | Construction-PPE, `final_split = val` (orijinal val'in 196 görüntülük alt kümesi) | 196 | Erken durdurma, hiperparametre ve güven eşiği seçimi, model karşılaştırma. Sonuçlar **"geliştirme ölçümü (dağılım içi, kısmen sentetik)"** olarak raporlanır | Nihai başarı iddiası |
| **test (bağımsız saha testi)** | Hedef ortamdan yeni toplanacak gerçek görüntüler (§4) | hedef ≥ 300 (henüz yok) | Nihai, tek seferlik başarı ölçümü | Eğitim, eşik seçimi, hiperparametre ayarı, veri artırma kararları |
| kullanım dışı | 690 görüntüsü + kullanıcı kararıyla dışlanan 9 görüntü | 10 | Yeniden etiketleme/karantina kararı değişene kadar hiçbir bölümde yok | — |

Kurallar:

1. Bölmenin tek kaynağı manifestteki `final_split` sütunudur. Ham klasörler yeniden düzenlenmez. Eğitim için türetilmiş veri
   klasörü ve manifest tabanlı sınıf dönüşümü (12 → 4 sınıf), **Aşama 8 (ilk YOLO İSG modelinin eğitilmesi) hazırlığında** bu
   manifestten üretilir ve manifestin SHA-256 değeri kaydedilir. Birleşik veri için aynı işlem Aşama 9'da tekrarlanır.
2. Construction-PPE içinden ayrı bir iç test bölümü **ayrılmaz**. Gerekçe: 1.000 görüntülük tek kaynakta eğitim verisini azaltır ve
   val ile aynı dağılımdan gelir, yani bağımsızlık sağlamaz. İç testin rolünü bağımsız saha testi üstlenir.
3. Test kümesi bir kez dondurulur (sürüm etiketi ve dosya SHA-256 listesi). Test sonucuna bakılarak modelde değişiklik yapılırsa
   o test kümesi "kullanılmış" sayılır ve raporda belirtilir. Yeni bir nihai iddia için yeni bir oturum kümesi gerekir.
4. Mendeley ve Roboflow beklemede olduğu için hiçbir bölümde yer almaz. İleride onaylanırlarsa bölme birim düzeyinde yeniden yapılır
   (§3) ve saha testi yine ayrı tutulur.

## 3. Kaynak bazlı veri sızıntısı önlemleri

Politika: `configs/dataset_usage_policy.yaml` → `leakage.group_integrity_rules` (G1–G6). Uygulanan ve uygulanacak önlemler:

| Kural | İçerik | Bugünkü durum |
|---|---|---|
| G1 Bölmeden önce kaynak grafiği | Yeni her görüntü, bölmeden önce mevcut tüm görüntülere (beklemedeki kaynaklar dahil) karşı taranır: SHA-256, pHash/dHash adayları, ORB doğrulaması (tam ve karo düzeyi), küçük resim NCC taraması (≥ 0.85 aday) | Construction-PPE için yapıldı (2B ORB + 2C NCC taraması) |
| G2 Üretken varyantlar | Aynı taban kareden üretilmiş varyantlar sert kenardır ve aynı birimde tutulur | 2 birim (5 görüntü) kayıtlı |
| G3 Oturum gruplama | Saha verisinde bölme birimi en az kayıt oturumudur (kamera + gün + konum). Video kareleri tek tek bölünmez | Saha verisi toplanınca uygulanacak |
| G4 Atomik birim | Bölme birim düzeyindedir. Çelişen birim çoğunluk bölümüne taşınır (eşitlikte train). Görüntü kopyalanmaz | Uygulandı: SU00055 → train |
| G5 Test yalıtımı | Saha testi oturumları eğitim/val ile kaynak, kayıt oturumu veya kamera-gün paylaşmaz | Toplama protokolünde zorunlu (§4) |
| G6 Otomatik kapı | `unit_integrity_violations() == {}` ve onaylı kapsam denetimleri geçmeden manifest eğitimde kullanılmaz | `scripts/run_stage2c_checks.py`, `tests/test_stage2c_closure.py` |

Ek önlemler:

- Saha testi görüntüleri, eğitim verisine karşı da G1 taramasından geçirilir. İnternetten alınmış veya yayımlanmış görüntü test
  kümesine konmaz.
- İleride saha verisiyle ince ayar yapılırsa ince ayar oturumları ve test oturumları farklı gün, kamera ve mümkünse farklı
  konum/çalışan grubundan seçilir. Aynı kişinin farklı oturumlarda görünmesi bir sınırlılık olarak raporlanır.

## 4. Gerçek saha görüntülerinin toplanması

> Bu aşamada görüntü toplanmayacak. Aşağıdaki protokol, toplama başladığında uygulanacak kurallardır.

### 4.1 İzin ve veri koruma

- Saha sahibinden yazılı izin ve görüntülenen kişilerden 6698 sayılı KVKK'ya uygun aydınlatma/açık rıza alınır. Kişi görüntüleri
  kişisel veridir.
- Ham saha görüntüleri `data/` altında tutulur ve Git ile takip edilmez (`.gitignore` ve pre-commit kancası mevcut). Raporlarda
  kullanılacak örneklerde yüzler bulanıklaştırılır.
- KKD ihlali senaryoları (baretsiz/yeleksiz) **gerçek tehlike bölgesinde sahnelenmez**. Güvenli bir alanda, gönüllülerle ve
  saha sorumlusu gözetiminde canlandırılır.
- Zamanlama: saha test kümesi, Aşama 8'deki ilk bilimsel değerlendirmeden **önce** toplanır, etiketlenir ve dondurulur. Aşama 3'te
  geliştirilecek kamera altyapısı kayıt için kullanılabilir, ancak toplama kendi başına bir model değerlendirmesi değildir.

### 4.2 Örnekleme

| Boyut | Hedef |
|---|---|
| Toplam | ≥ 300 görüntü, ≥ 6 kayıt oturumu (en az 2 farklı gün, en az 2 kamera konumu/açısı) |
| Sınıf örnekleri (kutu) | helmet ≥ 100, no_helmet ≥ 100, vest ≥ 100, no_vest ≥ 100 |
| Kişi sayısı/görüntü | 1 kişi, 2–4 kişi ve 5+ kişi katmanlarının her biri ≥ %20 |
| Mesafe/ölçek | Yakın (baş ≥ 64 px), orta (32–64 px), uzak (< 32 px) |
| Koşullar | Gün ışığı, gölge, kapalı hava, iç mekân; arkadan/yandan görünüm; kısmi örtülme |
| Zor durumlar | Elde taşınan baret, şapka/kapüşon/bere, yelek olmayan hi-vis renkli giysi, kısmen kapalı yelek, farklı baret renkleri (beyaz/sarı/turuncu/mavi/kırmızı) |
| Video | Kareler en az 2 sn aralıkla veya sahne değişiminde alınır. Ardışık neredeyse aynı kareler tekilleştirilir |

Her görüntü için meta veri kaydı tutulur: `session_id, date, site, camera_id, mount_height_m, angle, lighting, weather, n_persons, notes`.

### 4.3 Etiketleme

- Sınıf tanımları eğitim verisiyle aynıdır (Aşama 2 anlamsal doğrulaması):
  - `helmet`: baret nesnesini sıkı saran kutu
  - `no_helmet`: baretsiz başın tamamı
  - `vest` / `no_vest`: gövde (göğüs–bel)
  - Elde taşınan baret de `helmet` kutusudur. Kişinin baret durumu ise §5'teki kişi düzeyi kurala göre `no_helmet`'tir.
- İki bağımsız etiketleyici kullanılır. Uyum, IoU ≥ 0.5 ile eşleşen kutularda sınıf uyumu (Cohen κ) ve eşleşmeyen kutu oranıyla
  ölçülür. Anlaşmazlıklar üçüncü bir kişi tarafından karara bağlanır.
- Belirlenemeyen durumlar (çok uzak, aşırı bulanık, ağır örtülme) `ignore` bölgesi olarak işaretlenir. Bu bölgelerdeki tespitler
  ne doğru ne yanlış sayılır.
- Dondurma: `test_manifest.csv` (dosya, SHA-256, oturum, meta veri) + sürüm etiketi. Sonradan yapılan düzeltmeler yeni sürüm
  numarasıyla kaydedilir.

## 5. Birden fazla insan içeren görüntülerin değerlendirilmesi

İki düzeyde ölçüm yapılır:

1. **Nesne düzeyi:** Görüntüdeki tüm hedef kutular için standart tespit ölçütleri (§7). Bir GT kutusu en fazla bir tahminle
   eşleşir. Fazladan (çift) tahminler yanlış pozitiftir.
2. **Kişi düzeyi (İSG kararı):** Her GT kişi için baret durumu ∈ {takılı, takılı değil, belirsiz} ve yelek durumu ∈ {giyili,
   giyili değil, belirsiz}.
   - Tahmin edilen baş/gövde kutuları, kişi kutusuna ilişkilendirme kuralıyla atanır.
   - Bu aşamada GT kişi kutusu kullanılır. İnsan tespiti/takibi eklendiğinde tahmin edilen kişi kutusu kullanılır.
   - Elde taşınan baret, baş bölgesiyle örtüşmüyorsa "takılı" sayılmaz.
   - "Belirsiz" kişiler kişi düzeyi doğruluğa katılmaz ama sayıları raporlanır.

Sonuçlar şu katmanlara göre ayrıca raporlanır:

- Görüntüdeki kişi sayısı (1 / 2–4 / 5+)
- Kişi boyu (baş < 32 px, 32–64 px, ≥ 64 px)
- Örtülme (yok / kısmi / ağır)
- Görünüm (ön / yan / arka)

Kalabalık sahnelerde ilişkilendirme hatası (doğru kutu, yanlış kişiye atanmış) ayrı bir hata türü olarak sayılır (§8).

## 6. Baret ve yelek sınıflarının test edilmesi

| Test | Yöntem |
|---|---|
| Sınıf bazlı başarı | helmet, no_helmet, vest, no_vest için ayrı AP, precision ve recall |
| Durum karışıklığı | helmet↔no_helmet ve vest↔no_vest karışıklık matrisi (IoU ≥ 0.5 ile eşleşen kutularda sınıf karşılaştırması) |
| İhlal yakalama (birincil güvenlik ölçütü) | Kişi düzeyinde "baretsiz" ve "yeleksiz" durumlarının recall değeri. Kaçırılan ihlal, yanlış alarmdan daha maliyetli kabul edilir. Bu nedenle eşik seçiminde ihlal recall'u için alt sınır tanımlanır (val üzerinde seçilir, testte değiştirilmez) |
| Zor durum alt kümeleri | Elde taşınan baret, şapka/kapüşon, hi-vis renkli giysi (yeleksiz), arkadan görünüm, renk çeşitleri. Her biri ≥ 20 örnekle ayrı raporlanır |
| Ölçek | Küçük/orta/büyük nesneler için ayrı AP |
| Sentetik ↔ gerçek farkı | Aynı model için val (kısmen sentetik) ve saha testi sonuçları yan yana raporlanır. Fark, alan kayması göstergesi olarak yorumlanır |

## 7. Ölçütler

### 7.1 En az %85 başarı hedefi

Projenin akademik hedefi değerlendirmede en az **%85 başarı** göstermektir. Bu hedef henüz tek bir ölçüte bağlanmamıştır ve
bugünkü performans durumu `not_measured` değerindedir. Model eğitilmediği için hedefe ulaşıldığı, yaklaşıldığı veya ulaşılacağı
iddia edilemez.

Türkçe rapor gösterimi: **Ölçülmedi (model yok)**; makine tarafından okunabilir durum kodu: `not_measured`.

- **Accuracy**, doğru sınıflandırılan kişi/durum kararlarının tüm değerlendirilen kararlar içindeki oranıdır; nesne tespiti için
  tek başına yeterli değildir ve ağır sınıf dengesizliğinde yanıltıcı olabilir.
- **Precision**, üretilen pozitif tahminlerin ne kadarının doğru olduğunu; **recall**, gerçek hedef/ihlallerin ne kadarının
  yakalandığını ölçer. Güvenlik açısından özellikle baretsiz ve yeleksiz kişi recall değerleri ayrıca raporlanır.
- **mAP@0.5** ve **mAP@0.5:0.95**, YOLO nesne tespiti için sınıf ve kutu konumunu birlikte değerlendirir. %85 hedefinin hangi
  mAP sürümüne veya kişi düzeyi ölçüte uygulanacağı, Aşama 8 deney protokolü dondurulmadan önce açıkça seçilecektir.
- YOLO KKD tespiti ve çalışan kimlik tanıma ayrı görevlerdir. Kimlik tanıma için accuracy/precision/recall, bilinmeyen kişi
  reddi ve kişi başına karışıklık; KKD tespiti için sınıf bazlı AP/precision/recall ayrı raporlanır.
- Tek kameradaki çoklu kişiler için doğru kimliğin korunması ve baret/yelek durumunun **doğru kişiye atanması** ayrıca ölçülür.
  `ASSOC` hataları, nesne doğru tespit edilse bile ayrı başarısızlık sayılır.
- Construction-PPE'nin bir bölümü sentetiktir. %85 hedefi yalnızca kısmen sentetik validation sonucuyla doğrulanamaz; gerçek
  saha performansı, eğitimden ve eşik seçiminden bağımsız dondurulmuş görüntü/oturumlarda ölçülmelidir.

| Ölçüt | Tanım / ayar |
|---|---|
| Precision | TP / (TP + FP); IoU ≥ 0.5 ve aynı sınıf eşleşmesi; val'de seçilip dondurulan tek bir güven eşiğinde |
| Recall | TP / (TP + FN); aynı eşik |
| F1 | Precision ve recall'un harmonik ortalaması |
| mAP@0.5 | Sınıf bazlı AP'lerin ortalaması; IoU = 0.5; 101 noktalı interpolasyon (COCO/Ultralytics uyumlu) |
| mAP@0.5:0.95 | IoU 0.50–0.95, 0.05 adım ortalaması (kutu konum kalitesi) |
| Kişi düzeyi ölçütler | Baret/yelek durumu için doğruluk, ihlal precision/recall değerleri ve "belirsiz" oranı |
| Güven aralığı | Görüntü (saha testinde **oturum**) düzeyinde bootstrap, 1000 yeniden örnekleme, %95 aralık |
| Sonraki aşamalar (bilgi) | Kamera FPS/gecikme (Aşama 3, teknik), takip tutarlılığı (Aşama 4), uçtan uca FPS/gecikme (Aşama 12) (§9) |

Raporlama kuralları:

- Her tabloda veri kümesinin adı ve sürümü, model ve ağırlık sürümü, eşik ve tarih yazılır.
- Val sonucu ile test sonucu aynı tabloda karıştırılmaz.
- Val ölçümleri ayrıca annotation-risk işaretli görüntüler hariç tutularak bir kez daha hesaplanır (duyarlılık analizi):
  güncel val bölümünde 12 görüntü işaretli, bkz. manifestteki `annotation_risk` sütunu.

Ölçüm durumu (bugün):

| Küme | mAP@0.5 | mAP@0.5:0.95 | Precision | Recall | Durum |
|---|---|---|---|---|---|
| val (Construction-PPE, 196) | — | — | — | — | `not_measured` (model yok) |
| Bağımsız saha testi | — | — | — | — | `not_measured` (küme henüz oluşturulmadı) |

## 8. Hatalı tespitlerin raporlanması

Her değerlendirme çalıştırması `results/<run_id>/errors.csv` dosyasını üretir:

| Alan | Açıklama |
|---|---|
| `image`, `session_id` | Görüntü ve kayıt oturumu |
| `gt_id`, `gt_class`, `pred_class`, `confidence`, `iou` | Eşleşme bilgisi (eşleşmeyen tarafta boş) |
| `error_type` | Aşağıdaki türlerden biri |
| `person_id`, `n_persons`, `scale_bin`, `occlusion`, `view` | Bağlam |
| `note` | Gözlem (ör. "elde baret", "kapüşon") |

Hata türleri:

| `error_type` | Anlamı |
|---|---|
| `FP_background` | Nesne olmayan bölgede tespit |
| `FP_duplicate` | Aynı nesneye ikinci tespit |
| `LOC` | Doğru sınıf, IoU 0.1–0.5 (konum hatası) |
| `CLS_state` | helmet↔no_helmet veya vest↔no_vest karışıklığı (güvenlik açısından en kritik tür) |
| `FN_small`, `FN_occluded`, `FN_truncated`, `FN_other` | Kaçırılan nesneler |
| `ASSOC` | Kişi düzeyinde yanlış kişiye atama |
| `GT_suspect` | İnceleme sonucunda etiketin hatalı olduğu düşünülen durum. Etiket düzeltmesi yeni test sürümüyle yapılır, sessizce değiştirilmez |

Rapor içeriği:

- Hata türü dağılımı (sınıf ve katman bazında)
- Karışıklık matrisi
- Her türden en yüksek güvenli 20 örneğin görsel galerisi (yüzler bulanıklaştırılmış)
- Kişi düzeyinde kaçırılan ihlallerin tam listesi

## 9. Gerçek kamera testlerinin zamanlaması

Aşama numaraları projenin 12 aşamalı planına göredir (`docs/project_plan.md`). Model doğruluğuna ilişkin bilimsel değerlendirmeler,
ilgili modeller eğitildikten sonra yapılır.

| Aşama | Ön koşul | Test |
|---|---|---|
| **Aşama 3** — Canlı kamera altyapısı | Aşama 2 kullanıcı onayı | Yalnızca kameranın **teknik işlev testleri**: OpenCV kamera erişimi, video akışı, FPS hesaplama, gecikme, kamera hatası/yeniden bağlanma, kare kaybı. **YOLO eğitimi, veri birleştirme ve model doğruluğu değerlendirmesi yok** |
| Aşama 4 — Çoklu insan tespiti ve takibi | Aşama 3 tamam | Kaydedilmiş videolarda insan tespiti/takibi ölçümleri (KKD değil) |
| Aşama 5–7 — Yüz tespiti, çalışan kaydı, kimlik tanıma | Önceki aşamalar | İlgili modüllerin kendi ölçümleri (KKD değil) |
| **Aşama 8** — İlk YOLO İSG modeli | Aşama 2 bölme manifesti denetimlerden geçti; görsel inceleme kararları tamamlandı (`check_visual_review.py --require-complete`); saha test kümesi toplanıp dondurulmuş (§4) | Eğitim hazırlığı (türetilmiş veri, sınıf dönüşümü), val ile geliştirme ölçümü, ardından dondurulmuş **bağımsız saha testi** (fotoğraf/kare tabanlı, §5–§8). İlk nihai KKD tespit başarısı burada ölçülür |
| Aşama 9 — Veri setlerinin birleştirilmesi | Lisans/hak doğrulaması (F01/F02) | Birleşik bölmenin G1–G6 denetimleri. Model değerlendirmesi yok |
| **Aşama 10** — Geliştirilmiş YOLO İSG modeli | Aşama 9 | Aynı dondurulmuş saha testiyle Aşama 8 modeline karşı karşılaştırma |
| **Aşama 11** — Kimlik tanıma + İSG kontrolü | Aşama 7 ve 8/10 modelleri | Kaydedilmiş video oturumlarında kişi/iz düzeyi uyum, kimlik değişimi, zamansal tutarlılık |
| **Aşama 12** — Canlı arayüz, sistem testleri, raporlama | Aşama 11 raporlandı; izinler ve KVKK süreci hazır | Canlı kamera uçtan uca sistem testleri: FPS, uçtan uca gecikme, kararlılık, yanlış alarm sıklığı (alarm/saat). Ölçüm süresi ve koşullar önceden yazılır |

Canlı kamera sistem testi (Aşama 12) ancak dondurulmuş saha testi (Aşama 8/10) ve kaydedilmiş video değerlendirmesi (Aşama 11)
tamamlandıktan sonra yapılır. Canlı testte etiketli GT bulunmadığından tespit doğruluğu iddiası canlı testten değil, dondurulmuş
saha test kümesinden yapılır.

## 10. Plan durumu özeti

| Bileşen | Durum |
|---|---|
| Bölme (train/val) ve sızıntı kuralları | Tanımlandı ve kodla denetleniyor |
| Bağımsız saha test protokolü | Tanımlandı |
| Saha görüntüsü toplama | **Yapılmadı** (bilinçli: Aşama 2 kapsamı dışında; Aşama 8 değerlendirmesinden önce tamamlanacak) |
| Model eğitimi ve ölçümler | **Yapılmadı**; hiçbir performans sonucu yok |
