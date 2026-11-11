# Kalan Açık Maddeler (Aşama 2C sonrası)

_Üretim tarihi 2026-11-11. 59 maddenin 56'i kapandı. Aşağıdaki maddeler kullanıcı/harici doğrulama bekler; hiçbiri onaylı eğitim kümesini ETKİLEMEZ (ilgili kaynaklar dışlandı)._

| No | Durum | Gereken eylem | İncelenecek dosya/bağlantı | Önerilen güvenli varsayılan |
|---|---|---|---|---|
| F01 | external_verification_pending_source_excluded | Tarayıcıda Roboflow v27 sayfasını açıp lisans satırını ve proje sahibini teyit edin (erişim kontrolü aşılmadı). | https://universe.roboflow.com/roboflow-universe-projects/construction-site-safety/dataset/27 | Teyit edilene kadar Roboflow `pending_license_verification` kalır (onaylı kümeye ALINMAZ). Teyit edilse bile üst kaynak/YouTube hakları ayrı sorudur. |
| F02 | external_verification_pending_source_excluded | Mendeley görsellerinin üst kaynak/hak durumunu belgelemek mümkün değil; yazarlara ulaşmak veya kaynak görselleri bağımsız doğrulamak gerekir. | https://data.mendeley.com/datasets/zkzghjvpn2/6 (yazarlar: Mei-Ling Huang, Ying Cheng) | Mendeley `rights_unverified` olarak dışlı kalır; filigranlı stok görseller (alamy, depositphotos vb.) hiçbir koşulda kullanılmaz. |
| G01 | pending_external_verification | Aşama 2'yi kapatma ve Aşama 3'e (canlı kamera altyapısı) geçme kararı (kullanıcı onayı). | reports/dataset_audit/STAGE_2_FINAL_REPORT.md | Önerilen: onaylı kapsamı yalnızca Construction-PPE ile sınırlı tutarak kapatmak. |

`external_verification_pending_source_excluded`: harici doğrulama hâlâ bekliyor; kaynak onaylı kapsam dışında tutulduğu için eğitim riski kontrol altında. Bu durum lisans/hak sorununun çözüldüğü anlamına GELMEZ ve kaynak bu durumdayken onaylı ilan edilemez.


## Kapanan maddelerdeki devam eden riskler (bilgi)

- Roboflow/Mendeley için `status_if_rights_verified` yalnızca hak doğrulanırsa geçerli bir ÖN karardır; bugün bu kaynaklardan hiçbir görüntü `approved` değildir.
- Bağımsız saha testi toplanmadan Construction-PPE ile eğitilen modelin başarısı alan dışında doğrulanmış sayılmaz (politika: `leakage.independent_field_test`).
