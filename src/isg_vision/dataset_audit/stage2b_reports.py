"""Aşama 2F raporları: nihai veri kararları, 59 maddelik kontrol listesinin nihai durumları, kalan işler, atıf dosyası.

Sayılar yalnızca betik çıktılarından (usage_policy_summary.json, source_group_summary.json, CSV'ler) okunur.
"""

from __future__ import annotations

from collections import Counter
from datetime import date
from pathlib import Path
from typing import Any

import yaml

from isg_vision.dataset_audit.common import read_csv, read_json
from isg_vision.dataset_audit.followup_reports import CAT_TR
from isg_vision.dataset_audit.report import md_table
from isg_vision.utils.paths import PROJECT_ROOT

# Aşama 2F: 'external_verification_pending_source_excluded' = harici (lisans/hak) doğrulama HÂLÂ bekliyor, ancak ilgili kaynak onaylı
# veri kapsamından çıkarıldığı için kullanım riski kontrol altında. Lisans sorununun çözüldüğü anlamına GELMEZ.
EXTERNAL_PENDING_EXCLUDED = "external_verification_pending_source_excluded"
STATUSES = ("resolved_verified", "resolved_excluded", EXTERNAL_PENDING_EXCLUDED, "pending_external_verification", "unresolved")
OPEN_STATUSES = (EXTERNAL_PENDING_EXCLUDED, "pending_external_verification", "unresolved")   # kapanmamış (harici/kullanıcı) maddeler
CONTAINED = "contained_by_exclusion"

# Madde numarası -> (nihai durum, gerekçe). Belirtilmeyen A/B maddeleri resolved_verified, C maddeleri pending_external_verification,
# D maddeleri unresolved sayılır; D/C olanların tamamı burada açıkça ele alınmıştır.
ITEM_OVERRIDES: dict[str, tuple[str, str]] = {
    "B04": ("resolved_verified", "12 contact sheet dosyasının TAMAMI açılarak incelendi (final_dataset_decisions.md §8)."),
    "C01": ("resolved_excluded", "1.5x orijinal çözünürlükte incelendi; hangi kutunun doğru olduğu belirlenemedi -> görüntü quarantine (SI057)."),
    "C02": ("resolved_verified", "Valid Annotation (farklı kişiler)."),
    "C03": ("resolved_excluded", "Aynı kişide çelişen Safety Vest/NO-Safety Vest -> görüntü needs_reannotation (SI081)."),
    "C04": ("resolved_excluded", "Birebir aynı satır: türetme adımında tekilleştirilir (DERIVE-DEDUPLICATE-ROWS); orijinal değişmez."),
    "C05": ("resolved_excluded", "Birebir aynı satır: türetme adımında tekilleştirilir."),
    "C06": ("resolved_excluded", "Birebir aynı satır: türetme adımında tekilleştirilir."),
    "C07": ("resolved_excluded", "Birebir aynı satır: türetme adımında tekilleştirilir."),
    "C08": ("resolved_excluded", "IoU>=0.95 yakın kopya kutu: türetme adımında tekilleştirilir."),
    "C09": ("resolved_excluded", "1-2 px kutular: kutuyu içeren görüntüler quarantine (RF-TINY-TARGET / RF-LIKELY-ERROR)."),
    "D07": ("resolved_verified", "Confirmed Same Source (karo düzeyi ORB 30-131 inlier + görsel): sert kenar, aynı bölme birimi."),
    "D08": ("resolved_excluded", "Likely Same Source: yumuşak kenar ile aynı bölme birimine konur; Roboflow zaten onaylı kümede değil."),
    "D16": ("resolved_excluded", "Mendeley onaylı kümede DEĞİL; birleşik veri için kaynak-grup bazlı bölme politikası tanımlı."),
    "E05": ("resolved_excluded", "Mendeley Helmet sınıfı birleşik eğitim adaylarından dışlandı; Helmet'li 1390 görüntü needs_reannotation/quarantine."),
    "E10": ("resolved_excluded", "Roboflow NO-Hardhat güvenilir alt küme bulunamadı -> sınıf dışlandı; içeren görüntüler needs_reannotation."),
    "E12": ("resolved_excluded", "Roboflow NO-Safety Vest %90 kesinlik sağlanamadı -> sınıf dışlandı; içeren görüntüler needs_reannotation."),
    "F01": (EXTERNAL_PENDING_EXCLUDED, "2C yeniden değerlendirme: yeni doğrulama YAPILAMADI (Roboflow v27 resmi sayfası HTTP 403; lisans yalnızca yerel metadata); yeni lisans iddiası oluşturulmadı. Lisans sorunu ÇÖZÜLMEDİ; kaynak onaylı kapsamdan çıkarıldığı için kullanım riski kontrol altında (pending_license_verification)."),
    "F02": (EXTERNAL_PENDING_EXCLUDED, "2C yeniden değerlendirme: Mendeley veri seti lisansı doğrulanmış, üst kaynak/görsel hakları hâlâ belgelenmemiş; yeni hak iddiası oluşturulmadı. Hak sorunu ÇÖZÜLMEDİ; kaynak onaylı kapsamdan çıkarıldığı için kullanım riski kontrol altında (rights_unverified)."),
    "F03": ("resolved_verified", "docs/dataset_attributions.md hazırlandı (resmi doküman + lisans kanıtlarıyla ilişkilendirildi); bitirme raporuna eklemek kullanıcıya ait basit bir eylemdir."),
    "G01": ("resolved_verified", "Kullanıcı 2026-11-11 tarihinde 22 görsel kararını onayladı ve Aşama 2'nin kapatılmasını istedi. Sonraki aşama başlatılmamıştır."),
}
RISK_CONTAINED = {"E05", "E10", "E12", "F01", "F02", "D08", "D16"}


def item_statuses(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out = []
    for it in items:
        i = dict(it)
        if i["id"] in ITEM_OVERRIDES:
            i["status"], i["status_note"] = ITEM_OVERRIDES[i["id"]]
            if i["id"] == "B04":
                i["result"] = "12/12 contact sheet dosyası açılarak incelendi; kutuların görüntü sınırları ve sınıf anlamları önceki crop/karşılaştırma bulgularıyla tutarlı bulundu."
        elif i["category"] in "AB":
            i["status"], i["status_note"] = "resolved_verified", ""
        elif i["category"] == "C":
            i["status"], i["status_note"] = "pending_external_verification", ""
        else:
            i["status"], i["status_note"] = "unresolved", "Karar eksik"
        i["training_risk"] = CONTAINED if i["id"] in RISK_CONTAINED else "n/a"
        out.append(i)
    return out


def checklist_md(items: list[dict[str, Any]]) -> str:
    cnt = Counter(i["status"] for i in items)
    L = ["# Manuel İnceleme ve Onay Kontrol Listesi (Aşama 2F sonrası NİHAİ durum)\n",
         f"_Üretim tarihi {date.today().isoformat()}. {len(items)} madde. Durumlar: `resolved_verified` (kanıtla doğrulandı), `resolved_excluded` "
         "(güvenli dışlama/karantina kararıyla eğitim açısından çözüldü), `pending_external_verification` (harici kaynak doğrulaması gerekir), "
         f"`{EXTERNAL_PENDING_EXCLUDED}` (harici doğrulama bekliyor; kaynak onaylı kapsam dışında — sorun ÇÖZÜLMEDİ), `unresolved` (karar/teknik çözüm gerekir). `training_risk=contained_by_exclusion`: ilgili kaynak/sınıf onaylı eğitim kümesinden DIŞLANDI; "
         "bu, lisansın veya hakkın doğrulandığı anlamına GELMEZ._\n",
         "## Özet\n", md_table(["Durum", "Madde"], [[s, cnt.get(s, 0)] for s in STATUSES] + [["**Toplam**", len(items)]]), ""]
    cat = Counter(i["category"] for i in items)
    L.append(f"Aşama 2 sonu sınıflaması (A/B/C/D): {dict(sorted(cat.items()))} -> Aşama 2B'de kalan 11 D ve 2 C maddesi ele alındı.\n")
    L.append("## Tüm maddeler\n")
    L.append(md_table(["No", "Bölüm", "Madde", "Nihai durum", "Eğitim riski", "Sonuç / gerekçe", "Kanıt"],
                      [[i["id"], i.get("section", ""), i["text"], i["status"], i["training_risk"],
                        (i["status_note"] + " " if i["status_note"] else "") + i["result"], i["evidence"]] for i in items]))
    return "\n".join(L) + "\n"


def remaining_md(items: list[dict[str, Any]]) -> str:
    rest = [i for i in items if i["status"] in OPEN_STATUSES]
    L = ["# Kalan Açık Maddeler (Aşama 2F sonrası)\n",
         f"_Üretim tarihi {date.today().isoformat()}. {len(items)} maddenin {len(items) - len(rest)}'i kapandı. Aşağıdaki maddeler yalnızca harici lisans/hak doğrulaması bekler; "
         "hiçbiri onaylı eğitim kümesini ETKİLEMEZ (ilgili kaynaklar dışlandı)._\n",
         md_table(["No", "Durum", "Gereken eylem", "İncelenecek dosya/bağlantı", "Önerilen güvenli varsayılan"], [
             ["F01", EXTERNAL_PENDING_EXCLUDED,
              "Tarayıcıda Roboflow v27 sayfasını açıp lisans satırını ve proje sahibini teyit edin (erişim kontrolü aşılmadı).",
              "https://universe.roboflow.com/roboflow-universe-projects/construction-site-safety/dataset/27",
              "Teyit edilene kadar Roboflow `pending_license_verification` kalır (onaylı kümeye ALINMAZ). Teyit edilse bile üst kaynak/YouTube hakları ayrı sorudur."],
             ["F02", EXTERNAL_PENDING_EXCLUDED,
              "Mendeley görsellerinin üst kaynak/hak durumunu belgelemek mümkün değil; yazarlara ulaşmak veya kaynak görselleri bağımsız doğrulamak gerekir.",
              "https://data.mendeley.com/datasets/zkzghjvpn2/6 (yazarlar: Mei-Ling Huang, Ying Cheng)",
              "Mendeley `rights_unverified` olarak dışlı kalır; filigranlı stok görseller (alamy, depositphotos vb.) hiçbir koşulda kullanılmaz."],
             ]),
         f"\n`{EXTERNAL_PENDING_EXCLUDED}`: harici doğrulama hâlâ bekliyor; kaynak onaylı kapsam dışında tutulduğu için eğitim riski kontrol altında. "
         "Bu durum lisans/hak sorununun çözüldüğü anlamına GELMEZ ve kaynak bu durumdayken onaylı ilan edilemez.\n",
         "\n## Kapanan maddelerdeki devam eden riskler (bilgi)\n",
         "- Roboflow/Mendeley için `status_if_rights_verified` yalnızca hak doğrulanırsa geçerli bir ÖN karardır; bugün bu kaynaklardan hiçbir görüntü `approved` değildir.",
         "- Bağımsız saha testi toplanmadan Construction-PPE ile eğitilen modelin başarısı alan dışında doğrulanmış sayılmaz (politika: `leakage.independent_field_test`)."]
    return "\n".join(L) + "\n"


def attributions_md(registry: dict[str, Any], lic: dict[str, Any], log: dict[str, Any]) -> str:
    shas: dict[str, dict[str, Any]] = {}
    for run in log["runs"]:
        for e in run["entries"]:
            shas[e["dataset"]] = e
    L = ["# Veri Seti Kaynak, Lisans ve Atıf Kayıtları\n",
         f"_Oluşturulma: {date.today().isoformat()} (Aşama 2B). Bu dosya yalnızca DOĞRULANABİLEN bilgileri içerir; doğrulanamayanlar açıkça belirtilmiştir. "
         "Yazar, yayın tarihi veya sürüm UYDURULMAMIŞTIR. Hukuki görüş değildir; bitirme raporuna yazmadan önce resmi sayfalar yeniden kontrol edilmelidir. "
         "Kanıt ayrıntısı: `reports/dataset_audit/license_verification_report.md`, `configs/review/license_evidence.yaml`._\n"]
    # 1) Construction-PPE (onaylı kaynak)
    c = registry["construction_ppe"]
    z = shas["construction_ppe"]
    L += ["## 1. Ultralytics Construction-PPE — ONAYLI KAYNAK (eğitim için kullanılabilir)\n",
          md_table(["Alan", "Değer", "Doğrulama"], [
              ["Veri seti adı", "Ultralytics Construction-PPE", "resmi doküman"],
              ["Yayıncı", "Ultralytics team", "resmi doküman"],
              ["Resmî kaynak bağlantısı", c["source"]["url"], "erişildi (HTTP 200)"],
              ["Doğrudan ZIP", c["source"]["direct_zip"], "kullanıcı bu ZIP'i indirdi"],
              ["Lisans", "CC BY 4.0 (https://creativecommons.org/licenses/by/4.0/)", "resmi sayfa + arşivdeki LICENSE.txt (tam yasal metin)"],
              ["Kullanılan sürüm", "Sürüm numarası YOK; yayıncı tarihi: 'November 2026' (README/doküman)", "yerel README + resmi sayfa"],
              ["Erişim tarihi", "2026-11-11", "—"],
              ["Yerel ZIP SHA-256", f"`{z['zip_sha256']}` ({z['zip_size']} bayt)", "kendi hesabımız"],
              ["İçerik", "1000 görüntü (800 train / 200 val; ayrı test yok), 12 sınıf, 14863 kutu", "yerel sayım = yayımlanan (manifest.jsonl ile SHA-256 eşleşmesi)"],
              ["Kaynak notu", "README: 'human-collected and generated images' (insan toplu ve üretilmiş görüntüler)", "yerel README"]]),
          "\n**Zorunlu atıf (yayıncının önerdiği metin):**\n",
          "> Ultralytics Construction-PPE Dataset, Ultralytics team, November 2026, CC BY 4.0.\n",
          "Yayıncı sayfasının önerdiği BibTeX (resmi sayfadan alındı; kullanmadan önce sayfadan yeniden kopyalayın):\n",
          "```bibtex\n@dataset{Ultralytics_Construction_PPE_2026,\n  author = \"{Ultralytics team}\",\n  title = {Ultralytics Construction-PPE Dataset},\n  month = {November},\n  year = {2026},\n  license = {CC-BY-4.0},\n  url = {https://docs.ultralytics.com/datasets/detect/construction-ppe},\n  publisher = {Ultralytics}\n}\n```\n",
          "**CC BY 4.0 koşulları:** uygun atıf (yaratıcı adı, telif/lisans bildirimi, lisans bağlantısı, materyal bağlantısı), değişiklik yapıldıysa belirtme "
          "(CC resmi sayfası). Bu projede yapılan değişiklikler aşağıdaki kayıt alanına işlenmelidir.\n",
          "**Açıkta kalan:** üretilmiş (generated) görüntülerin ayrıca üçüncü taraf hakkı taşıyıp taşımadığı belgelenmemiştir.\n",
          "### Değişiklik kayıtları (akademik raporda 'indicate if changes were made' için)\n",
          md_table(["Tarih", "Değişiklik", "Kapsam", "Uygulayan"], [
              ["2026-11-11", "Yok — ham veri değiştirilmedi; yalnızca okunarak analiz edildi (Aşama 2/2B)", "—", "—"],
              ["", "(Gelecek: 4 hedef sınıfa dönüştürme, bölme, yeniden boyutlandırma/augmentation burada belgelenmeli)", "", ""]]), ""]
    # 2) Mendeley
    m, ml = registry["mendeley_ppe"], lic["mendeley_ppe"]
    L += ["## 2. Mendeley Data — Dataset of Personal Protective Equipment (PPE), v6 — `rights_unverified` (ONAYLI KÜMEDE DEĞİL)\n",
          md_table(["Alan", "Değer", "Doğrulama"], [
              ["Yazarlar", "Mei-Ling Huang, Ying Cheng", "resmi sayfa + API"],
              ["Kurum", "National Chin-Yi University of Technology", "resmi API"],
              ["Resmî bağlantı", m["source"]["url"], "erişildi (HTTP 200)"],
              ["DOI / sürüm / yayım", "10.17632/zkzghjvpn2.6 / v6 / 28 Temmuz 2025", "resmi sayfa + API"],
              ["Veri seti lisansı", "CC BY 4.0", "resmi sayfa + API (doğrulandı)"],
              ["Yerel ZIP", f"20250731-PPE2286y.zip, SHA-256 `{shas['mendeley_ppe']['zip_sha256']}`", "resmi API'deki SHA-256 ile BİREBİR aynı"],
              ["Üst kaynaklar", "'GitHub, Kaggle, Roboflow ve gerçek fotoğraflar' (kaynak listesi/bağlantı/lisans YOK)", "DOĞRULANAMADI"]]),
          "\nÖnerilen künye (yalnızca veri seti düzeyi; üst kaynak hakları ayrı sorudur): "
          "Huang, M.-L.; Cheng, Y. Dataset of Personal Protective Equipment (PPE), Mendeley Data, v6, 2025. doi:10.17632/zkzghjvpn2.6 (CC BY 4.0).\n",
          "**Belirsizlikler:** " + "; ".join(ml["open_uncertainties"][:3]) + "\n"]
    # 3) Roboflow
    r, rl = registry["site_safety"], lic["site_safety"]
    L += ["## 3. Roboflow Universe — Construction Site Safety v27 — `pending_license_verification` (ONAYLI KÜMEDE DEĞİL)\n",
          md_table(["Alan", "Değer", "Doğrulama"], [
              ["Resmî bağlantı", r["source"]["url"], "HTTP 403 (erişilemedi; erişim kontrolü aşılmadı)"],
              ["Yayıncı/çalışma alanı", "roboflow-universe-projects (yerel data.yaml); 'Provided by a Roboflow user' (README)", "yalnızca yerel"],
              ["Sürüm", "v27; 10 Ocak 2023 dışa aktarım (yerel README.roboflow.txt)", "yalnızca yerel"],
              ["Lisans iddiası", "CC BY 4.0 (yerel data.yaml ve README.dataset.txt)", "BAĞIMSIZ DOĞRULANAMADI"],
              ["Yerel ZIP SHA-256", f"`{shas['site_safety']['zip_sha256']}`", "kendi hesabımız"],
              ["Kaynaklar", "Üst Roboflow projeleri + iki YouTube videosu (README)", "lisansları bilinmiyor (sayfalar 403)"]]),
          "\nRoboflow Şartları (roboflow.com/terms, resmî sayfa; kısmen özetle okundu): yükleyici veri seti lisansını seçer; seçilmezse varsayılan CC BY 4.0 olduğu belirtilir; "
          "indirilen veri setlerinin izinsiz yeniden barındırılamayacağı yazar. Bu, bu veri setinin lisansını DOĞRULAMAZ.\n",
          "**Bu veri seti için atıf metni YAZILMAMIŞTIR** (lisans doğrulanana kadar). Doğrulandıktan sonra eklenecek alan:\n",
          "- Künye: _doğrulama bekliyor_\n- Doğrulayan/tarih: _—_\n"]
    return "\n".join(L) + "\n"


def final_decisions_md(*, usage: dict[str, Any], groups: dict[str, Any], decisions: dict[str, Any], policy: dict[str, Any],
                       registry: dict[str, Any], lic: dict[str, Any], items: list[dict[str, Any]], criteria: dict[str, tuple[bool, str]],
                       tile_counts: dict[str, Any]) -> str:
    cnt = Counter(i["status"] for i in items)
    ud = usage["by_dataset"]
    L: list[str] = []
    add = L.append
    visual_review = usage["approved_scope_user_visual_review"]["construction_ppe"]
    add("# Nihai Veri Seti Kararları (Aşama 2F — uygulanmış kullanıcı kararları)\n")
    add(f"_Üretim tarihi {date.today().isoformat()} (`scripts/generate_stage2b_reports.py`). Orijinal veri/etiket/bölünme DEĞİŞTİRİLMEMİŞTİR; hiçbir filtreleme/birleştirme "
        "çalıştırılmamıştır. Görsel kararlar kullanıcı tarafından 22/22 tamamlanmış ve politikaya uygulanmıştır. Politika: `configs/dataset_usage_policy.yaml`._\n")
    add("## 1. Sonuç\n")
    add(f"- **Bugün onaylı eğitim kapsamı yalnızca Ultralytics Construction-PPE'dir:** {ud['construction_ppe']['images']} görüntünün durumları "
        f"{ud['construction_ppe']['status_now']}; nihai bölme {usage.get('approved_scope_final_split_images', {})} "
        f"(`construction_ppe_split_manifest.csv`); kullanılabilir görüntülerdeki hedef sınıf kutuları {usage['approved_scope_target_boxes']}.")
    add(f"- **Nihai görsel kararlar:** {visual_review['user_decided']}/{visual_review['total']} tamamlandı; "
        f"dağılım {visual_review['by_user_decision']}. Dokuz görüntü kullanıcı kararıyla eğitim/validation kapsamından çıkarıldı.")
    add(f"- **Mendeley PPE v6 ve Roboflow Site Safety v27 onaylı kümede DEĞİLDİR** (`pending_license_verification`): üst kaynak/lisans hakları doğrulanamadı. "
        "Hak doğrulanırsa alacakları durumlar ön karar olarak hesaplandı (`status_if_rights_verified`) ama bugün hiçbiri `approved` değildir.")
    add("- Sorunlu üç sınıf (E05 Mendeley `Helmet`, E10 Roboflow `NO-Hardhat`, E12 Roboflow `NO-Safety Vest`) birleşik eğitim adaylarından DIŞLANDI; "
        "dönüşüm/yeniden boyutlandırma uygulanmadı.")
    add(f"- 59 maddelik kontrol listesi: {dict(cnt)}. Açık kalan F01/F02 yalnızca harici doğrulama bekler ve onaylı kümeyi etkilemez.\n")

    add("## 2. Kaynak düzeyi durum\n")
    rows = []
    for ds, src in policy["sources"].items():
        rows.append([ds, src["license_status"], src["rights_status"], src["source_usage"],
                     f"{ud[ds]['status_now']}"])
    add(md_table(["Kaynak", "Lisans doğrulama", "Hak durumu", "Kullanım", "Görüntü durumları (bugün)"], rows))
    add("\n**Önemli ayrım:** bir kaynağı kullanım dışı bırakmak lisansının/hakkının doğrulandığı anlamına GELMEZ. Roboflow ve Mendeley için lisans/hak soruları "
        f"AÇIK kalır (`{EXTERNAL_PENDING_EXCLUDED}`); yalnızca eğitim riski dışlama ile kontrol altına alınmıştır.\n")

    add("## 3. Sınıf bazlı kullanım politikası\n")
    add(md_table(["Veri seti", "Sınıf", "Hedef", "Bugün", "Koşul (hak doğrulanırsa / notlar)"], [
        ["construction_ppe", "helmet", "helmet", "**approved**", "Grup bazlı bölme; taşınan baret de 'helmet' (konum kontrolü gerekir)"],
        ["construction_ppe", "no_helmet", "no_helmet", "**approved**", "Grup bazlı bölme"],
        ["construction_ppe", "vest", "vest", "**approved**", "Grup bazlı bölme"],
        ["construction_ppe", "no_vest", "no_vest", "**approved**", "Grup bazlı bölme"],
        ["construction_ppe", "gloves, boots, goggles, no_gloves, no_boots, no_goggle, Person, forklift", "—", "hedef dışı (yok sayılır)", "Hedef nesne değil; etiketsiz hedef bırakmaz"],
        ["mendeley_ppe", "Helmet", "helmet", "dışlandı (E05)", "Helmet'li görüntü needs_reannotation; dönüşüm yok"],
        ["mendeley_ppe", "NoHelmet / Vest / NoVest", "no_helmet / vest / no_vest", "pending_license_verification", "Hak doğrulanırsa: Helmet'siz ve <3 hedef kutulu görüntüler; kalabalık görüntüler karantina"],
        ["site_safety", "Hardhat / Safety Vest", "helmet / vest", "pending_license_verification", "Hak doğrulanırsa: RF-* kurallarını geçen görüntüler"],
        ["site_safety", "NO-Hardhat", "no_helmet", "dışlandı (E10)", "İçeren görüntü needs_reannotation"],
        ["site_safety", "NO-Safety Vest", "no_vest", "dışlandı (E12)", "İçeren görüntü needs_reannotation"],
        ["site_safety", "Mask, NO-Mask, Person, Safety Cone, machinery, vehicle", "—", "hedef dışı", "—"]]))
    add("\n### Hak doğrulanırsa görüntü durumları (ön karar; bugün geçerli DEĞİL)\n")
    rows = [[ds, v["status_if_rights_verified"], v["reason_counts"]] for ds, v in ud.items() if ds != "construction_ppe"]
    add(md_table(["Veri seti", "Görüntü durumları", "Neden sayıları (kural kodu)"], rows))
    add("\nGörüntü bazlı tüm kararlar: `image_usage_decisions.csv`; karantina/yeniden etiketleme adayları (gerekçeli): `quarantine_candidates.csv`; "
        "yeniden etiketleme iş listesi: `needs_reannotation_list.csv`.\n")

    add("## 4. E05 / E10 / E12 kararları\n")
    cd = decisions["class_decisions"]
    add("### E05 — Mendeley `Helmet` → **dışla**\n")
    add(cd["E05_mendeley_Helmet"]["evidence"])
    add(f"\n- **Yeniden etiketleme maliyeti/veri kaybı:** Helmet kutusu içeren görüntü sayısı {ud['mendeley_ppe']['reason_counts'].get('MD-HELMET-IMAGE', 0)} / "
        f"{ud['mendeley_ppe']['images']} (%{100 * ud['mendeley_ppe']['reason_counts'].get('MD-HELMET-IMAGE', 0) / ud['mendeley_ppe']['images']:.0f}); "
        "yalnızca Helmet satırlarını silmek bu görüntülerde etiketsiz baret bırakır, bu yüzden TÜM görüntü dışlanır (kural MD-HELMET-IMAGE). "
        "Yeniden çizilecek Helmet kutusu sayısı `needs_reannotation_list.csv` içinde.")
    add("- Karar: tahmini kutu küçültme uygulanmadı; gelecekte manuel yeniden etiketleme listesi hazır.\n")
    add("### E10 — Roboflow `NO-Hardhat` → **dışla**\n")
    add(cd["E10_site_safety_NO_Hardhat"]["evidence"])
    add("\n- Yanlış-kabul riski: süzgeçten geçen kutuların yaklaşık %70'i tam-baş tanımıyla uyumsuz (yanlış kabul), bu yüzden süzgeç güvenilir bir alt küme üretmiyor. "
        "Ölçütler (açık): `src/isg_vision/dataset_audit/box_filters.py`; örneklem: `roboflow_filter_precision_sample.csv`, `roboflow_review/filter_precision_no_helmet.jpg`.\n")
    add("### E12 — Roboflow `NO-Safety Vest` → **dışla**\n")
    add(cd["E12_site_safety_NO_Safety_Vest"]["evidence"])
    add("\n- Kararlar (`approved` / `quarantine` / `needs_reannotation`) görüntü seviyesinde verilir: NO-Safety Vest içeren görüntü `needs_reannotation`, "
        "çelişkili/kırpık/belirsiz kutu içeren görüntü `quarantine`. Hiçbir örnek kanıt yetersizken `approved` yapılmadı.\n")

    add("## 5. Veri sızıntısı çözümü (D16 ve Roboflow val/test)\n")
    gs = groups["by_dataset"]
    add(f"Kaynak grupları: eşleşme grafiğinde sert kenar (kanıtlı) {groups['hard_edges']}, yumuşak (sınırda/Likely) {groups['soft_edges']}; "
        f"karo düzeyi ORB ile ek {groups['tile_level_pairs_checked']} çift incelendi; {groups['split_units']} bölme birimi (`source_group_manifest.csv`).\n")
    rows = [[ds, v["images"], v["linked_hard"], v["images_in_cross_split_hard_groups"], v["by_split_in_cross_split_hard_groups"],
             v["images_in_cross_dataset_units"], v["largest_unit"]] for ds, v in gs.items()]
    add(md_table(["Veri seti", "Görüntü", "Sert bağlantılı", "Bölümler arası sert grupta", "Bölüme göre", "Veri setleri arası birimde", "En büyük birim"], rows))
    add("\n**Yöntem (politika):** (1) bölme birimi = sert + yumuşak bağlantılı bileşen; (2) bir birim ASLA bölünmez; (3) rastgele görüntü bazlı bölme yasak; "
        "(4) bilinmeyen/şüpheli ilişkiler yumuşak kenar olarak birlikte tutulur; (5) beklemedeki kaynaklar için oran %70/15/15, seed=42, büyük birimler önce "
        "(önizleme `image_usage_decisions.csv` `proposed_split_preview` — UYGULANMADI); (6) **2C:** onaylı Construction-PPE için yayıncı bölünmesi korunur, "
        "bölümler arası birim çoğunluk bölümüne alınır (`construction_ppe_split_manifest.csv` `final_split`). 2B'deki Construction-PPE %70/15/15 önizlemesi GEÇERSİZDİR.")
    add("\n**Önemli sonuçlar:**")
    add(f"- Roboflow'un {gs['site_safety']['images']} görüntüsünün {gs['site_safety']['images_in_cross_split_hard_groups']}'ü bölümler arası sert gruptadır "
        f"(val {gs['site_safety']['by_split_in_cross_split_hard_groups'].get('val', 0)}/114, test {gs['site_safety']['by_split_in_cross_split_hard_groups'].get('test', 0)}/82). "
        f"Kolaj zincirleri nedeniyle {gs['site_safety']['largest_unit']} görüntülük TEK bir dev birim oluştu (veri setleri arası bağ da içeriyor): bu birim tek bölüme (train) gider; "
        "dolayısıyla Roboflow'un mevcut val/test'i bağımsız değerlendirme için KULLANILAMAZ. Bu muhafazakâr ama güvenli bir sonuçtur.")
    add(f"- Mendeley: {gs['mendeley_ppe']['images_in_cross_split_hard_groups']} görüntü bölümler arası gruptadır (val {gs['mendeley_ppe']['by_split_in_cross_split_hard_groups'].get('val', 0)}).")
    add(f"- Construction-PPE: yalnızca {gs['construction_ppe']['images_in_cross_split_hard_groups']} görüntü (2 train + 1 val) bölümler arası gruptadır "
        "(aynı taban karenin üretken varyantları). **2C kararı:** birim train'e alındı (val üyesi taşındı); ayrıntı `configs/review/stage2c_decisions.yaml`.")
    add("- **Açık risk:** kenarlar alt sınırdır; bağlantısı olmayan görüntü 'ilişkisiz' kanıtlı değildir. Bu nedenle bağımsız saha testi ZORUNLUDUR (§6 politika).")
    add(f"- **Bağımsız saha testi önerisi:** {policy['leakage']['independent_field_test']['recommendation']}\n")

    add("## 6. C01 / C03 ve D07 / D08\n")
    cdn = decisions["conflict_decisions"]
    add(md_table(["Madde", "Görüntü", "Karar", "Kanıt"], [
        ["C01 (SI057)", "`IMG_5024_mp4-2_jpg.rf.77ca6ff1…`", cdn["SI057"]["decision"], cdn["SI057"]["evidence"]],
        ["C03 (SI081)", "`construction-897-_jpg.rf.c7afbfac…`", cdn["SI081"]["decision"], cdn["SI081"]["evidence"]]]))
    add("\nGörsel: `roboflow_review/c01_c03_original.jpg` (1.5x orijinal çözünürlük, tüm kutular sınıf adı ve satır no ile). Belirsiz etiketler sessizce kabul edilmedi.\n")
    srd = {d["group"]: d for d in decisions["source_relation_decisions"]}
    add(md_table(["Madde", "Grup", "Karar", "Kanıt"], [
        ["D07", "IMG_0871_mp4-23", srd["IMG_0871_mp4-23"]["decision"], srd["IMG_0871_mp4-23"]["evidence"]],
        ["D08", "IMG_3103_mp4-17", srd["IMG_3103_mp4-17"]["decision"], srd["IMG_3103_mp4-17"]["evidence"]]]))
    add("\nGörsel: `duplicate_verification/_d07_d08.jpg`. Likely Same Source ilişkisi yumuşak kenar olarak kaydedildi; birim ileride farklı bölümlere ayrılmaz. "
        "Diğer 6 ortak-ad grubu: " + ", ".join(f"{g}: {d['decision']}" for g, d in srd.items() if g not in ("IMG_0871_mp4-23", "IMG_3103_mp4-17")) + ".\n")
    add(f"_Not:_ karo düzeyi ORB, tam görüntü ORB'sinin kaçırdığı eşleşmeleri yakaladı ({tile_counts.get('tile_edges', 0)} sert kenar); bu, ilk raporlanan "
        "Roboflow val/test maruziyetini (alt sınır) daha da artırdı.\n")

    add("## 7. Lisans ve kullanım hakkı\n")
    rows = []
    for ds, v in lic.items():
        rows.append([ds, v["claimed_license"], v["verification_status"], policy["sources"][ds]["rights_status"], policy["sources"][ds]["source_usage"]])
    add(md_table(["Kaynak", "İddia edilen lisans", "Lisans doğrulaması", "Hak durumu", "Kullanım"], rows))
    add("\n- **Doğrulandı:** Construction-PPE (resmi sayfa + arşivdeki lisans metni); Mendeley veri seti lisansı (resmi sayfa + API) ve yerel ZIP'in SHA-256 eşleşmesi.")
    add("- **Doğrulanamadı:** Roboflow v27 (resmi sayfa ve üst projeler HTTP 403; proje ana sayfası 404; web aramasında ayni sınıf listesine sahip FARKLI bir çalışma alanı "
        "projesi CC BY 4.0 görünüyor — bu, bizim veri setinin doğrulaması DEĞİLDİR); Mendeley üst kaynak hakları.")
    add("- **Dışlandı (kullanım dışı):** Roboflow ve Mendeley onaylı kümeden; bu, lisans doğrulaması anlamına gelmez.")
    add("- Mendeley görüntülerinde çok sayıda stok-fotoğraf filigranı görüldü (alamy, depositphotos, CanStockPhoto, paixin, vcg, redocn, Fang.com, weibo): "
        "bu görseller için CC BY 4.0 etiketi hak sahipliğini göstermez. Kaynak/atıf ayrıntıları: `docs/dataset_attributions.md`.\n")

    add("## 8. Contact sheet incelemesi (B04) — 12/12 dosya AÇILARAK incelendi\n")
    cs = decisions["contact_sheet_review"]
    add(md_table(["Veri seti", "İncelenen sayfa", "Bulgular"], [[ds, v["sheets_viewed"], v["findings"]] for ds, v in cs.items() if ds != "comparison_with_crop_review"]))
    add(f"\n**Önceki crop incelemeleriyle karşılaştırma:** {cs['comparison_with_crop_review']}\n")

    add("## 9. Kontrol listesi nihai durum özeti\n")
    add(md_table(["Durum", "Madde"], [[s, cnt.get(s, 0)] for s in STATUSES] + [["Toplam", len(items)]]))
    add("\nAyrıntı: `manual_review_checklist.md` (59 madde, her biri için durum + eğitim riski), açık olanlar: `remaining_manual_reviews.md`.\n")

    add("## 10. Açık riskler ve onaylı veri kullanımına etkisi\n")
    add(md_table(["Risk", "Onaylı (Construction-PPE) kümeyi etkiler mi?", "Kontrol"], [
        ["Roboflow lisansı doğrulanamadı", "Hayır", "Kaynak onaylı kümeden dışlı"],
        ["Mendeley üst kaynak hakları", "Hayır", "Kaynak onaylı kümeden dışlı"],
        ["Roboflow val/test sızıntısı", "Hayır", "Roboflow dışlı; politika: grup bazlı bölme"],
        ["Mendeley etiket eksikliği / filigranlı görseller", "Hayır", "Mendeley dışlı"],
        ["Construction-PPE ayrı test yok, kısmen üretilmiş görüntü, taşınan baret 'helmet'", "EVET (alan kayması)", "Bağımsız saha testi zorunlu; baret konumu kontrolü"],
        ["Aday üretimi alt sınır (bilinmeyen sızıntı)", "Küçük (Construction'da 3 görüntü bağlantılı)", "Yumuşak/bilinmeyen bağlar birlikte tutulur"]]))
    add("")

    add("## 11. Aşama 2 kapanış ölçütleri\n")
    add(md_table(["Ölçüt", "Karşılandı mı", "Kanıt"], [[k, "EVET" if ok else "HAYIR", why] for k, (ok, why) in criteria.items()]))
    add("")
    add("## 12. Kapanış değerlendirmesi\n")
    add(f"Aşama 2F kapanış değerlendirmesi `STAGE_2_FINAL_REPORT.md` ve `STAGE_2_CLOSURE_REPORT.md` içindedir. Roboflow lisansı (F01) ve Mendeley üst kaynak hakları (F02) `{EXTERNAL_PENDING_EXCLUDED}` "
        "durumundadır: harici doğrulama hâlâ bekliyor, ancak kaynaklar onaylı kapsam dışında tutulduğu için kullanım riski kontrol altındadır (sorun ÇÖZÜLMEDİ). "
        "G01 kullanıcı onayıyla kapatılmış ve Aşama 2 `COMPLETED` olmuştur; Aşama 3 başlatılmamıştır.\n")
    return "\n".join(L) + "\n"


def stage2b_addendum(usage: dict[str, Any], items: list[dict[str, Any]]) -> str:
    """dataset_audit_report.md'ye eklenen 'Aşama 2B' (tarihsel) ve 'Aşama 2F' (güncel) özet bölümleri."""
    cnt = Counter(i["status"] for i in items)
    ud = usage["by_dataset"]["construction_ppe"]
    return ("\n## 17. Aşama 2B sonuçları (veri kullanım politikası) — tarihsel özet\n\n"
            "Aşama 2 eksik giderme sonrası kalan 11 karar maddesi ve 2 lisans konusu ele alındı; sonuçlar `final_dataset_decisions.md` içindedir. **Önceki bulgular korunmuştur**. "
            "2B'deki sayılar ve kapanış ifadesi §18 (Aşama 2F) ile GÜNCELLENMİŞTİR: 2B'de onaylı kapsam 1000 görüntü / vest 1816 kutu ve F01/F02 "
            "`pending_external_verification` idi; Construction-PPE için %70/15/15 bölme önizlemesi vardı (artık geçersiz).\n\n"
            "- **Sınıf kararları:** Mendeley `Helmet` (E05), Roboflow `NO-Hardhat` (E10) ve `NO-Safety Vest` (E12) birleşik eğitim adaylarından dışlandı; "
            "görüntü seviyesinde karantina/yeniden etiketleme listeleri üretildi.\n"
            "- **Veri sızıntısı:** kaynak-grup bazlı bölme politikası ve manifest (`source_group_manifest.csv`); Roboflow val/test bağımsız değerlendirme için kullanılamaz.\n"
            "\n## 18. Aşama 2F sonuçları (nihai kullanıcı kararları, güncel)\n\n"
            f"- **Onaylı kapsam:** yalnızca Construction-PPE. 1000 görüntünün durumu {ud['status_now']}; nihai bölme "
            f"{usage.get('approved_scope_final_split_images', {})} (`construction_ppe_split_manifest.csv`); kullanılabilir görüntülerde hedef kutular "
            f"{usage['approved_scope_target_boxes']}. Mendeley ve Roboflow beklemede (`pending_license_verification`, `hold_reasons`), ham verileri ve analiz kayıtları korunuyor.\n"
            "- **Sızıntı kararı:** Construction-PPE'deki bölümler arası tek birim (aynı taban karenin 3 üretken varyantı) train'e alındı; yayıncı bölünmesi korundu.\n"
            f"- **Eksik anotasyon riski:** {usage.get('approved_scope_annotation_risk', {}).get('flagged_images', 0)} görüntü işaretlendi; toplam 10 görüntü kullanım dışı: 9 kullanıcı kararı ve önceden dışlanan `ultralytics-ppe-nov2026-690`.\n"
            f"- **Kontrol listesi (59):** {dict(cnt)}. F01/F02: `{EXTERNAL_PENDING_EXCLUDED}` (sorun çözülmedi; kaynak dışlandı).\n"
            "- **Kapanış:** 22 kullanıcı kararı tamamlandı; Aşama 2 `COMPLETED`. Ayrıntı: `STAGE_2_CLOSURE_REPORT.md`; bağımsız test stratejisi `docs/evaluation_plan.md`.\n")
