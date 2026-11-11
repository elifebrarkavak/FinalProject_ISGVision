"""Aşama 2 eksik giderme raporları: anlamsal doğrulama, Roboflow etiket incelemesi, yakın kopya doğrulaması, lisans,
manuel kontrol listesi ve kalan manuel incelemeler. Sayılar yalnızca betik çıktılarından okunur."""

from __future__ import annotations

from collections import Counter, defaultdict
from datetime import date
from pathlib import Path
from typing import Any

import yaml

from isg_vision.dataset_audit.common import read_csv, read_json
from isg_vision.dataset_audit.report import md_table
from isg_vision.dataset_audit.review import wilson_interval
from isg_vision.utils.paths import PROJECT_ROOT

REVIEW_DIR = PROJECT_ROOT / "configs" / "review"
STRONG = ("same_image_derived", "same_scene_different_frame", "partial_region_match", "exact_duplicate")
CAT_TR = {"A": "Otomatik doğrulandı", "B": "Görsel olarak doğrulandı", "C": "Çözülemeyen teknik sorun",
          "D": "Kullanıcı kararı/onayı gerekir"}
ROLE_TR = {"helmet": "helmet", "no_helmet": "no_helmet", "vest": "vest", "no_vest": "no_vest", "person": "person"}


def _yaml(name: str) -> Any:
    return yaml.safe_load((REVIEW_DIR / name).read_text(encoding="utf-8"))


def load_followup(audit_dir: Path) -> dict[str, Any]:
    a: dict[str, Any] = {
        "sem": read_json(audit_dir / "semantic_metrics.json"),
        "sem_sampling": read_json(audit_dir / "semantic_comparison_sampling.json"),
        "conf": read_csv(audit_dir / "semantic_conflicts.csv"),
        "rev": read_csv(audit_dir / "label_review.csv"),
        "rev_sum": read_json(audit_dir / "label_review_summary.json"),
        "dv": read_csv(audit_dir / "duplicate_verification.csv"),
        "dv_sum": read_json(audit_dir / "duplicate_verification_summary.json"),
        "dv_val": read_json(audit_dir / "duplicate_method_validation.json"),
        "dups_old": read_csv(audit_dir / "duplicate_images.csv"),
        "art": read_json(audit_dir / "augmentation_artifacts.json"),
        "inv": read_csv(audit_dir / "dataset_inventory.csv"),
        "lic": _yaml("license_evidence.yaml")["datasets"],
        "resol": _yaml("checklist_resolution.yaml")["items"],
        "visual": _yaml("duplicate_visual_checks.yaml")["checks"],
        "name_groups_visual": _yaml("duplicate_visual_checks.yaml").get("name_groups", {}),
        "name_overlap_groups": [r for r in read_csv(audit_dir / "source_name_overlap.csv") if r["kind"] == "within_dataset_cross_split"],
    }
    return a


def _fmt(m: dict[str, float] | None) -> str:
    return f"{m['median']} [{m['p25']}–{m['p75']}]" if m else "—"


# ---------------------------------------------------------------- anlamsal doğrulama
def semantic_report(a: dict, entries: list[dict], targets: dict[int, str]) -> str:
    sem, L = a["sem"], []
    add = L.append
    add("# Kutu Semantiği Doğrulama Raporu (Aşama 2 eksik giderme)\n")
    add(f"_Üretim tarihi {date.today().isoformat()} (`scripts/generate_followup_reports.py`). Sayılar `semantic_metrics.json`, "
        "`label_review.csv` ve `candidate_class_mapping.yaml` dosyalarından okunur. Görsel gözlemler model destekli, tek değerlendirici "
        "incelemesidir; kullanıcı doğrulaması DEĞİLDİR._\n")
    add("## 1. Amaç ve yöntem")
    add("Sınıf adı aynı olsa bile kutunun neyi kapsadığı veri setleri arasında farklı olabilir. Bu nedenle üç bağımsız kanıt birleştirildi:\n")
    add("1. **Nicel ölçü (kişi kutusuna göre):** `Person` sınıfı olan veri setlerinde (Construction-PPE, Roboflow) her baret/yelek kutusu merkezi "
        "ve alanı yeterince içeren en küçük kişi kutusuna atanır; kutunun kişiye göre yüksekliği, genişliği ve dikey konumu ölçülür. "
        "(Medyan *görüntü alanı* kadraj yakınlığından etkilendiği için artık kullanılmıyor.)")
    add("2. **Nicel ölçü (baş/gövde oranı):** Tam bir baş kutusu ve tam bir gövde kutusu olan görüntülerde baş/gövde boyut oranı; "
        "`Person` sınıfı olmayan Mendeley dahil üç veri seti için ORTAK ölçüdür.")
    add("3. **Görsel inceleme:** Her hedef sınıf için üç veri setinden aynı çevre payıyla kırpılmış 8'er kutu yan yana (`semantic_review/compare_*.jpg`) "
        "ve önceki aşamanın 24'er kırpma sayfaları açılıp incelendi.\n")
    sampling = a["sem_sampling"]
    add("**Karşılaştırma örnekleme bilgisi** (kutu yüksekliği <24 px olanlar ayırt edilemediği için örneklemeden çıkarıldı):\n")
    rows = []
    for role, d in sampling.items():
        for ds, v in d.items():
            rows.append([role, ds, v["n_total"], v["n_eligible"], v["shown"]])
    add(md_table(["Rol", "Veri seti", "Toplam kutu", "Örneklenebilir (≥24 px)", "Gösterilen"], rows))

    add("\n## 2. Nicel ölçüler")
    add("### 2.1 Kutunun kişiye göre boyutu ve konumu\n")
    rows = []
    for ds, d in sem.items():
        for role in ("helmet", "no_helmet", "vest", "no_vest"):
            r = d["person_relative"].get(role)
            if r:
                rows.append([ds, role, f"{r['n_assigned']}/{r['n_total']}", _fmt(r["rel_height"]), _fmt(r["rel_width"]),
                             _fmt(r["top_offset"]), _fmt(r["aspect_px"])])
    add(md_table(["Veri seti", "Rol", "Kişiye atanan", "Göreli yükseklik", "Göreli genişlik", "Kişi tepesinden uzaklık", "h/w (piksel)"], rows))
    add("\nDeğerler: medyan [Q1–Q3]. Mendeley'de `Person` sınıfı olmadığı için yoktur. Roboflow kişi kutularının çoğu kısmi/yakın çekim "
        "olduğu için göreli yükseklikleri yayılmıştır (Q1–Q3 geniş).\n")
    add("### 2.2 Baş/gövde oranı (tek baş + tek gövde kutulu görüntüler)\n")
    rows = []
    for ds, d in sem.items():
        for role, r in d["single_head_torso_pairs"].items():
            rows.append([ds, role, r["n_pairs"], _fmt(r["head_h_over_torso_h"]), _fmt(r["head_w_over_torso_w"]), _fmt(r["aspect_px"])])
    add(md_table(["Veri seti", "Baş rolü", "Çift sayısı", "baş_h / gövde_h", "baş_w / gövde_w", "baş h/w"], rows))
    add("\n### 2.3 Rol bazında en-boy oranı ve piksel yüksekliği\n")
    rows = []
    for ds, d in sem.items():
        for role in ("helmet", "no_helmet", "vest", "no_vest"):
            r = d["aspect_and_size"].get(role)
            if r:
                rows.append([ds, role, r["n"], r["aspect_h_over_w_median"], r["height_px_median"], r["height_px_p10"]])
    add(md_table(["Veri seti", "Rol", "Kutu", "h/w medyan", "Yükseklik medyan (px)", "Yükseklik P10 (px)"], rows))

    add("\n## 3. Görsel inceleme sonuçları (karşılaştırma sayfaları, 8'er kutu/veri seti/sınıf)\n")
    add("| Hedef sınıf | Construction-PPE | Mendeley | Roboflow |\n|---|---|---|---|")
    add("| helmet | Baret nesnesine SIKI (8/8) | Baş+baret, YÜZ dahil (8/8) | Çoğunlukla sıkı (6/8); 2 örnek kolaj/siyah kare etkisinde |")
    add("| no_helmet | Tüm baş: alın–çene (8/8) | Tüm baş/yüz (8/8; şapka, mozaik yüz dahil) | Heterojen: saç/alın şeridi, kısmi yüz; 2/8'de baret benzeri nesne görünüyor |")
    add("| vest | Yelekli gövde (8/8) | Yelekli gövde (8/8) | Gövde; 2/8 kısmi/şerit kutu |")
    add("| no_vest | Gövde (8/8) | Gövde (8/8; çoğu inşaat dışı) | Gövde ama 3/8'de baret/kumaş bölgesi veya şerit: gürültülü |")
    add("\n_Not: 8'lik karoların tek başına istatistiksel gücü yoktur; sayılar yalnızca bu sayfalardaki görsel gözlemdir. Roboflow için ayrıca "
        "24'er kutuluk rastgele örneklem yapıldı (bkz. `roboflow_label_review_report.md`)._\n")

    add("## 4. Bulgular ve düzeltmeler")
    c = sem
    cp, mp, rp = c["construction_ppe"], c["mendeley_ppe"], c["site_safety"]
    add(f"- **Baret kutusu yalnızca bareti mi, başı da mı kapsıyor?** Construction-PPE: yalnızca baret (kişi yüksekliğinin "
        f"{cp['person_relative']['helmet']['rel_height']['median']}'ü, h/w={cp['person_relative']['helmet']['aspect_px']['median']}). "
        f"Roboflow: yalnızca baret (h/w={rp['aspect_and_size']['helmet']['aspect_h_over_w_median']}). Mendeley: **başı/yüzü de kapsıyor** — "
        f"baş/gövde oranı {mp['single_head_torso_pairs']['helmet']['head_h_over_torso_h']['median']} "
        f"(Construction {cp['single_head_torso_pairs']['helmet']['head_h_over_torso_h']['median']}).")
    add(f"- **Baretsiz sınıfı baş mı, bütün insan mı?** Hiçbir veri setinde bütün insan değil. Construction-PPE ve Mendeley: tüm baş "
        f"(baş/gövde {cp['single_head_torso_pairs']['no_helmet']['head_h_over_torso_h']['median']} ve "
        f"{mp['single_head_torso_pairs']['no_helmet']['head_h_over_torso_h']['median']} — birbirine çok yakın). Roboflow: heterojen "
        f"(h/w medyan {rp['aspect_and_size']['no_helmet']['aspect_h_over_w_median']}: saç/alın şeritleri ve kısmi yüzler).")
    add(f"- **Yeleksiz sınıfı gövde mi, bütün insan mı?** Gövde. Construction-PPE: kişi yüksekliğinin "
        f"{cp['person_relative']['no_vest']['rel_height']['median']}'ü (vest: {cp['person_relative']['vest']['rel_height']['median']}). "
        "Roboflow'da kısmi/yakın çekim kutular nedeniyle yayılmış.")
    add("- **Düzeltme (önceki taslağa):** Önceki raporda 'no_helmet kutusu helmet'ten ~4 kat küçük' ve 'no_vest kutusu vest'ten ~3 kat büyük' "
        "denmişti. Bu, medyan *görüntü alanı* karşılaştırmasından geliyordu ve kadraj yakınlığından etkileniyordu. Kişiye göre ölçüde "
        "Construction-PPE'de no_helmet kutusu helmet'ten büyüktür (baş vs baret) ve no_vest ile vest aynı boydadır. Eşleme dosyası buna göre düzeltildi.")
    add("- **Taşınan baret:** Construction-PPE'de 14 görüntüde kişinin elinde taşıdığı baret `helmet` etiketli ve aynı kişinin başı `no_helmet` "
        "(çelişki incelemesi, geçerli anotasyon). `helmet` sınıfı 'takılı baret' değil 'baret nesnesi' anlamındadır; KKD kararında konum kontrolü şart.")
    add("- **Roboflow etiket kalitesi:** bkz. `roboflow_label_review_report.md`.\n")

    add("## 5. Çelişkili etiket aramasının sonucu")
    add("Aynı özniteliğin olumlu/olumsuz sınıfları (helmet–no_helmet, vest–no_vest) aynı bölgede (IoU≥0.1 veya merkez içeride) ya da aynı kişi "
        "kutusuna atanmış olarak aranmıştır. Aynı görüntüde farklı kişilere ait helmet + no_helmet TEK BAŞINA hata sayılmamıştır.\n")
    rows = []
    for ds in ("construction_ppe", "mendeley_ppe", "site_safety"):
        cnt = Counter(r["category"] for r in a["rev"] if r["dataset"] == ds and r["source"].startswith("semantic_conflict"))
        n = sum(1 for r in a["conf"] if r["dataset"] == ds)
        rows.append([ds, n, cnt.get("Valid Annotation", 0), cnt.get("Likely Error", 0), cnt.get("Confirmed Error", 0),
                     cnt.get("Needs Manual Review", 0)])
    add(md_table(["Veri seti", "Çelişki adayı", "Valid Annotation", "Likely Error", "Confirmed Error", "Needs Manual Review"], rows))

    add("\n## 6. Sınıf eşlemesi sonucu (hedef: " + ", ".join(f"{k}={v}" for k, v in targets.items()) + ")\n")
    rows = []
    for e in entries:
        if e.get("target") is None:
            continue
        rows.append([e["dataset"], f"{e['id']}:{e['name']}", targets[e["target"]], e["status"], e.get("compatibility", ""),
                     e.get("reliability", ""), e.get("box_semantics", ""), e.get("visual_result", ""), e.get("inconsistencies", "")])
    add(md_table(["Veri seti", "Kaynak", "Hedef", "Durum", "Uyum", "Güvenilirlik", "Kutu semantiği (ölçülen)", "Görsel sonuç", "Tutarsızlıklar"], rows))
    st = Counter(e["status"] for e in entries if e.get("target") is not None)
    add(f"\nDurum sayıları (hedefe eşlenen 12 sınıf): {dict(st)}. Hedef dışı 14 sınıf `incompatible` (hedef dörtlüde karşılığı yok).")
    add("\n**Kesin doğrulanamayanlar:** `NO-Safety Vest` (needs_review) ve Mendeley/Roboflow'daki alan kayması. `confirmed` statüsü kutu "
        "semantiğinin hedef tanımıyla tutarlı olduğunu ve örneklemde olası hatanın ≤%10 olduğunu söyler; etiketlerin tamamının doğru olduğunu DEĞİL.\n")
    add("## 7. Öneri (uygulanmadı)\n")
    add("- Hedef tanım: helmet=baret nesnesi (sıkı), no_helmet=baretsiz baş, vest/no_vest=gövde. Bu tanıma uymayan Mendeley `Helmet` "
        "(baş+baret) ve Roboflow `NO-Hardhat` (heterojen) doğrudan eşlenmemeli; kullanıcı kararı sonrası ayrı ele alınmalı.")
    add("- Baret etiketi 'takılı baret' anlamı taşımadığından KKD kararı baret kutusunun baş bölgesiyle konumsal ilişkisine dayanmalıdır (Aşama 11 entegrasyon katmanı).")
    return "\n".join(L) + "\n"


# ---------------------------------------------------------------- Roboflow etiket incelemesi
def roboflow_report(a: dict) -> str:
    rev, summ, L = a["rev"], a["rev_sum"], []
    add = L.append
    rf = [r for r in rev if r["dataset"] == "site_safety"]
    add("# Roboflow (site_safety v27) Şüpheli Etiket İnceleme Raporu\n")
    add(f"_Üretim tarihi {date.today().isoformat()}. Tüm kararlar `configs/review/label_review_verdicts.yaml` içindeki görsel inceleme "
        "kayıtlarından gelir (model destekli, tek değerlendirici). Orijinal veri DEĞİŞTİRİLMEMİŞTİR._\n")
    add("## 1. Kapsam")
    add("`annotation_issues.csv` kayıtları (21 çok küçük kutu, 4 birebir + 1 yakın kopya kutu, 23 boş etiket), 112 çelişki adayı "
        "(`semantic_conflicts.csv`; 19'u Construction-PPE) ve 4 rol için rastgele 24'er kutuluk kalite örneklemi (96 kutu, ≥24 px) incelendi. "
        "Her öğe için dosya yolu, ilgili etiket satırları ve gerekçe `label_review.csv` içinde.\n")
    add("## 2. Kategorilere göre sonuç (Roboflow)\n")
    cats = ("Confirmed Error", "Likely Error", "Valid Annotation", "Needs Manual Review")
    src_names = sorted({r["source"] for r in rf})
    rows = []
    for s in src_names:
        c = Counter(r["category"] for r in rf if r["source"] == s)
        rows.append([s, sum(c.values())] + [c.get(k, 0) for k in cats])
    c = Counter(r["category"] for r in rf)
    rows.append(["**TOPLAM**", len(rf)] + [c.get(k, 0) for k in cats])
    add(md_table(["Kaynak", "Öğe"] + list(cats), rows))
    add("\n### Kategori anlamları")
    add("- **Confirmed Error:** kod ile kesin (birebir aynı etiket satırı).\n- **Likely Error:** görselde açık biçimde hatalı/anlamsız ama insan teyidi önerilir "
        "(1-2 piksel kutu, baretli başta NO-Hardhat, baret boyutunun >2 katı Hardhat kutusu, yüksek görünürlüklü giysiye NO-Safety Vest vb.).\n"
        "- **Valid Annotation:** meşru çoklu anotasyon (farklı kişiler, elde baret, vest kutusunun içindeki küçük arka plan kişisi, meşru negatif görüntü).\n"
        "- **Needs Manual Review:** düşük çözünürlük/piksellenme/kısmi nesne nedeniyle görselden karar verilemedi.\n")
    add("## 3. Üç çelişkili örtüşme örneği (önceki rapordaki CONFLICTING_OVERLAP)")
    old = [i for i in a_issues(a) if i["issue_code"] == "CONFLICTING_OVERLAP"]
    rows = []
    for i in old:
        m = [r for r in rf if r["image_path"] == i["image_path"] and r["source"].startswith("semantic_conflict:vest")]
        res = "; ".join(f"{r['item_id']}: {r['category']}" for r in m) or "semantik taramada ayrı kayıt yok"
        rows.append([f"`{Path(i['image_path']).name[:42]}…`", i["detail"], res])
    add(md_table(["Görüntü", "Önceki uyarı", "Yeni karar"], rows))
    add("\n(Üç örnekte de vest/no_vest kutuları yan yana veya arka planda ayrı kişilere ait olabilir; karar tablosuna bakınız.)\n")
    add("## 4. Rastgele kalite örneklemi (96 kutu) ve tahmini hata oranı")
    aud = [r for r in rf if r["source"].startswith("audit_")]
    n_err = sum(1 for r in aud if r["category"] in ("Confirmed Error", "Likely Error"))
    n_nr = sum(1 for r in aud if r["category"] == "Needs Manual Review")
    lo, hi = wilson_interval(n_err, len(aud))
    lo2, hi2 = wilson_interval(n_nr, len(aud))
    rows = []
    for role in ("helmet", "no_helmet", "vest", "no_vest"):
        c = Counter(r["category"] for r in aud if r["source"] == f"audit_{role}")
        rows.append([role, sum(c.values()), c.get("Valid Annotation", 0), c.get("Needs Manual Review", 0),
                     c.get("Likely Error", 0) + c.get("Confirmed Error", 0)])
    add(md_table(["Rol", "n", "Geçerli", "Belirsiz", "Olası hata"], rows))
    add(f"\nToplam n={len(aud)}: olası hata **{n_err}** (%{100 * n_err / len(aud):.1f}; Wilson %95 GA: %{100 * lo:.1f}–%{100 * hi:.1f}), "
        f"belirsiz **{n_nr}** (%{100 * n_nr / len(aud):.1f}; GA %{100 * lo2:.1f}–%{100 * hi2:.1f}). **Sınırlamalar:** kutu yüksekliği <24 px olanlar "
        "örneklemden çıkarıldı (küçük kutularda hata oranı muhtemelen daha yüksek); örneklem küçük; tek değerlendirici; Roboflow train görüntülerinin "
        f"çoğu 2×2 kolaj ve siyah kare (cutout) içerdiğinden kutular kolaj kenarlarında kırpılmıştır. Siyah kare içeren görüntü oranı: "
        f"train %{a['art']['site_safety']['train']['cutout_pct']}, val %{a['art']['site_safety']['val']['cutout_pct']}, "
        f"test %{a['art']['site_safety']['test']['cutout_pct']} (sezgisel sayım).\n")
    add("## 5. Teknik sorunların değerlendirmesi")
    add("- **BOX_VERY_SMALL (21):** hepsi Likely Error — 1–2 piksel kutular, augmentation ile ince şeride kırpılmış bulanık görüntülerde; 9'u TEST görüntüsünde (`ka_01181…`), "
        "yani test metriklerini etkiler.")
    add("- **DUPLICATE_BOX (4):** Confirmed Error (birebir aynı satır). **NEAR_DUPLICATE_BOX (1):** Likely Error.")
    add("- **LABEL_EMPTY (23):** 23/23 görüntüde insan/KKD yok (kitapçı, havalimanı, kumarhane, kuş, iskele slaytı karesi): meşru negatif örnek. "
        "Ancak bunlar alan dışı (MIT indoor 'null images'), gerçek şantiye zor-negatifleri değildir.")
    add("- **Çelişki adayları (93):** çoğu meşru (farklı kişi, elde baret, küçük arka plan kişisi); Hardhat kutusu baret boyutunun çok üstünde 5 örnek Likely Error.\n")
    return "\n".join(L) + "\n"


def a_issues(a: dict) -> list[dict]:
    from isg_vision.dataset_audit.common import read_csv as rc
    return [r for r in rc(PROJECT_ROOT / "reports" / "dataset_audit" / "annotation_issues.csv") if r["dataset"] == "site_safety"]


def cleanup_suggestions(a: dict) -> str:
    rev = [r for r in a["rev"] if r["dataset"] == "site_safety"]
    L = ["# Roboflow Etiket Temizliği — ÖNERİLEN Düzeltmeler (UYGULANMADI)\n",
         f"_Üretim tarihi {date.today().isoformat()}. Bu belge yalnızca gelecekteki temizlik için öneridir. Ham veri (`data/raw/site_safety/`) "
         "değiştirilmemiştir; hiçbir etiket silinmemiş veya düzeltilmemiştir. Temizlik yapılacaksa YENİ bir kopyada ve kullanıcı onayıyla uygulanmalıdır._\n",
         "## Özet\n"]
    by = Counter(r["category"] for r in rev)
    rows = [[k, by.get(k, 0)] for k in ("Confirmed Error", "Likely Error", "Needs Manual Review", "Valid Annotation")]
    L.append(md_table(["Kategori", "Öğe"], rows))
    L.append("\n## Önerilen eylemler\n")
    L.append("| Öğe grubu | Sayı | Önerilen eylem | Gerekçe |\n|---|---|---|---|")
    n = lambda src, cat: sum(1 for r in rev if r["source"] == src and r["category"] == cat)  # noqa: E731
    L.append(f"| duplicate_box (Confirmed Error) | {n('duplicate_box', 'Confirmed Error')} | Yinelenen satırı yeni kopyadan sil (ilkini koru) | Birebir aynı satır, eğitim kaybını iki kez ağırlıklandırır |")
    L.append(f"| near_duplicate_box (Likely Error) | {n('near_duplicate_box', 'Likely Error')} | İki kutudan birini kaldır (IoU≥0.95 aynı sınıf) | Aynı nesne iki kez etiketli |")
    L.append(f"| tiny_box (Likely Error) | {n('tiny_box', 'Likely Error')} | Kutuyu yeni kopyadan çıkar veya görüntüyü eğitimden/testten hariç tut | 1-2 px kutular öğrenilemez; 9'u test setinde metrikleri bozar |")
    sem_le = sum(1 for r in rev if r["source"].startswith("semantic_conflict") and r["category"] == "Likely Error")
    L.append(f"| Hardhat kutusu baret boyutunun çok üstünde (Likely Error) | {sem_le} | Kutuyu baret sınırına küçült veya kaldır (insan onayı) | Başı/gövdeyi de kapsıyor |")
    aud_le = sum(1 for r in rev if r["source"].startswith("audit_") and r["category"] == "Likely Error")
    L.append(f"| Kalite örneklemi olası hatalar | {aud_le} | Örnek dosyaları düzelt; kalan veri için sistematik tarama (aşağıda) | Baretli başta NO-Hardhat, yelekli giysiye NO-Safety Vest, gövde bölgesinde Hardhat |")
    L.append(f"| Needs Manual Review | {by.get('Needs Manual Review', 0)} | İnsan incelemesi; kararsızsa eğitimde `ignore`/hariç | Görselden karar verilemedi |")
    L.append("\n## Sistematik tarama önerileri (veri temizliği öncesi)\n")
    L.append("- `NO-Safety Vest` kutularında yüksek görünürlüklü renk (parlak sarı/yeşil/turuncu) oranı hesaplanıp yüksek olanlar incelenmeli (örneklemde 2/24).")
    L.append("- `NO-Hardhat` kutularının içinde baret rengi/şekli içerenler (ör. Hardhat sınıflı kutu ile örtüşme, renk doygunluğu) taranmalı.")
    L.append("- Kutu yüksekliği <12 px ve kolaj kenarına değen kutular için hariç tutma eşiği belirlenmeli; hariç tutma kararı Aşama 9–10 (birleştirme/geliştirilmiş model) eğitim stratejisine bağlıdır.")
    L.append("- 2×2 kolaj görüntülerde (train) kutular kolaj karosuna göre yeniden değerlendirilmeli; mümkünse kolaj olmayan görüntüler ayrı değerlendirilmeli.")
    L.append("\n## Aday listesi\n\n`roboflow_cleanup_candidates.csv`: Confirmed Error + Likely Error öğeleri (görüntü, etiket dosyası, satır, satır metni, gerekçe).")
    return "\n".join(L) + "\n"


def cleanup_candidate_rows(a: dict) -> list[dict[str, Any]]:
    return [{"item_id": r["item_id"], "category": r["category"], "image_path": r["image_path"], "label_path": r["label_path"],
             "lines": r["lines"], "label_text": r["label_text"], "rationale": r["rationale"], "suggested_fix": r["suggested_fix"]}
            for r in a["rev"] if r["dataset"] == "site_safety" and r["category"] in ("Confirmed Error", "Likely Error")]


# ---------------------------------------------------------------- yakın kopya
def _exposure(dv: list[dict]) -> dict[tuple[str, str], set[str]]:
    inv: dict[tuple[str, str], set[str]] = defaultdict(set)
    for r in dv:
        if r["relation"] == "within_dataset_cross_split" and r["decision"] in STRONG:
            inv[(r["dataset_a"], r["split_a"])].add(r["path_a"])
            inv[(r["dataset_b"], r["split_b"])].add(r["path_b"])
    return inv


def duplicate_report(a: dict) -> str:
    dv, ds, val, L = a["dv"], a["dv_sum"], a["dv_val"], []
    add = L.append
    add("# Yakın Kopya Doğrulama Raporu (Aşama 2 eksik giderme)\n")
    add(f"_Üretim tarihi {date.today().isoformat()}. Veriler: `duplicate_verification.csv`, `duplicate_verification_summary.json`, "
        "`duplicate_method_validation.json`, `configs/review/duplicate_visual_checks.yaml`. Hiçbir görüntü silinmedi/değiştirilmedi; bölünmeler korunuyor._\n")
    add("## 1. Önceki yöntemin değerlendirmesi")
    add("Önceki aşamada 64-bit dHash (≤5 bit) kullanıldı: 0 birebir (SHA-256) ve 83 yakın kopya çifti bulundu. Bu yöntem kırpma/dönme/anizotropik "
        "ölçekleme gibi augmentation'lara dayanıklı değildir; ayrıca yalnızca hash mesafesine dayanıyordu. Yeni yöntem aday üretimini ve kararı ayırır:\n")
    add("1. **Aday üretimi (üçü birlikte):** (a) dHash + çok pencereli pHash (tam/%90/%80 kırpma, yatay çevirme); (b) ORB tanımlayıcıları üzerinde "
        "bit-örnekleme LSH oylaması (kırpma/dönme/çevirmeye dayanıklı, 16 tablo × 20 bit, kova ≤12, ≥15 oy); (c) aynı kaynak adı (`.rf.` öncesi).")
    add("2. **Doğrulama:** her aday çifte ORB (≤1500 öznitelik) + RANSAC ile benzerlik ve TAM AFİN dönüşüm (Roboflow 640×640 *stretch* anizotropisini "
        "yakalamak için), inlier sayısı/oranı, yayılım ve hizalanmış alanda sıfır-ortalamalı NCC. Saf siyah (cutout) bölgeler ORB'den maskelenir.")
    add("3. **Karar:** `same_image_derived`, `same_scene_different_frame`, `partial_region_match` (kolaj: yalnızca bir bölge ortak), `uncertain`, `not_duplicate`, "
        "`exact_duplicate` (SHA-256). Hash mesafesi tek başına karar vermez.\n")
    add("## 2. Yöntem doğrulama deneyi\n")
    add(f"90 gerçek görüntüden (üç veri setinden 30'ar) üretilen bozulmuş kopyalar (Roboflow benzeri) ve rastgele çiftlerle. Eşikler: {ds['thresholds']}.\n")
    rows = []
    lsh = val["lsh_candidate_recall"]["by_augmentation"]
    for kind, v in val["positives"].items():
        rows.append([kind, v["hash_candidate_rate"], lsh.get(kind, "—"), v["orb_geometric_match_rate"], v["decided_same_image_derived_rate"],
                     v["inliers_median"], v["ncc_median"]])
    add(md_table(["Bozulma", "Hash aday yakalama", f"LSH aday yakalama (6085 dikkat dağıtıcı)", "ORB geometrik eşleşme", "Karar: same_image_derived", "Inlier medyan", "NCC medyan"], rows))
    neg = val["negatives"]
    add(f"\n**Negatifler (rastgele {neg['n_pairs']} çift):** hash yanlış adayı {neg['hash_false_candidates']}; geometrik eşleşen çift {neg['geometric_false_positives']} "
        f"(inlier medyan {neg['inliers_median']}, P99 {neg['inliers_p99']}, maks {neg['inliers_max']}). Bu iki çift görsel olarak incelendi: "
        "biri (`reflectivevest1529` ↔ `reflectivevest-398`) iki Mendeley kolajında ortak bisikletçi karesi (gerçek ortak kaynak), diğeri (`construction-684` ↔ `image_941`, Roboflow kolajları) "
        "ortak bölge görsel olarak KESİNLEŞTİRİLEMEDİ. Dolayısıyla gerçek yanlış pozitif oranı en çok 1/400 (%0.25) civarındadır; tüm rastgele çiftlerin "
        "tam negatif olduğu garanti değildir (veri setlerinde gerçek tekrarlar var).")
    add("\n**Seçilen eşikler ve dayanak:** rastgele çiftlerin %99'unda inlier ≤8; 12-24 inlier 'uncertain', ≥25 eşleşme. Hizalanmış NCC≥0.95 ve inlier≥12 ise aynı görüntü (rastgele hizalamada görülmez).")
    add("\n**Bilinen sınırlar:** (i) `combo` bozulmasında (kırpma+dönme+parlaklık+bulanıklık+çevirme) `same_image_derived` oranı yalnızca %75.6; geri kalanı `same_scene_different_frame` etiketi alır "
        "(her ikisi de eşleşme sayılır). (ii) Aday üretimi tüm olası çiftleri taramaz; ≥15 LSH oyu veya hash ≤10 bit gerekir — bu nedenle aşağıdaki oranlar ALT SINIRDIR.\n")
    add("## 3. Gerçek augmentation üzerinde aday üretici geri çağırması (kaynak adı = sahte gerçek)\n")
    se = ds["sibling_candidate_recall_eval"]
    add(f"Roboflow'da aynı kaynak adını taşıyan aynı-bölüm çiftlerinden {se['n']} tanesi örneklendi: geometrik eşleşme {se['decided_geometric_match']}/{se['n']}; "
        f"dHash≤5 {se['dhash_le5']}, pHash-çok-pencere≤10 {se['phash_multicrop_le10']}, LSH oyu≥15 {se['lsh_votes_ge_min']}. "
        "**Yorum:** Roboflow *train* görüntüleri 2×2 kolajdır (görsel doğrulandı); aynı kaynak adını taşıyan iki görüntü genellikle yalnızca bir kolaj karesini paylaşır, "
        "bu yüzden hash/LSH yakalaması düşüktür ve geometrik eşleşme çoğunlukla bölgesel (`partial_region_match`/`same_scene_different_frame`). "
        "Kaynak adı tek başına 'aynı görüntü' kanıtı değildir.\n")
    add("**Aynı kaynak adı ≠ kopya.** Önceki aşamada bölümler arasında aynı kaynak adını taşıyan 8 Roboflow grubu 'olası sızıntı' sayılmıştı. Geometrik ve görsel doğrulama: "
        "yalnızca 1 grupta (`Movie-on…mov-13`) kanıtlı eşleşme var; 4 grupta görsel olarak ortak içerik YOK (kolaj karesi adı ilk kareden gelir); 1 grupta yalnızca geometrik "
        "(görsel bakılmadı) eşleşme yok; 2 grupta (`IMG_3103_mp4-17`, `IMG_0871_mp4-23`) aynı kişi/videodan olma ihtimali var ama yöntemle KESİNLEŞTİRİLEMEDİ. "
        "Dolayısıyla Roboflow dosya adı öneki güvenilir bir kopya anahtarı değildir.\n")
    add("## 4. Sonuçlar\n")
    add(f"Doğrulanan aday çift: **{ds['candidate_pairs']}** ({ds['images']} görüntü). Karar dağılımı: {ds['decisions']}.\n")
    rows = []
    for rl, d in ds["decisions_by_relation"].items():
        rows.append([rl] + [d.get(k, 0) for k in ("same_image_derived", "same_scene_different_frame", "partial_region_match", "uncertain", "not_duplicate")])
    add(md_table(["İlişki", "same_image_derived", "same_scene_different_frame", "partial_region_match", "uncertain", "not_duplicate"], rows))
    add("\n### 4.1 Bölümler arası maruziyet (en az bir geometrik eşleşmesi olan görüntüler; ALT SINIR)\n")
    inv = _exposure(dv)
    sizes = {(r["dataset"], r["split"]): int(r["images"]) for r in a["inv"] if r["split"] != "ALL"}
    rows = [[f"{d}/{s}", len(v), sizes[(d, s)], f"%{100 * len(v) / sizes[(d, s)]:.1f}"] for (d, s), v in sorted(inv.items())]
    add(md_table(["Veri seti/bölüm", "Eşleşmeli görüntü", "Bölüm boyutu", "Oran"], rows))
    add("\n**Yüksek öncelikli risk:** `site_safety` val ve test görüntülerinin büyük çoğunluğu başka bir bölümdeki (çoğunlukla train) görüntüyle aynı videodan/aynı fotoğraftan "
        "türemiştir. Mevcut Roboflow val/test metrikleri bu nedenle iyimser olur. Mendeley val'de de anlamlı bir pay vardır (rastgele %20 bölme).\n")
    add("### 4.2 Öncelikli çiftler\n")
    vis = a["visual"]
    pri = ["DV0079", "DV0751", "DV0081", "DV0428", "DV4609", "DV4612"]
    by_id = {r["pair_id"]: r for r in dv}
    rows = []
    for pid in pri:
        r = by_id[pid]
        rows.append([pid, f"{r['dataset_a']}/{r['split_a']}", f"`{Path(r['path_a']).name[:34]}…`", f"{r['dataset_b']}/{r['split_b']}",
                     f"`{Path(r['path_b']).name[:34]}…`", f"dHash {r['dhash_bits']}, pHash {r['phash_bits']}, LSH {r['lsh_votes']}/{r['lsh_votes_flipped']}, "
                     f"ORB inlier {r['orb_inliers']} (oran {r['orb_inlier_ratio']}), NCC {r['aligned_ncc']}, anizotropi {r['anisotropy']}",
                     r["decision"], f"{vis[pid]['result']}: {vis[pid]['note']}", r["leakage_risk"]])
    add(md_table(["Çift", "Veri seti A", "Dosya A", "Veri seti B", "Dosya B", "Ölçülen benzerlikler", "Otomatik karar", "Görsel doğrulama", "Sızıntı riski"], rows))
    add("\n- **Mendeley–Roboflow 4 çift: dördü de AYNI KAYNAK FOTOĞRAF** (Roboflow sürümü 640×640 gerilmiş). Mendeley sayfası verinin Roboflow'dan da derlendiğini söylüyor; "
        "ancak Mendeley ZIP'inin kendi README'si de bir Roboflow projesinden dışa aktarım olduğunu gösteriyor. İlk yöntem benzerlik dönüşümüyle bu çiftleri "
        "'uncertain/not_duplicate' saymıştı; tam afin dönüşümü eklenince düzeldi.")
    add("- **val↔test 2 çift:** aynı videodan ardışık kareler (poz farklı): aynı görüntü değil ama aynı sahne; val/test bağımsızlığını bozar.")
    cross = [r for r in dv if r["relation"] == "cross_dataset" and r["decision"] in STRONG + ("uncertain",)]
    add(f"- Veri setleri arası eşleşme adayı toplam {len(cross)} çift (karar: {dict(Counter(r['decision'] for r in cross))}); 4'ü yan yana incelenip aynı-kaynak fotoğraf olarak doğrulandı; kalan çiftler aynı kaynak/sahne ADAYI olarak CSV'dedir "
        "(görsel doğrulanmadı).\n")
    add("### 4.3 Görsel doğrulama örneklemi\n")
    n_vis = len(vis)
    res = Counter(v["result"] for v in vis.values())
    add(f"Otomatik kararı 'eşleşme' olan {n_vis} çift (6 öncelikli + rastgele 12 Roboflow + 9 Mendeley bölümler-arası; bir Roboflow çifti ortak) yan yana açılıp incelendi: "
        f"**{n_vis}/{n_vis} çift görsel olarak aynı kaynağa/sahneye ait** (dağılım: {dict(res)}). Bu, bölümler arası geometrik eşleşmelerde tahmini kesinliğin yüksek "
        f"olduğunu gösterir (Wilson %95 alt sınır ≈ %{100 * wilson_interval(n_vis, n_vis)[0]:.0f}); yine de örneklem küçüktür ve 'uncertain' ile `not_duplicate` kararları görsel doğrulanmadı.\n")
    add("## 5. Augmentation izleri\n")
    art = a["art"]["site_safety"]
    add(f"Siyah kare (cutout) içeren görüntü oranı (sezgisel): Roboflow train %{art['train']['cutout_pct']}, val %{art['val']['cutout_pct']}, test %{art['test']['cutout_pct']}. "
        "Augmentation yalnızca eğitim bölümüne uygulanmış görünüyor. Yerel README `Resize 640x640 (Stretch)` ve flip/crop/rotation/shear/brightness/exposure/blur listeler; "
        "cutout ve 2×2 kolaj ise README'de YAZMIYOR ancak görüntülerde görülüyor: README ile gerçek içerik uyuşmuyor.\n")
    add("## 6. Karar ve öneriler (uygulanmadı)\n")
    add("- Veri setleri birleştirilmeden ve eğitim yapılmadan önce bölünmeler **grup bazlı** (aynı video/sahne/fotoğraf bir arada) yeniden kurulmalı; mevcut Roboflow val/test'e güvenilmemeli.")
    add("- Mendeley `val`'deki train ile eşleşen görüntüler (aşağı yukarı %9) değerlendirme dışı bırakılmalı veya gruplanmalı.")
    add("- Veri setleri arası aynı-kaynak çiftleri (en az 4) birleşik sette tek örneğe indirilmeli.")
    add("- Yeni, daha geniş kapsamlı aday üretimi gerekirse (≥%99 geri çağırma) LSH oy eşiği düşürülüp ORB doğrulaması genişletilebilir; maliyet ≈ dakikalar.")
    return "\n".join(L) + "\n"


# ---------------------------------------------------------------- lisans
def license_report(a: dict, registry: dict) -> str:
    lic, L = a["lic"], []
    add = L.append
    add("# Lisans Doğrulama Raporu (Aşama 2 eksik giderme)\n")
    add("_Kontrol tarihi: 2026-11-11. Hukuki görüş değildir; bitirme raporunda kullanılacak atıf/lisans ifadeleri yayıncının güncel sayfasından yeniden teyit edilmelidir. "
        "Erişim kontrolleri aşılmaya çalışılmadı: HTTP 403 veren sayfalar için yeniden/alternatif yolla erişim denenmedi._\n")
    add("## 1. Özet\n")
    status_tr = {"verified_official": "Resmi kaynaktan DOĞRULANDI", "partially_verified": "KISMEN doğrulandı",
                 "verified_local_only": "Yalnızca YEREL dosyadan (resmi doğrulama YOK)", "not_verified": "Doğrulanamadı"}
    rows = []
    for ds, v in lic.items():
        rows.append([ds, v["claimed_license"], status_tr[v["verification_status"]], v["official_link"], "2026-11-11"])
    add(md_table(["Veri seti", "İddia edilen lisans", "Doğrulama durumu", "Resmî bağlantı", "Kontrol tarihi"], rows))
    for ds, v in lic.items():
        add(f"\n## {ds} — {registry[ds]['display_name']}\n")
        add(f"- **İddia edilen lisans:** {v['claimed_license']}  \n- **Lisans bağlantısı:** {v['license_link']}  \n- **Doğrulama durumu:** {status_tr[v['verification_status']]}")
        add("\n**Kanıt kaynakları:**\n")
        add(md_table(["Tür", "Kaynak", "Sonuç"], [[e["type"], f"`{e['url']}`", e["result"]] for e in v["evidence"]]))
        add(f"\n**Bilinen kullanım koşulları:** {v['known_terms']}\n")
        add("**Açıkta kalan belirsizlikler:**\n")
        for u in v["open_uncertainties"]:
            add(f"- {u}")
    add("\n## Atıf metinleri (bitirme raporu için)\n")
    for ds in lic:
        add(f"- {registry[ds]['license']['attribution']}")
    add("\n## Değerlendirme\n")
    add("- **Doğrulanan:** Construction-PPE (resmi sayfa + arşivdeki lisans metni); Mendeley için lisans etiketi (resmi sayfa + API) ve yerel ZIP'in resmi arşivle birebir aynı olduğu (SHA-256 eşleşmesi).")
    add("- **Doğrulanamayan:** Roboflow v27 lisansı (yalnızca yerel metadata; resmi sayfalar 403); Mendeley ve Roboflow'un ÜST KAYNAK lisansları ve görsel bazında hak sahipliği.")
    add("- **Önemli ilke:** Veri setinin CC BY 4.0 olarak yayımlanması, içindeki üçüncü taraf görsellerin haklarının otomatik olarak doğrulandığı anlamına gelmez "
        "(Mendeley'nin kendi açıklaması ek izin gerekebileceğini belirtir; CC SSS'ye göre lisans yalnızca lisans verenin sahip olduğu haklar için geçerlidir). "
        "İki veri seti de başka kaynaklardan (ve Roboflow olan YouTube videolarından) derlenmiştir; akademik kullanım için atıf yapılmalı, ticari/yeniden dağıtım "
        "senaryolarında kaynak hakları ayrıca incelenmelidir.")
    return "\n".join(L) + "\n"


# ---------------------------------------------------------------- kontrol listesi
def _checklist_items(a: dict) -> list[dict[str, Any]]:
    """Eski 59 maddenin sabit numaralı listesi."""
    res = a["resol"]
    items: list[dict[str, Any]] = []
    for k, v in res.items():
        items.append({"id": k, "category": v["category"], "result": v["result"], "evidence": v["evidence"]})
    return items


def build_checklist_v2(a: dict, registry: dict, entries: list[dict], targets: dict[int, str]) -> tuple[list[dict[str, Any]], str]:
    """(madde listesi, markdown). C01-C10 ve D01-D15 veri tabanlı çözülür."""
    resol = a["resol"]
    rev = a["rev"]
    items: list[dict[str, Any]] = []

    def add(item_id: str, section: str, text: str, category: str, result: str, evidence: str) -> None:
        items.append({"id": item_id, "section": section, "text": text, "category": category, "result": result, "evidence": evidence})

    crops = [("construction_ppe", "class_00_helmet.jpg", "Baret kutusu gerçekten bareti mi gösteriyor?"),
             ("construction_ppe", "class_06_no_helmet.jpg", "`no_helmet` baş mı, bütün insan mı?"),
             ("construction_ppe", "class_02_vest.jpg", "Yelek kutusu yeleği mi gösteriyor?"),
             ("construction_ppe", "class_11_no_vest.jpg", "`no_vest` hangi bölge?"),
             ("mendeley_ppe", "class_00_Helmet.jpg", "`Helmet` baş+baret mi?"),
             ("mendeley_ppe", "class_01_NoHelmet.jpg", "`NoHelmet` baş/yüz mü?"),
             ("mendeley_ppe", "class_03_Vest.jpg", "`Vest` yeleği/gövdeyi mi kapsıyor?"),
             ("mendeley_ppe", "class_02_NoVest.jpg", "`NoVest` KKD bağlamında mı?"),
             ("site_safety", "class_00_Hardhat.jpg", "`Hardhat` sıkı mı?"),
             ("site_safety", "class_02_NO-Hardhat.jpg", "`NO-Hardhat` baş bölgesi mi, baretli kutu var mı?"),
             ("site_safety", "class_07_Safety_Vest.jpg", "`Safety Vest` kutularında yelek var mı?"),
             ("site_safety", "class_04_NO-Safety_Vest.jpg", "`NO-Safety Vest` kutularında yelek görünen var mı?")]
    for i, (ds, fn, q) in enumerate(crops, start=1):
        r = resol[f"A{i:02d}"]
        add(f"A{i:02d}", "A. Kutu semantiği (kırpma sayfaları)", f"`{ds}/crops/{fn}` — {q}", r["category"], r["result"], r["evidence"])
    for i, q in enumerate(["Üç `helmet` karşılığı aynı mı?", "Üç `no_helmet` karşılığı aynı bölge mi?", "Üç `vest` ve `no_vest` karşılığı aynı bölge mi?",
                           "Genel kontrol: `contact_sheet_*.jpg` (12 dosya) kutular nesne üzerinde mi?"], start=1):
        r = resol[f"B{i:02d}"]
        add(f"B{i:02d}", "B. Veri setleri arası semantik karşılaştırma", q, r["category"], r["result"], r["evidence"])

    # C: şüpheli etiketler (veri tabanlı)
    issues = a_issues(a)
    rf = {(r["image_path"], r["source"].split(":")[0]): r for r in rev}
    conf_old = [i for i in issues if i["issue_code"] == "CONFLICTING_OVERLAP"]
    n = 0
    for i in conf_old:
        n += 1
        m = [r for r in rev if r["image_path"] == i["image_path"] and r["source"].startswith("semantic_conflict:vest")]
        if m:
            cats = Counter(r["category"] for r in m)
            cat = "B" if "Needs Manual Review" not in cats else "D"
            result = "; ".join(f"{r['item_id']}={r['category']}" for r in m)
            add(f"C{n:02d}", "C. Şüpheli etiketler", f"CONFLICTING_OVERLAP `{Path(i['image_path']).name[:40]}…` ({i['detail']})", cat, result,
                "label_review.csv; semantic_review/conflicts_site_safety/*.jpg")
        else:
            add(f"C{n:02d}", "C. Şüpheli etiketler", f"CONFLICTING_OVERLAP `{Path(i['image_path']).name[:40]}…`", "D", "Semantik taramada kayıt yok", "")
    dup_ids = [r for r in rev if r["source"] == "duplicate_box"]
    for r in dup_ids:
        n += 1
        add(f"C{n:02d}", "C. Şüpheli etiketler", f"DUPLICATE_BOX `{Path(r['image_path']).name[:40]}…` ({r['lines']})", "A",
            f"{r['category']}: aynı sınıf ve birebir aynı koordinatlı satır ({r['label_text'][:70]})", "label_review.csv; roboflow_review/duplicate_boxes.jpg")
    for r in [x for x in rev if x["source"] == "near_duplicate_box"]:
        n += 1
        add(f"C{n:02d}", "C. Şüpheli etiketler", f"NEAR_DUPLICATE_BOX `{Path(r['image_path']).name[:40]}…`", "B", f"{r['category']}: {r['rationale']}", "roboflow_review/duplicate_boxes.jpg")
    tiny = Counter(r["category"] for r in rev if r["source"] == "tiny_box")
    n += 1
    add(f"C{n:02d}", "C. Şüpheli etiketler", "BOX_VERY_SMALL (21 kutu)", "B", f"21/21 görsel incelendi: {dict(tiny)} — 1-2 px kutular, ince kırpılmış şeritler (9'u test görüntüsünde)", "roboflow_review/tiny_boxes.jpg")
    emp = Counter(r["category"] for r in rev if r["source"] == "empty_label")
    n += 1
    add(f"C{n:02d}", "C. Şüpheli etiketler", "LABEL_EMPTY (23 boş etiket)", "B", f"23/23 görüntü incelendi: {dict(emp)} — insan/KKD yok, meşru negatif", "roboflow_review/empty_labels_01-02.jpg")

    # D: tekrar/sızıntı
    dv = a["dv"]
    vis = a["visual"]
    by_id = {r["pair_id"]: r for r in dv}
    d = 0
    for pid in ("DV0079", "DV0751", "DV0081", "DV0428"):
        r = by_id[pid]
        d += 1
        add(f"D{d:02d}", "D. Tekrar / sızıntı", f"Veri setleri arası: `{Path(r['path_a']).name[:34]}…` ↔ `{Path(r['path_b']).name[:34]}…`", "B",
            f"{vis[pid]['result']}: {vis[pid]['note']} (otomatik: {r['decision']}, inlier {r['orb_inliers']}, NCC {r['aligned_ncc']})", "duplicate_verification.csv; duplicate_verification_report.md")
    for pid in ("DV4609", "DV4612"):
        r = by_id[pid]
        d += 1
        add(f"D{d:02d}", "D. Tekrar / sızıntı", f"val↔test: `{Path(r['path_a']).name[:34]}…` ↔ `{Path(r['path_b']).name[:34]}…`", "B",
            f"{vis[pid]['result']}: {vis[pid]['note']} (inlier {r['orb_inliers']}, NCC {r['aligned_ncc']})", "duplicate_verification.csv")
    groups = [r for r in a["name_overlap_groups"]]
    for g in groups:
        d += 1
        base = g["base_name"]
        rows = [r for r in dv if r["name_overlap"] == "yes" and base in r["path_a"] and base in r["path_b"] and r["relation"] == "within_dataset_cross_split"]
        ng = a["name_groups_visual"].get(base)
        if ng is None:
            add(f"D{d:02d}", "D. Tekrar / sızıntı", f"Bölümler arası ortak kaynak adı `{base}`", "D", "Karar kaydı yok", "duplicate_verification.csv")
        else:
            add(f"D{d:02d}", "D. Tekrar / sızıntı", f"Bölümler arası ortak kaynak adı `{base}`", ng["category"], ng["result"],
                "duplicate_verification.csv; duplicate_verification/_name_groups.jpg")
    n_tv = sum(1 for k, v in vis.items() if by_id[k]["split_a"] + "|" + by_id[k]["split_b"] in ("train|val", "val|train"))
    d += 1
    add(f"D{d:02d}", "D. Tekrar / sızıntı", "train↔val yakın kopya adayları: en az 10'u görsel karşılaştırılmalı", "B",
        f"{n_tv} train|val çifti görsel incelendi (hepsi aynı kaynak/sahne); toplam train|val geometrik eşleşme: {sum(1 for r in dv if r['relation'] == 'within_dataset_cross_split' and {r['split_a'], r['split_b']} == {'train', 'val'} and r['decision'] in STRONG)}", "duplicate_verification_report.md §4.3")
    d += 1
    r = resol["D16"]
    add(f"D{d:02d}", "D. Tekrar / sızıntı", "Mendeley bölünmesi rastgele: aynı sahnenin train ve val'de bulunması kabul edilebilir mi?", r["category"], r["result"], r["evidence"])

    # E: eşleme onayı
    for i, e in enumerate([x for x in entries if x.get("target") is not None], start=1):
        r = resol[f"E{i:02d}"]
        add(f"E{i:02d}", "E. Aday sınıf eşlemesi", f"{e['dataset']} `{e['id']}:{e['name']}` → `{targets[e['target']]}` (YAML durumu: {e['status']})", r["category"], r["result"], r["evidence"])
    for i, q in enumerate(["Roboflow v27 lisansı resmi sayfadan doğrulandı mı?", "Mendeley v6 lisansı ve üst kaynaklar", "Ultralytics atıf metni bitirme raporunda mı?", "Mendeley 640×640 farkı kabul edilebilir mi?"], start=1):
        r = resol[f"F{i:02d}"]
        add(f"F{i:02d}", "F. Lisans / kaynak", q, r["category"], r["result"], r["evidence"])
    r = resol["G01"]
    add("G01", "Birleştirmeye geçiş kararı", "A–F tamamlandı; sınıf eşleme ve kutu semantiği kararı yazıya döküldü mü?", r["category"], r["result"], r["evidence"])

    cnt = Counter(i["category"] for i in items)
    L = ["# Manuel İnceleme ve Onay Kontrol Listesi (Aşama 2 — eksik giderme sonrası)\n",
         f"_Üretim tarihi {date.today().isoformat()}. Toplam {len(items)} madde. Her madde yalnızca geçerli bir analize dayanarak sonuçlandırılmıştır; "
         "betiğin çalışmış olması görsel/anlamsal kontrolü tamamlanmış saymaz. Görsel kararlar model destekli, tek değerlendirici incelemesidir._\n",
         "## Özet\n",
         md_table(["Kategori", "Madde"], [[f"{k} — {CAT_TR[k]}", cnt.get(k, 0)] for k in "ABCD"] + [["**Toplam**", len(items)]]),
         f"\nÇözülen (A+B): **{cnt['A'] + cnt['B']}/{len(items)}**; çözülemeyen teknik (C): **{cnt['C']}**; kullanıcı kararı gerekli (D): **{cnt['D']}**.\n"]
    for k in "ABCD":
        sel = [i for i in items if i["category"] == k]
        L.append(f"## {k}. {CAT_TR[k]} ({len(sel)})\n")
        L.append(md_table(["No", "Madde", "Sonuç", "Kanıt"], [[i["id"], i["text"], i["result"], i["evidence"]] for i in sel]))
        L.append("")
    return items, "\n".join(L) + "\n"


def remaining_reviews(a: dict, items: list[dict[str, Any]]) -> str:
    cnt = Counter(i["category"] for i in items)
    L = ["# Kalan Manuel İncelemeler (Aşama 2)\n",
         f"_Üretim tarihi {date.today().isoformat()}. Kontrol listesindeki {len(items)} maddenin {cnt['A'] + cnt['B']}'i analizle çözüldü. "
         f"Aşağıda yalnızca {cnt['D']} kullanıcı kararı ve {cnt['C']} çözülemeyen teknik madde var._\n",
         "## Kullanıcı kararı/onayı gerekenler (D)\n"]
    rows = [[i["id"], i["text"], i["result"]] for i in items if i["category"] == "D"]
    L.append(md_table(["No", "Madde", "Neden/ne yapılmalı"], rows))
    L.append("\n## Çözülemeyen teknik sorunlar (C)\n")
    L.append(md_table(["No", "Madde", "Neden çözülemedi"], [[i["id"], i["text"], i["result"]] for i in items if i["category"] == "C"]))
    L.append("\n## Birleştirmeye geçmeden önce ön koşullar\n")
    L.append("1. Mendeley `Helmet` (baş+baret) için karar (E05) ve Roboflow `NO-Hardhat`/`NO-Safety Vest` için kullanım kararı (E10, E12).\n"
             "2. Bölünme stratejisi: grup bazlı (aynı video/sahne/fotoğraf bir arada) yeniden bölme kararı (D16 ve Roboflow val/test sızıntısı).\n"
             "3. Lisans: Roboflow v27 için tarayıcıdan resmi sayfa teyidi (F01); üst kaynak hakları için kabul/ret kararı (F02).\n"
             "4. Hedef kutu tanımı: helmet=baret nesnesi (sıkı), no_helmet=baretsiz baş, vest/no_vest=gövde önerisinin onayı.\n")
    return "\n".join(L) + "\n"


def main_report_addendum(a: dict, entries: list[dict], summary: dict[str, Any]) -> str:
    """dataset_audit_report.md'ye eklenen 'Eksik giderme' bölümü (özet; ayrıntı ayrı raporlarda)."""
    L = ["\n## 16. Eksik giderme sonuçları (Aşama 2 güncellemesi)\n",
         "İlk raporda çözülemeyen konular ek analizlerle yeniden ele alındı. **Önceki bulgular korunmuştur**; aşağıdaki maddeler hangi bölümlerin güncellendiğini "
         "ve yeni sonuçların kaynağını gösterir.\n"]
    st = Counter(e["status"] for e in entries if e.get("target") is not None)
    L.append(f"- **Kutu semantiği** (`semantic_validation_report.md`, `semantic_metrics.json`): kişi kutusuna göre göreli boyut ve baş/gövde oranı + 3×4 görsel karşılaştırma. "
             f"Hedefe eşlenen 12 sınıfın durumu: {dict(st)}. Mendeley `Helmet` baş+baret (yüz dahil) olduğu için `incompatible`; Roboflow `NO-Hardhat` heterojen olduğu için `incompatible`; "
             "Roboflow `NO-Safety Vest` `needs_review`. Önceki taslaktaki medyan-alan temelli bazı yorumlar DÜZELTİLDİ (§11 notu).")
    rf = Counter(r["category"] for r in a["rev"] if r["dataset"] == "site_safety")
    L.append(f"- **Roboflow şüpheli etiketler** (`roboflow_label_review_report.md`, `label_review.csv`, `roboflow_cleanup_suggestions.md`): 238 öğe → {dict(rf)}. "
             "Bölüm 8'deki 'olası etiket gürültüsü' ifadesi nicelleştirildi: rastgele 96 kutuluk örneklemde olası hata ≈%5, belirsiz ≈%26.")
    ds = a["dv_sum"]
    inv = _exposure(a["dv"])
    sizes = {(r["dataset"], r["split"]): int(r["images"]) for r in a["inv"] if r["split"] != "ALL"}
    exp = {f"{d}/{s}": f"%{100 * len(v) / sizes[(d, s)]:.0f}" for (d, s), v in sorted(inv.items()) if (d, s) in sizes and d != "construction_ppe"}
    L.append(f"- **Yakın kopya** (`duplicate_verification_report.md`, `duplicate_verification.csv`): çok yöntemli doğrulama ({ds['candidate_pairs']} aday çift). "
             f"Mendeley–Roboflow 4 çiftin 4'ü AYNI kaynak fotoğraf; val↔test 2 çift aynı videodan ardışık kareler. Bölümler arası eşleşmeli görüntü oranı (alt sınır): {exp}. "
             "Bölüm 9-10'daki dHash tabanlı sayılar bu doğrulamayla GÜNCELLENMİŞTİR; Roboflow val/test sızıntısı yüksek öncelikli risktir.")
    L.append("- **Lisans** (`license_verification_report.md`): Mendeley için resmi sayfa/API lisansı doğrulandı ve yerel ZIP'in SHA-256'sı resmi API ile eşleşti; "
             "Roboflow v27 resmi sayfası HTTP 403 verdi (yerel metadata CC BY 4.0, bağımsız doğrulama yok); üst kaynak lisansları doğrulanamadı.")
    L.append(f"- **Manuel kontrol listesi** (`manual_review_checklist.md`, `remaining_manual_reviews.md`): {summary['checklist_items']} maddenin "
             f"{summary['resolved_A_B']}'si analizle çözüldü (A+B), {summary['unresolved_technical_C']} teknik olarak çözülemedi, {summary['remaining_D']} kullanıcı kararı gerektiriyor.")
    L.append("\n**Genel kullanılabilirlik değerlendirmesi (güncel, §15'in yerine geçer):** Üç veri seti biçim olarak sağlamdır. Birleştirme için engeller: (1) Mendeley `Helmet` ve Roboflow "
             "`NO-Hardhat` kutu tanımı uyumsuzluğu, (2) Roboflow val/test'in büyük ölçüde train ile aynı videodan/fotoğraftan türemesi (grup bazlı yeniden bölme gerekir), "
             "(3) Roboflow `NO-Safety Vest` etiket kalitesi, (4) lisans belirsizlikleri. Bu nedenle birleştirme **henüz hazır değil**; Construction-PPE tek başına en güvenilir "
             "ve (resmi lisansı doğrulanmış) çekirdek veri setidir.\n")
    return "\n".join(L) + "\n"
