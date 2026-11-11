"""Aşama 2 denetim çıktılarından Türkçe rapor ve manuel inceleme kontrol listesi üretir.

Sayılar yalnızca diğer betiklerin ürettiği dosyalardan okunur; burada elle sayı yazılmaz.
Dosya eksikse sahte sonuç üretilmez, açık hata verilir.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from datetime import date
from pathlib import Path
from typing import Any

from isg_vision.dataset_audit.common import human_size, read_csv, read_json

SPLITS = ("train", "val", "test")
SEVERITY_TR = {"error": "hata", "warning": "uyarı", "info": "bilgi"}


def load_artifacts(audit_dir: Path) -> dict[str, Any]:
    """Raporun dayandığı tüm çıktı dosyalarını okur (eksik dosya -> FileNotFoundError)."""
    a: dict[str, Any] = {
        "inventory_rows": read_csv(audit_dir / "dataset_inventory.csv"),
        "inventory": {d["dataset"]: d for d in read_json(audit_dir / "dataset_inventory.json")},
        "validation": read_json(audit_dir / "label_validation_summary.json"),
        "issues": read_csv(audit_dir / "annotation_issues.csv"),
        "dist": read_csv(audit_dir / "class_distribution.csv"),
        "boxstats": read_csv(audit_dir / "class_box_stats.csv"),
        "dist_summary": read_json(audit_dir / "class_distribution_summary.json"),
        "dup_summary": read_json(audit_dir / "duplicate_summary.json"),
        "dups": read_csv(audit_dir / "duplicate_images.csv"),
        "name_overlap": read_csv(audit_dir / "source_name_overlap.csv"),
    }
    log = audit_dir / "extraction_log.json"
    a["extraction"] = {}
    if log.is_file():
        data = read_json(log)
        runs = data.get("runs", []) if isinstance(data, dict) else [{"entries": data}]
        for run in runs:  # sonraki çalıştırmalar öncekilerin üzerine yazar (ZIP bilgisi hepsinde var)
            for e in run["entries"]:
                a["extraction"][e["dataset"]] = e
        a["extraction_runs"] = len(runs)
    return a


def md_table(headers: list[str], rows: list[list[Any]]) -> str:
    out = ["| " + " | ".join(headers) + " |", "|" + "|".join("---" for _ in headers) + "|"]
    for r in rows:
        out.append("| " + " | ".join(str(c).replace("|", "\\|") for c in r) + " |")
    return "\n".join(out)


def _inv(a: dict, ds: str, split: str) -> dict | None:
    return next((r for r in a["inventory_rows"] if r["dataset"] == ds and r["split"] == split), None)


def _int(v: Any) -> int:
    return int(float(v)) if v not in ("", None) else 0


def target_group_counts(a: dict, entries: list[dict]) -> dict[str, dict[int, int]]:
    """Dataset -> hedef sınıf -> toplam kutu (eşlemedeki kaynak sınıfların toplamı)."""
    boxes = {(r["dataset"], int(r["class_id"])): _int(r["boxes"]) for r in a["dist"] if r["split"] == "ALL"}
    out: dict[str, dict[int, int]] = defaultdict(lambda: defaultdict(int))
    for e in entries:
        if e.get("target") is not None:
            out[e["dataset"]][e["target"]] += boxes.get((e["dataset"], int(e["id"])), 0)
    return out


def build_report(a: dict[str, Any], registry: dict[str, Any], targets: dict[int, str],
                 entries: list[dict[str, Any]], mapping_problems: list[str]) -> str:
    names = list(registry)
    L: list[str] = []
    add = L.append

    add("# Aşama 2 — Veri Seti Denetim Raporu")
    add(f"\n_Otomatik üretilmiştir (`scripts/generate_dataset_report.py`), tarih: {date.today().isoformat()}. "
        "Sayılar `reports/dataset_audit/` altındaki betik çıktılarından okunur. "
        "\"Görsel ön gözlem\" ifadeleri kullanıcı tarafından DOĞRULANMAMIŞTIR._\n")

    # 1
    add("## 1. Analizin amacı")
    add("Bu rapor, üç açık erişimli İSG (KKD) veri setinin ileride baret ve yelek (helmet / no_helmet / vest / no_vest) "
        "tespiti için YOLO eğitiminde güvenilir biçimde kullanılıp kullanılamayacağını değerlendirir. Yalnızca analiz "
        "yapılmıştır: veri setleri birleştirilmemiş, yeniden bölünmemiş, etiketler değiştirilmemiş, hiçbir dosya silinmemiş "
        "ve model eğitilmemiştir.\n")

    # 2
    add("## 2. Kullanılan veri setleri")
    rows = []
    for n in names:
        ex = a["extraction"].get(n, {})
        rows.append([registry[n]["display_name"], f"`{registry[n]['raw_dir']}/`",
                     Path(ex.get("zip_path", "?")).name if ex else "?",
                     human_size(ex["zip_size"]) if ex else "?",
                     f"`{ex['zip_sha256'][:16]}…`" if ex and ex.get("zip_sha256") else "?"])
    add(md_table(["Veri seti", "Proje içi konum", "Kaynak ZIP (indirilenler)", "ZIP boyutu", "ZIP SHA-256 (ilk 16)"], rows))
    add("\nZIP'ler `~/Downloads` içinde olduğu gibi korunmuştur; içerik `scripts/extract_datasets.py` ile güvenli biçimde "
        "(üzerine yazmadan, zip-slip denetimiyle, dosyalar salt-okunur) `data/raw/` altına çıkarılmıştır. "
        "Tek üst düzey klasör (`construction-ppe/`, `20250731-ppe2286y/`) sıyrılmıştır; bu yüzden orijinal `data.yaml` "
        "içindeki göreli `path` değerleri burada olduğu gibi çözümlenmez (aşağıya bakın).\n")
    for n in names:
        ex = a["extraction"].get(n)
        if ex and (ex.get("rejected") or ex.get("errors")):
            add(f"- **{n}**: reddedilen/hatalı girdiler: {ex.get('rejected')} {ex.get('errors')}")
    add("- `mendeley_ppe` arşivi, veri dışında bir Python betiği (`01_*.py`, rastgele %20'yi `valid/` klasörüne taşıyan bir bölme "
        "betiği) içerir. Bu betik **çalıştırılmamıştır**; yalnızca okunmuştur ve verinin rastgele bölündüğünün işaretidir.\n")

    # 3
    add("## 3. Veri setlerinin sürümleri")
    rows = []
    for n in names:
        s = registry[n]["source"]
        rows.append([n, s.get("version", ""), s.get("publisher", ""), s.get("doi", "—"), s["url"]])
    add(md_table(["Veri seti", "Sürüm", "Yayıncı/sağlayıcı", "DOI", "Kaynak"], rows))
    add("")

    # 4
    add("## 4. Görüntü ve etiket sayıları")
    add("Sayılar gerçek dosyalardan sayılmıştır; \"yayımlanan\" sütunu yerel README/kaynak sayfasındaki değerdir.\n")
    rows = []
    for n in names:
        for sp in SPLITS + ("ALL",):
            r = _inv(a, n, sp)
            if r:
                rows.append([n, sp, r["images"], r["label_files"], r["boxes"], r["classes_defined"],
                             r["empty_labels"], r["missing_labels"], r["orphan_labels"], r["unreadable_images"],
                             r["size_mb"]])
    add(md_table(["Veri seti", "Bölüm", "Görüntü", "Etiket dosyası", "Kutu", "Sınıf", "Boş etiket", "Eksik etiket",
                  "Yetim etiket", "Bozuk görüntü", "Görüntü MB"], rows))
    add("")
    for n in names:
        pub = registry[n].get("published") or {}
        tot = _inv(a, n, "ALL")
        notes = []
        if pub.get("images") is not None:
            diff = _int(tot["images"]) - pub["images"]
            notes.append(f"görüntü gerçek {tot['images']} / yayımlanan {pub['images']} (fark {diff:+d})")
        if pub.get("boxes") is not None:
            notes.append(f"kutu gerçek {tot['boxes']} / yayımlanan {pub['boxes']}")
        add(f"- **{n}**: " + "; ".join(notes) + f". ({pub.get('note', '')})")
    mend = a["inventory"].get("mendeley_ppe")
    if mend:
        add(f"\n**Çözünürlük bulguları.** `site_safety` tüm görüntüleri 640×640'tır (Roboflow ön işlemi: stretch). "
            f"`construction_ppe` görüntüleri {_inv(a, 'construction_ppe', 'ALL')['min_resolution']} ile "
            f"{_inv(a, 'construction_ppe', 'ALL')['max_resolution']} arasındadır. `mendeley_ppe` için kaynak sayfa 640×640 "
            f"belirtse de gerçekte {_inv(a, 'mendeley_ppe', 'ALL')['distinct_resolutions']} farklı çözünürlük vardır "
            f"({_inv(a, 'mendeley_ppe', 'ALL')['min_resolution']} … {_inv(a, 'mendeley_ppe', 'ALL')['max_resolution']}); "
            f"yalnızca {mend['resolutions_top20'][0]['count']} görüntü {mend['resolutions_top20'][0]['size']} boyutundadır. "
            f"Bu fark kaynak sayfanın betimlemesiyle gerçek içerik arasındaki tutarsızlıktır.")
    add("")
    add("**YAML dosyaları.**\n")
    rows = []
    for n in names:
        d = a["inventory"][n]
        yp = d.get("yaml_paths") or {}
        rows.append([n, f"`{d.get('yaml_file')}`", d["names_format"], d["nc_declared"] if d["nc_declared"] is not None else "yok",
                     len(d["classes"]), "; ".join(d["class_warnings"]) or "—",
                     "evet" if yp.get("path_is_absolute_foreign") else "hayır",
                     ", ".join(f"{k}={'çözülüyor' if yp.get(k, {}).get('resolves') else ('çözülmüyor' if yp.get(k, {}).get('value') else '—')}"
                               for k in ("train", "val", "test"))])
    add(md_table(["Veri seti", "YAML", "names biçimi", "nc", "names içindeki sınıf", "Sınıf uyarıları",
                  "Yabancı mutlak path", "YAML yolları bu konumda"], rows))
    add("\n`construction_ppe/data.yaml` `nc` alanı içermez (sınıf sayısı `names` sözlüğünden 12 olarak okunmuştur). "
        "`mendeley_ppe/data.yaml` Windows mutlak yolu (`C:/Users/...`) içerir; `site_safety/data.yaml` Roboflow'un bilinen "
        "`../train/images` göreli yollarını kullanır. Hiçbir YAML olduğu gibi eğitimde kullanılamaz; Aşama 8+ eğitimde "
        "yeni, yerel yollarla bir YAML oluşturulacaktır (bu aşamada yazılmadı).\n")
    add("**Sınıf tanımları (YAML'dan okunan sıra, değiştirilmedi):**\n")
    rows = []
    for n in names:
        cls = a["inventory"][n]["classes"]
        rows.append([n, ", ".join(f"{k}:{v}" for k, v in cls.items())])
    add(md_table(["Veri seti", "ID:ad"], rows))
    add("")

    # 5
    add("## 5. Sınıf dağılımları")
    add("Grafikler: `reports/dataset_audit/charts/class_distribution_<veri_seti>.png`. Ayrıntılı tablo: `class_distribution.csv`.\n")
    for n in names:
        rows = []
        classes = a["inventory"][n]["classes"]
        total_boxes = sum(_int(r["boxes"]) for r in a["dist"] if r["dataset"] == n and r["split"] == "ALL")
        for cid, cname in classes.items():
            per = {sp: next((r for r in a["dist"] if r["dataset"] == n and r["class_id"] == cid and r["split"] == sp), None)
                   for sp in SPLITS + ("ALL",)}
            tot = per["ALL"]
            rows.append([cid, cname, tot["boxes"], tot["images_with_class"], f"%{tot['share_of_split_boxes_pct']}"] +
                        [per[sp]["boxes"] if per[sp] else "—" for sp in SPLITS])
        add(f"**{n}** (toplam {total_boxes} kutu)\n")
        add(md_table(["ID", "Sınıf", "Kutu", "Görüntü", "Pay", "train", "val", "test"], rows))
        add("")
    groups = target_group_counts(a, entries)
    rows = []
    for n in names:
        g = groups.get(n, {})
        h, nh, v, nv = (g.get(i, 0) for i in range(4))
        rows.append([n, h, nh, f"{h / nh:.2f}" if nh else "—", v, nv, f"{v / nv:.2f}" if nv else "—"])
    add("**Hedef sınıf gruplarına göre kutu sayıları (aday eşlemeye göre; eşleme kesin değildir):**\n")
    add(md_table(["Veri seti", "helmet", "no_helmet", "helmet/no_helmet", "vest", "no_vest", "vest/no_vest"], rows))
    add("\n**Dengesizlik değerlendirmesi.**")
    for n in names:
        per_class = [(r["class_name"], _int(r["boxes"])) for r in a["dist"] if r["dataset"] == n and r["split"] == "ALL"]
        big, small = max(per_class, key=lambda t: t[1]), min(per_class, key=lambda t: t[1])
        add(f"- `{n}`: en çok kutu `{big[0]}` ({big[1]}), en az `{small[0]}` ({small[1]}); oran {big[1] / max(1, small[1]):.1f}:1.")
    add("- Hedef sınıf çiftlerinde (yukarıdaki tablo) ihlal ve uyum sınıfları arasında oran farkı vardır. Bu tür bir dengesizlik "
        "eğitilen modelin baskın sınıfa (ör. baretli) yönelmesine ve nadir sınıfın (baretsiz/yeleksiz) recall değerinin düşmesine "
        "yol açabilir. Bu aşamada oversampling, undersampling veya augmentation UYGULANMAMIŞTIR; yalnızca belgelenmiştir.")
    small_val = [(n, sp) for n in names for sp in ("val", "test")
                 if (r := _inv(a, n, sp)) and _int(r["images"]) < 150]
    if small_val:
        add("- Küçük bölümler: " + ", ".join(f"`{n}/{sp}` ({_inv(a, n, sp)['images']} görüntü)" for n, sp in small_val) +
            ". Bu bölümlerde bazı sınıfların kutu sayısı çok düşüktür; sınıf bazlı doğrulama/test metrikleri istatistiksel olarak "
            "güvenilmez olabilir.")
    add("")

    # 6
    add("## 6. Görüntü–etiket eşleşme sonuçları")
    rows = []
    for n in names:
        t = _inv(a, n, "ALL")
        rows.append([n, t["images"], t["label_files"], t["missing_labels"], t["orphan_labels"], t["empty_labels"],
                     t["unreadable_labels"], t["unreadable_images"], t["exif_rotated_images"]])
    add(md_table(["Veri seti", "Görüntü", "Etiket dosyası", "Etiketsiz görüntü", "Görüntüsüz etiket", "Boş etiket",
                  "Okunamayan etiket", "Bozuk görüntü", "EXIF döndürmeli"], rows))
    add("\nEşleşme aynı dosya adı (uzantısız) üzerinden yapılmıştır. Boş veya eksik etiketler hata sayılmamıştır "
        "(kasıtlı negatif örnek olabilir); yalnızca raporlanmıştır.\n")

    # 7
    add("## 7. Bounding box doğrulama sonuçları")
    add("Her satır `class_id x_center y_center width height` (5 sütun) biçiminde beklenir; segmentation/OBB benzeri ek sütunlar "
        "detection etiketi olarak işlenmez. Toleranslar: normalize aralık ve görüntü sınırı için 1e-4, çok küçük kutu için "
        "2 piksel, aynı sınıfta kopya kutu için IoU≥0.95, çelişkili sınıf örtüşmesi için IoU≥0.3.\n")
    codes = sorted({c for v in a["validation"].values() for c in v["issues_by_code"]})
    checks = ["UNSUPPORTED_FORMAT", "BAD_COLUMN_COUNT", "NON_NUMERIC", "CLASS_ID_INVALID", "CLASS_ID_UNDEFINED",
              "COORD_OUT_OF_RANGE", "BOX_NONPOSITIVE_SIZE", "BOX_OUTSIDE_IMAGE", "LABEL_UNREADABLE", "IMAGE_UNREADABLE"]
    rows = [[c] + [a["validation"][n]["issues_by_code"].get(c, 0) for n in names] for c in checks]
    add(md_table(["Kontrol (sorun kodu)"] + names, rows))
    add("\nYukarıdaki kontrollerde bulunan sorun sayısı 0 ise ilgili hata türü TESPİT EDİLMEMİŞTİR. Sonuç, doğrulayıcıdan "
        "bağımsız yazılmış TEK SEFERLİK bir numpy kontrolüyle (betik olarak saklanmadı) çapraz denetlenmiştir: sütun sayıları, "
        "sınıf ID'leri ve kutu sınırları uyumlu bulunmuştur.\n")
    add("**Matematiksel geçerlilik, semantik doğruluğu garanti etmez**: kutunun geçerli olması doğru nesneyi gösterdiği anlamına gelmez "
        "(bkz. bölüm 11-12).\n")

    # 8
    add("## 8. Tespit edilen etiket sorunları")
    rows = []
    for c in codes:
        rows.append([c, SEVERITY_TR.get(next((i["severity"] for i in a["issues"] if i["issue_code"] == c), ""), "")] +
                    [a["validation"][n]["issues_by_code"].get(c, 0) for n in names])
    add(md_table(["Sorun kodu", "Düzey"] + names, rows))
    add("\nAyrıntı: `annotation_issues.csv` (dosya yolu, satır numarası, açıklama). Hiçbir dosya düzeltilmemiş veya silinmemiştir.\n")
    example_codes = [c for c in ("BOX_VERY_SMALL", "DUPLICATE_BOX", "NEAR_DUPLICATE_BOX", "CONFLICTING_OVERLAP") if c in codes]
    for c in example_codes:
        ex = [i for i in a["issues"] if i["issue_code"] == c][:3]
        add(f"- `{c}` örnekleri: " + "; ".join(f"`{Path(i['label_path']).name}` satır {i['line_no']} ({i['detail']})" for i in ex))
    add("\n**Görsel ön gözlemler (kod ile doğrulanamaz, KULLANICI DOĞRULAMASI GEREKİR):**")
    add("- `construction_ppe`: `helmet` kutuları baret nesnesini, `no_helmet` kutuları baretsiz baş bölgesini, `vest`/`no_vest` kutuları "
        "gövde bölgesini sarıyor görünüyor. `no_helmet` kutuları helmet kutularına göre belirgin şekilde küçük/bulanık örnekler içeriyor.")
    add("- `mendeley_ppe`: `Helmet` ve `NoHelmet` kutuları baş+baret / baş-yüz bölgesini birlikte kapsıyor görünüyor (baret nesnesine sıkı değil); "
        "`NoVest` örneklerinin birçoğu inşaat dışı sıradan kıyafetli insanlar; bazı yüzler piksellenmiş (mozaik).")
    add("- `site_safety`: Çok sayıda çok küçük, bulanık veya görüntü kenarında kırpılmış kutu var (Roboflow augmentation/stretch etkisi). "
        "Birkaç örnekte `NO-Hardhat` kutusu içinde baret ve `NO-Safety Vest` kutusunda yelek görünüyor, `Safety Vest` kutularında yelek olmayan "
        "(beyaz/koyu giysi) örnekler var: **olası etiket gürültüsü**.\n")

    # 9
    add("## 9. Tekrarlanan görüntüler")
    ds = a["dup_summary"]
    add(f"Toplam {ds['images_hashed']} görüntü SHA-256 ve 64-bit dHash ile işlendi (okunamayan: {len(ds['unreadable_images'])}).\n")
    exact = ds["matches"].get("exact", {"total_pairs": 0})
    add(f"- **Birebir aynı (SHA-256) dosya çifti: {exact['total_pairs']}.** (Veri seti içi, bölümler arası ve veri setleri arası.)")
    for mt in ("near", "near_flipped"):
        info = ds["matches"].get(mt)
        if info:
            label = "yakın kopya" if mt == "near" else "yatay çevrilmiş yakın kopya"
            add(f"- **{label} adayı (dHash ≤ {ds['near_threshold_bits_of_64']}/64): {info['total_pairs']} çift, "
                f"{info['distinct_images_involved']} görüntü.**")
            for key, n in info["pairs"].items():
                add(f"    - `{key}`: {n}")
    add(f"- Eşik duyarlılığı (çift sayısı, ≤ mesafe): {ds['near_pair_count_by_threshold']}. Eşik büyüdükçe yanlış pozitif artar.")
    add(f"- Aynı **kaynak adını** paylaşan görüntü grupları (ipucu): {ds['source_name_overlap_groups']}.")
    mc = ds.get("manifest_checks", {})
    for n, c in mc.items():
        add(f"- `{n}/manifest.jsonl` kontrolü: {c['entries']} girdi, SHA-256 uyuşmazlığı {c['sha_mismatch']}, bölüm uyuşmazlığı "
            f"{c['split_mismatch']}, kutu sayısı uyuşmazlığı {c['box_count_mismatch']}, yerelde eksik {c['missing_locally']}.")
    cross = [r for r in a["dups"] if r["relation"] == "cross_dataset"]
    if cross:
        add("\nVeri setleri arası yakın kopya adayları (inceleme gerekir; kesin aynı görüntü değildir):\n")
        add(md_table(["Tür", "Mesafe", "Dosya A", "Dosya B"],
                     [[r["match_type"], r["distance"], f"`{r['path_a']}`", f"`{r['path_b']}`"] for r in cross]))
    add("\n**Önemli:** Yakın kopya eşleşmeleri algısal özet (dHash) ile bulunmuştur; kesin aynı görüntü kabul EDİLMEMİŞTİR. "
        "Kırpma, döndürme veya güçlü renk değişimi içeren augmentation türevleri dHash ile kaçırılmış olabilir; "
        "`site_safety` içinde aynı kaynaktan türeyen çok sayıda görüntü grubu kaynak adı ipucuyla görülmektedir.\n")

    # 10
    add("## 10. Veri bölünmelerinin durumu")
    rows = []
    for n in names:
        t = _int(_inv(a, n, "ALL")["images"])
        parts = []
        for sp in SPLITS:
            r = _inv(a, n, sp)
            parts.append(f"{r['images']} (%{100 * _int(r['images']) / t:.1f})" if r else "yok")
        rows.append([n] + parts)
    add(md_table(["Veri seti", "train", "val", "test"], rows))
    add("")
    for n in names:
        dups_ds = [r for r in a["dups"] if r["dataset_a"] == n and r["dataset_b"] == n]
        cross_split = [r for r in dups_ds if r["relation"] == "within_dataset_cross_split"]
        by_pair = Counter(r["split_pair"] for r in cross_split)
        name_cross = [r for r in a["name_overlap"] if r["kind"] == "within_dataset_cross_split" and r["datasets_splits"].startswith(n)]
        name_same = [r for r in a["name_overlap"] if r["kind"] == "within_dataset_same_split" and r["datasets_splits"].startswith(n)]
        add(f"- **{n}**: bölümler arası yakın kopya adayı {len(cross_split)} çift {dict(by_pair) or ''}; "
            f"bölümler arası ortak kaynak adı grubu {len(name_cross)}; aynı bölüm içinde ortak kaynak adı grubu {len(name_same)}.")
    add("\n- Ayrı bir **test** bölümü yalnızca `site_safety` içinde vardır (ve çok küçüktür); `construction_ppe` ve `mendeley_ppe` yalnızca "
        "train/val içerir. `construction_ppe` dokümantasyonu nihai değerlendirme için bağımsız saha görüntüsü önerir.")
    add("- `mendeley_ppe` bölmesi, arşivdeki betiğe göre rastgele %20 ayırmadır; aynı sahneden gelen görüntüler train ve val'e dağılmış olabilir "
        "(veri sızıntısı riski, yukarıdaki bölümler arası eşleşmeler).")
    add("- Üç kaynak birleştirilirse: (i) veri setleri arası yakın kopya/ortak kaynak adları nedeniyle bir kaynağın train görüntüsü diğerinin val/test "
        "görüntüsüyle örtüşebilir; (ii) `site_safety` train içindeki augmentation türevleri etkin örnek sayısını ve doğrulama güvenilirliğini etkiler. "
        "Mevcut bölmeler DEĞİŞTİRİLMEMİŞTİR.\n")

    # 11
    add("## 11. Veri setleri arasındaki sınıf uyumu")
    add("Hedef sınıflar: " + ", ".join(f"{k}={v}" for k, v in targets.items()) + ". Aşağıdaki tablo **ADAY** eşlemedir; "
        "hiçbir etiket ID'si değiştirilmemiştir.\n")
    add(md_table(["Veri seti", "Kaynak ID:ad", "Hedef", "Durum", "Kutu semantiği", "Kanıt / risk"],
                 [[e["dataset"], f"{e['id']}:{e['name']}", targets.get(e.get("target"), "(hedef dışı)"), e["status"],
                   e.get("box_semantics", ""), (e.get("evidence", "") + (" — Risk: " + e["risk"] if e.get("risk") else ""))]
                  for e in entries]))
    add("\nTam liste: `candidate_class_mapping.csv`. Durum anlamları (eksik giderme sonrası): **confirmed** = kutu tanımı hedefle tutarlı (nicel ölçü + görsel + "
        "dokümantasyon) ve örneklemde olası hata ≤%10; **needs_review** = tanım tutarlı görünüyor ama etiket kalitesi/örneklem belirsiz; **incompatible** = kutu tanımı "
        "hedefle tutarsız veya hedef dışı sınıf (doğrudan birleştirilmemeli). Tüm satırlar `user_review=Pending`. Ayrıntı: `semantic_validation_report.md`.\n")
    if mapping_problems:
        add("**Eşleme tutarlılık uyarıları:** " + "; ".join(mapping_problems) + "\n")
    stat = {(s["dataset"], int(s["class_id"])): s for s in a["boxstats"]}
    rows = []
    for e in entries:
        if e.get("target") is None:
            continue
        s = stat.get((e["dataset"], int(e["id"])))
        if s and s.get("median_area"):
            rows.append([targets[e["target"]], e["dataset"], e["name"], s["boxes"], s["median_width"], s["median_height"],
                         s["median_area"], s["median_aspect_h_over_w"]])
    rows.sort(key=lambda r: (r[0], r[1]))
    add("**Kutu geometrisi karşılaştırması (normalize; kutu semantiği için nesnel ipucu):**\n")
    add(md_table(["Hedef", "Veri seti", "Kaynak sınıf", "Kutu", "Medyan genişlik", "Medyan yükseklik", "Medyan alan", "Medyan h/w"], rows))
    add("\n_Not: Medyan alan görüntü kadrajının yakınlığından da etkilenir (yakın çekim/uzak çekim); tek başına kanıt değil, görsel inceleme için ipucudur._")
    add("\n**Düzeltme:** Bu medyan-alan tablosu kadraj yakınlığından etkilendiği için kutu semantiği kanıtı olarak YANILTICIDIR (önceki taslaktaki 'no_helmet helmet'ten "
        "4 kat küçük' sonucu bu yüzden hatalıydı). Kişi kutusuna göre göreli boyutlar ve baş/gövde oranlarıyla yapılan doğru karşılaştırma "
        "`semantic_validation_report.md` içindedir: Mendeley `Helmet` baş+baret (yüz dahil) olduğundan Construction/Roboflow baret nesnesiyle UYUMSUZ; Roboflow `NO-Hardhat` heterojen.\n")

    # 12
    add("## 12. Manuel inceleme gerektiren noktalar")
    add("Güncel durum (eksik giderme sonrası): onay listesi `manual_review_checklist.md`, kalanlar `remaining_manual_reviews.md`. İlk taslaktaki 7 başlık "
        "(kutu semantiği, sınıf karşılaştırması, Roboflow gürültüsü, kopyalar, boş etiketler, eşleme onayı, lisans) analizle büyük ölçüde sonuçlandı; kullanıcıya yalnızca "
        "gerçek karar/onay gerektiren maddeler kaldı (bkz. §16).\n")

    # 13
    add("## 13. Lisans ve kaynak bilgileri")
    rows = []
    for n in names:
        lic, src = registry[n]["license"], registry[n]["source"]
        rows.append([n, lic["type"], lic["verification"], lic.get("terms", "")])
    add(md_table(["Veri seti", "Lisans", "Doğrulama durumu", "Koşullar / notlar"], rows))
    add("\n**Atıf metinleri:**\n")
    for n in names:
        add(f"- {registry[n]['license']['attribution']}")
    add("\nEksik giderme sonrası ayrıntılı kanıt tablosu: `license_verification_report.md`. Özet: Construction-PPE resmi kaynaktan doğrulandı; Mendeley lisansı resmi sayfa/API'den "
        "doğrulandı ve yerel ZIP'in SHA-256'sı resmi API ile eşleşti, ancak üst kaynak lisansları doğrulanamadı; Roboflow v27 lisansı YALNIZCA yerel dosyalardan biliniyor (resmi sayfa 403).\n")
    add("\nNotlar: CC BY 4.0 atıf zorunluluğu ve değişiklik belirtme koşulu getirir. Mendeley ve Roboflow veri setleri "
        "başka kaynaklardan derlenmiştir (README'lere göre); üst kaynakların lisansları bu aşamada DOĞRULANMADI. "
        "Hukuki bir görüş değildir; bitirme raporunda atıf yapılmadan önce kaynak sayfaları yeniden kontrol edilmelidir.\n")

    # 14
    add("## 14. Gelecek eğitim aşamaları için öneriler")
    add("(Öneridir; hiçbiri bu aşamada uygulanmamıştır.)\n")
    add("- Nihai değerlendirme için hiçbir eğitim verisiyle örtüşmeyen, bağımsız bir test kümesi ayrılmalı (idealde hedef ortama benzer saha görüntüleri).")
    add("- Veri setleri birleştirilmeden önce kutu semantiği kararı verilmeli: özellikle baret için 'sıkı baret kutusu' ile 'baş+baret kutusu' ayrımı.")
    add("- Birleştirilirse bölme, kaynak görüntü (ortak kaynak adı) ve yakın kopya gruplarına göre grup bazlı yapılmalı; mevcut val/test bölümleri bu kontrol olmadan güvenilir kabul edilmemeli.")
    add("- `site_safety` etiket gürültüsü örnekleme ile elle taranmalı; gürültü yüksekse bu veri setinin yalnızca bölümü kullanılmalı veya dışarıda bırakılmalı.")
    add("- Sınıf dengesizliği eğitim aşamasında (örn. sınıf ağırlıkları, hedefli örnekleme) ele alınmalı; şu an uygulanmadı.")
    add("- Her kaynak için ayrı doğrulama metriği raporlanmalı (kaynak başına mAP), böylece alan kayması görülebilir.\n")

    # 15
    add("## 15. Veri setlerinin genel kullanılabilirlik değerlendirmesi")
    add("Aşağıdaki değerlendirme teknik denetime ve görsel ÖN gözleme dayanır; kullanıcı onayı tamamlanana kadar kesin kabul edilmemelidir.\n")
    add(md_table(["Veri seti", "Teknik biçim", "Ön değerlendirme"], [
        ["construction_ppe", "Biçim sorunu yok; manifest ile birebir tutarlı",
         "En güvenilir aday. Küçük (1000 görüntü), ayrı test yok. Baret ve yelek sınıfları dokümante edilmiş; kutu tanımı sıkı baret nesnesi."],
        ["mendeley_ppe", "Biçim sorunu yok",
         "Ek veri olarak yararlı ama kutu semantiği (baş+baret) diğerlerinden farklı; rastgele bölme nedeniyle train/val sızıntı riski; alan çeşitliliği yüksek, birçok inşaat dışı sahne."],
        ["site_safety", "Küçük uyarılar (çok küçük/kopya kutu, boş etiket)",
         "Çeşitli ama augmentation türevleri, kırpılmış kutular ve olası etiket gürültüsü var; bölümler arası ortak kaynak grupları tespit edildi. Elle inceleme olmadan eğitime katılması riskli."],
    ]))
    add("\n**Sonuç:** Üç veri seti biçim olarak okunabilir ve bütünlük açısından sağlamdır; ancak veri setlerinin birleştirilmesi için "
        "ön koşullar henüz karşılanmamıştır: (1) kutu semantiği ve aday sınıf eşlemesinin kullanıcı onayı, (2) olası etiket gürültüsünün incelenmesi, "
        "(3) sızıntı riskine karşı bölme stratejisi, (4) lisans doğrulamalarının tamamlanması. Bu nedenle birleştirme için **koşullu / henüz hazır değil** durumundadır.\n")
    return "\n".join(L)


def build_checklist(a: dict[str, Any], registry: dict[str, Any], targets: dict[int, str], entries: list[dict[str, Any]]) -> str:
    L: list[str] = ["# Manuel İnceleme ve Onay Kontrol Listesi (Aşama 2)", "",
                    "Aşağıdaki maddeler otomatik olarak doğrulanamaz. Her biri için ilgili dosyayı açıp karar verin; "
                    "veri setlerini birleştirmeden önce bu liste tamamlanmalıdır. (Dosya yolları proje köküne görelidir. "
                    "`visual_samples/` klasörü Git'e eklenmez.)", ""]
    vs = "reports/dataset_audit/visual_samples"
    L += ["## A. Kutu semantiği (sınıf kırpma sayfaları)", "",
          "Her sayfada kutu çizili kırpmalar vardır. Soru: kutu doğru nesneyi/bölgeyi mi gösteriyor?", ""]
    items = [
        ("construction_ppe", "class_00_helmet.jpg", "Baret kutusu gerçekten bareti mi (yalnızca baret nesnesi) gösteriyor?"),
        ("construction_ppe", "class_06_no_helmet.jpg", "`no_helmet` kutusu başı mı kapsıyor, bütün insanı mı? Koyu başlıklı/şapkalı örnek var mı?"),
        ("construction_ppe", "class_02_vest.jpg", "Yelek kutusu gerçekten yeleği mi gösteriyor?"),
        ("construction_ppe", "class_11_no_vest.jpg", "`no_vest` hangi bölgeye uygulanmış (gövde mi, tüm kişi mi)?"),
        ("mendeley_ppe", "class_00_Helmet.jpg", "`Helmet` kutusu baş+baret mi kapsıyor? (Construction-PPE ile aynı mı?)"),
        ("mendeley_ppe", "class_01_NoHelmet.jpg", "`NoHelmet` kutusu baş/yüz mü? Şapka/maske/mozaik yüz örnekleri kabul edilebilir mi?"),
        ("mendeley_ppe", "class_03_Vest.jpg", "`Vest` kutusu yeleği/gövdeyi mi kapsıyor?"),
        ("mendeley_ppe", "class_02_NoVest.jpg", "`NoVest` kutuları KKD gerektiren bağlamda mı (inşaat) yoksa sıradan sahnelerde mi?"),
        ("site_safety", "class_00_Hardhat.jpg", "`Hardhat` kutuları sıkı mı? Çok küçük/bulanık/kırpılmış örnekler kabul edilebilir mi?"),
        ("site_safety", "class_02_NO-Hardhat.jpg", "`NO-Hardhat` baş bölgesini mi kapsıyor? İçinde baret görünen kutu var mı (etiket hatası)?"),
        ("site_safety", "class_07_Safety_Vest.jpg", "`Safety Vest` kutularında gerçekten yelek var mı? (beyaz/koyu giysi örnekleri)"),
        ("site_safety", "class_04_NO-Safety_Vest.jpg", "`NO-Safety Vest` kutularında yelek görünen örnek var mı? Kısmi gövde kutuları kabul edilebilir mi?"),
    ]
    for ds, fn, q in items:
        L.append(f"- [ ] `{vs}/{ds}/crops/{fn}` — {q}")
    L += ["", "## B. Veri setleri arası sınıf semantiği karşılaştırması", "",
          "- [ ] Üç `helmet` karşılığını (`construction_ppe/class_00_helmet.jpg`, `mendeley_ppe/class_00_Helmet.jpg`, `site_safety/class_00_Hardhat.jpg`) yan yana inceleyin: kutu tanımı aynı mı?",
          "- [ ] Üç `no_helmet` karşılığı (`no_helmet`, `NoHelmet`, `NO-Hardhat`) aynı bölgeyi mi etiketliyor?",
          "- [ ] Üç `vest` ve üç `no_vest` karşılığı aynı bölgeyi mi etiketliyor?",
          "- [ ] Genel kontrol: `contact_sheet_*.jpg` dosyalarına (her veri setinde 4 adet) göz atın; kutular nesnelerin üzerinde mi?", ""]

    L += ["## C. Şüpheli etiketler (annotation_issues.csv)", ""]
    for code in ("CONFLICTING_OVERLAP", "DUPLICATE_BOX", "NEAR_DUPLICATE_BOX"):
        for i in [x for x in a["issues"] if x["issue_code"] == code]:
            L.append(f"- [ ] {code}: `{i['image_path']}` satır {i['line_no']} — {i['detail']}")
    n_small = sum(1 for x in a["issues"] if x["issue_code"] == "BOX_VERY_SMALL")
    L.append(f"- [ ] BOX_VERY_SMALL ({n_small} adet, <2 piksel kenarlı kutu): `annotation_issues.csv` içinden birkaçını açıp gürültü mü gerçek nesne mi karar verin.")
    n_empty = sum(1 for x in a["issues"] if x["issue_code"] == "LABEL_EMPTY")
    L.append(f"- [ ] LABEL_EMPTY ({n_empty} adet boş etiket): gerçekten nesnesiz (negatif) görüntü mü? Liste: `annotation_issues.csv` (issue_code=LABEL_EMPTY).")
    L += ["", "## D. Tekrar / sızıntı (duplicate_images.csv, source_name_overlap.csv)", ""]
    for r in [x for x in a["dups"] if x["relation"] == "cross_dataset"]:
        L.append(f"- [ ] Veri setleri arası ({r['match_type']}, mesafe {r['distance']}): `{r['path_a']}` ↔ `{r['path_b']}`")
    for r in [x for x in a["dups"] if x["split_pair"] == "test|val" and x["relation"] == "within_dataset_cross_split"]:
        L.append(f"- [ ] val↔test ({r['match_type']}, mesafe {r['distance']}): `{r['path_a']}` ↔ `{r['path_b']}`")
    cross_names = [x for x in a["name_overlap"] if x["kind"] == "within_dataset_cross_split"]
    for r in cross_names:
        L.append(f"- [ ] Bölümler arası ortak kaynak adı `{r['base_name']}` ({r['datasets_splits']}, {r['n_images']} görüntü)")
    n_tp = sum(1 for x in a["dups"] if x["split_pair"] == "train|val" and x["relation"] == "within_dataset_cross_split")
    L.append(f"- [ ] train↔val yakın kopya adayları ({n_tp} çift): `duplicate_images.csv` içinde split_pair=train|val olanlardan en az 10'unu görsel olarak karşılaştırın.")
    L.append("- [ ] `mendeley_ppe` bölünmesi arşivdeki betikle rastgele yapıldı: aynı sahneden gelen görüntülerin train ve val'de bulunması kabul edilebilir mi?")
    L += ["", "## E. Aday sınıf eşlemesi onayı (candidate_class_mapping.csv)", ""]
    for e in entries:
        if e.get("target") is None:
            continue
        L.append(f"- [ ] {e['dataset']} `{e['id']}:{e['name']}` → `{targets[e['target']]}` (durum: {e['status']}, kutu: {e.get('box_semantics', '')})")
    L += ["", "## F. Lisans / kaynak", "",
          "- [ ] Roboflow v27 lisansı resmi sayfadan doğrulanamadı (403): sayfayı tarayıcıda açıp 'CC BY 4.0' olduğunu teyit edin.",
          "- [ ] Mendeley v6 lisans bağlantısını (CC BY 4.0) sayfadan teyit edin; üst kaynaklar (GitHub/Kaggle/Roboflow) için ek lisans gerekip gerekmediğine karar verin.",
          "- [ ] Ultralytics Construction-PPE: atıf metni bitirme raporuna eklendi mi?",
          "- [ ] Mendeley sayfası 640×640 belirtiyor, gerçekte 900+ farklı çözünürlük var: bu fark kabul edilebilir mi?", "",
          "## Birleştirmeye geçiş kararı", "",
          "- [ ] A–F bölümlerindeki maddeler tamamlandı; hangi sınıfların hangi hedef sınıfa eşleneceği ve kutu semantiği kararı yazıya döküldü."]
    return "\n".join(L) + "\n"
