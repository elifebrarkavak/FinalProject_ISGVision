"""Veri kullanım politikasını (configs/dataset_usage_policy.yaml) doğrular ve görüntü seviyesinde karar üretir.

Bu modül veriyi DEĞİŞTİRMEZ: yalnızca okur ve karar tablosu üretir. Karar ilkeleri:
- Hakları/lisansı doğrulanmamış kaynak asla 'approved' olmaz (kaynak düzeyinde pending_license_verification).
- Etiketsiz hedef nesne bırakmamak için başarısız kutu silinmez; görüntü quarantine / needs_reannotation olur.
- Belirsiz etiket sessizce kabul edilmez.
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any, Iterator

import yaml

from isg_vision.dataset_audit.imageio import read_image_info
from isg_vision.dataset_audit.labels import parse_label_file
from isg_vision.dataset_audit.layout import DatasetLayout, SPLIT_ORDER, iter_samples
from isg_vision.dataset_audit.semantic import HEAD_ROLES, TORSO_ROLES, RoleBox, assign_to_person, class_roles
from isg_vision.utils.paths import PROJECT_ROOT

POLICY_FILE = PROJECT_ROOT / "configs" / "dataset_usage_policy.yaml"
STATUSES = ("approved", "quarantine", "needs_reannotation", "pending_license_verification")
SOURCE_USAGES = ("approved_scope", "pending_license_verification")
LICENSE_STATUSES_APPROVABLE = ("verified_official",)
RIGHTS_STATUSES_APPROVABLE = ("verified_publisher_original",)
# Aşama 2C: beklemedeki (onaylı olmayan) kaynakların bekleme gerekçeleri
HOLD_REASONS = ("license_unverified", "rights_unverified", "annotation_quality", "label_noise", "split_leakage",
                "class_incompatibility")
# Onaylı kaynak için bölme yöntemi: yayıncı bölünmesi korunur; bölümler arası kaynak birimi çoğunluk bölümüne taşınır
SPLIT_BASES = ("original_publisher_split_with_unit_fix",)
# Görüntüyü kullanım dışı bırakmayan, yalnızca işaretleyen kural durumu
FLAG_STATUS = "annotation_risk_flag"


def load_policy(path: Path = POLICY_FILE) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"Politika dosyası bulunamadı: {path}")
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def validate_policy(policy: dict[str, Any], registry: dict[str, Any], mapping_entries: list[dict[str, Any]],
                    classes_by_dataset: dict[str, dict[int, str]]) -> list[str]:
    """Politikanın iç tutarlılığını ve eşleme/lisans kayıtlarıyla uyumunu denetler. Sorun listesi döner (boşsa geçerli)."""
    problems: list[str] = []
    targets = {int(k): v for k, v in policy["target_classes"].items()}
    if set(policy["sources"]) != set(registry):
        problems.append(f"Politika kaynakları kayıt defteriyle uyuşmuyor: {sorted(policy['sources'])} / {sorted(registry)}")
    mapped = {(e["dataset"], int(e["id"])): e for e in mapping_entries}
    for ds, src in policy["sources"].items():
        if src["source_usage"] not in SOURCE_USAGES:
            problems.append(f"{ds}: geçersiz source_usage {src['source_usage']!r}")
        if src["source_usage"] == "approved_scope" and src["license_status"] not in LICENSE_STATUSES_APPROVABLE:
            problems.append(f"{ds}: lisansı '{src['license_status']}' olan kaynak onaylı kapsamda olamaz")
        if src["source_usage"] == "approved_scope" and src.get("rights_status", "") not in RIGHTS_STATUSES_APPROVABLE:
            problems.append(f"{ds}: hakları doğrulanmamış kaynak onaylı kapsamda olamaz")
        if src["source_usage"] == "approved_scope":
            if "conditional_class_map" in src or "hold_reasons" in src:
                problems.append(f"{ds}: onaylı kaynakta koşullu eşleme/bekleme gerekçesi olamaz")
            sp = src.get("split_policy") or {}
            if sp.get("basis") not in SPLIT_BASES:
                problems.append(f"{ds}: onaylı kaynak için geçerli split_policy.basis yok")
            uvr = src.get("user_visual_review")
            if uvr is not None and uvr.get("rule") not in policy["rules"]:
                problems.append(f"{ds}: user_visual_review.rule tanımsız: {uvr.get('rule')!r}")
            for img, ov in (src.get("image_overrides") or {}).items():
                if ov.get("status") not in ("quarantine", "needs_reannotation"):
                    problems.append(f"{ds}: {img} geçersiz override durumu {ov.get('status')!r}")
                if ov.get("rule") not in policy["rules"]:
                    problems.append(f"{ds}: {img} tanımsız kural {ov.get('rule')!r}")
                if not str(img).startswith(f"data/raw/{ds}/"):
                    problems.append(f"{ds}: override başka kaynağın görüntüsünü gösteriyor: {img}")
        else:
            reasons = src.get("hold_reasons") or []
            if not reasons:
                problems.append(f"{ds}: beklemedeki kaynak için hold_reasons zorunlu")
            for r in reasons:
                if r not in HOLD_REASONS:
                    problems.append(f"{ds}: geçersiz hold_reason {r!r}")
            if src.get("class_map"):
                problems.append(f"{ds}: beklemedeki kaynakta koşulsuz class_map olamaz (yalnızca conditional_class_map)")
            if src.get("image_overrides"):
                problems.append(f"{ds}: image_overrides yalnızca onaylı kaynakta kullanılabilir")
            if src.get("raw_data_retention") != "keep":
                problems.append(f"{ds}: beklemedeki kaynağın ham verisi ve analiz kayıtları korunmalı (raw_data_retention: keep)")
        cmap = {int(k): int(v) for k, v in (src.get("class_map") or src.get("conditional_class_map") or {}).items()}
        excluded = {int(k) for k in (src.get("excluded_classes") or {})}
        ignored = {int(k) for k in (src.get("ignored_classes") or [])}
        if cmap.keys() & excluded:
            problems.append(f"{ds}: hem eşlenen hem dışlanan sınıf: {sorted(cmap.keys() & excluded)}")
        if cmap.keys() & ignored or excluded & ignored:
            problems.append(f"{ds}: çakışan sınıf kümeleri (eşlenen/dışlanan/yok sayılan)")
        covered = set(cmap) | excluded | ignored
        missing = set(classes_by_dataset.get(ds, {})) - covered
        if missing:
            problems.append(f"{ds}: politikada karşılığı olmayan sınıflar: {sorted(missing)}")
        for cid, tgt in cmap.items():
            if tgt not in targets:
                problems.append(f"{ds}: sınıf {cid} geçersiz hedefe eşlenmiş: {tgt}")
            e = mapped.get((ds, cid))
            if e is None:
                problems.append(f"{ds}: sınıf {cid} eşleme dosyasında yok")
            else:
                if e.get("target") != tgt:
                    problems.append(f"{ds}: sınıf {cid} hedefi eşleme dosyasıyla uyuşmuyor ({e.get('target')} != {tgt})")
                if e["status"] != "confirmed":
                    problems.append(f"{ds}: sınıf {cid} eşleme durumu '{e['status']}' (yalnızca confirmed kullanılabilir)")
        for cid in excluded:
            e = mapped.get((ds, cid))
            if e is not None and e["status"] == "confirmed":
                problems.append(f"{ds}: confirmed sınıf {cid} dışlanmış (tutarsız)")
        for rid in src.get("conditional_rules", []):
            if rid not in policy["rules"]:
                problems.append(f"{ds}: tanımsız kural {rid}")
    for rid, rule in policy["rules"].items():
        if rule["status"] not in STATUSES + ("derivation_step", FLAG_STATUS, "per_user_decision"):
            problems.append(f"kural {rid}: geçersiz durum {rule['status']!r}")
    ratios = policy["leakage"]["ratios"]
    if abs(sum(ratios.values()) - 1.0) > 1e-6:
        problems.append("bölme oranları 1'e toplanmıyor")
    if not policy["leakage"].get("never_split_a_unit"):
        problems.append("leakage.never_split_a_unit true olmalı")
    return problems


def iter_image_records(layout: DatasetLayout, registry_entry: dict[str, Any]) -> Iterator[dict[str, Any]]:
    """Her görüntü için kutular (sınıf, satır, piksel yüksekliği), rol kutuları ve kişi öznitelik eksikliği."""
    roles = class_roles(registry_entry)
    for split in SPLIT_ORDER:
        sd = layout.splits.get(split)
        if sd is None:
            continue
        for s in iter_samples(sd):
            if s.image_path is None:
                continue
            info = read_image_info(s.image_path, full_decode=False)
            res = parse_label_file(s.label_path) if s.label_path else None
            boxes = res.boxes if res else []
            rb = [RoleBox(roles[b.class_id], b) for b in boxes if b.class_id in roles and b.w > 0 and b.h > 0]
            issues = person_attribute_issues(rb, info.height) if info.readable else []
            yield {"image_path": s.image_path, "split": split, "height": info.height if info.readable else 0,
                   "readable": info.readable, "boxes": boxes, "n_lines": len(boxes),
                   "incomplete_persons": len(issues), "person_issues": issues, "has_label": s.label_path is not None}


def _inside_ratio(b: RoleBox, region: tuple[float, float, float, float]) -> float:
    ix = max(0.0, min(b.x2, region[2]) - max(b.x1, region[0]))
    iy = max(0.0, min(b.y2, region[3]) - max(b.y1, region[1]))
    return ix * iy / max(b.area, 1e-12)


def person_attribute_issues(rb: list[RoleBox], img_h: int, min_person_px: int = 60, edge: float = 0.01) -> list[list[str]]:
    """Yüksekliği >= min_person_px olan ve baş (helmet/no_helmet) veya gövde (vest/no_vest) etiketi atanamayan her kişi için
    olası açıklama kodları. Kodlar yalnızca SEZGİSELDİR (eksik etiket kanıtı değil, inceleme önceliği):
      *_label_present_unassigned : kişinin (yukarı genişletilmiş) kutusunda ilgili etiket var ama atama başka kişiye/hiçbirine yaptı
      head_out_of_frame_top      : kişi kutusu görüntünün üst kenarına değiyor (baş kadraj dışı olabilir)
      person_truncated_at_edge   : kişi kutusu sol/sağ kenara değiyor (kısmi görünür)
      torso_small_person         : kişi <120 px; gövde özniteliği güvenilir etiketlenemeyebilir
      *_unexplained              : yukarıdakilerin hiçbiri değil -> öncelikli insan incelemesi
    """
    return [codes for _, codes in iter_person_issues(rb, img_h, min_person_px, edge)]


def iter_person_issues(rb: list[RoleBox], img_h: int, min_person_px: int = 60, edge: float = 0.01
                       ) -> list[tuple[RoleBox, list[str]]]:
    """person_attribute_issues ile aynı kural; işaretli kişi kutusunu da döndürür: [(kişi, kodlar)]."""
    out: list[tuple[RoleBox, list[str]]] = []
    for p in (b for b in rb if b.role == "person"):
        if p.box.h * img_h < min_person_px:
            continue
        has_head = any(b.role in HEAD_ROLES and assign_to_person(b, [p]) is p for b in rb)
        has_torso = any(b.role in TORSO_ROLES and assign_to_person(b, [p]) is p for b in rb)
        if has_head and has_torso:
            continue
        region = (p.x1 - 0.1 * p.box.w, p.y1 - 0.15 * p.box.h, p.x2 + 0.1 * p.box.w, p.y2)
        side_cut = p.x1 <= edge or p.x2 >= 1 - edge
        why: list[str] = []
        if not has_head:
            if any(b.role in HEAD_ROLES and _inside_ratio(b, region) > 0.5 for b in rb):
                why.append("head_label_present_unassigned")
            elif p.y1 <= edge:
                why.append("head_out_of_frame_top")
            elif side_cut:
                why.append("person_truncated_at_edge")
            else:
                why.append("head_unexplained")
        if not has_torso:
            if any(b.role in TORSO_ROLES and _inside_ratio(b, region) > 0.5 for b in rb):
                why.append("torso_label_present_unassigned")
            elif side_cut:
                why.append("person_truncated_at_edge")
            elif p.box.h * img_h < 120:
                why.append("torso_small_person")
            else:
                why.append("torso_unexplained")
        out.append((p, sorted(set(why))))
    return out


def final_split_assignment(images: list[tuple[str, str, str]], excluded: set[str] | frozenset[str] = frozenset()
                           ) -> tuple[dict[str, str], list[dict[str, Any]]]:
    """Onaylı kaynak için nihai bölme: yayıncı bölünmesi korunur, yalnızca bölümler arası dağılmış kaynak birimleri düzeltilir.

    images: (image_path, original_split, split_unit). excluded: kullanım dışı (quarantine/needs_reannotation) görüntüler;
    bölümsüz kalırlar ve çoğunluk hesabına katılmazlar. Bir birimin KULLANILAN üyeleri farklı bölümlerdeyse birim, üyelerinin
    çoğunluğunun bulunduğu bölüme taşınır (eşitlikte 'train': değerlendirme kümesi küçülür, eğitim/değerlendirme ayrımı korunur).
    Hiçbir görüntü kopyalanmaz; bir birim ASLA bölünmez. Dönüş: (path -> final_split | '', birim düzeltme kayıtları).
    """
    by_unit: dict[str, list[tuple[str, str]]] = {}
    for path, split, unit in images:
        if path not in excluded:
            by_unit.setdefault(unit, []).append((path, split))
    final: dict[str, str] = {p: "" for p, _, _ in images}
    fixes: list[dict[str, Any]] = []
    for unit, members in sorted(by_unit.items()):
        cnt = Counter(s for _, s in members)
        if len(cnt) == 1:
            target = members[0][1]
        else:
            top = max(cnt.values())
            winners = sorted(s for s, n in cnt.items() if n == top)
            target = "train" if "train" in winners else winners[0]
            fixes.append({"split_unit": unit, "members": len(members), "original_splits": dict(sorted(cnt.items())),
                          "final_split": target,
                          "moved": sorted(p for p, s in members if s != target)})
        for p, _ in members:
            final[p] = target
    return final, fixes


def unit_integrity_violations(rows: list[dict[str, Any]], split_key: str = "final_split", unit_key: str = "split_unit"
                              ) -> dict[str, set[str]]:
    """Birden çok bölüme dağılmış bölme birimleri (boş bölme = kullanım dışı, sayılmaz). Boş sözlük = kural korunuyor."""
    seen: dict[str, set[str]] = {}
    for r in rows:
        if r.get(split_key):
            seen.setdefault(r[unit_key], set()).add(r[split_key])
    return {u: s for u, s in seen.items() if len(s) > 1}


def decide_conditional(dataset: str, rec: dict[str, Any], src: dict[str, Any], policy: dict[str, Any],
                       review_by_image: dict[str, list[dict[str, str]]], verified_negatives: set[str],
                       conflict_overrides: dict[str, dict[str, str]], rel_path: str) -> tuple[str, list[str]]:
    """Hak doğrulanırsa görüntünün alacağı durum ve nedenleri: (status, reason_codes)."""
    cmap = {int(k) for k in src["conditional_class_map"]}
    excluded = {int(k) for k in (src.get("excluded_classes") or {})}
    ignored = {int(k) for k in (src.get("ignored_classes") or [])}
    boxes = rec["boxes"]
    target_boxes = [b for b in boxes if b.class_id in cmap]
    reasons_na: list[str] = []   # needs_reannotation nedenleri
    reasons_q: list[str] = []    # quarantine nedenleri
    rules = src.get("conditional_rules", [])
    if not rec["readable"]:
        reasons_q.append("IMAGE-UNREADABLE")
    if "MD-HELMET-IMAGE" in rules and any(b.class_id == 0 for b in boxes):
        reasons_na.append("MD-HELMET-IMAGE")
    if "MD-CROWD-INCOMPLETE" in rules and len(target_boxes) >= policy["rules"]["MD-CROWD-INCOMPLETE"]["params"]["min_target_boxes"]:
        reasons_q.append("MD-CROWD-INCOMPLETE")
    if "RF-EXCLUDED-CLASS-IMAGE" in rules and any(b.class_id in excluded for b in boxes):
        reasons_na.append("RF-EXCLUDED-CLASS-IMAGE")
    if "RF-TINY-TARGET" in rules:
        min_h = policy["rules"]["RF-TINY-TARGET"]["params"]["min_target_height_px"]
        if any(b.h * rec["height"] < min_h for b in target_boxes):
            reasons_q.append("RF-TINY-TARGET")
    if "RF-PERSON-INCOMPLETE" in rules and rec["incomplete_persons"] > 0:
        reasons_q.append("RF-PERSON-INCOMPLETE")
    if "RF-UNVERIFIED-NEGATIVE" in rules and not target_boxes and not any(b.class_id in excluded for b in boxes):
        if rel_path not in verified_negatives:
            reasons_q.append("RF-UNVERIFIED-NEGATIVE")
    for it in review_by_image.get(rel_path, []):
        ov = conflict_overrides.get(it["item_id"])
        if ov is not None:  # açık (görsel) karar kaydı varsa genel kural yerine o uygulanır
            if ov["decision"] == "quarantine":
                reasons_q.append(f"RF-CONFLICT-OVERRIDE:{it['item_id']}")
            elif ov["decision"] == "needs_reannotation":
                reasons_na.append(f"RF-CONFLICT-OVERRIDE:{it['item_id']}")
            continue
        if "RF-LIKELY-ERROR" in rules and it["category"] in ("Confirmed Error", "Likely Error") \
                and it["source"] not in ("duplicate_box", "near_duplicate_box"):
            reasons_q.append(f"RF-LIKELY-ERROR:{it['item_id']}")
        if "RF-CONFLICT-UNRESOLVED" in rules and it["source"].startswith("semantic_conflict") and it["category"] == "Needs Manual Review":
            reasons_q.append(f"RF-CONFLICT-UNRESOLVED:{it['item_id']}")
    if reasons_q:
        return "quarantine", reasons_q + reasons_na
    if reasons_na:
        return "needs_reannotation", reasons_na
    return "approved", []
