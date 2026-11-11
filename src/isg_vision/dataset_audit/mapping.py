"""Aday sınıf eşleme taslağı (configs/candidate_class_mapping.yaml) okuma ve tutarlılık kontrolü."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from isg_vision.utils.paths import PROJECT_ROOT

MAPPING_FILE = PROJECT_ROOT / "configs" / "candidate_class_mapping.yaml"
VALID_STATUSES = {"confirmed", "needs_review", "incompatible"}
VALID_COMPATIBILITY = {"Uyumlu", "Uyumsuz", "Karar verilemedi"}
VALID_RELIABILITY = {"Yüksek", "Orta", "Düşük"}
MAPPING_FIELDS = ["source_dataset", "source_class_id", "source_class_name", "proposed_target_id",
                  "proposed_target_class", "box_semantics", "validation_status", "compatibility", "reliability",
                  "visual_result", "inconsistencies", "evidence", "risk_or_note", "user_review"]


def load_mapping(path: Path = MAPPING_FILE) -> tuple[dict[int, str], list[dict[str, Any]]]:
    if not path.is_file():
        raise FileNotFoundError(f"Sınıf eşleme dosyası bulunamadı: {path}")
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    return {int(k): v for k, v in data["target_classes"].items()}, data["mappings"]


def mapping_rows(targets: dict[int, str], entries: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """YAML kayıtlarını CSV satırlarına çevirir ve kayıt biçimini doğrular."""
    rows = []
    for e in entries:
        if e["status"] not in VALID_STATUSES:
            raise ValueError(f"Geçersiz doğrulama durumu: {e['status']!r} ({e['dataset']}/{e['name']})")
        if e.get("compatibility", "Karar verilemedi") not in VALID_COMPATIBILITY:
            raise ValueError(f"Geçersiz uyumluluk: {e.get('compatibility')!r} ({e['dataset']}/{e['name']})")
        if e.get("reliability", "Düşük") not in VALID_RELIABILITY:
            raise ValueError(f"Geçersiz güvenilirlik: {e.get('reliability')!r} ({e['dataset']}/{e['name']})")
        target = e.get("target")
        if target is not None and target not in targets:
            raise ValueError(f"Bilinmeyen hedef sınıf: {target} ({e['dataset']}/{e['name']})")
        rows.append({
            "source_dataset": e["dataset"], "source_class_id": e["id"], "source_class_name": e["name"],
            "proposed_target_id": "" if target is None else target,
            "proposed_target_class": "(hedef dışı)" if target is None else targets[target],
            "box_semantics": e.get("box_semantics", ""), "validation_status": e["status"],
            "compatibility": e.get("compatibility", "Karar verilemedi"), "reliability": e.get("reliability", "Düşük"),
            "visual_result": e.get("visual_result", ""), "inconsistencies": e.get("inconsistencies", ""),
            "evidence": e.get("evidence", ""), "risk_or_note": e.get("risk", ""), "user_review": "Pending",
        })
    return rows


def check_completeness(entries: list[dict[str, Any]], classes_by_dataset: dict[str, dict[int, str]]) -> list[str]:
    """Her veri setindeki her YAML sınıfının eşlemede tam bir kez yer aldığını ve adların tuttuğunu denetler."""
    problems: list[str] = []
    seen: dict[tuple[str, int], str] = {}
    for e in entries:
        key = (e["dataset"], int(e["id"]))
        if key in seen:
            problems.append(f"Eşlemede tekrar eden sınıf: {key}")
        seen[key] = e["name"]
        actual = classes_by_dataset.get(e["dataset"], {}).get(int(e["id"]))
        if actual is None:
            problems.append(f"Eşlemedeki sınıf YAML'da yok: {key}")
        elif actual != e["name"]:
            problems.append(f"Sınıf adı uyuşmuyor {key}: eşleme '{e['name']}' / YAML '{actual}'")
    for ds, names in classes_by_dataset.items():
        for cid, cname in names.items():
            if (ds, cid) not in seen:
                problems.append(f"Eşlemede EKSİK sınıf: {ds} {cid}:{cname}")
    return problems
