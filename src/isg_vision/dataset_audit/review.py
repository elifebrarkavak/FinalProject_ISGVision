"""Görsel inceleme kararlarının (configs/review/label_review_verdicts.yaml) tabloya birleştirilmesi.

Kararlar insan benzeri (model destekli) görsel incelemenin KAYDIDIR; kod yalnızca kayıtları ilgili şüpheli
etiketlerle eşler, kendisi görsel karar üretmez. Karar kaydı olmayan her öğe "Needs Manual Review" kalır.
"""

from __future__ import annotations

import math
import re
from pathlib import Path
from typing import Any

from isg_vision.dataset_audit.common import read_csv
from isg_vision.utils.paths import PROJECT_ROOT

CATEGORIES = ("Confirmed Error", "Likely Error", "Valid Annotation", "Needs Manual Review")
REVIEW_FIELDS = ["item_id", "source", "dataset", "split", "image_path", "label_path", "lines", "label_text",
                 "category", "rationale", "evidence", "suggested_fix"]
AUDIT_PREFIX = {"helmet": "AH", "no_helmet": "AN", "vest": "AV", "no_vest": "AX"}
_ID = re.compile(r"^([A-Za-z]+)(\d+)$")

SUGGESTED_FIX = {
    "Confirmed Error": "Yeni (temizlenmiş) kopyada düzeltilmeli/kaldırılmalı; ham veri DEĞİŞTİRİLMEZ.",
    "Likely Error": "Yeni kopyada kutu düzeltilmeli veya çıkarılmalı; önce insan onayı alınmalı.",
    "Valid Annotation": "Düzeltme gerekmez.",
    "Needs Manual Review": "İnsan incelemesi gerekir; karar verilene kadar eğitimde dikkatle ele alınmalı.",
}


def wilson_interval(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson %95 güven aralığı (oran = k/n)."""
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (round(max(0.0, center - half), 3), round(min(1.0, center + half), 3))


def _expand_ids(spec: str) -> list[str]:
    """'SI029-SI036, SI042' -> ['SI029', ..., 'SI036', 'SI042'] (aynı önek ve rakam sayısı korunur)."""
    out: list[str] = []
    for part in [p.strip() for p in str(spec).split(",") if p.strip()]:
        if "-" in part:
            a, b = [x.strip() for x in part.split("-", 1)]
            ma, mb = _ID.match(a), _ID.match(b)
            if not ma or not mb or ma.group(1) != mb.group(1):
                raise ValueError(f"Geçersiz aralık: {part}")
            width = len(ma.group(2))
            out += [f"{ma.group(1)}{i:0{width}d}" for i in range(int(ma.group(2)), int(mb.group(2)) + 1)]
        else:
            out.append(part)
    return out


def expand_verdicts(raw: dict[str, Any]) -> dict[str, dict[str, str]]:
    """YAML kararlarını {item_id: {category, reason, evidence}} sözlüğüne açar; geçersiz kategori ve tekrarları reddeder."""
    out: dict[str, dict[str, str]] = {}
    for rule in raw.get("verdicts", []):
        if rule["category"] not in CATEGORIES:
            raise ValueError(f"Geçersiz kategori: {rule['category']}")
        for item_id in _expand_ids(rule["ids"]):
            if item_id in out:
                raise ValueError(f"Aynı öğe için birden çok karar: {item_id}")
            out[item_id] = {"category": rule["category"], "reason": rule.get("reason", ""),
                            "evidence": rule.get("evidence", "")}
    return out


def _label_text(label_path: str, lines: list[int]) -> str:
    p = PROJECT_ROOT / label_path
    if not p.is_file() or not lines:
        return ""
    content = p.read_text(encoding="utf-8").splitlines()
    return " | ".join(f"L{n}: {content[n - 1].strip()}" for n in lines if 0 < n <= len(content))


def build_review_rows(audit_dir: Path, verdicts: dict[str, dict[str, str]]) -> list[dict[str, Any]]:
    """Şüpheli etiket kayıtlarını (çelişkiler, küçük/kopya kutular, boş etiketler, kalite örneklemi) karar kayıtlarıyla eşler."""
    rows: list[dict[str, Any]] = []

    def add(item_id: str, source: str, dataset: str, split: str, image: str, label: str, lines: list[int]) -> None:
        v = verdicts.get(item_id)
        cat = v["category"] if v else "Needs Manual Review"
        rows.append({
            "item_id": item_id, "source": source, "dataset": dataset, "split": split, "image_path": image,
            "label_path": label, "lines": " ".join(map(str, lines)), "label_text": _label_text(label, lines),
            "category": cat, "rationale": v["reason"] if v else "Karar kaydı yok",
            "evidence": v["evidence"] if v else "", "suggested_fix": SUGGESTED_FIX[cat]})

    for c in read_csv(audit_dir / "semantic_conflicts.csv"):
        lines = [int(c["pos_line"]), int(c["neg_line"])] + ([int(c["person_line"])] if c["person_line"] else [])
        add(c["conflict_id"], f"semantic_conflict:{c['attribute']}:{c['type']}", c["dataset"], c["split"],
            c["image_path"], c["label_path"], lines)
    issues = read_csv(audit_dir / "annotation_issues.csv")
    prefix = {"BOX_VERY_SMALL": ("TB", "tiny_box"), "LABEL_EMPTY": ("EL", "empty_label"),
              "DUPLICATE_BOX": ("DB", "duplicate_box"), "NEAR_DUPLICATE_BOX": ("ND", "near_duplicate_box")}
    for code, (pre, source) in prefix.items():
        for i, r in enumerate([x for x in issues if x["dataset"] == "site_safety" and x["issue_code"] == code], start=1):
            line = int(r["line_no"]) if r["line_no"] else None
            add(f"{pre}{i:02d}", source, r["dataset"], r["split"], r["image_path"], r["label_path"], [line] if line else [])
    sample_file = audit_dir / "roboflow_audit_sample.csv"
    if sample_file.is_file():
        for r in read_csv(sample_file):
            pre = AUDIT_PREFIX[r["role"]]
            label = r["image_path"].replace("/images/", "/labels/").rsplit(".", 1)[0] + ".txt"
            add(f"{pre}{int(r['tile']):02d}", f"audit_{r['role']}", r["dataset"], r["split"], r["image_path"], label,
                [int(r["line_no"])])
    return rows
