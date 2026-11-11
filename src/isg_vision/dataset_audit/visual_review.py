"""Aşama 2D: insan doğrulaması için görsel inceleme paketi (sayfa çizimi) ve kullanıcı karar CSV'sinin doğrulanması.

Ham veri yalnızca OKUNUR. Model önerileri (approved_candidate / needs_reannotation / quarantine / inconclusive) kullanıcı kararı
DEĞİLDİR; politika yalnızca 'user_decision' sütununu uygular.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFont

from isg_vision.dataset_audit.labels import parse_label_file
from isg_vision.dataset_audit.semantic import HEAD_ROLES, TORSO_ROLES, RoleBox, assign_to_person, class_roles
from isg_vision.dataset_audit.usage_policy import iter_person_issues

MODEL_DECISIONS = ("approved_candidate", "needs_reannotation", "quarantine", "inconclusive")
USER_DECISIONS = ("approved", "needs_reannotation", "quarantine")      # boş = karar bekliyor
REVIEW_FIELDS = ["review_id", "image_path", "split", "risk_code", "suspect_person_box_px", "suspect_label", "nearby_labels",
                 "model_decision", "model_rationale", "contact_sheet_page", "user_decision", "user_note"]
USER_FIELDS = ("user_decision", "user_note")

PAGE_W, PAGE_H = 2480, 1754          # A4 yatay, 300 dpi
FONT_PATHS = ("/System/Library/Fonts/Supplemental/Arial Unicode.ttf", "/System/Library/Fonts/Supplemental/Arial.ttf",
              "/Library/Fonts/Arial Unicode.ttf", "DejaVuSans.ttf")
ROLE_COLORS = {"helmet": (0, 170, 0), "no_helmet": (230, 0, 0), "vest": (0, 110, 255), "no_vest": (220, 0, 220),
               "person": (255, 200, 0)}
OTHER_COLOR = (150, 150, 150)
SUSPECT_COLOR = (255, 255, 0)


@dataclass
class ReviewItem:
    image_path: Path
    rel_path: str
    split: str
    risk_code: str
    width: int
    height: int
    boxes: list[tuple[str, Any]]                  # (sınıf adı, Box)
    persons: list[RoleBox]
    suspect: RoleBox
    suspect_index: int
    codes: list[str]
    nearby: list[dict[str, Any]] = field(default_factory=list)


def font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    for p in FONT_PATHS:
        try:
            return ImageFont.truetype(p, size)
        except OSError:
            continue
    return ImageFont.load_default()


def _iou(a: RoleBox, b: RoleBox) -> float:
    ix = max(0.0, min(a.x2, b.x2) - max(a.x1, b.x1))
    iy = max(0.0, min(a.y2, b.y2) - max(a.y1, b.y1))
    inter = ix * iy
    return inter / max(a.area + b.area - inter, 1e-12)


def collect_items(manifest_rows: list[dict[str, str]], registry_entry: dict[str, Any], project_root: Path,
                  code_filter: str = "unexplained") -> list[ReviewItem]:
    """Manifestte annotation_risk'i code_filter içeren görüntüler için işaretli kişi + yakındaki baş/gövde etiketleri."""
    roles = class_roles(registry_entry)
    names = registry_entry["expected_class_names"]
    items: list[ReviewItem] = []
    for r in manifest_rows:
        if code_filter not in r.get("annotation_risk", ""):
            continue
        img_path = project_root / r["image_path"]
        lbl = img_path.parent.parent.parent / "labels" / img_path.parent.name / (img_path.stem + ".txt")
        with Image.open(img_path) as im:
            w, h = im.size
        boxes = parse_label_file(lbl).boxes
        rb = [RoleBox(roles[b.class_id], b) for b in boxes if b.class_id in roles and b.w > 0 and b.h > 0]
        persons = [b for b in rb if b.role == "person"]
        flagged = [(p, c) for p, c in iter_person_issues(rb, h) if any(code_filter in x for x in c)]
        for p, codes in flagged:
            region = (p.x1 - 0.1 * p.box.w, p.y1 - 0.15 * p.box.h, p.x2 + 0.1 * p.box.w, p.y2)
            nearby = []
            for b in rb:
                if b.role not in HEAD_ROLES + TORSO_ROLES:
                    continue
                ix = max(0.0, min(b.x2, region[2]) - max(b.x1, region[0]))
                iy = max(0.0, min(b.y2, region[3]) - max(b.y1, region[1]))
                if ix * iy <= 0:
                    continue
                owner = assign_to_person(b, persons)
                nearby.append({"class": names[b.box.class_id], "line": b.box.line_no,
                               "inside_ratio": round(ix * iy / max(b.area, 1e-12), 2),
                               "assigned_person": (persons.index(owner) + 1) if owner is not None else 0})
            items.append(ReviewItem(img_path, r["image_path"], r["final_split"] or r["original_split"], "|".join(codes), w, h,
                                    [(names[b.class_id], b) for b in boxes], persons, p, persons.index(p) + 1, codes, nearby))
    return items


def _px(b: Any, w: int, h: int) -> tuple[int, int, int, int]:
    return (int((b.xc - b.w / 2) * w), int((b.yc - b.h / 2) * h), int((b.xc + b.w / 2) * w), int((b.yc + b.h / 2) * h))


def _annotate(img: Image.Image, item: ReviewItem, crop: tuple[int, int, int, int] | None, out_size: tuple[int, int],
              labels: bool, line_scale: float = 1.0) -> Image.Image:
    """Görüntüyü (veya kırpıntıyı) out_size içine sığdırır ve kutuları/etiketleri ÇIKTI çözünürlüğünde çizer."""
    x0, y0, x1, y1 = crop if crop else (0, 0, item.width, item.height)
    region = img.crop((x0, y0, x1, y1))
    s = min(out_size[0] / region.width, out_size[1] / region.height)
    region = region.resize((max(1, int(region.width * s)), max(1, int(region.height * s))), Image.LANCZOS)
    if not labels and crop is None:
        return region
    d = ImageDraw.Draw(region)
    f = font(int(22 * line_scale) if crop else 18)
    roles = {"helmet": "helmet", "no_helmet": "no_helmet", "vest": "vest", "no_vest": "no_vest", "Person": "person"}
    person_no = {id(p.box): i + 1 for i, p in enumerate(item.persons)}
    for name, b in item.boxes:
        bx = _px(b, item.width, item.height)
        rx = [(bx[0] - x0) * s, (bx[1] - y0) * s, (bx[2] - x0) * s, (bx[3] - y0) * s]
        if rx[2] < 0 or rx[3] < 0 or rx[0] > region.width or rx[1] > region.height:
            continue
        role = roles.get(name)
        color = ROLE_COLORS.get(role, OTHER_COLOR)
        is_suspect = b is item.suspect.box
        wdt = max(2, int((6 if is_suspect else (3 if role else 1)) * line_scale))
        d.rectangle(rx, outline=SUSPECT_COLOR if is_suspect else color, width=wdt)
        if labels:
            txt = f"P{person_no[id(b)]}" if role == "person" else name
            if is_suspect:
                txt = f"P{person_no[id(b)]} ŞÜPHELİ"
            tw = d.textlength(txt, font=f)
            ty = max(0, rx[1] - f.size - 4) if role == "person" else rx[1] + 2
            tx = max(0, min(rx[0], region.width - tw - 4))
            d.rectangle([tx, ty, tx + tw + 4, ty + f.size + 4], fill=(0, 0, 0))
            d.text((tx + 2, ty + 1), txt, fill=SUSPECT_COLOR if is_suspect else color, font=f)
    return region


def _zoom_boxes(item: ReviewItem) -> tuple[tuple[int, int, int, int], tuple[int, int, int, int], str]:
    """(kişi yakınlaştırması, şüpheli bölge yakınlaştırması, bölge adı)."""
    p = item.suspect
    W, H = item.width, item.height
    pw, ph = p.box.w * W, p.box.h * H
    m = max(pw, ph) * 0.35
    person = (max(0, int(p.x1 * W - m)), max(0, int(p.y1 * H - m)), min(W, int(p.x2 * W + m)), min(H, int(p.y2 * H + m * 0.3)))
    head_issue = any(c.startswith("head") for c in item.codes)
    if head_issue:
        cy0, cy1 = p.y1 * H - 0.25 * ph, p.y1 * H + 0.40 * ph
        name = "baş bölgesi"
    else:
        cy0, cy1 = p.y1 * H + 0.10 * ph, p.y1 * H + 0.70 * ph
        name = "gövde bölgesi"
    cx0, cx1 = p.x1 * W - 0.3 * pw, p.x2 * W + 0.3 * pw
    side = max(cx1 - cx0, cy1 - cy0, 48)
    cx, cy = (cx0 + cx1) / 2, (cy0 + cy1) / 2
    region = (max(0, int(cx - side / 2)), max(0, int(cy - side / 2)), min(W, int(cx + side / 2)), min(H, int(cy + side / 2)))
    return person, region, name


def render_page(item: ReviewItem, review_id: str, decision: str = "", rationale: str = "") -> Image.Image:
    page = Image.new("RGB", (PAGE_W, PAGE_H), (255, 255, 255))
    d = ImageDraw.Draw(page)
    with Image.open(item.image_path) as im0:
        img = im0.convert("RGB")
    f_title, f_txt, f_cap = font(40), font(27), font(28)
    pbox = _px(item.suspect.box, item.width, item.height)
    d.text((40, 24), f"{review_id}  |  {item.rel_path}  |  bölüm: {item.split}  |  {item.width}x{item.height}", fill=(0, 0, 0), font=f_title)
    d.text((40, 78), f"Risk kodu: CP-PERSON-ATTR-INCOMPLETE:{item.risk_code}   |   Şüpheli kişi: P{item.suspect_index} "
                     f"(kutu {pbox[2]-pbox[0]}x{pbox[3]-pbox[1]} px)", fill=(150, 0, 0), font=f_txt)
    near = "; ".join(f"{n['class']} (satır {n['line']}, kişi bölgesinde %{int(100*n['inside_ratio'])}, atanan: "
                     f"{'P' + str(n['assigned_person']) if n['assigned_person'] else 'yok'})" for n in item.nearby) or "yok"
    d.text((40, 116), f"Şüpheli bölge çevresindeki baş/gövde etiketleri: {near}"[:230], fill=(0, 0, 0), font=f_txt)
    top_y, top_h, col_w = 170, 760, (PAGE_W - 120) // 2
    a = _annotate(img, item, None, (col_w, top_h), labels=False)
    b = _annotate(img, item, None, (col_w, top_h), labels=True)
    page.paste(a, (40, top_y))
    page.paste(b, (80 + col_w, top_y))
    d.text((40, top_y + top_h + 6), "A) Orijinal görüntü", fill=(0, 0, 0), font=f_cap)
    d.text((80 + col_w, top_y + top_h + 6), "B) Tüm kutular (sarı kalın = şüpheli kişi; P# = kişi no)", fill=(0, 0, 0), font=f_cap)
    pz, rz, rname = _zoom_boxes(item)
    bot_y, bot_h = top_y + top_h + 56, 640
    zw = (PAGE_W - 160) // 3
    c = _annotate(img, item, pz, (zw, bot_h), labels=True, line_scale=1.3)
    z_raw = _annotate(img, item, rz, (zw, bot_h), labels=False)
    z_lab = _annotate(img, item, rz, (zw, bot_h), labels=True, line_scale=1.3)
    page.paste(c, (40, bot_y))
    page.paste(z_raw, (80 + zw, bot_y))
    page.paste(z_lab, (120 + 2 * zw, bot_y))
    cap_y = bot_y + bot_h + 6
    d.text((40, cap_y), "C) Şüpheli kişi (kutulu)", fill=(0, 0, 0), font=f_cap)
    d.text((80 + zw, cap_y), f"D) {rname} yakınlaştırma (kutusuz)", fill=(0, 0, 0), font=f_cap)
    d.text((120 + 2 * zw, cap_y), f"E) {rname} yakınlaştırma (kutulu)", fill=(0, 0, 0), font=f_cap)
    if decision:
        d.text((40, PAGE_H - 46), f"Model önerisi (kullanıcı kararı DEĞİL): {decision} — {rationale}"[:200], fill=(0, 0, 160), font=f_txt)
    return page


# ---------------------------------------------------------------- kullanıcı karar CSV'si
def read_review_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh))


def write_review_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=REVIEW_FIELDS)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in REVIEW_FIELDS})


def validate_review_rows(rows: list[dict[str, str]], manifest_paths: set[str], expected_ids: set[str] | None = None) -> list[str]:
    """Karar CSV'sinin şema/değer denetimi. Boş liste = geçerli. user_decision boş olabilir (karar bekliyor)."""
    problems: list[str] = []
    if not rows:
        return ["CSV boş"]
    missing_cols = [c for c in REVIEW_FIELDS if c not in rows[0]]
    if missing_cols:
        return [f"eksik sütun(lar): {missing_cols}"]
    seen: set[str] = set()
    for r in rows:
        rid, path = r["review_id"], r["image_path"]
        if rid in seen or not rid:
            problems.append(f"yinelenen/boş review_id: {rid!r}")
        seen.add(rid)
        if path not in manifest_paths:
            problems.append(f"{rid}: görüntü nihai manifestte yok: {path}")
        if r["model_decision"] not in MODEL_DECISIONS:
            problems.append(f"{rid}: geçersiz model_decision {r['model_decision']!r}")
        ud = r["user_decision"].strip()
        if ud and ud not in USER_DECISIONS:
            problems.append(f"{rid}: geçersiz user_decision {ud!r} (geçerli: {', '.join(USER_DECISIONS)} veya boş)")
        if ud and ud != "approved" and not r["user_note"].strip():
            problems.append(f"{rid}: '{ud}' kararı için user_note (kısa gerekçe) zorunlu")
    if expected_ids is not None and seen != expected_ids:
        problems.append(f"beklenen inceleme kümesiyle uyuşmuyor: eksik {sorted(expected_ids - seen)}, fazla {sorted(seen - expected_ids)}")
    return problems


def user_overrides(rows: list[dict[str, str]]) -> dict[str, dict[str, str]]:
    """Kullanıcının kullanım dışı bıraktığı görüntüler -> politika override kaydı. 'approved' ve boş karar override üretmez."""
    out = {}
    for r in rows:
        ud = r["user_decision"].strip()
        if ud in ("needs_reannotation", "quarantine"):
            out[r["image_path"]] = {"status": ud, "rule": "USER-VISUAL-REVIEW", "evidence": f"{r['review_id']}: {r['user_note'].strip()}"}
    return out


def review_state(rows: list[dict[str, str]]) -> dict[str, Any]:
    """İnceleme ilerlemesi: kullanıcı kararı verilmiş/bekleyen sayıları."""
    decided = [r for r in rows if r["user_decision"].strip()]
    return {"total": len(rows), "user_decided": len(decided), "pending_user": len(rows) - len(decided),
            "by_user_decision": {d: sum(r["user_decision"].strip() == d for r in rows) for d in USER_DECISIONS},
            "complete": len(decided) == len(rows)}
