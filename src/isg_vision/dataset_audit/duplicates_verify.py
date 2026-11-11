"""Yakın kopya aday üretimi (dHash + çok ölçekli pHash) ve geometrik doğrulama (ORB + RANSAC afin + hizalı NCC).

Karar mantığı hash mesafesine TEK BAŞINA dayanmaz:
  1) Aday: pHash/dHash (kırpma ve yatay çevirmeye dayanıklı çok pencereli imza) veya aynı kaynak adı.
  2) Doğrulama: ORB eşleşmeleri üzerinde RANSAC ile afin dönüşüm; inlier sayısı/oranı + hizalanmış alanda NCC.
  3) Karar: same_image_derived / same_scene_different_frame / uncertain / not_duplicate (+ exact_duplicate: SHA-256).
Eşikler sentetik+gerçek doğrulama deneyiyle (validate_method) seçilir ve raporlanır. Dosyalar yalnızca okunur.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import cv2
import numpy as np

MAX_SIDE = 512
_POP16 = np.array([bin(i).count("1") for i in range(1 << 16)], dtype=np.uint8)

# Doğrulama deneyinde seçilen varsayılan eşikler (bkz. duplicate_method_validation.json)
DEFAULT_THRESHOLDS = {"min_inliers": 25, "min_inlier_ratio": 0.30, "uncertain_inliers": 12, "min_ncc": 0.80,
                      "candidate_hash_bits": 10}


def load_gray(path: Path, max_side: int = MAX_SIDE) -> np.ndarray | None:
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        return None
    s = max_side / max(img.shape[:2])
    return cv2.resize(img, (int(img.shape[1] * s), int(img.shape[0] * s)), interpolation=cv2.INTER_AREA) if s < 1 else img


def phash64(gray: np.ndarray) -> int:
    """64-bit DCT tabanlı algısal özet (32x32 -> 8x8 düşük frekans, DC hariç medyan eşiği)."""
    small = cv2.resize(gray, (32, 32), interpolation=cv2.INTER_AREA).astype(np.float32)
    low = cv2.dct(small)[:8, :8].flatten()
    med = np.median(low[1:])
    bits = (low > med)
    value = 0
    for b in bits:
        value = (value << 1) | int(b)
    return value


def dhash64(gray: np.ndarray) -> int:
    small = cv2.resize(gray, (9, 8), interpolation=cv2.INTER_AREA).astype(np.int16)
    bits = (small[:, 1:] > small[:, :-1]).flatten()
    value = 0
    for b in bits:
        value = (value << 1) | int(b)
    return value


def feature_mask(gray: np.ndarray) -> np.ndarray | None:
    """Saf siyah (0) bölgeleri (cutout kareleri, döndürme/kırpma kenarları) ORB'den hariç tutan maske.

    Aynı boyutlu siyah karelerin köşeleri farklı görüntülerde sahte eşleşme üretir; bu yüzden maskelenir.
    Siyah bölge yoksa None döner.
    """
    zero = (gray == 0).astype(np.uint8)
    solid = cv2.erode(zero, np.ones((5, 5), np.uint8))
    if not solid.any():
        return None
    return 255 - cv2.dilate(solid, np.ones((11, 11), np.uint8)) * 255


def center_crop(gray: np.ndarray, frac: float) -> np.ndarray:
    h, w = gray.shape
    ch, cw = int(h * frac), int(w * frac)
    y, x = (h - ch) // 2, (w - cw) // 2
    return gray[y:y + ch, x:x + cw]


def signature(gray: np.ndarray) -> dict[str, int]:
    """Tam görüntü + %90/%80 merkez kırpma pHash'leri, yatay çevrilmiş sürümleri ve dHash."""
    sig = {"d": dhash64(gray), "dflip": dhash64(gray[:, ::-1])}
    for name, frac in (("p0", 1.0), ("p1", 0.9), ("p2", 0.8)):
        g = gray if frac == 1.0 else center_crop(gray, frac)
        sig[name] = phash64(g)
        sig[name + "f"] = phash64(np.ascontiguousarray(g[:, ::-1]))
    return sig


def _popcount(x: np.ndarray) -> np.ndarray:
    return _POP16[x.view(np.uint16).reshape(-1, 4)].sum(axis=1)


_COMBOS = [("p0", "p0"), ("p0", "p0f"), ("p0", "p1"), ("p0", "p2"), ("p0", "p1f"), ("p0", "p2f"),
           ("p1", "p0"), ("p2", "p0"), ("p1", "p0f"), ("p2", "p0f"), ("d", "d"), ("d", "dflip")]


def hash_distance(a: dict[str, int], b: dict[str, int]) -> tuple[int, str]:
    """İki imza arasında (en küçük mesafe, hangi karşılaştırma). Kırpma ve yatay çevirme için çok pencereli."""
    best, how = 65, ""
    for ka, kb in _COMBOS:
        d = bin(a[ka] ^ b[kb]).count("1")
        if d < best:
            best, how = d, f"{ka}~{kb}"
    return best, how


def candidate_pairs(sigs: list[dict[str, int]], max_bits: int) -> list[tuple[int, int, int, str]]:
    """Tüm çiftler için (i, j, mesafe, yöntem); mesafe <= max_bits olanlar. numpy ile vektörleştirilmiş."""
    n = len(sigs)
    keys = ("p0", "p1", "p2", "p0f", "p1f", "p2f", "d", "dflip")
    arr = {k: np.array([s[k] for s in sigs], dtype=np.uint64) for k in keys}
    pairs: list[tuple[int, int, int, str]] = []
    combos = _COMBOS
    for i in range(n - 1):
        best = np.full(n - i - 1, 99, dtype=np.int32)
        how = np.zeros(n - i - 1, dtype=np.int8)
        for c, (ka, kb) in enumerate(combos):
            dist = _popcount(np.bitwise_xor(arr[ka][i], arr[kb][i + 1:])).astype(np.int32)
            better = dist < best
            best[better] = dist[better]
            how[better] = c
        for k in np.nonzero(best <= max_bits)[0]:
            ka, kb = combos[int(how[k])]
            pairs.append((i, i + 1 + int(k), int(best[k]), f"{ka}~{kb}"))
    return pairs


@dataclass
class OrbCache:
    nfeatures: int = 1500
    store: dict[Any, tuple[Any, Any]] = field(default_factory=dict)

    def features(self, key: Any, gray: np.ndarray, flipped: bool = False):
        k = (key, flipped)
        if k not in self.store:
            img = np.ascontiguousarray(gray[:, ::-1]) if flipped else gray
            orb = cv2.ORB_create(nfeatures=self.nfeatures, fastThreshold=10)
            kps, des = orb.detectAndCompute(img, feature_mask(img))
            self.store[k] = (kps, des)
        return self.store[k]


def _ncc_aligned(a: np.ndarray, b: np.ndarray, M: np.ndarray, size: int = 256) -> float | None:
    """a'yı b uzayına afin M ile taşır; ortak geçerli alanda sıfır-ortalamalı NCC (parlaklık/kontrasta dayanıklı)."""
    h, w = b.shape
    warped = cv2.warpAffine(a, M, (w, h), flags=cv2.INTER_LINEAR, borderValue=0)
    mask = cv2.warpAffine(np.full(a.shape, 255, np.uint8), M, (w, h), flags=cv2.INTER_NEAREST, borderValue=0) > 0
    if mask.sum() < 0.15 * h * w:
        return None
    s = size / max(h, w)
    wa = cv2.resize(warped, None, fx=s, fy=s, interpolation=cv2.INTER_AREA).astype(np.float32)
    wb = cv2.resize(b, None, fx=s, fy=s, interpolation=cv2.INTER_AREA).astype(np.float32)
    mk = cv2.resize(mask.astype(np.uint8), (wa.shape[1], wa.shape[0]), interpolation=cv2.INTER_NEAREST) > 0
    wa, wb = cv2.GaussianBlur(wa, (0, 0), 1.0), cv2.GaussianBlur(wb, (0, 0), 1.0)
    va, vb = wa[mk], wb[mk]
    if va.size < 100 or va.std() < 1e-3 or vb.std() < 1e-3:
        return None
    return float(np.corrcoef(va, vb)[0, 1])


def orb_verify(a: np.ndarray, b: np.ndarray, cache: OrbCache | None = None, key_a: Any = "a", key_b: Any = "b",
               ratio: float = 0.75) -> dict[str, Any]:
    """İki gri görüntü arasında geometrik doğrulama; normal ve yatay çevrilmiş b denenir, en iyi sonuç döner."""
    cache = cache or OrbCache()
    kpa, desa = cache.features(key_a, a)
    best: dict[str, Any] = {"good_matches": 0, "inliers": 0, "inlier_ratio": 0.0, "ncc": None, "flipped": False,
                            "scale": None, "rotation_deg": None, "spread": 0.0, "model": "", "anisotropy": None}
    if desa is None or len(kpa) < 8:
        return best
    for flipped in (False, True):
        kpb, desb = cache.features(key_b, b, flipped)
        if desb is None or len(kpb) < 8:
            continue
        bf = cv2.BFMatcher(cv2.NORM_HAMMING)
        good = []
        for pair in bf.knnMatch(desa, desb, k=2):
            if len(pair) == 2 and pair[0].distance < ratio * pair[1].distance:
                good.append(pair[0])
        if len(good) < 6:
            if len(good) > best["good_matches"] and best["inliers"] == 0:
                best.update(good_matches=len(good), flipped=flipped)
            continue
        pa = np.float32([kpa[m.queryIdx].pt for m in good])
        pb = np.float32([kpb[m.trainIdx].pt for m in good])
        # Benzerlik dönüşümü (ölçek+dönme) ve tam afin (anizotropik ölçek: Roboflow 'stretch' yeniden boyutlandırması) denenir
        for model, fn in (("similarity", cv2.estimateAffinePartial2D), ("affine", cv2.estimateAffine2D)):
            M, mask = fn(pa, pb, method=cv2.RANSAC, ransacReprojThreshold=4.0, maxIters=2000)
            if M is None:
                continue
            inl = int(mask.sum())
            if inl <= best["inliers"]:
                continue
            pts = pa[mask.ravel() == 1]
            spread = 0.0
            if len(pts) >= 3:
                spread = float(cv2.contourArea(cv2.convexHull(pts)) / (a.shape[0] * a.shape[1]))
            sx, sy = float(np.hypot(M[0, 0], M[1, 0])), float(np.hypot(M[0, 1], M[1, 1]))
            rot = float(np.degrees(np.arctan2(M[1, 0], M[0, 0])))
            bb = np.ascontiguousarray(b[:, ::-1]) if flipped else b
            n_ncc = _ncc_aligned(a, bb, M)
            best.update(good_matches=len(good), inliers=inl, inlier_ratio=round(inl / len(good), 3), flipped=flipped,
                        scale=round(sx, 3), rotation_deg=round(rot, 1), spread=round(spread, 3), model=model,
                        anisotropy=round(sx / sy, 3) if sy > 0 else None,
                        ncc=None if n_ncc is None else round(n_ncc, 3))
    return best


def decide(v: dict[str, Any], t: dict[str, float] = DEFAULT_THRESHOLDS) -> str:
    """Geometrik doğrulama sonucundan karar. (SHA-256 eşitliği çağıran tarafından exact_duplicate yapılır.)

    same_image_derived          : yüksek inlier sayısı/oranı + hizalı NCC yüksek (aynı görüntünün türevi / ardışık kare)
    same_scene_different_frame  : yüksek inlier + düşük hizalı NCC (aynı sahne, farklı kare veya ağır bozulma)
    partial_region_match        : çok sayıda inlier ama oran düşük (kolaj/mozaik: görüntünün yalnızca bir bölgesi ortak)
    uncertain                   : sınırda inlier sayısı; yalnızca görsel incelemeyle karar verilir
    not_duplicate               : geometrik kanıt yok (hash adayı yanlış pozitif)
    """
    inl = v["inliers"]
    # Hizalanmış NCC çok yüksekse (rastgele hizalamada görülmez) az sayıda inlier bile aynı görüntü kanıtıdır
    if inl >= t["uncertain_inliers"] and v["inlier_ratio"] >= 0.4 and v["ncc"] is not None and v["ncc"] >= 0.95:
        return "same_image_derived"
    if inl >= t["min_inliers"]:
        if v["inlier_ratio"] >= t["min_inlier_ratio"] and v["spread"] >= 0.02:
            if v["ncc"] is not None and v["ncc"] >= t["min_ncc"]:
                return "same_image_derived"
            return "same_scene_different_frame"
        return "partial_region_match"
    if inl >= t["uncertain_inliers"]:
        return "uncertain"
    return "not_duplicate"


def leakage_risk(decision: str, relation: str, split_a: str, split_b: str) -> str:
    """Mevcut bölünmelere göre sızıntı riski (kaynak eşleşmesi + bölüm ilişkisi)."""
    if decision in ("not_duplicate",):
        return "none"
    eval_involved = "val" in (split_a, split_b) or "test" in (split_a, split_b)
    cross_split = split_a != split_b
    strong = decision in ("exact_duplicate", "same_image_derived", "same_scene_different_frame", "partial_region_match")
    if relation == "within_dataset_same_split":
        return "low (aynı bölüm içi: etkin örnek sayısını şişirir)"
    if relation == "within_dataset_cross_split":
        if cross_split and eval_involved:
            return "HIGH (train/val/test sızıntısı)" if strong else "HIGH (aynı sahne/video, farklı bölüm)"
        return "medium"
    if relation == "cross_dataset":
        if eval_involved:
            return "HIGH (birleştirmede doğrulama/test sızıntısı)" if strong else "HIGH (birleştirmede aynı sahne)"
        return "medium (birleştirmede yinelenen eğitim örneği)" if strong else "low-medium"
    return "unknown"


# ---------------- Yöntem doğrulama deneyi ----------------

def augment(gray: np.ndarray, kind: str, rng: random.Random) -> np.ndarray:
    """Roboflow benzeri bozulmalar (kırpma <=%20, ±12° dönme, parlaklık ±%25, bulanıklık, JPEG, ölçek, çevirme)."""
    h, w = gray.shape
    img = gray.copy()
    if kind == "identity_resize50":
        return cv2.resize(img, (w // 2, h // 2), interpolation=cv2.INTER_AREA)
    if kind == "jpeg_q30":
        ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 30])
        return cv2.imdecode(buf, cv2.IMREAD_GRAYSCALE)
    if kind in ("brightness+25", "brightness-25"):
        f = 1.25 if kind.endswith("+25") else 0.75
        return np.clip(img.astype(np.float32) * f, 0, 255).astype(np.uint8)
    if kind == "blur":
        return cv2.GaussianBlur(img, (0, 0), 1.5)
    if kind == "flip":
        return np.ascontiguousarray(img[:, ::-1])
    if kind == "crop20":
        dx, dy = int(w * rng.uniform(0.05, 0.2)), int(h * rng.uniform(0.05, 0.2))
        x0, y0 = rng.randint(0, dx), rng.randint(0, dy)
        return img[y0:h - (dy - y0), x0:w - (dx - x0)]
    if kind == "rot12":
        ang = rng.choice([-12, -8, 8, 12])
        M = cv2.getRotationMatrix2D((w / 2, h / 2), ang, 1.0)
        return cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_CONSTANT)
    if kind == "stretch640":
        return cv2.resize(img, (640, 640), interpolation=cv2.INTER_AREA)  # Roboflow 'Resize to 640x640 (Stretch)'
    if kind == "combo":
        out = augment(img, "crop20", rng)
        oh, ow = out.shape
        M = cv2.getRotationMatrix2D((ow / 2, oh / 2), rng.choice([-10, 10]), 1.0)
        out = cv2.warpAffine(out, M, (ow, oh))
        out = np.clip(out.astype(np.float32) * rng.uniform(0.8, 1.2), 0, 255).astype(np.uint8)
        out = cv2.GaussianBlur(out, (0, 0), 1.0)
        return np.ascontiguousarray(out[:, ::-1])
    raise ValueError(kind)


AUGMENTATIONS = ("identity_resize50", "jpeg_q30", "brightness+25", "brightness-25", "blur", "flip", "crop20", "rot12",
                 "stretch640", "combo")


def validate_method(gray_images: list[np.ndarray], seed: int = 42, n_negative: int = 400,
                    thresholds: dict[str, float] = DEFAULT_THRESHOLDS) -> dict[str, Any]:
    """Pozitifler: aynı görüntünün bozulmuş sürümleri. Negatifler: farklı görüntü çiftleri.

    Ölçüler: hash adayı yakalama oranı, ORB doğrulama + karar doğruluğu, negatiflerde yanlış pozitif sayısı.
    """
    rng = random.Random(seed)
    cache = OrbCache()
    result: dict[str, Any] = {"n_base_images": len(gray_images), "thresholds": thresholds, "positives": {}, "negatives": {}}
    sigs = [signature(g) for g in gray_images]
    for kind in AUGMENTATIONS:
        cand = ok = derived = 0
        inl, ncc = [], []
        for i, g in enumerate(gray_images):
            aug = augment(g, kind, rng)
            aug_sig = signature(aug)
            dist, _ = hash_distance(sigs[i], aug_sig)
            cand += dist <= thresholds["candidate_hash_bits"]
            v = orb_verify(g, aug, cache, ("base", i), ("aug", kind, i))
            d = decide(v, thresholds)
            ok += d in ("same_image_derived", "same_scene_different_frame", "partial_region_match")
            derived += d == "same_image_derived"
            inl.append(v["inliers"])
            if v["ncc"] is not None:
                ncc.append(v["ncc"])
        n = len(gray_images)
        result["positives"][kind] = {
            "hash_candidate_rate": round(cand / n, 3), "orb_geometric_match_rate": round(ok / n, 3),
            "decided_same_image_derived_rate": round(derived / n, 3),
            "inliers_median": int(np.median(inl)), "inliers_p10": int(np.percentile(inl, 10)),
            "ncc_median": round(float(np.median(ncc)), 3) if ncc else None,
            "ncc_p10": round(float(np.percentile(ncc, 10)), 3) if ncc else None}
    n_img = len(gray_images)
    pairs = set()
    while len(pairs) < min(n_negative, n_img * (n_img - 1) // 2):
        i, j = rng.sample(range(n_img), 2)
        pairs.add((min(i, j), max(i, j)))
    fp_geo = fp_hash = 0
    max_inl = []
    fp_pairs: list[dict[str, int]] = []
    for i, j in pairs:
        dist, _ = hash_distance(sigs[i], sigs[j])
        fp_hash += dist <= thresholds["candidate_hash_bits"]
        v = orb_verify(gray_images[i], gray_images[j], cache, ("base", i), ("base", j))
        max_inl.append(v["inliers"])
        is_fp = decide(v, thresholds) in ("same_image_derived", "same_scene_different_frame", "partial_region_match")
        fp_geo += is_fp
        if is_fp:
            fp_pairs.append({"i": i, "j": j, "inliers": v["inliers"], "ncc": v["ncc"]})
    result["negatives"] = {"n_pairs": len(pairs), "hash_false_candidates": fp_hash, "geometric_false_positives": fp_geo,
                           "inliers_max": int(max(max_inl)), "inliers_p99": int(np.percentile(max_inl, 99)),
                           "inliers_median": int(np.median(max_inl)),
                           "geometric_matches_among_random_pairs": fp_pairs}
    return result


# ---------------- Yerel özellik tabanlı aday üretimi (kırpma/dönme dayanıklı) ----------------

INDEX_SIDE = 320
INDEX_FEATURES = 300


def index_descriptors(gray: np.ndarray) -> np.ndarray | None:
    """Aday indeksi için küçük bir ORB tanımlayıcı kümesi (N x 32 uint8)."""
    s = INDEX_SIDE / max(gray.shape[:2])
    small = cv2.resize(gray, None, fx=s, fy=s, interpolation=cv2.INTER_AREA) if s < 1 else gray
    _, des = cv2.ORB_create(nfeatures=INDEX_FEATURES, fastThreshold=10).detectAndCompute(small, feature_mask(small))
    return des


def lsh_candidate_votes(descs: list[np.ndarray | None], tables: int = 16, bits: int = 20, bucket_cap: int = 12,
                        seed: int = 1) -> dict[tuple[int, int], int]:
    """Bit-örnekleme LSH: birden çok tabloda aynı kovaya düşen ORB tanımlayıcıları görüntü çiftlerine OY verir.

    Kırpılmış/döndürülmüş/bozulmuş kopyalar çok sayıda ortak tanımlayıcı paylaştığı için yüksek oy alır; farklı görüntüler
    tesadüfen ancak 1-2 oy alır. Dönen sözlük: (i, j) -> oy sayısı (i < j).
    """
    rng = np.random.default_rng(seed)
    parts = [(i, d) for i, d in enumerate(descs) if d is not None and len(d)]
    if not parts:
        return {}
    ids = np.concatenate([np.full(len(d), i, dtype=np.int64) for i, d in parts])
    data = np.vstack([d for _, d in parts])
    n_img = len(descs)
    codes_all: list[np.ndarray] = []
    for _ in range(tables):
        positions = rng.choice(256, bits, replace=False)
        keys = np.zeros(len(data), dtype=np.uint32)
        for pos in positions:
            keys = (keys << 1) | ((data[:, pos // 8] >> (7 - pos % 8)) & 1).astype(np.uint32)
        order = np.argsort(keys, kind="stable")
        sk = keys[order]
        change = np.flatnonzero(np.diff(sk)) + 1
        starts = np.concatenate([[0], change])
        sizes = np.diff(np.concatenate([starts, [len(sk)]]))
        for s in range(2, bucket_cap + 1):
            sel = starts[sizes == s]
            if sel.size == 0:
                continue
            members = ids[order[sel[:, None] + np.arange(s)[None, :]]]  # G x s görüntü kimlikleri
            for a in range(s):
                for b in range(a + 1, s):
                    x, y = members[:, a], members[:, b]
                    keep = x != y
                    lo, hi = np.minimum(x[keep], y[keep]), np.maximum(x[keep], y[keep])
                    codes_all.append(lo * n_img + hi)
    if not codes_all:
        return {}
    uniq, counts = np.unique(np.concatenate(codes_all), return_counts=True)
    return {(int(c // n_img), int(c % n_img)): int(n) for c, n in zip(uniq, counts)}


def lsh_votes_with_flip(descs: list[np.ndarray | None], descs_flipped: list[np.ndarray | None], **kw
                        ) -> dict[tuple[int, int], tuple[int, int]]:
    """Normal ve yatay çevrilmiş tanımlayıcılarla LSH oyları. (i, j) -> (normal_oy, çevrilmiş_oy)."""
    n = len(descs)
    votes = lsh_candidate_votes(list(descs) + list(descs_flipped), **kw)
    out: dict[tuple[int, int], list[int]] = {}
    for (a, b), v in votes.items():
        if a < n and b < n:
            out.setdefault((a, b), [0, 0])[0] += v
        elif a < n <= b and a != b - n:
            i, j = min(a, b - n), max(a, b - n)
            out.setdefault((i, j), [0, 0])[1] += v
    return {k: (v[0], v[1]) for k, v in out.items()}


def apply_visual_checks(rows: list[dict[str, Any]], checks: dict[str, Any]) -> int:
    """configs/review/duplicate_visual_checks.yaml içindeki GÖRSEL inceleme kayıtlarını satırlara işler.

    Kayıtlar aday küme ve parametreler sabitken geçerli olan ``pair_id`` ile eşlenir; yol parçaları (a/b) da tutarlılık
    için kontrol edilir, uyuşmazsa kayıt uygulanmaz (yanlış çifte karar yazılmasın diye).
    """
    applied = 0
    by_id = {r["pair_id"]: r for r in rows}
    for pair_id, c in (checks.get("checks") or {}).items():
        r = by_id.get(pair_id)
        if r is None or c["a"] not in r["path_a"] or c["b"] not in r["path_b"]:
            continue
        r["visual_check"] = f"{c['result']}: {c['note']}"
        applied += 1
    return applied


# ---------------- Kolaj (mozaik) için karo düzeyinde doğrulama ----------------

def tile_windows(h: int, w: int) -> dict[str, tuple[int, int, int, int]]:
    """Kolaj görüntüsünde aranacak pencereler: tam, 4 çeyrek, 2 yarı (üst/alt/sol/sağ). (x, y, genişlik, yükseklik)."""
    wins = {"full": (0, 0, w, h)}
    for i, (x, y) in enumerate([(0, 0), (w // 2, 0), (0, h // 2), (w // 2, h // 2)]):
        wins[f"q{i}"] = (x, y, w // 2, h // 2)
    wins.update({"top": (0, 0, w, h // 2), "bottom": (0, h // 2, w, h // 2), "left": (0, 0, w // 2, h), "right": (w // 2, 0, w // 2, h)})
    return wins


def tile_level_verify(collage: np.ndarray, single: np.ndarray, cache: OrbCache | None = None, key: Any = "t") -> dict[str, Any]:
    """Bir kolajın pencerelerini tek görüntüyle karşılaştırır; en çok inlier veren pencereyi döndürür.

    Kolaj karelerinin farklı ölçek/konumda olması nedeniyle tam görüntü ORB'sinin kaçırdığı bölgesel eşleşmeleri yakalar.
    """
    cache = cache or OrbCache()
    h, w = collage.shape
    best: dict[str, Any] = {"window": "", "inliers": 0, "inlier_ratio": 0.0, "ncc": None}
    for name, (x, y, ww, hh) in tile_windows(h, w).items():
        v = orb_verify(collage[y:y + hh, x:x + ww], single, cache, (key, name), (key, "single"))
        if v["inliers"] > best["inliers"]:
            best = {"window": name, "inliers": v["inliers"], "inlier_ratio": v["inlier_ratio"], "ncc": v["ncc"]}
    return best


def tile_match(v: dict[str, Any], min_inliers: int = 40, min_ratio: float = 0.5) -> bool:
    return v["inliers"] >= min_inliers and v["inlier_ratio"] >= min_ratio
