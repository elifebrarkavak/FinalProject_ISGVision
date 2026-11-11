"""Kaynak grubu (veri sızıntısı) çıkarımı: eşleşme grafiği -> bağlantılı bileşenler -> bölme birimleri.

- SERT kenar: kanıtlı aynı kaynak (aynı görüntü türevi / aynı sahne / bölgesel-kolaj eşleşmesi / karo eşleşmesi / görsel doğrulama).
- YUMUŞAK kenar: kanıt sınırda (uncertain) veya "Likely Same Source" görsel kararı. Yumuşak kenarlar bölme birimini
  birleştirir (muhafazakâr: birlikte tutulur) ama sert grup sayılmaz.
Dosya adı öneklerinden kesin kaynak eşleşmesi ÜRETİLMEZ; ad yalnızca aday üretiminde kullanıldı ve doğrulama gerektirdi.
"""

from __future__ import annotations

import random
from collections import defaultdict
from typing import Any, Iterable

STRONG_DECISIONS = ("same_image_derived", "same_scene_different_frame", "partial_region_match", "exact_duplicate")
SOFT_DECISIONS = ("uncertain",)


class UnionFind:
    def __init__(self) -> None:
        self.parent: dict[str, str] = {}

    def find(self, x: str) -> str:
        self.parent.setdefault(x, x)
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a: str, b: str) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[max(ra, rb)] = min(ra, rb)  # deterministik: en küçük yol kök olur


def build_groups(images: Iterable[str], hard_edges: Iterable[tuple[str, str]], soft_edges: Iterable[tuple[str, str]]
                 ) -> dict[str, dict[str, Any]]:
    """images -> {hard_group, split_unit, hard_size, unit_size}. Tek başına kalan görüntü kendi grubudur (singleton)."""
    imgs = sorted(set(images))
    hard, unit = UnionFind(), UnionFind()
    for p in imgs:
        hard.find(p)
        unit.find(p)
    for a, b in hard_edges:
        hard.union(a, b)
        unit.union(a, b)
    for a, b in soft_edges:
        unit.union(a, b)
    hard_members: dict[str, list[str]] = defaultdict(list)
    unit_members: dict[str, list[str]] = defaultdict(list)
    for p in imgs:
        hard_members[hard.find(p)].append(p)
        unit_members[unit.find(p)].append(p)
    ids_h = {root: f"HG{i:05d}" for i, root in enumerate(sorted(hard_members))}
    ids_u = {root: f"SU{i:05d}" for i, root in enumerate(sorted(unit_members))}
    return {p: {"hard_group": ids_h[hard.find(p)], "hard_size": len(hard_members[hard.find(p)]),
                "split_unit": ids_u[unit.find(p)], "unit_size": len(unit_members[unit.find(p)])} for p in imgs}


def assign_unit_splits(units: dict[str, int], ratios: dict[str, float], seed: int = 42) -> dict[str, str]:
    """Bölme birimlerini (birim -> görüntü sayısı) hedef oranlara göre deterministik olarak train/val/test'e atar.

    Büyük birimler önce yerleştirilir; her birim, hedefe göre en çok açığı olan bölüme gider. Bir birim ASLA bölünmez.
    Bu bir ÖNİZLEME algoritmasıdır; mevcut bölünmeleri değiştirmez.
    """
    total = sum(units.values())
    target = {s: r * total for s, r in ratios.items()}
    filled = {s: 0 for s in ratios}
    rng = random.Random(seed)
    order = sorted(units, key=lambda u: (-units[u], rng.random()))
    out: dict[str, str] = {}
    for u in order:
        split = max(ratios, key=lambda s: (target[s] - filled[s]) / max(1.0, target[s]))
        out[u] = split
        filled[split] += units[u]
    return out
