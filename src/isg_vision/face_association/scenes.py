"""Yüz–kişi ilişkilendirme için SENTETİK sahne üretici (test ve ölçüm içindir; çalışma zamanında kullanılmaz).

**Bu bir MONTAJDIR, gerçek çoklu-kişi videosu DEĞİLDİR.** Ultralytics paketinin örnek fotoğraflarından (``bus.jpg``, ``zidane.jpg``; indirme yok, ``ultralytics``
İÇE AKTARILMADAN bulunur) kırpılan ÜÇ gerçek kişi (``A``, ``B``, ``C``), gri bir tuval üzerinde bilinen konumlarda hareket ettirilir. Her kare için gerçek
(ground-truth) kişi ve yüz kutuları bilinir; bu yüzden ilişkilendirmenin DOĞRULUĞU (yanlış eşleşme var mı?) ölçülebilir. Sınırlar:

* kişiler tek bir fotoğraf karesinden kırpılmıştır (poz/ışık/arka plan çeşitliliği YOK, hareket yalnızca kaydırmadır; yürüme/dönme/ölçek değişimi yok);
* aynı yüz kareden kareye PİKSEL OLARAK aynıdır (gerçek video gürültüsü, hareket bulanıklığı ve poz değişimi yoktur);
* "arkasını dönme" / "yüz görünmüyor" yüz bölgesinin koyu bir dolguyla kapatılmasıyla TAKLİT edilir (gerçek bir arkası dönük kişi değildir);
* "poster/fotoğraf" bir yüz kırpıntısının veya küçültülmüş kişi kırpıntısının tuvale yerleştirilmesidir (gerçek poster/basılı kâğıt/telefon ekranı değildir).

Yani bu sahneler, algoritmanın geometrik ve süreklilik mantığını GERÇEK YOLO + GERÇEK ByteTrack + GERÇEK YuNet ile deterministik sınamak içindir;
gerçek kamerada/gerçek çoklu-kişi videosunda başarının KANITI DEĞİLDİR. Gerçek farklı kişilerin hareket ettiği bir video ile aynı yüzün çoğaltıldığı bir
montaj eşdeğer sayılmaz (``docs/face_association.md``).

Kaynak kutuları (``ACTOR_SOURCES``) gerçek YOLO11n ve YuNet çıktılarından alınmış ve görüntüde GÖZLE doğrulanmıştır (4A/5A testlerindeki referans kutular
gibi); bağımsız etiket değildir.
"""

from __future__ import annotations

import importlib.util
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

CANVAS = (1280, 720)  # (genişlik, yükseklik)
BACKGROUND = 100
FACE_MASK_COLOR = (38, 32, 30)  # saç/kapüşon benzeri koyu dolgu ("yüz görünmüyor" taklidi)
MARGIN_PX = 6
PHOTO_MARGIN = 0.5

# (kaynak görüntü, kişi kutusu, yüz kutusu): bus.jpg 810x1080, zidane.jpg 1280x720; gerçek modellerin çıktısı, gözle doğrulandı.
ACTOR_SOURCES: dict[str, tuple[str, tuple[int, int, int, int], tuple[int, int, int, int]]] = {
    "A": ("bus", (47, 400, 239, 904), (113, 416, 155, 470)),  # krem montlu kişi
    "B": ("bus", (223, 409, 344, 860), (268, 423, 306, 473)),  # koyu ceketli kişi
    "C": ("zidane", (748, 42, 1148, 711), (917, 99, 1053, 273)),  # takım elbiseli kişi
}
DEFAULT_HEIGHT = {"A": 470, "B": 470, "C": 480}  # tuvalde kişi kutusu yüksekliği (px)

Box = tuple[float, float, float, float]


class SceneError(RuntimeError):
    """Örnek görüntüler bulunamadı/okunamadı."""


@dataclass(frozen=True)
class Actor:
    """Bir kaynak kişi: kırpıntı ve kırpıntı koordinatlarında kişi/yüz kutuları."""

    name: str
    crop: np.ndarray
    person_box: Box
    face_box: Box


@dataclass(frozen=True)
class Placement:
    """Bir karede bir kişinin yerleşimi.

    Attributes:
        name: Kişi adı (``A``/``B``/``C``) veya ``photo:<ad>`` / ``poster:<ad>`` (yüz kırpıntısı / küçültülmüş kişi kırpıntısı).
        x, y: Kişi KUTUSUNUN tuvaldeki sol-üst köşesi (negatif/taşan değerler kadraj dışına çıkma demektir).
        height: Kişi kutusunun tuvaldeki yüksekliği (px).
        face_visible: ``False`` ise yüz bölgesi koyu dolguyla kapatılır (arkasını dönme/yüz görünmüyor TAKLİDİ).
    """

    name: str
    x: float
    y: float
    height: float | None = None
    face_visible: bool = True


@dataclass(frozen=True)
class ActorTruth:
    """Bir karede bir kişinin gerçek durumu (render sonrası hesaplanır).

    Attributes:
        person_box: Kişi kutusu (kadraja kırpılmış). visible_fraction: Kutunun, sonradan çizilen başka bir kişi/nesne tarafından KAPANMAMIŞ ve kadraj içindeki oranı.
        face_box: Yüz kutusu; yüz maskelendiyse, kadraj dışındaysa veya ≥ %50 kapalıysa ``None`` (GÖRÜNMEZ).
    """

    person_box: Box
    visible_fraction: float
    face_box: Box | None


@dataclass
class SceneFrame:
    image: np.ndarray
    truth: dict[str, ActorTruth] = field(default_factory=dict)  # yalnızca kişiler (poster/photo yok)
    decoys: dict[str, Box] = field(default_factory=dict)  # photo/poster kırpıntılarının kutuları (yüz/kişi değil; "sahte yüz" kaynağı)


def _assets_dir() -> Path:
    spec = importlib.util.find_spec("ultralytics")  # ultralytics İÇE AKTARILMAZ (OpenCV iş parçacığı etkisi)
    if spec is None or spec.origin is None:
        raise SceneError("ultralytics kurulu değil; paketle gelen örnek görüntüler bulunamadı.")
    return Path(spec.origin).parent / "assets"


def load_actors(names: Sequence[str] = ("A", "B", "C")) -> dict[str, Actor]:
    """Örnek görüntülerden kişi kırpıntılarını yükler (indirme yok)."""
    assets = _assets_dir()
    images: dict[str, np.ndarray] = {}
    actors: dict[str, Actor] = {}
    for name in names:
        source, person, face = ACTOR_SOURCES[name]
        if source not in images:
            image = cv2.imread(str(assets / f"{source}.jpg"))
            if image is None:
                raise SceneError(f"Örnek görüntü okunamadı: {assets / (source + '.jpg')}")
            images[source] = image
        image = images[source]
        h, w = image.shape[:2]
        x1, y1 = max(0, person[0] - MARGIN_PX), max(0, person[1] - MARGIN_PX)
        x2, y2 = min(w, person[2] + MARGIN_PX), min(h, person[3] + MARGIN_PX)
        crop = image[y1:y2, x1:x2].copy()
        actors[name] = Actor(
            name, crop,
            (person[0] - x1, person[1] - y1, person[2] - x1, person[3] - y1),
            (face[0] - x1, face[1] - y1, face[2] - x1, face[3] - y1),
        )
    return actors


def _scaled(box: Box, scale: float, ox: float, oy: float) -> Box:
    return (box[0] * scale + ox, box[1] * scale + oy, box[2] * scale + ox, box[3] * scale + oy)


def _clip(box: Box, width: int, height: int) -> Box | None:
    x1, y1, x2, y2 = max(0.0, box[0]), max(0.0, box[1]), min(float(width), box[2]), min(float(height), box[3])
    return (x1, y1, x2, y2) if x2 - x1 >= 1.0 and y2 - y1 >= 1.0 else None


def render_frame(actors: dict[str, Actor], placements: Sequence[Placement], canvas_size: tuple[int, int] = CANVAS) -> SceneFrame:
    """Yerleşimleri SIRAYLA (sonra gelen öndedir) tuvale çizer ve gerçek (ground-truth) kutuları hesaplar."""
    width, height = canvas_size
    canvas = np.full((height, width, 3), BACKGROUND, np.uint8)
    drawn: list[tuple[str, Box, Box | None, tuple[int, int, int, int], str]] = []  # (ad, kişi kutusu, yüz kutusu, ayak izi, tür)
    for placement in placements:
        kind, _, base = placement.name.partition(":")
        name = base if kind in ("photo", "poster") else placement.name
        actor = actors[name]
        ph = actor.person_box[3] - actor.person_box[1]
        target = placement.height or DEFAULT_HEIGHT.get(name, 400)
        scale = target / ph
        crop = actor.crop
        if kind == "photo":  # yalnızca yüz (+ bağlam) kırpıntısı: bedeni olmayan fotoğraf/poster yüzü
            fx1, fy1, fx2, fy2 = actor.face_box
            mx, my = (fx2 - fx1) * PHOTO_MARGIN, (fy2 - fy1) * PHOTO_MARGIN
            cx1, cy1 = max(0, int(fx1 - mx)), max(0, int(fy1 - my))
            cx2, cy2 = min(crop.shape[1], int(fx2 + mx)), min(crop.shape[0], int(fy2 + my))
            patch = crop[cy1:cy2, cx1:cx2]
            scale = placement.height / (fy2 - fy1) if placement.height else scale
            origin_box = (fx1 - cx1, fy1 - cy1, fx2 - cx1, fy2 - cy1)
            person_box = face_box = origin_box
        else:
            patch = crop.copy()
            person_box, face_box = actor.person_box, actor.face_box
            if not placement.face_visible:
                fx1, fy1, fx2, fy2 = (int(v) for v in face_box)
                px, py = int((fx2 - fx1) * 0.25), int((fy2 - fy1) * 0.25)
                patch[max(0, fy1 - py) : fy2 + py, max(0, fx1 - px) : fx2 + px] = FACE_MASK_COLOR
                patch = cv2.GaussianBlur(patch, (0, 0), 1.2)
        new_w, new_h = max(1, int(round(patch.shape[1] * scale))), max(1, int(round(patch.shape[0] * scale)))
        resized = cv2.resize(patch, (new_w, new_h), interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_CUBIC)
        # placement.x/y = kişi kutusunun sol-üstü -> kırpıntının sol-üstü
        ox, oy = placement.x - person_box[0] * scale, placement.y - person_box[1] * scale
        ix, iy = int(round(ox)), int(round(oy))
        x1, y1, x2, y2 = max(0, ix), max(0, iy), min(width, ix + new_w), min(height, iy + new_h)
        if x2 > x1 and y2 > y1:
            canvas[y1:y2, x1:x2] = resized[y1 - iy : y2 - iy, x1 - ix : x2 - ix]
        pbox = _scaled(person_box, scale, ix, iy)
        fbox = _scaled(face_box, scale, ix, iy) if placement.face_visible or kind == "photo" else None
        drawn.append((placement.name, pbox, fbox, (x1, y1, x2, y2), kind))

    truth: dict[str, ActorTruth] = {}
    decoys: dict[str, Box] = {}
    for index, (name, pbox, fbox, _, kind) in enumerate(drawn):
        later = [fp for (_, _, _, fp, _) in drawn[index + 1 :]]
        if kind in ("photo", "poster"):
            clipped = _clip(pbox, width, height)
            if clipped is not None:
                decoys[name] = clipped
            continue
        clipped = _clip(pbox, width, height)
        if clipped is None:
            continue
        visible = _visible_fraction(clipped, later)
        face = None
        if fbox is not None:
            fclip = _clip(fbox, width, height)
            full = (fbox[2] - fbox[0]) * (fbox[3] - fbox[1])
            if fclip is not None and (fclip[2] - fclip[0]) * (fclip[3] - fclip[1]) >= 0.5 * full and _visible_fraction(fclip, later) >= 0.5:
                face = fclip
        truth[name] = ActorTruth(clipped, visible, face)
    return SceneFrame(canvas, truth, decoys)


def _visible_fraction(box: Box, covers: Sequence[tuple[int, int, int, int]]) -> float:
    x1, y1, x2, y2 = int(box[0]), int(box[1]), max(int(box[0]) + 1, int(box[2])), max(int(box[1]) + 1, int(box[3]))
    mask = np.zeros((y2 - y1, x2 - x1), dtype=bool)
    for cx1, cy1, cx2, cy2 in covers:
        ix1, iy1, ix2, iy2 = max(x1, cx1), max(y1, cy1), min(x2, cx2), min(y2, cy2)
        if ix2 > ix1 and iy2 > iy1:
            mask[iy1 - y1 : iy2 - y1, ix1 - x1 : ix2 - x1] = True
    return 1.0 - float(mask.mean())


def _lerp(a: float, b: float, i: int, n: int) -> float:
    return a + (b - a) * (i / (n - 1) if n > 1 else 0.0)


# ---------------------------------------------------------------------- senaryolar (her biri kare başına yerleşim listeleri döndürür)
Scenario = list[list[Placement]]


def single_person(n: int = 40) -> Scenario:
    """T1: tek kişi (A), hafif sağa kayar."""
    return [[Placement("A", _lerp(520, 560, i, n), 130)] for i in range(n)]


def side_by_side(n: int = 40) -> Scenario:
    """T2: iki kişi yan yana (A, B), örtüşme yok."""
    return [[Placement("A", _lerp(320, 340, i, n), 130), Placement("B", _lerp(760, 740, i, n), 130)] for i in range(n)]


def three_people(n: int = 40) -> Scenario:
    """T3: üç kişi aynı anda (A, B, C), örtüşme yok."""
    return [[Placement("A", _lerp(80, 100, i, n), 130), Placement("B", _lerp(500, 490, i, n), 130), Placement("C", _lerp(760, 740, i, n), 120)] for i in range(n)]


def swap_places(n: int = 60) -> Scenario:
    """T4: iki kişi yer değiştirir: A soldan sağa, B sağdan sola; A daha yakın (daha aşağıda, üstte çizilir). Ortada kutular büyük ölçüde örtüşür."""
    out: Scenario = []
    for i in range(n):
        out.append([
            Placement("B", _lerp(820, 280, i, n), 150, 400),
            Placement("A", _lerp(280, 820, i, n), 200, 480),
        ])
    return out


def pass_in_front(n: int = 60) -> Scenario:
    """T5: B sabit durur; A önünden geçer ve bir süre B'yi (yüzü dahil) tamamen örter."""
    return [[Placement("B", 560, 140, 440), Placement("A", _lerp(60, 1000, i, n), 100, 520)] for i in range(n)]


def leave_and_return(n: int = 80, absent: int = 25) -> Scenario:
    """T6: A sağ kenardan kadrajdan çıkar, ``absent`` kare yok kalır, aynı yerden geri girer; B sabit."""
    out: Scenario = []
    leave_start, leave_end = n // 5, n // 5 + 12
    back_start = leave_end + absent
    for i in range(n):
        if i < leave_start:
            x = 700.0
        elif i < leave_end:
            x = _lerp(700, 1320, i - leave_start, leave_end - leave_start)
        elif i < back_start:
            x = 1400.0  # tamamen kadraj dışı
        elif i < back_start + 12:
            x = _lerp(1320, 700, i - back_start, 12)
        else:
            x = 700.0
        out.append([Placement("B", 200, 130), Placement("A", x, 130)])
    return out


def turn_away(n: int = 60) -> Scenario:
    """T7: A'nın yüzü ortadaki üçte birde görünmez (arkasını dönme TAKLİDİ); beden hareket etmeye devam eder; B sabit."""
    lo, hi = n // 3, 2 * n // 3
    return [[Placement("B", 220, 130), Placement("A", _lerp(640, 760, i, n), 130, face_visible=not lo <= i < hi)] for i in range(n)]


def no_visible_face(n: int = 40) -> Scenario:
    """T8: A'nın yüzü hiç görünmez (yüz maskeli); B'nin yüzü görünür."""
    return [[Placement("B", 260, 130), Placement("A", _lerp(640, 660, i, n), 130, face_visible=False)] for i in range(n)]


def photo_decoys(n: int = 40) -> Scenario:
    """T9: gerçek kişi B + arka planda bedeni olmayan bir yüz FOTOĞRAFI (A'nın yüzü) ve küçültülmüş bir kişi "POSTERİ" (C)."""
    return [[Placement("poster:C", 60, 150, 260), Placement("photo:A", 980, 120, 150), Placement("B", _lerp(500, 520, i, n), 130)] for i in range(n)]


def photo_held_up(n: int = 40, fraction: float = 0.45, face_px: float = 70.0) -> Scenario:
    """T9b: B bir fotoğrafı (A'nın yüzü) göğsünün önünde tutuyor: B'nin kutusunun içinde, kendi yüzünün altında ikinci bir yüz.

    ``fraction``: fotoğraf yüzünün üst kenarının B kutusu içindeki dikey konumu (0 = baş hizası, 1 = ayak hizası)."""
    return [[Placement("B", 520, 130, 470), Placement("photo:A", 540, 130 + fraction * 470, face_px)] for _ in range(n)]


SCENARIOS = {
    "single_person": single_person,
    "side_by_side": side_by_side,
    "three_people": three_people,
    "swap_places": swap_places,
    "pass_in_front": pass_in_front,
    "leave_and_return": leave_and_return,
    "turn_away": turn_away,
    "no_visible_face": no_visible_face,
    "photo_decoys": photo_decoys,
    "photo_held_up": photo_held_up,
}


def render_scenario(actors: dict[str, Actor], scenario: Scenario) -> list[SceneFrame]:
    return [render_frame(actors, placements) for placements in scenario]
