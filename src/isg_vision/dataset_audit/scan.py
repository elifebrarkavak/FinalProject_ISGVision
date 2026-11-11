"""Bir veri setini örnek örnek (akış halinde) tarar: görüntü bilgisi + etiket doğrulama.

Tüm betikler bu ortak taramayı kullanır; böylece doğrulama mantığı tek yerde kalır ve
bellekte aynı anda yalnızca bir görüntü/etiket dosyası bulunur.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterator

from isg_vision.dataset_audit.imageio import ImageInfo, read_image_info
from isg_vision.dataset_audit.labels import Issue, LabelResult, conflicting_overlaps, parse_label_file
from isg_vision.dataset_audit.layout import DatasetLayout, Sample, iter_samples, SPLIT_ORDER


@dataclass
class Record:
    sample: Sample
    image: ImageInfo | None
    label: LabelResult | None
    issues: list[Issue]
    stem_collision: bool = False


def scan_dataset(
    layout: DatasetLayout,
    *,
    full_decode: bool = False,
    read_labels: bool = True,
    conflict_pairs: list[tuple[int, int]] | None = None,
    progress: Callable[[str, int], None] | None = None,
) -> Iterator[Record]:
    """Düzendeki tüm bölümleri sırayla tarar ve her örnek için Record üretir."""
    valid_ids = set(layout.classes.names) if layout.classes.names else None
    count = 0
    for split in SPLIT_ORDER:
        sd = layout.splits.get(split)
        if sd is None:
            continue
        previous_stem = None
        for sample in iter_samples(sd):
            issues: list[Issue] = []
            image_info = None
            label_result = None
            collision = previous_stem == sample.stem and sample.image_path is not None
            previous_stem = sample.stem
            if collision:
                issues.append(Issue("STEM_COLLISION", None, "Aynı dosya adına sahip birden çok görüntü (farklı uzantı)"))

            if sample.image_path is not None:
                image_info = read_image_info(sample.image_path, full_decode=full_decode)
                if not image_info.readable:
                    issues.append(Issue("IMAGE_UNREADABLE", None, image_info.error))
                elif image_info.exif_orientation != 1:
                    issues.append(Issue("EXIF_ORIENTATION", None,
                                        f"EXIF yönü={image_info.exif_orientation}; etiketler döndürülmüş görüntüye göre mi kontrol edilmeli"))
            else:
                issues.append(Issue("ORPHAN_LABEL", None, "Etiket var ama aynı adlı görüntü yok"))

            if sample.label_path is None:
                issues.append(Issue("LABEL_MISSING", None, "Etiket dosyası yok (kasıtlı negatif örnek olabilir)"))
            elif read_labels:
                size = (image_info.width, image_info.height) if image_info and image_info.readable else None
                label_result = parse_label_file(sample.label_path, valid_ids, size)
                issues.extend(label_result.issues)
                if conflict_pairs:
                    issues.extend(conflicting_overlaps(label_result.boxes, conflict_pairs))
            count += 1
            if progress and count % 500 == 0:
                progress(layout.name, count)
            yield Record(sample, image_info, label_result, issues, collision)
