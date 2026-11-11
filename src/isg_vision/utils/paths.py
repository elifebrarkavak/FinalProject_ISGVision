"""Proje dosya yollarının tek merkezde tanımı.

Yollar kullanıcının bilgisayarına sabitlenmez; proje kökü bu dosyanın
konumundan hesaplanır.
"""

from __future__ import annotations

from pathlib import Path

PROJECT_ROOT: Path = Path(__file__).resolve().parents[3]

REQUIRED_DIRS: tuple[str, ...] = (
    "data/raw/construction_ppe",
    "data/raw/mendeley_ppe",
    "data/raw/site_safety",
    "data/processed",
    "data/merged_ppe",
    "data/employees",
    "models/human_detection",
    "models/tracking",
    "models/face_detection",
    "models/face_recognition",
    "models/ppe_detection",
    "src/isg_vision/camera",
    "src/isg_vision/human_detection",
    "src/isg_vision/tracking",
    "src/isg_vision/face_detection",
    "src/isg_vision/face_recognition",
    "src/isg_vision/ppe_detection",
    "src/isg_vision/integration",
    "src/isg_vision/storage",
    "src/isg_vision/utils",
    "configs",
    "scripts",
    "tests",
    "results",
    "reports",
    "docs",
)

REQUIRED_FILES: tuple[str, ...] = (
    "requirements.txt",
    ".gitignore",
    "README.md",
    "docs/architecture.md",
    "configs/default.yaml",
    ".env.example",
    "scripts/check_mps.py",
    "scripts/check_environment.py",
    "scripts/git-hooks/pre-commit",
    ".vscode/settings.json",
    "configs/datasets.yaml",
    "configs/candidate_class_mapping.yaml",
    "scripts/extract_datasets.py",
    "scripts/inspect_datasets.py",
    "scripts/validate_labels.py",
    "scripts/visualize_annotations.py",
    "scripts/analyze_class_distribution.py",
    "scripts/detect_duplicates.py",
    "scripts/generate_dataset_report.py",
    "scripts/analyze_box_semantics.py",
    "scripts/review_roboflow_labels.py",
    "scripts/verify_duplicates.py",
    "scripts/analyze_augmentation_artifacts.py",
    "scripts/generate_followup_reports.py",
    "configs/review/label_review_verdicts.yaml",
    "configs/review/duplicate_visual_checks.yaml",
    "configs/review/license_evidence.yaml",
    "configs/review/checklist_resolution.yaml",
    "configs/review/stage2b_decisions.yaml",
    "configs/dataset_usage_policy.yaml",
    "scripts/build_source_groups.py",
    "scripts/apply_usage_policy.py",
    "scripts/generate_stage2b_reports.py",
    "docs/dataset_attributions.md",
)

CONFIG_FILES: tuple[str, ...] = ("configs/default.yaml", ".env.example")


def project_path(*parts: str) -> Path:
    """Proje köküne göre göreli bir yol döndürür."""
    return PROJECT_ROOT.joinpath(*parts)
