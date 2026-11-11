"""Aşama 2: İSG veri setlerinin analizi, etiket doğrulaması ve raporlanması.

Bu paket yalnızca OKUMA yapar (ham veriyi değiştirmez, birleştirmez, yeniden etiketlemez).
Çıktılar ``reports/dataset_audit/`` altına yazılır.
"""

from pathlib import Path

from isg_vision.utils.paths import PROJECT_ROOT

AUDIT_DIR: Path = PROJECT_ROOT / "reports" / "dataset_audit"
