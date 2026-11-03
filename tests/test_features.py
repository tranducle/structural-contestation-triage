from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

SRC_ROOT = Path(__file__).resolve().parents[1] / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from features import LMD_PRIMARY_FEATURES, build_salad_features, lmd_label_name


def test_lmd_label_semantics_match_source_documentation() -> None:
    assert lmd_label_name(0) == "Normal"
    assert lmd_label_name(1) == "EoRS"
    assert lmd_label_name(2) == "EoHT"


def test_primary_lmd_features_exclude_station_and_calendar_identity() -> None:
    forbidden = ("compsta", "station", "year", "month", "week", "day", "hour", "minute")
    lowered = [name.lower() for name in LMD_PRIMARY_FEATURES]
    assert lowered
    assert all(not any(token in name for token in forbidden) for name in lowered)


def test_salad_text_proxy_contains_no_label_bearing_terms() -> None:
    frame = pd.DataFrame(
        {
            "alert_id": ["a1"],
            "timestamp": ["2017-01-01T00:00:00"],
            "source_dataset": ["CICIDS2017"],
            "src_ip": ["10.0.0.1"],
            "dst_ip": ["192.168.1.5"],
            "src_port": [50000],
            "dst_port": [445],
            "protocol": ["tcp"],
            "flow_duration": [1.2],
            "total_fwd_packets": [10],
            "total_bwd_packets": [8],
            "flow_bytes_per_sec": [2500.0],
            "network_segment": ["workstation"],
            "is_malicious": [True],
            "attack_category": ["Lateral Movement"],
            "triage_decision": ["escalate"],
            "mitre_tactic": ["Lateral Movement"],
            "alert_description": ["MITRE attack detected"],
            "alert_count_1h": [3],
            "alert_count_24h": [12],
            "is_repeated_target": [True],
        }
    )
    engineered = build_salad_features(frame)
    text = engineered.loc[0, "text_proxy"].lower()
    for forbidden in ["malicious", "benign", "mitre", "escalate", "suppress", "attack_category"]:
        assert forbidden not in text
    assert "tcp" in text
    assert "service" in text
