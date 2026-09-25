import json
import math
from pathlib import Path
import sys


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from _paper_batch import weighted_metrics


def test_paper_manifests_have_reported_sequence_counts():
    root = Path(__file__).resolve().parents[1] / "configs/datasets"
    aeva = json.loads((root / "aevascenes_paper_sequences.json").read_text())
    hercules = json.loads((root / "hercules_paper_sequences.json").read_text())
    assert len(aeva["highway"]) == 44
    assert len(aeva["city"]) == 43
    assert len(set(aeva["highway"] + aeva["city"])) == 87
    assert set(hercules) == {
        "bridge",
        "library",
        "parking_lot",
        "river_island",
        "stream",
        "street",
    }


def test_weighted_metrics_uses_frame_pair_counts():
    records = [
        {
            "method": {
                "pairs": 1,
                "rte_mean_m": 1.0,
                "rte_rmse_m": 1.0,
                "rre_mean_deg": 2.0,
                "rre_rmse_deg": 2.0,
            }
        },
        {
            "method": {
                "pairs": 3,
                "rte_mean_m": 3.0,
                "rte_rmse_m": 3.0,
                "rre_mean_deg": 4.0,
                "rre_rmse_deg": 4.0,
            }
        },
    ]
    result = weighted_metrics(records)["method"]
    assert result["pairs"] == 4
    assert result["rte_mean_m"] == 2.5
    assert result["rte_rmse_m"] == math.sqrt(7.0)
    assert result["rre_mean_deg"] == 3.5
    assert result["rre_rmse_deg"] == math.sqrt(13.0)
    assert result["rte_rmse_sequence_mean_m"] == 2.0
    assert result["rre_rmse_sequence_mean_deg"] == 3.0
