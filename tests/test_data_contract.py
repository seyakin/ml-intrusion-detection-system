import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

from ids_pipeline import NUMERIC_FEATURES, _normalise_columns, _validate_dataset


def _minimal_frame() -> pd.DataFrame:
    values = {column: [0, 1] for column in NUMERIC_FEATURES}
    values.update({"proto": ["tcp", "udp"], "state": ["FIN", "CON"], "service": ["http", "dns"]})
    values["label"] = [0, 1]
    return pd.DataFrame(values)


def test_historical_column_aliases_are_canonicalised() -> None:
    frame = _minimal_frame().rename(
        columns={"smean": "smeansz", "dmean": "dmeansz", "response_body_len": "res_bdy_len"}
    )
    normalised = _normalise_columns(frame)
    assert {"smean", "dmean", "response_body_len"}.issubset(normalised.columns)


def test_validation_rejects_non_binary_labels() -> None:
    frame = _minimal_frame()
    frame.loc[0, "label"] = 2
    with pytest.raises(ValueError, match="binary label"):
        _validate_dataset(frame, source="fixture")


def test_validation_checks_expected_row_count() -> None:
    with pytest.raises(ValueError, match="expected 3"):
        _validate_dataset(_minimal_frame(), source="fixture", expected_rows=3)
