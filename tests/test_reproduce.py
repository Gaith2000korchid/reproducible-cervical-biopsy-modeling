"""Check that reruns cannot overwrite existing published evidence."""

from pathlib import Path

import pytest

from cervical_biopsy_modeling.reproduce import reproduce


def test_existing_output_is_rejected_before_any_write(tmp_path):
    output = tmp_path / "existing"
    output.mkdir()
    protected = output / "performance_summary.csv"
    protected.write_text("published evidence\n")
    source = Path(__file__).resolve().parents[1]
    with pytest.raises(ValueError, match="not empty"):
        reproduce(source, output, "full")
    assert protected.read_text() == "published evidence\n"
    assert list(output.iterdir()) == [protected]
