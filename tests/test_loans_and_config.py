"""Parsing sentinels, and the config validation that refuses a meaningless backtest."""

import numpy as np
import pandas as pd
import pytest

from src.config import load_config
from src.loans import load_origination, period_add, data_as_of


def test_period_arithmetic_crosses_the_year_boundary():
    assert period_add(200511, 1) == 200512
    assert period_add(200511, 2) == 200601
    assert period_add(200501, 24) == 200701


def test_data_as_of_is_the_latest_period_present():
    frame = pd.DataFrame({"period": pd.array([200501, 200712, 200606], dtype="Int64")})
    assert data_as_of(frame) == 200712


def test_sentinels_become_missing_rather_than_the_top_of_the_range(tmp_path):
    """A credit score of 9999 read naively is the best applicant in the book."""
    row = "L1|200501|9999|80|80|999|200000|6.0|360|P|P|R|SF|1|2|N|0|CA"
    (tmp_path / "origination_2005.txt").write_text(row + "\n", encoding="utf-8")
    frame = load_origination(tmp_path, "origination_*.txt")
    assert np.isnan(frame.loc[0, "credit_score"])
    assert np.isnan(frame.loc[0, "dti"])
    assert frame.loc[0, "vintage"] == 2005


def test_duplicate_loan_ids_are_refused(tmp_path):
    row = "L1|200501|700|80|80|35|200000|6.0|360|P|P|R|SF|1|2|N|0|CA"
    (tmp_path / "origination_2005.txt").write_text(f"{row}\n{row}\n", encoding="utf-8")
    with pytest.raises(ValueError, match="duplicate loan ids"):
        load_origination(tmp_path, "origination_*.txt")


def test_a_missing_extract_says_what_to_do_about_it(tmp_path):
    with pytest.raises(FileNotFoundError, match="make fixture"):
        load_origination(tmp_path, "origination_*.txt")


def test_the_real_config_loads_and_validates():
    config = load_config()
    assert set(config.vintages.fit).issubset(set(config.vintages.backtest))
    assert config.target.performance_window_months > 0


def test_a_fit_vintage_outside_the_backtest_set_is_refused(tmp_path):
    import yaml
    from src.config import load_config as loader

    source = load_config()
    raw = yaml.safe_load((source.root / "config" / "config.yaml").read_text())
    raw["vintages"]["fit"] = [1999]
    directory = tmp_path / "config"
    directory.mkdir()
    (directory / "config.yaml").write_text(yaml.safe_dump(raw), encoding="utf-8")
    with pytest.raises(ValueError, match="not in the backtest set"):
        loader(directory / "config.yaml")


def test_a_zero_balance_code_cannot_be_both_a_loss_and_a_prepayment(tmp_path):
    import yaml
    from src.config import load_config as loader

    source = load_config()
    raw = yaml.safe_load((source.root / "config" / "config.yaml").read_text())
    raw["target"]["prepaid_zero_balance_codes"] = ["01", "03"]
    directory = tmp_path / "config"
    directory.mkdir()
    (directory / "config.yaml").write_text(yaml.safe_dump(raw), encoding="utf-8")
    with pytest.raises(ValueError, match="both a credit loss and a voluntary prepayment"):
        loader(directory / "config.yaml")


def test_a_config_outside_the_documented_layout_is_refused_not_guessed(tmp_path):
    import yaml
    from src.config import load_config as loader

    source = load_config()
    raw = yaml.safe_load((source.root / "config" / "config.yaml").read_text())
    stray = tmp_path / "somewhere.yaml"
    stray.write_text(yaml.safe_dump(raw), encoding="utf-8")
    with pytest.raises(ValueError, match="cannot derive the project root"):
        loader(stray)
