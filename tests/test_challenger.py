"""The information lag. This is the guard that keeps the backtest honest."""

import pytest

from src.config import load_config
from src.challenger import available_training_vintages, oracle_training_vintages, outcome_known_by

CONFIG = load_config()
ALL = [2004, 2005, 2006, 2007, 2008, 2015, 2016, 2017]


def test_a_vintage_outcome_is_known_two_years_after_the_vintage_closes():
    assert outcome_known_by(2006, 24) == 200812


def test_a_challenger_cannot_be_trained_on_a_vintage_whose_outcome_did_not_exist_yet():
    """The whole point. Training the 2007 challenger on 2006 hands it two years it never had.

    At the start of 2007 the 2005 book is eight months from having a 24 month outcome and the
    2006 book is nearly three years away. Only 2004 is finished.
    """
    assert available_training_vintages(2007, ALL, CONFIG) == [2004]
    assert available_training_vintages(2008, ALL, CONFIG) == [2004, 2005]


def test_the_earliest_vintages_have_no_feasible_challenger_at_all():
    assert available_training_vintages(2004, ALL, CONFIG) == []
    assert available_training_vintages(2006, ALL, CONFIG) == []


def test_the_feasible_challenger_reaches_back_across_a_gap_in_the_vintages():
    """With no 2009 to 2014 in the extract, the 2015 challenger is stuck with crisis data.

    That is not a defect in the code. It is the situation, and it produces the finding that a
    challenger retrained on 2007 and 2008 is wildly over conservative afterwards.
    """
    assert available_training_vintages(2015, ALL, CONFIG) == [2007, 2008]


def test_the_oracle_ignores_the_lag_and_is_labelled_infeasible_wherever_it_appears():
    assert oracle_training_vintages(2007, ALL, CONFIG) == [2005, 2006]
    assert oracle_training_vintages(2007, ALL, CONFIG) != available_training_vintages(2007, ALL, CONFIG)


def test_no_challenger_is_ever_trained_on_the_vintage_it_is_evaluated_on():
    for vintage in ALL:
        assert vintage not in available_training_vintages(vintage, ALL, CONFIG)
        assert vintage not in oracle_training_vintages(vintage, ALL, CONFIG)
