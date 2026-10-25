"""Termly outlook: term dates, flags from flow, the blend, and the ensemble on the stored record."""
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from flag_predictor import termly as T

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize('term,acad,start', [
    ('Michaelmas', 2026, '2026-10-11'), ('Michaelmas', 2023, '2023-10-08'), ('Michaelmas', 2024, '2024-10-13'),
    ('Hilary', 2025, '2026-01-18'), ('Hilary', 2023, '2024-01-14'), ('Hilary', 2018, '2019-01-13'),
    ('Trinity', 2024, '2025-04-27'), ('Trinity', 2018, '2019-04-28'), ('Trinity', 2023, '2024-04-21'),
])
def test_term_start(term, acad, start):
    assert T.term_start(term, acad) == pd.Timestamp(start)


def test_race_days():
    # Torpids 2026: 4-7 March (EA river restriction notice); Summer Eights 2025: 28-31 May
    assert list(T.race_days('torpids', 2025).strftime('%m-%d')) == ['03-04', '03-05', '03-06', '03-07']
    assert T.race_days('eights', 2024)[0] == pd.Timestamp('2025-05-28')
    assert T.race_days('torpids', 2026)[0] == pd.Timestamp('2027-02-24')     # 6th week (OURCs)
    assert T.race_days('eights', 2026)[0] == pd.Timestamp('2027-05-26')
    w0, w1 = T.term_window('Hilary', 2025)
    assert (w1 - w0).days + 1 == 63 and w0.dayofweek == 6


def test_flag_ranks_two_day_mean():
    q = np.array([10.0, 30.0, 40.0, 40.0, 50.0])
    # 2-day means: 10, 20, 35, 40, 45 against the Isis lines 22 / 34.5 / 39.5
    assert T.flag_ranks(q, 'isis').tolist() == [0, 0, 2, 3, 3]
    assert T.flag_ranks(q, 'godstow').tolist() == [0, 0, 1, 1, 2]


def test_blend_and_weighted_quantile():
    assert T.blend_weight(-3) == T.blend_weight(0) == 0.9
    assert T.blend_weight(400) == 0.2
    w0, w1 = T.term_window('Christmas Vac', 2026)
    assert (w0, w1) == (pd.Timestamp('2026-12-06'), pd.Timestamp('2027-01-09'))
    x, w = np.arange(11.0), np.full(11, 1 / 11)
    assert T.wquantile(x, w, 50) == pytest.approx(5.0)
    assert T.wcrps(x, w, 5.0) == pytest.approx(T.crps(x, 5.0))


def test_ensemble_on_record():
    q = T.load_flow(ROOT)
    model = T.Outlook(q, exclude=2019)
    ens = model.ensemble(pd.Timestamp('2019-11-01'))
    assert ens.members.shape == ens.climatology.shape == (len(ens.years), T.HORIZON)
    assert 2019 not in ens.years and len(ens.years) >= 30
    assert np.isfinite(ens.members).all() and (ens.members > 0).all()
    # autumn 2019 was wet: the conditioned members start higher than an average year
    assert np.median(ens.members[:, :14]) > np.median(ens.climatology[:, :14])
    samples, wts = T.mix(ens, 30)
    assert len(samples) == 2 * len(ens.years) and wts.sum() == pytest.approx(1.0)
