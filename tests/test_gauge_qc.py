import numpy as np
import pandas as pd

from flag_predictor.data import gauge_qc


def _frame(days=20, seed=0):
    """Hourly rain at every gauge: the same showery pattern everywhere, scaled by each gauge's ratio."""
    rng = np.random.default_rng(seed)
    idx = pd.date_range('2026-10-01', periods=24 * days, freq='1h', tz='UTC')
    base = np.where(rng.random(len(idx)) < 0.15, rng.gamma(1.5, 1.2, len(idx)), 0.0)
    return pd.DataFrame({g: base * gauge_qc.RATIO[g] for g in gauge_qc.COORDS}, index=idx)


def test_healthy_gauges_not_flagged():
    clean, report = gauge_qc.clean_rain(_frame(), verbose=False)
    assert report == {}
    pd.testing.assert_frame_equal(clean, _frame())


def test_blocked_gauge_flagged_and_filled():
    raw = _frame()
    raw.loc['2026-10-12':, 'Stanford'] *= 0.02          # funnel blocked from 12 Oct
    clean, report = gauge_qc.clean_rain(raw, verbose=False)
    assert set(report) == {'Stanford'}
    assert report['Stanford']['faulty_now']
    assert report['Stanford']['since'] <= '2026-10-13'
    truth = _frame().loc['2026-10-12':, 'Stanford'].sum()
    assert abs(clean.loc['2026-10-12':, 'Stanford'].sum() - truth) < 0.15 * truth
    assert clean['Abingdon'].equals(raw['Abingdon'])    # neighbours untouched


def test_late_gauge_not_flagged_but_filled():
    raw = _frame()
    raw.iloc[-5:, raw.columns.get_loc('Stanford')] = np.nan   # last 5 hours not reported yet
    clean, report = gauge_qc.clean_rain(raw, verbose=False)
    assert report == {}
    assert clean['Stanford'].notna().all()


def test_recovered_gauge_not_faulty_now():
    raw = _frame(days=40)
    raw.loc['2026-10-05':'2026-10-15', 'Stanford'] = 0.0       # blocked, then cleared
    _, report = gauge_qc.clean_rain(raw, verbose=False)
    assert 'Stanford' in report and not report['Stanford']['faulty_now']
