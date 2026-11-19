import numpy as np
import pandas as pd

from flag_predictor.config import get_flag_thresholds, get_live_flag_thresholds, live_line_shift
from flag_predictor.prediction import official_flag as OF


def test_live_thresholds_shift_only_the_isis_lines():
    s = live_line_shift('isis')
    raw, live = get_flag_thresholds('isis'), get_live_flag_thresholds('isis')
    assert live['green'][0] == -float('inf') and live['red'][1] == float('inf')
    assert np.isclose(live['red'][0], raw['red'][0] + s)
    assert np.isclose(live['light_blue'][0], raw['light_blue'][0] + s)
    assert get_live_flag_thresholds('godstow') == get_flag_thresholds('godstow')


def test_shifted_lines_equal_shifted_differential():
    """The flag model sees only margins: lines lowered by c == differential raised by c."""
    from tests.test_official_flag import params
    t0 = pd.Timestamp('2026-11-19 08:00', tz='UTC')
    hist = np.linspace(0.40, 0.47, 49)
    paths = np.linspace(0.47, 0.56, 73)[None, :] + np.linspace(-0.03, 0.03, 20)[:, None]
    c = 0.06
    a = OF.simulate(params(), 'isis', 3, 30.0, hist, paths, t0, rng=np.random.default_rng(1), shift=-c)
    b = OF.simulate(params(), 'isis', 3, 30.0, hist + c, paths + c, t0, rng=np.random.default_rng(1))
    assert (a == b).all()
    assert OF.sync_note('isis', 'amber', hist, -c) == OF.sync_note('isis', 'amber', hist + c)
