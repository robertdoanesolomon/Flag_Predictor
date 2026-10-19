"""
Checks for the continuous-state hybrid, the model-error spread and the forecast archive.

The continuous hybrid must keep the September hybrid's guarantees for any
parameters and any simulated starting state, so those are checked on randomly
initialised models with random inputs.

Run: python -m unittest discover tests   (or pytest)
"""

import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))

from flag_predictor.models.continuous_hybrid import (  # noqa: E402
    N_STATE, ContinuousHybrid, ReservoirCore,
)
from flag_predictor.prediction import archive  # noqa: E402
from flag_predictor.prediction.model_error import (  # noqa: E402
    ErrorLibrary, crps_samples, fill_trajectory, start_features, total_uncertainty,
)

B, SEQ, F, H = 12, 100, 10, 240


def random_model(seed: int, et_mode: str = 'cos') -> ContinuousHybrid:
    torch.manual_seed(seed)
    core = ReservoirCore(et_mode=et_mode, direct=True)
    model = ContinuousHybrid(F, d_lo=-0.15, d_hi=1.5, core=core)
    with torch.no_grad():
        for p in core.params_raw.values():
            p.add_(torch.randn(()) * 1.5)
        core.split_logits.add_(torch.randn(3))
        model.rating.slope_raw.add_(torch.randn_like(model.rating.slope_raw))
        # A non-zero head, so the corrections and parameter shifts are exercised.
        model.head.weight.normal_(0, 0.3)
        model.head.bias.normal_(0, 0.5)
    return model.eval()


def random_inputs(seed: int, rain_scale: float = 1.0, recent: bool = True):
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(B, SEQ, F, generator=g)
    rain = torch.rand(B, H, generator=g) ** 4 * 5 * rain_scale
    season = torch.cos(torch.linspace(0, 6, H)).expand(B, H)
    rain24 = torch.rand(B, generator=g) * 20 * (1.0 if recent else 0.0)
    rain168 = rain24 + torch.rand(B, generator=g) * 40 * (1.0 if recent else 0.0)
    d0 = torch.rand(B, generator=g) * 1.2 - 0.05
    state = torch.rand(B, N_STATE, generator=g) * torch.tensor([1, 5, 5, 5, 5, 300, 2.0])
    q_sim0 = torch.rand(B, generator=g) * 0.2
    et0 = torch.rand(B, H, generator=g) * 0.3
    return x, rain, season, rain24, rain168, d0, state, q_sim0, et0


class ContinuousHybridGuarantees(unittest.TestCase):
    def test_no_rise_without_any_rain(self):
        for et_mode in ('cos', 'pet'):
            for seed in range(4):
                m = random_model(seed, et_mode)
                x, rain, season, r24, r168, d0, st, q, et0 = random_inputs(seed, recent=False)
                with torch.no_grad():
                    d = m(x, rain * 0, season, r24, r168, d0, st, q, et0=et0)
                full = torch.cat([d0[:, None], d], 1)
                self.assertTrue(torch.all(full[:, 1:] - full[:, :-1] <= 1e-5),
                                f'{et_mode} seed {seed}: climbed with no rain at all')

    def test_more_rain_never_lowers(self):
        for seed in range(4):
            m = random_model(seed)
            x, rain, season, r24, r168, d0, st, q, et0 = random_inputs(seed)
            with torch.no_grad():
                lo = m(x, rain, season, r24, r168, d0, st, q)
                hi = m(x, rain * 1.5 + 0.1, season, r24, r168, d0, st, q)
            self.assertTrue(torch.all(hi >= lo - 1e-5))

    def test_core_mass_balance(self):
        """With no ET, water in = water out + change in storage."""
        core = ReservoirCore(et_mode='cos', direct=True)
        T = 500
        rain = torch.rand(1, T) ** 3 * 4
        state0 = torch.zeros(1, N_STATE)
        state0[0, 0] = 0.5
        P = core.params()
        q, final = core.simulate(rain, torch.zeros(1, T), state0, P)
        smax = float(P['smax'])
        storage = lambda s: s[0, 0] * smax + s[0, 1:].sum()  # noqa: E731
        balance = float(rain.sum() - q.sum() - (storage(final) - storage(state0)))
        self.assertAlmostEqual(balance, 0.0, delta=1e-2)


class ModelErrorTests(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(0)
        n = 300
        self.feats = np.stack([start_features(float(rng.uniform(0, 1)), float(rng.uniform(0, 1)),
                                              rng.gamma(0.3, 1, 240), float(rng.uniform(0, 50)),
                                              pd.Timestamp('2025-01-01') + pd.Timedelta(days=int(i)))
                               for i in range(n)])
        res = np.cumsum(rng.normal(0, 0.005, (n, 240)), axis=1)
        res[5, 100:120] = np.nan
        self.lib = ErrorLibrary.build('isis', 'test', pd.date_range('2025-01-01', periods=n, freq='6h').to_numpy(),
                                      self.feats, res)

    def test_gaps_filled(self):
        self.assertTrue(np.isfinite(self.lib.residuals).all())
        r = fill_trajectory(np.array([np.nan, 1.0, np.nan, 3.0, np.nan]))
        np.testing.assert_allclose(r, [1, 1, 2, 3, 3])

    def test_total_uncertainty_shapes_and_start(self):
        members = np.linspace(0.1, 0.3, 5)[:, None] + np.zeros((5, 241))
        out = total_uncertainty(members, self.feats[:5], self.lib, n=7)
        self.assertEqual(out.shape, (35, 241))
        # Hour 0 is the observation: no error added there.
        np.testing.assert_allclose(out[:, 0], np.repeat(members[:, 0], 7))

    def test_save_load_roundtrip(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / 'lib.npz'
            self.lib.save(p)
            lib2 = ErrorLibrary.load(p)
            self.assertEqual(lib2.residuals.shape, self.lib.residuals.shape)
            np.testing.assert_allclose(lib2.residuals, self.lib.residuals, atol=1e-3)

    def test_crps_of_point_is_mae(self):
        obs = np.array([0.1, 0.2, 0.3])
        s = np.array([[0.15, 0.1, 0.5]])
        np.testing.assert_allclose(crps_samples(s, obs), np.abs(s[0] - obs))


class ArchiveTests(unittest.TestCase):
    @staticmethod
    def payload(t0: str, d0: float, n: int = 5):
        times = pd.date_range(t0, periods=241, freq='1h')
        members = [[d0 + 0.001 * i * j for j in range(241)] for i in range(n)]
        hist_t = pd.date_range(pd.Timestamp(t0) - pd.Timedelta(hours=48), t0, freq='1h')
        return {
            'forecast_start': times[0].isoformat(),
            'current': {'differential': d0},
            'historical': {'times': [t.isoformat() for t in hist_t], 'differential': [d0] * len(hist_t)},
            'ensemble': {'times': [t.isoformat() for t in times], 'members': members,
                         'total': {'p10': [d0 - 0.01] * 241, 'p90': [d0 + 0.05] * 241}},
            'model': 'test',
        }

    def test_slots(self):
        self.assertEqual(archive.slot_id(pd.Timestamp('2026-10-17 17:59', tz='UTC')), '2026-10-17T12')
        self.assertEqual(archive.slot_id(pd.Timestamp('2026-10-17 18:00')), '2026-10-17T18')

    def test_update_cycle(self):
        with tempfile.TemporaryDirectory() as d:
            docs = Path(d) / 'docs'
            raw = Path(d) / 'raw'
            rain = pd.DataFrame(np.random.default_rng(0).gamma(0.2, 1, (240, 5)),
                                index=pd.date_range('2026-10-15 13:00', periods=240, freq='1h'))
            r1 = archive.update_archive(docs, {'isis': self.payload('2026-10-15 12:00', 0.1)}, 'test',
                                        member_rain={'isis': rain}, raw_dir=raw,
                                        now=pd.Timestamp('2026-10-15 12:30', tz='UTC'))
            self.assertTrue(r1['wrote_issue'])
            # A second run in the same slot does not overwrite the issue.
            r2 = archive.update_archive(docs, {'isis': self.payload('2026-10-15 13:00', 0.2)}, 'test',
                                        now=pd.Timestamp('2026-10-15 13:30', tz='UTC'))
            self.assertFalse(r2['wrote_issue'])
            # Two days later: new slot; observations accumulate.
            archive.update_archive(docs, {'isis': self.payload('2026-10-17 12:00', 0.3)}, 'test',
                                   now=pd.Timestamp('2026-10-17 12:10', tz='UTC'))
            index = json.loads((docs / 'archive' / 'index.json').read_text())
            self.assertEqual([e['slot'] for e in index['issues']], ['2026-10-15T12', '2026-10-17T12'])
            self.assertTrue(index['issues'][0]['has_total'])
            obs = archive.observed_series(json.loads((docs / 'archive' / 'observed.json').read_text())['isis'])
            self.assertEqual(obs.index[0], pd.Timestamp('2026-10-13 12:00'))
            self.assertEqual(obs.index[-1], pd.Timestamp('2026-10-17 12:00'))
            now = json.loads((docs / 'archive' / 'now.json').read_text())['locations']['isis']
            self.assertEqual(now['target'], '2026-10-17T12:00:00')
            self.assertEqual(now['forecasts'][0]['lead_h'], 48)
            self.assertIn('total_p90', now['forecasts'][0])
            z = np.load(r1['raw'])
            self.assertEqual(z['isis_rain_cmm'].shape, (5, 240))
            np.testing.assert_allclose(z['isis_rain_cmm'] / 100, rain.to_numpy().T, atol=0.006)


class ArchiveGapFillTests(unittest.TestCase):
    def test_fill_gaps_from_git(self):
        import os
        import subprocess
        with tempfile.TemporaryDirectory() as d:
            repo = Path(d)
            (repo / 'docs').mkdir()
            env = dict(os.environ, GIT_AUTHOR_NAME='t', GIT_AUTHOR_EMAIL='t@t', GIT_COMMITTER_NAME='t',
                       GIT_COMMITTER_EMAIL='t@t')
            subprocess.run(['git', 'init', '-q'], cwd=repo, check=True)
            for i, when in enumerate(['2026-10-15T13:05:00Z', '2026-10-15T14:00:00Z', '2026-10-15T19:30:00Z',
                                      '2026-10-16T02:00:00Z']):
                p = ArchiveTests.payload(when[:16].replace('T', ' '), 0.1 * i)
                (repo / 'docs' / 'forecast_data.json').write_text(json.dumps(
                    {'generated_at': when, 'location_order': ['isis'], 'locations': {'isis': p}}))
                subprocess.run(['git', 'add', '-A'], cwd=repo, check=True)
                subprocess.run(['git', 'commit', '-q', '-m', f'c{i}'], cwd=repo, check=True,
                               env=dict(env, GIT_COMMITTER_DATE=when, GIT_AUTHOR_DATE=when))
            written = archive.fill_gaps_from_git(repo / 'docs', repo, now=pd.Timestamp('2026-10-16 07:00', tz='UTC'))
            self.assertEqual(written, ['2026-10-15T12', '2026-10-15T18', '2026-10-16T00'])
            issue = json.loads((repo / 'docs' / 'archive' / 'issues' / '2026-10-15T12.json').read_text())
            self.assertEqual(issue['locations']['isis']['d0'], 0.0)   # first commit in the slot
            # Nothing new to fill on a second call.
            self.assertEqual(archive.fill_gaps_from_git(repo / 'docs', repo,
                                                        now=pd.Timestamp('2026-10-16 07:00', tz='UTC')), [])


if __name__ == '__main__':
    unittest.main()


class OutageFillTests(unittest.TestCase):
    def test_outage_bridged_leading_hole_and_long_runs_kept(self):
        from flag_predictor.prediction.hybrid_live import fill_outage
        idx = pd.date_range('2026-10-01', periods=400, freq='1h', tz='UTC')
        d = pd.Series(np.nan, idx)
        d.iloc[50:100] = 0.3                     # leading hole 0-49 (before the API window)
        d.iloc[250:] = 0.5                       # 150 h outage 100-249, gauge back at 250
        f = fill_outage(d)
        self.assertTrue(f.iloc[:50].isna().all())
        self.assertTrue(f.iloc[100:250].notna().all())
        self.assertAlmostEqual(f.iloc[175], 0.3 + 0.2 * 76 / 151, places=6)
        self.assertTrue(fill_outage(d, max_hours=100).iloc[100:250].isna().all())

    def test_hours_since_outage(self):
        from flag_predictor.prediction.hybrid_live import _hours_since_outage
        idx = pd.date_range('2026-10-01', periods=300, freq='1h', tz='UTC')
        d = pd.Series(0.3, idx)
        d.iloc[:40] = np.nan                      # leading hole: not an outage
        self.assertIsNone(_hours_since_outage(d))
        d.iloc[100:120] = np.nan                  # under a day: the encoder copes
        self.assertIsNone(_hours_since_outage(d))
        d.iloc[150:295] = np.nan                  # gauge back 5 h ago
        self.assertEqual(_hours_since_outage(d), 5)

    def test_outage_shaped_by_neighbour(self):
        from flag_predictor.prediction.hybrid_live import fill_outage
        idx = pd.date_range('2026-09-01', periods=600, freq='1h', tz='UTC')
        nb = pd.Series(0.2 + 0.3 * np.exp(-((np.arange(600) - 450) / 30.0) ** 2), idx)   # a flood peak
        d = 0.05 + 1.5 * nb + 0.01 * np.sin(np.arange(600) / 7.0)
        gappy = d.copy()
        gappy.iloc[400:500] = np.nan                                     # outage over the peak
        shaped = fill_outage(gappy, nb, slope=1.4)
        straight = fill_outage(gappy)
        err = lambda f: float((f.iloc[400:500] - d.iloc[400:500]).abs().mean())
        self.assertLess(err(shaped), 0.02)
        self.assertGreater(err(straight), 0.15)
        self.assertTrue(np.allclose(shaped.iloc[:400], d.iloc[:400]))
