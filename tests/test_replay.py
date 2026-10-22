import unittest

import numpy as np

from endless.replay import NStep, Replay, Reservoir, SumTree


class TestSumTree(unittest.TestCase):
    def test_total_and_proportional_sampling(self):
        t = SumTree(8)
        p = np.array([1.0, 0.0, 3.0, 0.0, 4.0, 0.0, 2.0, 0.0])
        t.update(np.arange(8), p)
        self.assertAlmostEqual(t.total, 10.0)
        rng = np.random.default_rng(0)
        idx = np.concatenate([t.sample(100, rng, 8) for _ in range(200)])
        freq = np.bincount(idx, minlength=8) / len(idx)
        np.testing.assert_allclose(freq, p / p.sum(), atol=0.01)

    def test_duplicate_indices_and_single_set_agree(self):
        a, b = SumTree(16), SumTree(16)
        a.update(np.array([3, 3, 5]), np.array([2.0, 2.0, 1.0]))
        b.set(3, 2.0)
        b.set(5, 1.0)
        np.testing.assert_allclose(a.tree, b.tree)


class TestNStep(unittest.TestCase):
    def test_returns_and_discounts(self):
        g = 0.9
        n = NStep(3, g)
        s = [np.full(2, i, dtype=np.float32) for i in range(6)]
        out = []
        for i in range(4):
            out += n.push(s[i], i, 1.0, s[i + 1], terminal=(i == 3))
        # First transition: three rewards, bootstrapped from s3 with g^3.
        _, _, R, s_last, disc = out[0]
        self.assertAlmostEqual(R, 1 + g + g * g, places=6)
        self.assertAlmostEqual(disc, g**3, places=6)
        np.testing.assert_array_equal(s_last, s[3])
        # The transitions flushed at the terminal step have no bootstrap.
        self.assertEqual(out[-1][4], 0.0)
        self.assertEqual(len(out), 4)

    def test_truncation_bootstraps(self):
        n = NStep(3, 0.5)
        x = np.zeros(1, dtype=np.float32)
        out = n.push(x, 0, 1.0, x, terminal=False, truncated=True)
        self.assertEqual(out[0][4], 0.5)


class TestReplay(unittest.TestCase):
    def test_priorities_follow_td(self):
        r = Replay(64, 2, alpha=1.0)
        for _ in range(64):
            r.add(np.zeros(2), 0, 0.0, np.zeros(2), 0.99)
        r.update(np.arange(64), np.where(np.arange(64) == 7, 100.0, 0.0))
        idx, w = r.sample(64, 1.0, np.random.default_rng(0))
        self.assertGreater((idx == 7).mean(), 0.8)
        self.assertLessEqual(w.max(), 1.0)

    def test_reservoir_keeps_old_items(self):
        res = Reservoir(100, 1)
        rng = np.random.default_rng(0)
        for i in range(10_000):
            res.offer(np.array([i]), 0, 0.0, np.array([i]), 0.0, rng)
        old = (res.obs[:, 0] < 5000).mean()
        self.assertTrue(0.3 < old < 0.7)


if __name__ == "__main__":
    unittest.main()
