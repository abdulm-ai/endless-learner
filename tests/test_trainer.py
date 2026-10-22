import os
import tempfile
import unittest

import numpy as np

from endless.curriculum import Curriculum
from endless.failure import DeathBank
from endless.metrics import read_only
from endless.trainer import Trainer, TrainConfig


def small_cfg():
    return TrainConfig(
        warmup=300, log_every=500, eval_every=1000, plasticity_every=1000, replay_size=5000,
        reservoir_size=1000, eps_steps=1000, checkpoint_every_s=1e9, latest_every_s=0.0,
    )


class TestTrainer(unittest.TestCase):
    def test_trains_checkpoints_and_resumes_exactly(self):
        with tempfile.TemporaryDirectory() as d:
            t = Trainer(d, small_cfg(), log=lambda *a: None)
            t.train(max_steps=2000)
            steps, params = t.steps, [p.copy() for p in t.agent.online]
            rng_state = t.rng.bit_generator.state
            self.assertTrue(os.path.exists(os.path.join(d, "checkpoint.pkl")))
            self.assertTrue(os.path.exists(os.path.join(d, "latest_net.npz")))

            t2 = Trainer(d, log=lambda *a: None)
            self.assertEqual(t2.steps, steps)
            for a, b in zip(params, t2.agent.online):
                np.testing.assert_array_equal(a, b)
            self.assertEqual(t2.rng.bit_generator.state, rng_state)
            t2.train(max_steps=steps + 500)
            con = read_only(os.path.join(d, "metrics.sqlite"))
            kinds = [r[0] for r in con.execute("SELECT kind FROM events")]
            evals = con.execute("SELECT COUNT(*) FROM evals").fetchone()[0]
            con.close()
            self.assertIn("resume", kinds)
            self.assertGreater(evals, 0)


class TestCurriculum(unittest.TestCase):
    def test_promotes_and_only_demotes_after_a_collapse(self):
        c = Curriculum(window=10, promote=0.8, demote=0.2, collapse_from=0.5)
        for _ in range(10):
            c.record(True)
        self.assertEqual(c.tier, 1)
        for _ in range(30):  # a new tier going badly from the start is not a collapse
            c.record(False)
        self.assertEqual(c.tier, 1)
        for i in range(10):
            c.record(i % 2 == 0)
        self.assertEqual(c.tier, 1)  # 50%: neither promote nor demote, but now it has gone well once
        changes = [c.record(False) for _ in range(10)]
        self.assertIn("demote", changes)
        self.assertEqual(c.tier, 0)


class TestDeathBank(unittest.TestCase):
    def test_fixed_after_a_streak_and_gives_up(self):
        bank = DeathBank(rewind=2, fix_streak=2, max_attempts=3)
        hist = [(i, 0.0, 0.0) for i in range(5)]
        self.assertTrue(bank.add(0, 1, hist, 10, "low/air", 0))
        e = bank.entries[0]
        self.assertIsNone(bank.report(e, True))
        self.assertEqual(bank.report(e, True), "fixed")
        self.assertEqual(len(bank), 0)
        bank.add(0, 2, hist, 10, "gap/ground", 1)
        e = bank.entries[0]
        bank.report(e, False)
        bank.report(e, True)
        self.assertEqual(bank.report(e, False), "gave_up")
        self.assertFalse(bank.add(0, 3, hist[:1], 10, "low/air", 2))


if __name__ == "__main__":
    unittest.main()
