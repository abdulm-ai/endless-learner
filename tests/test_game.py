import unittest

import numpy as np

from endless.env import RunnerEnv, cast_rays
from endless.game import (
    DEAD,
    DONE,
    DUCK,
    GAP,
    LONG,
    LOW,
    NOOP,
    OVERHANG,
    OVERHANG_BOTTOM,
    OVERHANG_TOP,
    RUNNING,
    SHORT,
    START,
    Level,
    advance,
    airborne,
)
from endless.levels import eval_seed, generate, solve, train_seed


def make_level(boxes=(), gaps=(), speed=0.25, length=60.0):
    b = np.array(boxes, dtype=np.float64).reshape(-1, 6)
    g = np.array(gaps, dtype=np.float64).reshape(-1, 2)
    return Level(0, 0, speed, length, b[:, 0].astype(np.int64), b[:, 1], b[:, 2], b[:, 3], b[:, 4], b[:, 5], g[:, 0], g[:, 1])


def run(level, actions):
    st, out, cause = START, RUNNING, -1
    for a in actions:
        st, out, cause, _ = advance(level, st, a)
        if out != RUNNING:
            break
    return st, out, cause


class TestPhysics(unittest.TestCase):
    def test_jump_heights(self):
        lv = make_level()
        for action, lo, hi in ((SHORT, 2.0, 2.4), (LONG, 4.2, 4.7)):
            st, _, _, _ = advance(lv, START, action)
            peak = st[1]
            while airborne(st):
                st, _, _, _ = advance(lv, st, NOOP)
                peak = max(peak, st[1])
            self.assertTrue(lo < peak < hi, (action, peak))

    def test_running_into_a_block_kills(self):
        lv = make_level(boxes=[(LOW, 10, 11, 0, 1, 0)])
        _, out, cause = run(lv, [NOOP] * 200)
        self.assertEqual((out, cause), (DEAD, LOW))

    def test_overhang_needs_a_duck(self):
        lv = make_level(boxes=[(OVERHANG, 10, 14, OVERHANG_BOTTOM, OVERHANG_TOP, 0)], length=30)
        self.assertEqual(run(lv, [NOOP] * 200)[1:], (DEAD, OVERHANG))
        self.assertEqual(run(lv, [DUCK] * 200)[1], DONE)

    def test_gap_kills_on_the_ground(self):
        lv = make_level(gaps=[(10, 12)], length=30)
        self.assertEqual(run(lv, [NOOP] * 200)[1:], (DEAD, GAP))

    def test_passed_never_decreases(self):
        lv = generate(6, 12345)
        counts = [lv.passed(t) for t in range(0, int(lv.length / lv.speed) + 2, 3)]
        self.assertEqual(counts, sorted(counts))
        self.assertEqual(counts[-1], lv.n_features)


class TestLevels(unittest.TestCase):
    def test_same_seed_same_level(self):
        a, b = generate(5, 777), generate(5, 777)
        np.testing.assert_array_equal(a.x0, b.x0)
        np.testing.assert_array_equal(a.gx0, b.gx0)

    def test_eval_and_train_seeds_never_meet(self):
        rng = np.random.default_rng(0)
        evals = {eval_seed(t, i) for t in range(50) for i in range(10)}
        trains = {train_seed(rng) for _ in range(5000)}
        self.assertFalse(evals & trains)

    def test_solver_plan_finishes_the_level_in_the_env(self):
        # The solver and the environment share the physics, so the plan must replay exactly.
        for tier in range(0, 13, 2):
            for i in range(3):
                lv = generate(tier, eval_seed(tier, i))
                plan = solve(lv)
                env = RunnerEnv()
                env.reset(lv)
                out = RUNNING
                for a in plan:
                    _, _, out, _ = env.step(a)
                    if out != RUNNING:
                        break
                self.assertEqual(out, DONE, f"tier {tier} level {i}")


class TestRays(unittest.TestCase):
    def test_block_and_pit_are_seen(self):
        lv = make_level(boxes=[(LOW, 6, 7, 0, 1.0, 0)], gaps=[(3, 4.5)])
        dist, kind = cast_rays(lv, START, False)
        self.assertIn(2, kind.tolist())  # a block
        self.assertIn(1, kind.tolist())  # a pit
        self.assertTrue((dist > 0).all())

    def test_observation_shape_and_range(self):
        env = RunnerEnv()
        obs = env.reset(generate(4, 99))
        for _ in range(30):
            obs, _, out, _ = env.step(NOOP)
            if out != RUNNING:
                break
        self.assertEqual(obs.dtype, np.float32)
        self.assertTrue(np.isfinite(obs).all())


if __name__ == "__main__":
    unittest.main()
