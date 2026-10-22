import unittest

import numpy as np

from endless import nn


class TestNetwork(unittest.TestCase):
    def test_backprop_matches_numerical_gradients(self):
        for sizes in ([7, 6, 5, 4], [3, 8, 2]):
            errs = nn.gradcheck(sizes)
            self.assertLess(max(errs), 1e-5, f"{sizes}: {errs}")

    def test_adam_fits_a_linear_map(self):
        rng = np.random.default_rng(0)
        x = rng.standard_normal((256, 4)).astype(np.float32)
        y = x @ rng.standard_normal((4, 2)).astype(np.float32)
        params = nn.init_mlp([4, 16, 2], rng)
        opt = nn.Adam(params, lr=1e-2)
        first = None
        for _ in range(500):
            out, acts = nn.forward(params, x)
            loss = float(((out - y) ** 2).mean())
            first = first if first is not None else loss
            opt.step(params, nn.backward(params, acts, 2 * (out - y) / out.size))
        self.assertLess(loss, first * 0.05)

    def test_reset_unit_has_no_effect_until_trained(self):
        rng = np.random.default_rng(1)
        params = nn.init_mlp([5, 6, 6, 3], rng)
        x = rng.standard_normal((10, 5)).astype(np.float32)
        units = np.array([0, 3])
        params[4][units, :] = 0.0  # the network as it is once these units' outputs are cut
        before = nn.forward(params, x)[0]
        nn.init_layer(params, 1, units, rng)
        after = nn.forward(params, x)[0]
        # New incoming weights change nothing downstream while the outgoing weights are zero.
        np.testing.assert_allclose(before, after, atol=1e-6)
        self.assertTrue((params[4][units] == 0).all())


if __name__ == "__main__":
    unittest.main()
