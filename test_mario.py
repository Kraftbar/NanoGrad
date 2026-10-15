import math
import random
import unittest

try:
    import numpy as np
except ImportError:                     # mario.py needs numpy (as the emulator does)
    np = None

if np is not None:
    import mario
    from tensor import Tensor


@unittest.skipIf(np is None, "numpy is not installed")
class MarioTest(unittest.TestCase):
    def test_shrink_makes_an_84_by_84_grey_screen(self):
        screen = np.zeros((240, 256, 3), np.uint8)
        screen[:, 128:] = 255                                      # right half white
        small = mario.shrink(screen)
        self.assertEqual((small.shape, small.dtype), ((84, 84), np.uint8))
        self.assertEqual(small[0, 0], 0)
        self.assertGreaterEqual(small[0, -1], 254)                 # 255 * (0.299 + 0.587 + 0.114), in float32

    def test_clip_is_min_of_max(self):
        x = Tensor([0.5, 0.9, 1.0, 1.3], (4,), requires_grad=True)
        clipped = mario.clip(x, 0.8, 1.2)
        self.assertEqual([round(v, 12) for v in clipped.data], [0.8, 0.9, 1.0, 1.2])
        clipped.sum().backward()
        self.assertEqual(list(x.grad), [0.0, 1.0, 1.0, 0.0])

    def test_advantages_stop_at_episode_ends(self):
        # one game, three steps, the episode ends after step 1
        rewards, values, dones = [[1.0], [2.0], [3.0]], [[0.5], [0.25], [1.0]], [[False], [True], [False]]
        advantages, returns = mario.advantages_and_returns(rewards, values, dones, [2.0], gamma=0.5, lam=1.0)
        delta2 = 3.0 + 0.5 * 2.0 - 1.0
        delta1 = 2.0 - 0.25                                        # no bootstrap past the end
        delta0 = 1.0 + 0.5 * 0.25 - 0.5
        self.assertAlmostEqual(advantages[2][0], delta2)
        self.assertAlmostEqual(advantages[1][0], delta1)
        self.assertAlmostEqual(advantages[0][0], delta0 + 0.5 * delta1)
        self.assertAlmostEqual(returns[0][0], advantages[0][0] + 0.5)

    def test_sample_gives_log_probabilities_of_a_softmax(self):
        picks = mario.sample([0.0, math.log(3.0), 5.0, 5.0], 2, random.Random(0))
        self.assertAlmostEqual(math.exp(picks[0][1]), 0.25 if picks[0][0] == 0 else 0.75)
        self.assertAlmostEqual(picks[1][1], math.log(0.5))

    def test_ppo_loss_gradient_flows_to_every_parameter(self):
        model = mario.ActorCritic(7, seed=0)
        frames = np.random.default_rng(0).integers(0, 255, (2, 4, 84, 84), dtype=np.uint8)
        screens = mario.screens_tensor(frames)
        scores, values = model(screens)
        self.assertEqual((scores.shape, values.shape), ((2, 7), (2,)))
        old = [p for _, p in mario.sample(list(scores.data), 7, random.Random(1))]
        loss = mario.ppo_loss(model, screens, [0, 3], old, [1.0, -1.0], [0.5, 0.5], epsilon=0.2, beta=0.01)
        loss.backward()
        for parameter in model.parameters():
            self.assertTrue(any(g != 0.0 for g in parameter.grad))

    def test_unchanged_policy_has_ratio_one(self):
        # Right after play, pi = pi_old: the loss is -mean(A) + value loss - entropy bonus.
        model = mario.ActorCritic(7, seed=0)
        screens = mario.screens_tensor(np.zeros((1, 4, 84, 84), np.uint8))
        scores, values = model(screens)
        row = list(scores.data)
        top = max(row)
        log_total = top + math.log(sum(math.exp(v - top) for v in row))
        old = [row[2] - log_total]
        loss = mario.ppo_loss(model, screens, [2], old, [3.0], [values.data[0]], epsilon=0.2, beta=0.0)
        self.assertAlmostEqual(loss.item(), -3.0, places=9)


if __name__ == "__main__":
    unittest.main()
