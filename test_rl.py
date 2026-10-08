import importlib.util
import math
import random
import unittest

from rl import CartPole, Policy, discounted_returns, smooth_l1, update
from optim import TensorAdam
from tensor import Tensor


HAS_TORCH = importlib.util.find_spec("torch") is not None


class CartPoleTests(unittest.TestCase):
    def test_pushing_one_way_tips_the_pole_the_other(self) -> None:
        env = CartPole(random.Random(0))
        env.reset()
        env.state = [0.0, 0.0, 0.0, 0.0]
        for _ in range(5):
            state, reward, done = env.step(1)                  # push right
        self.assertGreater(state[1], 0.0)                      # the cart moves right
        self.assertLess(state[2], 0.0)                         # the pole falls back (left)
        self.assertEqual(reward, 1.0)

    def test_episode_ends_when_the_pole_falls(self) -> None:
        env = CartPole(random.Random(1))
        env.reset()
        steps, done = 0, False
        while not done:
            _, _, done = env.step(1)
            steps += 1
        self.assertLess(steps, 100)
        self.assertGreater(abs(env.state[2]), CartPole.max_angle - 0.1)

    def test_discounted_returns(self) -> None:
        self.assertEqual(discounted_returns([1.0, 1.0, 1.0], 0.5), [1.75, 1.5, 1.0])


@unittest.skipUnless(HAS_TORCH, "PyTorch not installed")
class GradientTests(unittest.TestCase):
    def test_smooth_l1_matches_pytorch(self) -> None:
        import torch

        prediction = Tensor([0.3, -2.0, 1.5], (3, 1), requires_grad=True)
        loss = smooth_l1(prediction, Tensor([0.0, 0.0, 1.0], (3, 1)))
        loss.backward()
        p = torch.tensor([[0.3], [-2.0], [1.5]], dtype=torch.float64, requires_grad=True)
        reference = torch.nn.functional.smooth_l1_loss(p, torch.tensor([[0.0], [0.0], [1.0]], dtype=torch.float64),
                                                        reduction="sum")
        reference.backward()
        self.assertAlmostEqual(loss.item(), reference.item(), places=12)
        for a, b in zip(prediction.grad, p.grad.reshape(-1).tolist()):
            self.assertAlmostEqual(a, b, places=12)

    def test_update_follows_the_policy_gradient(self) -> None:
        import torch

        for critic in (False, True):
            policy = Policy(hidden=8, critic=critic, seed=1)
            weights = [torch.tensor(list(p.data), dtype=torch.float64).reshape(p.shape).requires_grad_()
                       for p in policy.parameters()]
            rng = random.Random(2)
            states = [[rng.uniform(-0.1, 0.1) for _ in range(4)] for _ in range(6)]
            actions = [rng.randrange(2) for _ in range(6)]
            returns = discounted_returns([1.0] * 6, 0.9)
            optimizer = TensorAdam(policy.parameters(), lr=1e-2)
            update(policy, optimizer, states, actions, returns)

            g = torch.tensor(returns, dtype=torch.float64)
            g = (g - g.mean()) / (g.std() + 1e-8)
            features = torch.relu(torch.tensor(states, dtype=torch.float64) @ weights[0].T + weights[1])
            log_probs = torch.log_softmax(features @ weights[2].T + weights[3], dim=1)
            taken = log_probs[torch.arange(6), torch.tensor(actions)]
            if critic:
                values = (features @ weights[4].T + weights[5]).reshape(-1)
                loss = -(taken * (g - values.detach())).sum() + torch.nn.functional.smooth_l1_loss(
                    values, g, reduction="sum")
            else:
                loss = -(taken * g).sum()
            loss.backward()
            for parameter, reference in zip(policy.parameters(), weights):
                for a, b in zip(parameter.grad, reference.grad.reshape(-1).tolist()):
                    self.assertAlmostEqual(a, b, places=10)


if __name__ == "__main__":
    unittest.main()
