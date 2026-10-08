"""Reinforcement learning: balance a pole on a cart (CartPole).

There is no dataset. An agent acts in an environment, gets rewards, and
must work out which actions led to them. CartPole (Barto, Sutton and
Anderson, 1983; Gym's CartPole-v1) is the classic first problem: push a
cart left or right so the pole hinged on it stays up. Every step upright
is worth 1; an episode ends when the pole tips past 12 degrees, the cart
leaves the track, or after 500 steps. It counts as solved when the
running average return passes 475.

The agents are the two of PyTorch's ``examples/reinforcement_learning``:

REINFORCE (Williams, 1992). A policy network gives the probability of each
action; play a whole episode by sampling from it, then make every action
taken more likely in proportion to the (normalized) return that followed it:

    loss = -sum_t log pi(a_t | s_t) * G_t,     G_t = r_t + gamma r_{t+1} + gamma^2 r_{t+2} + ...

Actor-critic. The same network also predicts the return (a "critic"
value V(s_t)); actions are weighted by how much better the return was
than predicted, G_t - V(s_t), which is far less noisy, and the critic
learns from the returns it sees:

    loss = -sum_t log pi(a_t | s_t) * (G_t - V(s_t)) + sum_t smooth_l1(V(s_t), G_t)

    python3 rl.py --agent reinforce
    python3 rl.py --agent actor-critic --seeds 5
"""

from __future__ import annotations

import argparse
import math
import random
from time import perf_counter

from optim import TensorAdam
from tensor import Tensor, _add_grad, _grad_data, _op_result, _set_backward, gather_rows, no_grad
from tensor_nn import TensorLinear, TensorModule


class CartPole:
    """Gym's CartPole-v1 dynamics (Euler steps of 0.02 s)."""

    gravity, cart_mass, pole_mass, half_length, force, tau = 9.8, 1.0, 0.1, 0.5, 10.0, 0.02
    max_angle, max_position, max_steps = 12 * 2 * math.pi / 360, 2.4, 500

    def __init__(self, rng: random.Random) -> None:
        self.rng = rng

    def reset(self) -> list[float]:
        self.state = [self.rng.uniform(-0.05, 0.05) for _ in range(4)]   # x, x', theta, theta'
        self.steps = 0
        return list(self.state)

    def step(self, action: int) -> tuple[list[float], float, bool]:
        x, x_dot, theta, theta_dot = self.state
        force = self.force if action == 1 else -self.force
        total_mass = self.cart_mass + self.pole_mass
        pole_moment = self.pole_mass * self.half_length
        cos, sin = math.cos(theta), math.sin(theta)
        temp = (force + pole_moment * theta_dot ** 2 * sin) / total_mass
        theta_acc = (self.gravity * sin - cos * temp) / (
            self.half_length * (4.0 / 3.0 - self.pole_mass * cos ** 2 / total_mass))
        x_acc = temp - pole_moment * theta_acc * cos / total_mass
        x, x_dot = x + self.tau * x_dot, x_dot + self.tau * x_acc
        theta, theta_dot = theta + self.tau * theta_dot, theta_dot + self.tau * theta_acc
        self.state = [x, x_dot, theta, theta_dot]
        self.steps += 1
        fallen = abs(x) > self.max_position or abs(theta) > self.max_angle
        return list(self.state), 1.0, fallen or self.steps >= self.max_steps


class Policy(TensorModule):
    """state -> 128 hidden (ReLU) -> action scores, and (actor-critic) a value."""

    def __init__(self, *, hidden: int = 128, critic: bool = False, seed: int = 0) -> None:
        self.hidden = TensorLinear(4, hidden, seed=seed)
        self.actions = TensorLinear(hidden, 2, seed=seed + 1)
        self.value = TensorLinear(hidden, 1, seed=seed + 2) if critic else None

    def __call__(self, states: Tensor) -> tuple[Tensor, Tensor | None]:
        features = self.hidden(states).relu()
        return self.actions(features), (None if self.value is None else self.value(features))

    def parameters(self) -> list[Tensor]:
        layers = [self.hidden, self.actions] + ([self.value] if self.value else [])
        return [parameter for layer in layers for parameter in layer.parameters()]


def play(policy: Policy, env: CartPole, rng: random.Random) -> tuple[list[list[float]], list[int], list[float]]:
    """One episode, sampling each action from the policy."""

    states, actions, rewards = [], [], []
    state, done = env.reset(), False
    while not done:
        with no_grad():
            scores, _ = policy(Tensor(state, (1, 4)))
        top = max(scores.data)
        weights = [math.exp(value - top) for value in scores.data]
        action = 0 if rng.random() * sum(weights) < weights[0] else 1
        states.append(state)
        actions.append(action)
        state, reward, done = env.step(action)
        rewards.append(reward)
    return states, actions, rewards


def discounted_returns(rewards: list[float], gamma: float) -> list[float]:
    returns, total = [], 0.0
    for reward in reversed(rewards):
        total = reward + gamma * total
        returns.append(total)
    return returns[::-1]


def update(policy: Policy, optimizer: TensorAdam, states, actions, returns: list[float]) -> None:
    """One gradient step on the whole episode (the policy is recomputed on
    every state at once: the same numbers as during play, in one batch)."""

    count = len(states)
    mean = sum(returns) / count
    spread = math.sqrt(sum((g - mean) ** 2 for g in returns) / max(1, count - 1)) if count > 1 else 1.0
    normalized = [(g - mean) / (spread + 1e-8) for g in returns]          # as the PyTorch example does

    optimizer.zero_grad()
    scores, values = policy(Tensor([v for s in states for v in s], (count, 4)))
    if values is None:                                                    # REINFORCE
        weights = normalized
    else:                                                                 # actor-critic
        weights = [g - v for g, v in zip(normalized, values.data)]
    # -sum_t log pi(a_t|s_t) * w_t is the cross entropy of the taken actions,
    # each row weighted by w_t: written as a weighted sum of per-row losses.
    log_probs = -log_softmax_rows(scores)                                 # (count, 2)
    picked = gather_rows(log_probs.reshape((count * 2, 1)), [t * 2 + a for t, a in enumerate(actions)])
    loss = (picked * Tensor(weights, (count, 1))).sum()
    if values is not None:
        loss = loss + smooth_l1(values, Tensor(normalized, (count, 1)))
    loss.backward()
    optimizer.step()


def log_softmax_rows(scores: Tensor) -> Tensor:
    """log softmax over the last axis: x - logsumexp(x)."""

    probabilities = scores.softmax(axis=-1)
    return probabilities.log()


def smooth_l1(prediction: Tensor, target: Tensor) -> Tensor:
    """Huber loss, summed (PyTorch's smooth_l1_loss with beta 1 and reduction "sum"):

        huber(e) = e^2 / 2       if |e| < 1        d/de = e
                   |e| - 1 / 2   otherwise         d/de = sign(e)
    """

    error = prediction - target
    values = list(error.data)
    out = _op_result([sum(0.5 * e * e if abs(e) < 1.0 else abs(e) - 0.5 for e in values)], (1,), (error,),
                     "smooth_l1")

    def _backward() -> None:
        g = _grad_data(out)[0]
        _add_grad(error, [g * (e if abs(e) < 1.0 else math.copysign(1.0, e)) for e in values])

    _set_backward(out, _backward)
    return out


def train(args, seed: int) -> dict:
    rng = random.Random(seed)
    env = CartPole(random.Random(seed + 1))
    policy = Policy(hidden=args.hidden, critic=args.agent == "actor-critic", seed=seed)
    optimizer = TensorAdam(policy.parameters(), lr=args.lr)
    running, start = 10.0, perf_counter()
    for episode in range(1, args.episodes + 1):
        states, actions, rewards = play(policy, env, rng)
        update(policy, optimizer, states, actions, discounted_returns(rewards, args.gamma))
        running = 0.05 * sum(rewards) + 0.95 * running                    # the example's running reward
        if episode % args.log_every == 0:
            print(f"  episode {episode:4d}: last return {sum(rewards):5.0f}, running {running:6.1f}", flush=True)
        if running > args.solved:
            return {"solved_at": episode, "seconds": perf_counter() - start}
    return {"solved_at": None, "seconds": perf_counter() - start}


def main(argv: list[str] | None = None) -> dict:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--agent", choices=("reinforce", "actor-critic"), default="reinforce")
    parser.add_argument("--episodes", type=int, default=1000)
    parser.add_argument("--hidden", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-2)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--solved", type=float, default=475.0, help="running return that counts as solved")
    parser.add_argument("--seeds", type=int, default=1)
    parser.add_argument("--log-every", type=int, default=50)
    args = parser.parse_args(argv)
    results = []
    for seed in range(args.seeds):
        print(f"{args.agent}, seed {seed}", flush=True)
        result = train(args, seed)
        results.append(result)
        solved = f"solved after {result['solved_at']} episodes" if result["solved_at"] else "not solved"
        print(f"{args.agent}, seed {seed}: {solved}, {result['seconds']:.0f} s", flush=True)
    return {"results": results}


if __name__ == "__main__":
    main()
