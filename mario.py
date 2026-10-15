"""Super Mario Bros with PPO: learn to finish World 1-1 from the pixels.

The agent sees what a player sees: the screen, made grey and shrunk to
84 x 84, the last 4 of them stacked so motion shows. Every 4th frame it
picks one of 7 button combinations (gym-super-mario-bros's SIMPLE_MOVEMENT:
nothing, right, right+jump, right+run, right+run+jump, jump, left). The
reward is the environment's: pixels moved right, minus the clock ticking,
minus 15 for dying; plus 50 here for reaching the flag. An episode ends at
the flag or on the first death.

PPO (Schulman et al., 2017) is actor-critic with a safety catch. 8 Marios
play 512 steps each with the current policy; each step's advantage is how
much better things went than the critic predicted (GAE, Schulman et al.,
2016):

    delta_t = r_t + gamma V(s_{t+1}) - V(s_t),    A_t = sum_k (gamma lambda)^k delta_{t+k}

then the 4,096 steps are reused for 10 epochs of minibatches. Reusing old
data is only safe while the policy stays near the one that played, so the
gain is cut off once the probability ratio r = pi(a|s) / pi_old(a|s) leaves
[1 - eps, 1 + eps]:

    loss = -mean(min(r A, clip(r, 1 - eps, 1 + eps) A)) + 1/2 mean((V(s) - R)^2) - beta entropy

The network is DQN's (Mnih et al., 2015): three convolutions (32 8x8
stride 4, 64 4x4 stride 2, 64 3x3), a 512-wide layer, then 7 action scores
and a value. The hyperparameters are those of uvipen's
Super-mario-bros-PPO-pytorch, which finishes 1-1.

The emulator is nes-py, and the environments are gym-super-mario-bros:

    .venv/bin/pip install "numpy<2" gym==0.26.2 nes-py gym-super-mario-bros
    NANOGRAD_BACKEND=cuda NANOGRAD_DTYPE=float32 .venv/bin/python mario.py --out runs/mario
    .venv/bin/python mario.py --play runs/mario/model.json             # watch it in a window
    .venv/bin/python mario.py --play runs/mario/model.json --record mario.mp4
"""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import math
import multiprocessing
import os
import random
import subprocess
import time
import warnings
from array import array
from pathlib import Path

import numpy as np

import kernels
from optim import TensorAdam
from tensor import Tensor, gather_rows, no_grad
from tensor_nn import TensorConv2D, TensorLinear, TensorModule

SIZE, STACK, SKIP = 84, 4, 4
ROWS = np.linspace(0, 119, SIZE).round().astype(int)        # 240 x 256 -> 120 x 128 (2 x 2 means) -> 84 x 84
COLUMNS = np.linspace(0, 127, SIZE).round().astype(int)
GREY = np.array([0.299, 0.587, 0.114], np.float32)


def shrink(frame: np.ndarray) -> np.ndarray:
    """(240, 256, 3) RGB screen -> (84, 84) grey bytes."""

    grey = frame.astype(np.float32) @ GREY
    halved = grey.reshape(120, 2, 128, 2).mean(axis=(1, 3))
    return halved[ROWS][:, COLUMNS].astype(np.uint8)


class Mario:
    """One game: 4 frames per action, the last 4 screens stacked."""

    def __init__(self, stage: str, *, render: bool = False) -> None:
        with contextlib.redirect_stderr(io.StringIO()):       # gym 0.26 prints a notice on import
            import gym_super_mario_bros
        warnings.filterwarnings("ignore", module="gym")        # "v0 is out of date": v0 is the real graphics
        from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
        from nes_py.wrappers import JoypadSpace

        env = gym_super_mario_bros.make(f"SuperMarioBros-{stage}-v0", apply_api_compatibility=True,
                                        disable_env_checker=True, render_mode="human" if render else None)
        self.env = JoypadSpace(env, SIMPLE_MOVEMENT)
        self.actions = len(SIMPLE_MOVEMENT)
        self.frames = np.zeros((STACK, SIZE, SIZE), np.uint8)
        self.screen = None

    def reset(self) -> np.ndarray:
        screen, _ = self.env.reset()
        self.screen = screen
        self.frames[:] = shrink(screen)
        return self.frames.copy()

    def step(self, action: int) -> tuple[np.ndarray, float, bool, dict]:
        total, done = 0.0, False
        for _ in range(SKIP):
            screen, reward, terminated, truncated, info = self.env.step(action)
            total += reward
            if terminated or truncated:
                done = True
                break
        self.screen = screen
        self.frames[:-1] = self.frames[1:]
        self.frames[-1] = shrink(screen)
        if info["flag_get"]:
            total += 50.0
        return self.frames.copy(), total / 10.0, done, info


def _worker(remote, stage: str) -> None:
    """Plays one Mario for the trainer: receives actions, sends back (frames, reward, done, episode)."""

    game = Mario(stage)
    remote.send(game.reset())
    score = 0.0
    while True:
        action = remote.recv()
        if action is None:
            break
        frames, reward, done, info = game.step(action)
        score += reward
        episode = None
        if done:
            episode = {"x": int(info["x_pos"]), "flag": bool(info["flag_get"]), "return": score}
            frames, score = game.reset(), 0.0
        remote.send((frames, reward, done, episode))


class Games:
    """``count`` Marios in their own processes, stepped together."""

    def __init__(self, count: int, stage: str) -> None:
        context = multiprocessing.get_context("spawn")
        # The workers import this file again; they only need numpy and the
        # emulator, not a GPU context of their own.
        backend = os.environ.get("NANOGRAD_BACKEND")
        os.environ["NANOGRAD_BACKEND"] = "python"
        try:
            pipes = [context.Pipe() for _ in range(count)]
            self.processes = [context.Process(target=_worker, args=(child, stage), daemon=True)
                              for _, child in pipes]
            for process in self.processes:
                process.start()
        finally:
            if backend is None:
                del os.environ["NANOGRAD_BACKEND"]
            else:
                os.environ["NANOGRAD_BACKEND"] = backend
        self.remotes = [parent for parent, _ in pipes]
        self.frames = np.stack([remote.recv() for remote in self.remotes])

    def step(self, actions: list[int]) -> tuple[np.ndarray, list[float], list[bool], list[dict]]:
        for remote, action in zip(self.remotes, actions):
            remote.send(action)
        results = [remote.recv() for remote in self.remotes]
        self.frames = np.stack([frames for frames, _, _, _ in results])
        return (self.frames, [r for _, r, _, _ in results], [d for _, _, d, _ in results],
                [e for _, _, _, e in results if e is not None])

    def close(self) -> None:
        for remote in self.remotes:
            remote.send(None)
        for process in self.processes:
            process.join(timeout=5)


class ActorCritic(TensorModule):
    """DQN's convolutions, shared by a policy (action scores) and a critic (value)."""

    def __init__(self, actions: int, *, seed: int = 0) -> None:
        self.conv1 = TensorConv2D((32, STACK, 8, 8), stride=4, seed=seed)       # 84 -> 20
        self.conv2 = TensorConv2D((64, 32, 4, 4), stride=2, seed=seed + 1)      # 20 -> 9
        self.conv3 = TensorConv2D((64, 64, 3, 3), seed=seed + 2)                # 9 -> 7
        self.hidden = TensorLinear(64 * 7 * 7, 512, seed=seed + 3)
        self.policy = TensorLinear(512, actions, seed=seed + 4)
        self.value = TensorLinear(512, 1, seed=seed + 5)
        # A near-uniform first policy, so early play explores evenly.
        self.policy.weight.data = kernels.scale(self.policy.weight.data, 0.01)

    def layers(self) -> dict[str, TensorModule]:
        return {"conv1": self.conv1, "conv2": self.conv2, "conv3": self.conv3, "hidden": self.hidden,
                "policy": self.policy, "value": self.value}

    def __call__(self, screens: Tensor) -> tuple[Tensor, Tensor]:
        count = screens.shape[0]
        features = self.conv3(self.conv2(self.conv1(screens).relu()).relu()).relu()
        features = self.hidden(features.reshape((count, 64 * 7 * 7))).relu()
        return self.policy(features), self.value(features).reshape((count,))

    def parameters(self) -> list[Tensor]:
        return [parameter for layer in self.layers().values() for parameter in layer.parameters()]

    def state_dict(self) -> dict:
        return {name: layer.state_dict() for name, layer in self.layers().items()}

    def load_state_dict(self, state: dict) -> None:
        for name, layer in self.layers().items():
            layer.load_state_dict(state[name])


def screens_tensor(frames: np.ndarray) -> Tensor:
    """(count, 4, 84, 84) bytes -> a Tensor of values in [0, 1], in the backend's precision."""

    single = kernels.default_dtype() == "float32"
    values = frames.astype(np.float32 if single else np.float64) * (1 / 255)
    return Tensor(array("f" if single else "d", values.tobytes()), frames.shape)


def log_softmax(scores: Tensor) -> Tensor:
    return scores - scores.logsumexp(axis=-1).reshape((scores.shape[0], 1))


def sample(scores: list[float], actions: int, rng: random.Random) -> list[tuple[int, float]]:
    """(action, log probability) for each row of scores, drawn from its softmax."""

    picks = []
    for start in range(0, len(scores), actions):
        row = scores[start:start + actions]
        top = max(row)
        weights = [math.exp(value - top) for value in row]
        total = sum(weights)
        action = rng.choices(range(actions), weights)[0]
        picks.append((action, math.log(weights[action] / total)))
    return picks


def clip(x: Tensor, low: float, high: float) -> Tensor:
    """min(max(x, low), high), written with ReLUs."""

    return (x - low).relu() - (x - high).relu() + low


def ppo_loss(model: ActorCritic, screens: Tensor, actions: list[int], old_log_probs: list[float],
             advantages: list[float], returns: list[float], *, epsilon: float, beta: float) -> Tensor:
    count = len(actions)
    scores, values = model(screens)
    log_probs = log_softmax(scores)
    chosen = [0.0] * (count * model.policy.weight.shape[0])
    for row, action in enumerate(actions):
        chosen[row * model.policy.weight.shape[0] + action] = 1.0
    picked = (log_probs * Tensor(chosen, scores.shape)).sum(axis=1)
    ratio = (picked - Tensor(old_log_probs, (count,))).exp()
    advantage = Tensor(advantages, (count,))
    gain = ratio * advantage
    # min(r A, clip(r) A) = r A - relu(r A - clip(r) A)
    surrogate = gain - (gain - clip(ratio, 1 - epsilon, 1 + epsilon) * advantage).relu()
    error = values - Tensor(returns, (count,))
    entropy = -(log_probs.exp() * log_probs).sum(axis=1).mean()
    return -surrogate.mean() + (error * error).mean() * 0.5 - entropy * beta


def advantages_and_returns(rewards, values, dones, last_values, *, gamma: float, lam: float):
    """GAE over a (steps, games) rollout, as nested lists; the episode ends cut the sums."""

    steps, games = len(rewards), len(rewards[0])
    advantages = [[0.0] * games for _ in range(steps)]
    following, running = list(last_values), [0.0] * games
    for t in reversed(range(steps)):
        for g in range(games):
            keep = 0.0 if dones[t][g] else 1.0
            delta = rewards[t][g] + gamma * following[g] * keep - values[t][g]
            running[g] = delta + gamma * lam * keep * running[g]
            advantages[t][g] = running[g]
            following[g] = values[t][g]
    returns = [[a + v for a, v in zip(arow, vrow)] for arow, vrow in zip(advantages, values)]
    return advantages, returns


def save(path: Path, model: ActorCritic, meta: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps({"format": "nanograd.mario.v1", "meta": meta, "state": model.state_dict()}))
    temporary.replace(path)


def load(path: Path) -> tuple[ActorCritic, dict]:
    payload = json.loads(Path(path).read_text())
    if payload.get("format") != "nanograd.mario.v1":
        raise ValueError(f"{path}: not a mario.py checkpoint")
    model = ActorCritic(payload["meta"]["actions"])
    model.load_state_dict(payload["state"])
    return model, payload["meta"]


def train(args) -> dict:
    out = Path(args.out)
    rng = random.Random(args.seed)
    games = Games(args.games, args.stage)
    actions = 7
    model = ActorCritic(actions, seed=args.seed)
    meta = {"actions": actions, "stage": args.stage, "steps": 0, "updates": 0, "flags": 0, "best_x": 0}
    if args.resume and (out / "model.json").exists():
        model, meta = load(out / "model.json")
        print(f"resumed at update {meta['updates']}, {meta['steps']:,} steps", flush=True)
    optimizer = TensorAdam(model.parameters(), lr=args.lr, max_grad_norm=args.max_grad_norm)
    batch = args.games * args.steps
    minibatch = batch // args.minibatches
    recent: list[dict] = []
    start = time.perf_counter()
    out.mkdir(parents=True, exist_ok=True)
    log = (out / "log.jsonl").open("a")
    try:
        while meta["steps"] < args.total_steps:
            rollout_start = time.perf_counter()
            frames, rewards, dones, values, chosen, log_probs = [], [], [], [], [], []
            for _ in range(args.steps):
                with no_grad():
                    scores, value = model(screens_tensor(games.frames))
                picks = sample(list(scores.data), actions, rng)
                frames.append(games.frames)
                values.append(list(value.data))
                chosen.append([a for a, _ in picks])
                log_probs.append([p for _, p in picks])
                _, reward, done, episodes = games.step(chosen[-1])
                rewards.append(reward)
                dones.append(done)
                for episode in episodes:
                    meta["flags"] += episode["flag"]
                    meta["best_x"] = max(meta["best_x"], episode["x"])
                recent = (recent + episodes)[-20:]
            with no_grad():
                _, last = model(screens_tensor(games.frames))
            advantages, returns = advantages_and_returns(rewards, values, dones, list(last.data),
                                                         gamma=args.gamma, lam=args.lam)
            rollout_seconds = time.perf_counter() - rollout_start

            update_start = time.perf_counter()
            every = screens_tensor(np.concatenate(frames))            # (steps * games, 4, 84, 84), step-major
            every = every.reshape((batch, STACK * SIZE * SIZE))
            flat = lambda rows: [value for row in rows for value in row]
            all_actions, all_log_probs = flat(chosen), flat(log_probs)
            all_advantages, all_returns = flat(advantages), flat(returns)
            for _ in range(args.epochs):
                order = list(range(batch))
                rng.shuffle(order)
                for first in range(0, batch, minibatch):
                    rows = order[first:first + minibatch]
                    screens = gather_rows(every, rows).reshape((len(rows), STACK, SIZE, SIZE))
                    optimizer.zero_grad()
                    loss = ppo_loss(model, screens, [all_actions[i] for i in rows],
                                    [all_log_probs[i] for i in rows], [all_advantages[i] for i in rows],
                                    [all_returns[i] for i in rows], epsilon=args.epsilon, beta=args.beta)
                    loss.backward()
                    optimizer.step()
            del every
            update_seconds = time.perf_counter() - update_start

            meta["steps"] += batch
            meta["updates"] += 1
            window = recent
            row = {"update": meta["updates"], "steps": meta["steps"], "minutes": (time.perf_counter() - start) / 60,
                   "episodes": len(window), "mean_x": sum(e["x"] for e in window) / max(1, len(window)),
                   "flag_rate": sum(e["flag"] for e in window) / max(1, len(window)), "flags": meta["flags"],
                   "best_x": meta["best_x"], "loss": float(loss.item()),
                   "rollout_s": rollout_seconds, "update_s": update_seconds}
            log.write(json.dumps(row) + "\n")
            log.flush()
            if meta["updates"] % args.log_every == 0:
                print(f"update {row['update']:4d} | {row['steps']:9,} steps | {row['minutes']:6.1f} min | "
                      f"mean x {row['mean_x']:6.0f} | flags {row['flag_rate']:4.0%} (total {row['flags']}) | "
                      f"best x {row['best_x']} | rollout {rollout_seconds:4.1f} s, update {update_seconds:4.1f} s",
                      flush=True)
            if meta["updates"] % args.save_every == 0:
                save(out / "model.json", model, meta)
            if len(window) >= 20 and row["flag_rate"] >= args.solved:
                print(f"solved: {row['flag_rate']:.0%} of the last 20 episodes reached the flag", flush=True)
                break
        save(out / "model.json", model, meta)
    finally:
        log.close()
        games.close()
    return meta


def play(args) -> None:
    model, meta = load(Path(args.play))
    game = Mario(meta["stage"], render=not args.record)
    rng = random.Random(args.seed)
    recorder = None
    if args.record:
        recorder = subprocess.Popen(
            ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", "256x240",
             "-r", str(60 // SKIP), "-i", "-", "-vf", "scale=512:480:flags=neighbor", "-pix_fmt", "yuv420p",
             args.record], stdin=subprocess.PIPE)
    for episode in range(args.episodes):
        frames, done = game.reset(), False
        while not done:
            with no_grad():
                scores, _ = model(screens_tensor(frames[None]))
            if args.greedy:
                values = list(scores.data)
                action = values.index(max(values))
            else:
                action = sample(list(scores.data), meta["actions"], rng)[0][0]
            frames, _, done, info = game.step(action)
            if recorder:
                recorder.stdin.write(np.ascontiguousarray(game.screen).tobytes())
        print(f"episode {episode + 1}: x {info['x_pos']}, {'flag!' if info['flag_get'] else 'died'}", flush=True)
    if recorder:
        recorder.stdin.close()
        recorder.wait()
    game.env.close()


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--stage", default="1-1", help="world-stage, e.g. 1-1")
    parser.add_argument("--out", default="runs/mario")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--games", type=int, default=8)
    parser.add_argument("--steps", type=int, default=512, help="steps per game per rollout")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--minibatches", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--gamma", type=float, default=0.9)
    parser.add_argument("--lam", type=float, default=1.0, help="GAE lambda")
    parser.add_argument("--epsilon", type=float, default=0.2, help="PPO clip range")
    parser.add_argument("--beta", type=float, default=0.01, help="entropy bonus")
    parser.add_argument("--max-grad-norm", type=float, default=0.5)
    parser.add_argument("--total-steps", type=int, default=5_000_000)
    parser.add_argument("--solved", type=float, default=0.9, help="stop once this share of 20 episodes finish")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--log-every", type=int, default=1)
    parser.add_argument("--save-every", type=int, default=10)
    parser.add_argument("--play", help="a checkpoint to watch instead of training")
    parser.add_argument("--record", help="with --play: write an .mp4 instead of opening a window")
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--greedy", action="store_true", help="with --play: always the likeliest action")
    args = parser.parse_args(argv)
    if args.play:
        play(args)
    else:
        print(train(args), flush=True)


if __name__ == "__main__":
    main()
