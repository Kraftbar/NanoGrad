"""Tiny datasets used by demos and tests."""

from __future__ import annotations


def line_fitting() -> tuple[list[list[float]], list[float]]:
    """Small scalar regression dataset for learning y = 3x + 1."""

    xs = [
        [-1.0],
        [0.0],
        [1.0],
        [2.0],
    ]
    ys = [
        -2.0,
        1.0,
        4.0,
        7.0,
    ]
    return xs, ys


def sign_separator() -> tuple[list[list[float]], list[float]]:
    """Binary dataset for separating negative and positive scalar inputs."""

    xs = [
        [-2.0],
        [-1.0],
        [1.0],
        [2.0],
    ]
    ys = [
        0.0,
        0.0,
        1.0,
        1.0,
    ]
    return xs, ys


def xor_gate() -> tuple[list[list[float]], list[float]]:
    """XOR logic gate dataset for non-linear binary classification."""

    xs = [
        [0.0, 0.0],
        [0.0, 1.0],
        [1.0, 0.0],
        [1.0, 1.0],
    ]
    ys = [
        0.0,
        1.0,
        1.0,
        0.0,
    ]
    return xs, ys
