# Define a type for a named tensor with two dimensions
# ImagePlaneDetections = torch.TensorType[("x", float), ("y", float)]
import enum
from enum import Enum, IntEnum, auto
from typing import Any

import torch
from kornia.geometry import (
    convert_points_from_homogeneous,
    convert_points_to_homogeneous,
)
from torch import tensor


class StateIdx:
    x: int = 0
    y: int = 1
    z: int = 2
    vx: int = 3
    vy: int = 4
    vz: int = 5
    ax: int = 6
    ay: int = 7
    az: int = 8
    weight: int = 9


class Tracker:
    def __init__(
        self, num_particles: int = 1000, world_to_cam: torch.Tensor | None = None
    ) -> None:
        self.num_particles = num_particles
        # state: [x, y, z, vx, vy, zv, ax, ay, az, ax, ay, az, weight]
        self.states = torch.randn(
            (num_particles, 3 * 3 + 1),
            names=["num_particles", "state"],
        )
        self.states[:, StateIdx.weight] = torch.abs(
            self.normalized_weights(self.states)
        )

        self.H = torch.randn((3, 4))
        if world_to_cam is not None:
            self.H = world_to_cam.to(dtype=torch.float32)

    def set_states_to(self, point: torch.tensor):
        # tracker.states[:,0:3] = torch.tensor([1.0, 2.0, 3.0]).repeat((1000,1))
        self.states[:, StateIdx.x : StateIdx.z + 1] = point.repeat(
            (self.num_particles, 1)
        )

    @staticmethod
    def normalized_weights(states: torch.tensor) -> torch.tensor:
        return (
            states.select("state", StateIdx.weight)
            / states.select("state", StateIdx.weight).sum()
        )

    @staticmethod
    def state_to_observation(state, H):
        x_y_z_1_positions = convert_points_to_homogeneous(state[:, :3].rename(None))
        return convert_points_from_homogeneous((H @ x_y_z_1_positions.T).T)

    # # Define the likelihood function
    @staticmethod
    def likelihood(obs_state: torch.tensor, pred_state: torch.tensor):
        sigma = 0.1  # Standard deviation of the observation noise
        from torch.nn.functional import mse_loss

        mse = mse_loss(pred_state, obs_state, reduction="none").sum(dim=1)
        p = torch.min(torch.exp(0.5 * mse / sigma) ** 2, torch.tensor(1e-3))
        return p

    @staticmethod
    def update_weights(
        weights: torch.tensor, likelihoods: torch.tensor
    ) -> torch.tensor:
        return weights * likelihoods / (weights * likelihoods).sum()

    @staticmethod
    def resample(states: torch.tensor, weights: torch.tensor) -> torch.tensor:
        if weights.names is not None:
            weights = weights.rename(None)
        if states.names is not None:
            states = states.rename(None)

        return states[
            torch.multinomial(weights, len(weights), replacement=True)
        ].rename("num_particles", "state")
