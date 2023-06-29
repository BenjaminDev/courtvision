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


class StateIdx:
    """
    Named indices for the state tensor.
    """

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


class ParticleFilter:
    """_summary_"""

    def __init__(
        self,
        *,
        num_particles: int,
        court_size: torch.Tensor,
        world_to_cam: torch.Tensor | None = None,
        cam_to_image: torch.Tensor | None = None,
        # torch.tensor(
        # [PadelCourt.width, PadelCourt.length, PadelCourt.backall_fence_height]
        # ),
    ) -> None:
        self.reset(
            num_particles=num_particles,
            court_size=court_size,
            world_to_cam=world_to_cam,
            cam_to_image=cam_to_image,
        )

    def reset(
        self,
        *,
        num_particles: int,
        court_size: torch.tensor,
        world_to_cam: torch.Tensor | None = None,
        cam_to_image: torch.Tensor | None = None,
    ) -> None:
        self.num_particles = num_particles
        # Court size is a tensor of the form [width, length, height]
        self.court_size = court_size

        # state: [x, y, z, vx, vy, zv, ax, ay, az, ax, ay, az, weight]
        self.states = torch.randn(
            (num_particles, 3 * 3 + 1),
            names=["num_particles", "state"],
        )
        # Place a prior on the initial state X, Y, Z to be on the court

        self.states[:, StateIdx.x] = (
            self.states[:, StateIdx.x] * court_size[StateIdx.x]
            + court_size[StateIdx.x] / 2
        )
        self.states[:, StateIdx.y] = (
            self.states[:, StateIdx.y] * court_size[StateIdx.y]
            + court_size[StateIdx.y] / 2
        )
        self.states[:, StateIdx.z] = (
            self.states[:, StateIdx.z] * court_size[StateIdx.z]
            + court_size[StateIdx.z] / 2
        )
        self.states[:, StateIdx.weight] = torch.abs(
            self.normalized_weights(self.states)
        )
        # Set the initial velocity to be zero
        self.states[:, StateIdx.vx : StateIdx.vz + 1] = 0.0
        # Set the initial acceleration to be zero but -9.8 in the z direction
        self.states[:, StateIdx.ax : StateIdx.az + 1] = 0.0
        self.states[:, StateIdx.az] = -9.8

        self.cam_to_image = torch.randn((3, 3))
        if cam_to_image is not None:
            self.cam_to_image = cam_to_image.to(dtype=torch.float32)
        self.world_to_cam = torch.randn((4, 4))
        if world_to_cam is not None:
            self.world_to_cam = world_to_cam.to(dtype=torch.float32)

    def set_states_to(self, point: torch.tensor):
        # tracker.states[:,0:3] = torch.tensor([1.0, 2.0, 3.0]).repeat((1000,1))

        self.states[:, StateIdx.x : StateIdx.z + 1] = point.repeat(
            (self.num_particles, 1)
        )

    @property
    def xyz(self) -> torch.tensor:
        return self.states[:, StateIdx.x : StateIdx.z + 1].clone()

    @property
    def mean_image_plane_prediction(self) -> torch.tensor:
        return self.state_to_observation(
            self.xyz_mean,
            world_to_cam=self.world_to_cam,
            cam_to_image=self.cam_to_image,
        )

    @property
    def xyz_mean(self) -> torch.tensor:

        xyz_mean = (
            self.xyz.rename(None).T @ self.states[:, StateIdx.weight].rename(None)
        ) / self.states[:, StateIdx.weight].sum()
        return xyz_mean.unsqueeze(0)

    @staticmethod
    def normalized_weights(states: torch.tensor) -> torch.tensor:
        return (
            states.select("state", StateIdx.weight)
            / states.select("state", StateIdx.weight).sum()
        )

    @staticmethod
    def state_to_observation(state, *, world_to_cam, cam_to_image):

        x_y_z_1_positions = convert_points_to_homogeneous(
            state[:, : StateIdx.z + 1].rename(None)
        )
        x_y_z_1_positions_cam = convert_points_from_homogeneous(
            x_y_z_1_positions @ world_to_cam.T
        )
        return convert_points_from_homogeneous(x_y_z_1_positions_cam @ cam_to_image.T)

    def likelihood_in_state_space(
        self, obs_state: torch.tensor, pred_state: torch.tensor
    ):
        # unproject_points
        ...

    # # Define the likelihood function
    def likelihood(self, obs_state: torch.tensor, pred_state: torch.tensor):
        sigma = 20.1  # Standard deviation of the observation noise
        from torch.nn.functional import mse_loss

        pred_obs = self.state_to_observation(
            pred_state, world_to_cam=self.world_to_cam, cam_to_image=self.cam_to_image
        )
        mse = mse_loss(pred_obs, obs_state.expand_as(pred_obs), reduction="none").sum(
            dim=1
        )
        # p = torch.exp(-0.5 * mse / sigma)
        p = torch.max(torch.exp(-0.5 * mse / sigma) ** 2, torch.tensor(1e-3))
        return p

    def predict(self, dt: float = 1.0 / 30.0):
        # Define the transition model
        # state: [x, y, z, vx, vy, zv, ax, ay, az, ax, ay, az, weight]

        # Random walk in the x, y, and z directions
        self.states[:, StateIdx.x : StateIdx.z + 1] = (
            self.states[:, StateIdx.x : StateIdx.z + 1]
            + torch.randn((self.num_particles, 3)) * 5.0
        )

        # Update the state using the velocity
        # self.states[:, StateIdx.x : StateIdx.z + 1] += (
        #     self.states[:, StateIdx.vx : StateIdx.vz + 1]
        #     * dt
        #     # + torch.randn(
        #     #     (self.num_particles, 3)
        #     # )*5.0
        # )

        # Ensure state is on the court using clamp
        self.states[:, StateIdx.x] = torch.clamp(
            self.states[:, StateIdx.x], 0.0, self.court_size[StateIdx.x]
        )
        self.states[:, StateIdx.y] = torch.clamp(
            self.states[:, StateIdx.y], 0.0, self.court_size[StateIdx.y]
        )
        self.states[:, StateIdx.z] = torch.clamp(
            self.states[:, StateIdx.z], 0.0, self.court_size[StateIdx.z] * 1.5
        )
        # # Z cannot be less than zero
        # self.states[:, StateIdx.z] = torch.max(self.states[:, StateIdx.z].rename(None), torch.tensor(0.0))

        # self.states[:, StateIdx.vx : StateIdx.vz + 1] += (
        #     self.states[:, StateIdx.ax : StateIdx.az + 1] * dt
        # )
        # randomize the acceleration in the x and y directions
        # self.states[:, StateIdx.ax : StateIdx.az + 1] += torch.randn(
        #     (self.num_particles, 3)
        # )
        # Set the acceleration in the z direction to be -9.8
        # and the acceleration in the x and y directions to be zero
        # self.states[:, StateIdx.ax : StateIdx.az + 1] = 0.0
        # self.states[:, StateIdx.az] = -9.8

    def update(self, obs_state: torch.tensor, score: torch.tensor = torch.tensor(1.0)):
        # Define the likelihood function
        likelihoods = self.likelihood(obs_state, self.states)
        self.states[:, StateIdx.weight] = self.update_weights(
            self.states[:, StateIdx.weight], likelihoods
        )
        self.states = self.resample(self.states, self.states[:, StateIdx.weight])

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
