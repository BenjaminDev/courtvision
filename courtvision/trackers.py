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
    """
    Particle filter tracker.
    """

    def __init__(
        self,
        *,
        num_particles: int,
        court_size: torch.Tensor,
        world_to_cam: torch.Tensor | None = None,
        cam_to_image: torch.Tensor | None = None,
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
        # Set the initial acceleration to be zero but -9.8 m/s^2 in the z direction
        self.states[:, StateIdx.ax : StateIdx.az + 1] = 0.0
        self.states[:, StateIdx.az] = -0.98  # Note: units are in 1e-1 m/s^2

        self.cam_to_image = torch.randn((3, 3))
        if cam_to_image is not None:
            self.cam_to_image = cam_to_image.to(dtype=torch.float32)
        self.world_to_cam = torch.randn((4, 4))
        if world_to_cam is not None:
            self.world_to_cam = world_to_cam.to(dtype=torch.float32)

    def set_states_to(self, point: torch.tensor):
        """Set the state of the tracker to a single point.

        Args:
            point (torch.tensor): point in the world space. Shape: [3]
        """
        self.states[:, StateIdx.x : StateIdx.z + 1] = point.repeat(
            (self.num_particles, 1)
        )

    @property
    def xyz(self) -> torch.tensor:
        """Grab the xyz coordinates of the tracker.

        Returns:
            torch.tensor: [X,Y,Z] coordinates of the tracker. Shape: [N, 3]
        """
        return self.states[:, StateIdx.x : StateIdx.z + 1].clone()

    @property
    def mean_image_plane_prediction(self) -> torch.tensor:
        """Computes the weighted mean of the trackers state and
        projects it to the image plane.

        Returns:
            torch.tensor: [x,y] coordinates of the tracker mean estimate in the image plane. Shape: [1, 2]
        """
        return self.state_to_observation(
            self.xyz_mean,
            world_to_cam=self.world_to_cam,
            cam_to_image=self.cam_to_image,
        )

    @property
    def xyz_mean(self) -> torch.tensor:
        """Grab the weighted mean of the xyz coordinates of the tracker.

        Returns:
            torch.tensor: Weighted mean of the [X,Y,Z] coordinates of the tracker. Shape: [1, 3]
        """
        xyz_mean = (
            self.xyz.rename(None).T @ self.states[:, StateIdx.weight].rename(None)
        ) / self.states[:, StateIdx.weight].rename(None).sum()
        return xyz_mean.unsqueeze(0)

    @staticmethod
    def normalized_weights(states: torch.tensor) -> torch.tensor:

        return (
            states.select("state", StateIdx.weight).rename(None)
            / states.select("state", StateIdx.weight).rename(None).sum()
        )

    @staticmethod
    def state_to_observation(
        state: torch.Tensor,
        *,
        world_to_cam: torch.Tensor,
        cam_to_image: torch.Tensor,
    ) -> torch.Tensor:
        """Map the state to the observation space.
        This is from the 3D world space to the 2D image plane.

        Args:
            state (torch.Tensor): Tracker state. Shape: [N, state_dim]
            world_to_cam (torch.Tensor): _description_
            cam_to_image (torch.Tensor): _description_

        Returns:
            torch.Tensor: _description_
        """
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
    def likelihood(
        self, obs_state: torch.tensor, pred_state: torch.tensor
    ) -> torch.tensor:
        """Compute the likelihood of the observation given the predicted state.

        Args:
            obs_state (torch.tensor): Observation in the image plane. Shape: [2]
            pred_state (torch.tensor): Predicted state. Shape: [N, state_dim]

        Returns:
            orch.tensor: likelihood of the observation given the predicted state. Shape: [N]
        """
        sigma = 20.1  # Standard deviation of the observation noise
        from torch.nn.functional import mse_loss

        pred_obs = self.state_to_observation(
            pred_state, world_to_cam=self.world_to_cam, cam_to_image=self.cam_to_image
        )
        mse = mse_loss(pred_obs, obs_state.expand_as(pred_obs), reduction="none").sum(
            dim=1
        )
        p = torch.max(torch.exp(-0.5 * mse / sigma) ** 2, torch.tensor(1e-3))
        return p

    def predict(self, dt: float = 1.0 / 30.0):
        """
        Predict the next state using the current state and the time step.
        `p(x_t | x_{t-1}) ~ N(x_t; x_{t-1} + v_{t-1} * dt, sigma^2)`


        Args:
            dt (float, optional): Time step in [s]. Defaults to 1.0/30.0.
        """

        # state: [x, y, z, vx, vy, zv, ax, ay, az, ax, ay, az, weight]
        # Random walk in the x, y, and z directions
        # self.states[:, StateIdx.x : StateIdx.z + 1] = (
        #     self.states[:, StateIdx.x : StateIdx.z + 1]
        #     + torch.randn((self.num_particles, 3)) * 5.0
        # )

        # Update the state using the velocity
        self.states[:, StateIdx.x : StateIdx.z + 1] += (
            self.states[:, StateIdx.vx : StateIdx.vz + 1] * dt
        ) + 0.5 * (self.states[:, StateIdx.ax : StateIdx.az + 1] * dt**2)

        # Ensure state is on the court using clamp.
        self.states[:, StateIdx.x] = torch.clamp(
            self.states[:, StateIdx.x], 0.0, self.court_size[StateIdx.x]
        )
        self.states[:, StateIdx.y] = torch.clamp(
            self.states[:, StateIdx.y], 0.0, self.court_size[StateIdx.y]
        )
        self.states[:, StateIdx.z] = torch.clamp(
            self.states[:, StateIdx.z], 0.0, self.court_size[StateIdx.z] * 1.5
        )

        # Update the velocity using the acceleration + jitter
        self.states[:, StateIdx.vx : StateIdx.vz + 1] += (
            self.states[:, StateIdx.ax : StateIdx.az + 1] * dt
            + torch.randn((self.num_particles, 3)) * 1.5
        )

        self.states[:, StateIdx.ax : StateIdx.az + 1] = (
            torch.randn((self.num_particles, 3)) * 0.9
        )
        self.states[:, StateIdx.az] = -0.98  # Note: units are in 1e-1 m/s^2

    def update(self, obs_state: torch.tensor, score: torch.tensor = torch.tensor(1.0)):
        """Update the state using the observation and it's associated score.

        Args:
            obs_state (torch.tensor): measurement in the image plane. Shape: [2]
            score (torch.tensor, optional): If the detector assigns a score this
                                            can be used in the update step.
                                            Defaults to torch.tensor(1.0).
        """
        likelihoods = self.likelihood(obs_state, self.states) * score
        self.states[:, StateIdx.weight] = self.update_weights(
            self.states[:, StateIdx.weight], likelihoods
        )
        self.states = self.resample(self.states, self.states[:, StateIdx.weight])

    @staticmethod
    def update_weights(
        weights: torch.tensor, likelihoods: torch.tensor
    ) -> torch.tensor:
        """Given the current weights and the likelihoods, update the weights.

        Args:
            weights (torch.tensor): Nx1 tensor of weights
            likelihoods (torch.tensor): Nx1 tensor of likelihoods

        Returns:
            torch.tensor: Nx1 tensor of updated weights
        """
        return (
            weights.rename(None)
            * likelihoods
            / (weights.rename(None) * likelihoods).sum()
        )

    @staticmethod
    def resample(states: torch.tensor, weights: torch.tensor) -> torch.tensor:
        """Given a set of states and associated weights, resample the states.

        Args:
            states (torch.tensor): Tracker state. Shape: [N, state_dim]
            weights (torch.tensor): weights associated with each state. Shape: [N x 1]

        Returns:
            torch.tensor: returns the resampled states. Shape: [N, state_dim]
        """
        if weights.names is not None:
            weights = weights.rename(None)
        if states.names is not None:
            states = states.rename(None)

        return states[
            torch.multinomial(weights, len(weights), replacement=True)
        ].rename("num_particles", "state")
