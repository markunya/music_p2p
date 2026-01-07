from typing import List, Optional, Union

import numpy as np
import torch
from acestep.schedulers.scheduling_flow_match_euler_discrete import (
    FlowMatchEulerDiscreteScheduler,
    FlowMatchEulerDiscreteSchedulerOutput,
)


class FlowMatchEulerInverseDiscreteScheduler(FlowMatchEulerDiscreteScheduler):
    def set_timesteps(
        self,
        num_inference_steps: int = None,
        device: Union[str, torch.device] = None,
        sigmas: Optional[List[float]] = None,
        mu: Optional[float] = None,
    ):
        if self.config.use_dynamic_shifting and mu is None:
            raise ValueError("Need `mu` if use_dynamic_shifting=True")

        if sigmas is None:
            self.num_inference_steps = num_inference_steps
            timesteps = np.linspace(
                self._sigma_to_t(self.sigma_min),
                self._sigma_to_t(self.sigma_max),
                num_inference_steps,
            )
            sigmas = timesteps / self.config.num_train_timesteps

        if self.config.use_dynamic_shifting:
            sigmas = self.time_shift(mu, 1.0, sigmas)
        else:
            sigmas = self.config.shift * sigmas / (1 + (self.config.shift - 1) * sigmas)

        sigmas = torch.from_numpy(sigmas).to(dtype=torch.float32, device=device)
        timesteps = sigmas * self.config.num_train_timesteps

        self.timesteps = timesteps.to(device=device)
        self.sigmas = torch.cat([torch.zeros(1, device=sigmas.device), sigmas])

        self._step_index = None
        self._begin_index = None

    def _omega_rescale(self, omega):
        def logistic_function(x, L=0.9, U=1.1, x_0=0.0, k=1):
            if isinstance(x, torch.Tensor):
                device_ = x.device
                x = x.to(torch.float).cpu().numpy()
            new_x = L + (U - L) / (1 + np.exp(-k * (x - x_0)))
            if isinstance(new_x, np.ndarray):
                new_x = torch.from_numpy(new_x).to(device_)
            return new_x

        self.omega_bef_rescale = omega
        omega = logistic_function(omega, k=0.1)
        self.omega_aft_rescale = omega
        return omega

    def step(
        self,
        model_output: torch.FloatTensor,
        timestep: Union[float, torch.FloatTensor],
        sample: torch.FloatTensor,
        s_churn: float = 0.0,
        s_tmin: float = 0.0,
        s_tmax: float = float("inf"),
        s_noise: float = 1.0,
        generator: Optional[torch.Generator] = None,
        return_dict: bool = True,
        omega: Union[float, np.array] = 0.0,
    ):
        if (
            isinstance(timestep, int)
            or isinstance(timestep, torch.IntTensor)
            or isinstance(timestep, torch.LongTensor)
        ):
            raise ValueError(
                "Pass actual scheduler.timesteps value, not integer index."
            )

        if self.step_index is None:
            self._init_step_index(timestep)

        sample = sample.to(torch.float32)
        omega = self._omega_rescale(omega)

        sigma = self.sigmas[self.step_index]
        if self.step_index + 1 >= len(self.sigmas):
            raise IndexError(
                "Inverse step out of bounds: no sigma_next. Check timesteps length."
            )
        sigma_next = self.sigmas[self.step_index + 1]

        dx = (sigma_next - sigma) * model_output
        m = dx.mean()
        dx_ = (dx - m) * omega + m
        next_sample = sample + dx_

        next_sample = next_sample.to(model_output.dtype)

        self._step_index += 1

        if not return_dict:
            return (next_sample,)

        return FlowMatchEulerDiscreteSchedulerOutput(prev_sample=next_sample)
