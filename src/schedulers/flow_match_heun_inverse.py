from typing import Optional, Tuple, Union

import numpy as np
import torch
from acestep.schedulers.scheduling_flow_match_heun_discrete import (
    FlowMatchHeunDiscreteScheduler,
    FlowMatchHeunDiscreteSchedulerOutput,
)


class FlowMatchHeunInverseDiscreteScheduler(FlowMatchHeunDiscreteScheduler):
    def set_timesteps(
        self,
        num_inference_steps: int = None,
        device: Union[str, torch.device] = None,
        **kwargs,
    ):
        super().set_timesteps(
            num_inference_steps=num_inference_steps, device=device, **kwargs
        )

        self.timesteps = torch.flip(self.timesteps, [0])
        self.sigmas = torch.flip(self.sigmas, [0])

        self.prev_derivative = None
        self.dt = None
        self.sample = None
        self._step_index = None
        self._begin_index = None

    def index_for_timestep(self, timestep, schedule_timesteps=None):
        if schedule_timesteps is None:
            schedule_timesteps = self.timesteps
        indices = (schedule_timesteps == timestep).nonzero()
        pos = 0
        return indices[pos].item()

    def _init_step_index(self, timestep):
        if self.begin_index is None:
            if isinstance(timestep, torch.Tensor):
                timestep = timestep.to(self.timesteps.device)
            self._step_index = self.index_for_timestep(timestep)
        else:
            self._step_index = self._begin_index

    @property
    def state_in_first_order(self):
        return self.dt is None

    def _omega_rescale(self, omega):
        def logistic_function(x, L=0.9, U=1.1, x_0=0.0, k=1):
            if isinstance(x, torch.Tensor):
                device_ = x.device
                x = x.to(torch.float).cpu().numpy()
            y = L + (U - L) / (1 + np.exp(-k * (x - x_0)))
            if isinstance(y, np.ndarray):
                y = torch.from_numpy(y).to(device_)
            return y

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
        omega: Union[float, np.array] = 1.0,
    ) -> Union[FlowMatchHeunDiscreteSchedulerOutput, Tuple]:
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

        if self.state_in_first_order:
            sigma = self.sigmas[self.step_index]
            sigma_next = self.sigmas[self.step_index + 1]

            v1 = model_output
            dt = sigma_next - sigma

            self.prev_derivative = v1
            self.dt = dt
            self.sample = sample

            dx = v1 * dt
            m = dx.mean()
            dx = (dx - m) * omega + m
            next_sample = sample + dx

        else:
            sigma = self.sigmas[self.step_index - 1]
            sigma_next = self.sigmas[self.step_index]

            v1 = self.prev_derivative
            v2 = model_output
            v = 0.5 * (v1 + v2)

            dt = self.dt
            sample = self.sample

            self.prev_derivative = None
            self.dt = None
            self.sample = None

            dx = v * dt
            m = dx.mean()
            dx = (dx - m) * omega + m
            next_sample = sample + dx

        next_sample = next_sample.to(model_output.dtype)

        self._step_index += 1

        if not return_dict:
            return (next_sample,)

        return FlowMatchHeunDiscreteSchedulerOutput(prev_sample=next_sample)
