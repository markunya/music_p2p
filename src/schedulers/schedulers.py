from enum import Enum, auto

from acestep.schedulers.scheduling_flow_match_euler_discrete import (
    FlowMatchEulerDiscreteScheduler,
)
from acestep.schedulers.scheduling_flow_match_heun_discrete import (
    FlowMatchHeunDiscreteScheduler,
)

from src.schedulers.flow_match_euler_inverse import (
    FlowMatchEulerInverseDiscreteScheduler,
)
from src.schedulers.flow_match_heun_inverse import FlowMatchHeunInverseDiscreteScheduler


class SchedulerType(Enum):
    Euler = auto()
    Heun = auto()


_default_scheduler_params = {"num_train_timesteps": 1000, "shift": 3.0}
_invalid_scheduler_type_msg = "Unsupported scheduler type: {scheduler_type}"


def get_inverse_scheduler(scheduler_type: SchedulerType):
    match scheduler_type:
        case SchedulerType.Euler:
            return FlowMatchEulerInverseDiscreteScheduler(**_default_scheduler_params)
        case SchedulerType.Heun:
            return FlowMatchHeunInverseDiscreteScheduler(**_default_scheduler_params)
        case _:
            raise ValueError(
                _invalid_scheduler_type_msg.format(scheduler_type=scheduler_type)
            )


def get_direct_scheduler(scheduler_type: SchedulerType):
    match scheduler_type:
        case SchedulerType.Euler:
            return FlowMatchEulerDiscreteScheduler(**_default_scheduler_params)
        case SchedulerType.Heun:
            return FlowMatchHeunDiscreteScheduler(**_default_scheduler_params)
        case _:
            raise ValueError(
                _invalid_scheduler_type_msg.format(scheduler_type=scheduler_type)
            )
