from acestep.schedulers.scheduling_flow_match_euler_discrete import (
    FlowMatchEulerDiscreteScheduler,
)
from acestep.schedulers.scheduling_flow_match_heun_discrete import (
    FlowMatchHeunDiscreteScheduler,
)
from acestep.schedulers.scheduling_flow_match_pingpong import (
    FlowMatchPingPongScheduler,
)
from schedulers.flow_match_euler_inverse import FlowMatchEulerInverseDiscreteScheduler
from schedulers.flow_match_heun_inverse import FlowMatchHeunInverseDiscreteScheduler

def get_scheduler(scheduler_type):
    if scheduler_type == "euler":
        return FlowMatchEulerDiscreteScheduler(
            num_train_timesteps=1000,
            shift=3.0,
        )
    elif scheduler_type == 'euler_inverse':
        return FlowMatchEulerInverseDiscreteScheduler(
            num_train_timesteps=1000,
            shift=3.0,
        )
    elif scheduler_type == "heun":
        return FlowMatchHeunDiscreteScheduler(
            num_train_timesteps=1000,
            shift=3.0,
        )
    elif scheduler_type == "heun_inverse":
        return FlowMatchHeunInverseDiscreteScheduler(
            num_train_timesteps=1000,
            shift=3.0,
        )
    elif scheduler_type == "pingpong":
        return FlowMatchPingPongScheduler(
            num_train_timesteps=1000,
            shift=3.0,
        )
    else:
        raise ValueError("Invalid scheduler type")