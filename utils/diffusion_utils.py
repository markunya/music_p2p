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
from acestep.apg_guidance import (
    apg_forward,
    project,
    MomentumBuffer,
    cfg_forward,
    cfg_zero_star,
)

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
    
def mix_guidance(
    cfg_type,
    noise_cond,
    noise_null,
    gscale,
    momentum_buffer=None,
    i=None
):
    if cfg_type == "apg":
        return apg_forward(
            pred_cond=noise_cond,
            pred_uncond=noise_null,
            guidance_scale=gscale,
            momentum_buffer=momentum_buffer,
        )
    elif cfg_type == "cfg":
        return cfg_forward(
            cond_output=noise_cond,
            uncond_output=noise_null,
            cfg_strength=gscale,
        )
    elif cfg_type == "cfg_star":
        return cfg_zero_star(
            noise_pred_with_cond=noise_cond,
            noise_pred_uncond=noise_null,
            guidance_scale=gscale,
            i=0 if i is None else i,
            zero_steps=1,
            use_zero_init=True,
        )
    else:
        raise ValueError(f"Unknown cfg_type: {cfg_type}")
    
def compute_current_guidance(
    i,
    start_idx,
    end_idx,
    guidance_scale,
    min_guidance_scale,
    guidance_interval_decay
):
    if not (start_idx <= i < end_idx):
        return 1.0, False
    if guidance_interval_decay > 0 and end_idx - start_idx > 1:
        progress = (i - start_idx) / (end_idx - start_idx - 1)
        cur = guidance_scale - (guidance_scale - min_guidance_scale) * progress * guidance_interval_decay
    else:
        cur = guidance_scale
    return float(cur), True
