import torch
from acestep.apg_guidance import cfg_forward

from src.utils.structures import CfgType, GuidanceParams


def mix_guidance(
    cfg_type: CfgType,
    noise_cond: torch.Tensor,
    noise_null: torch.Tensor,
    gscale: float,
    # momentum_buffer=None,
    # i=None,
):
    match cfg_type:
        case CfgType.CFG:
            return cfg_forward(
                cond_output=noise_cond,
                uncond_output=noise_null,
                cfg_strength=gscale,
            )
        # case CfgType.APG:
        #     return apg_forward(
        #         pred_cond=noise_cond,
        #         pred_uncond=noise_null,
        #         guidance_scale=gscale,
        #         momentum_buffer=momentum_buffer,
        #     )
        # case CfgType.CFG_STAR:
        #     return cfg_zero_star(
        #         noise_pred_with_cond=noise_cond,
        #         noise_pred_uncond=noise_null,
        #         guidance_scale=gscale,
        #         i=0 if i is None else i,
        #         zero_steps=1,
        #         use_zero_init=True,
        #     )
        case _:
            raise ValueError(f"Unsupported cfg_type: {cfg_type}")


def compute_current_guidance(
    i: int, start_idx: int, end_idx: int, guidance_params: GuidanceParams
):
    """
    Compute the guidance scale for step `i` within an active guidance interval.

    Returns:
        (scale, is_active):
          - scale (float): guidance scale for the current step. If `i` is outside
            [start_idx, end_idx), returns 1.0.
          - is_active (bool): True if `i` is inside [start_idx, end_idx), else False.

    Decay logic:
        If `guidance_interval_decay > 0` and the interval length is > 1, the scale
        linearly decays from `guidance_scale` toward `min_guidance_scale` over the
        interval, with strength controlled by `guidance_interval_decay`.
    """
    active = start_idx <= i < end_idx
    if not active:
        return 1.0, False

    gs = guidance_params.guidance_scale
    decay = guidance_params.guidance_interval_decay
    min_gs = guidance_params.min_guidance_scale

    if decay <= 0 or end_idx - start_idx <= 1:
        return float(gs), True

    denom = end_idx - start_idx - 1
    progress = (i - start_idx) / denom
    cur = gs - (gs - min_gs) * progress * decay

    return float(cur), True
