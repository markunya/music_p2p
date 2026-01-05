import torch
import utils.diffusion_utils as diffusion_utils
from tqdm import tqdm

from diffusers.pipelines.stable_diffusion_3.pipeline_stable_diffusion_3 import (
    retrieve_timesteps,
)
from src.utils.diffusion_utils import DiffusionParams
from src.schedulers import get_inverse_scheduler

@torch.no_grad()
def build_pivot_trajectory(
    ace_step_transformer,
    target_latents,
    encoder_text_hidden_states,
    text_attention_mask,
    speaker_embds,
    lyric_token_ids,
    lyric_mask,
    diffusion_params: DiffusionParams,
    random_generators=None
):
    bsz = encoder_text_hidden_states.shape[0]
    
    scheduler = get_inverse_scheduler(diffusion_params.scheduler_type)
    frame_length = target_latents.shape[-1]

    timesteps, num_inference_steps = retrieve_timesteps(
        scheduler,
        num_inference_steps=diffusion_params.num_steps,
        device=ace_step_transformer.device,
        timesteps=None,
    )

    attention_mask = torch.ones(
        bsz,
        frame_length,
        device=ace_step_transformer.device,
        dtype=ace_step_transformer.dtype
    )

    encoder_hidden_states, encoder_hidden_mask = ace_step_transformer.encode(
        encoder_text_hidden_states,
        text_attention_mask,
        speaker_embds,
        lyric_token_ids,
        lyric_mask,
    )

    trajectory = [target_latents.detach().clone()]
    for i, t in tqdm(enumerate(timesteps), total=num_inference_steps, desc="Building pivot trajectory..."):
        timestep = t.expand(target_latents.shape[0])

        noise_pred = ace_step_transformer.decode(
            hidden_states=target_latents,
            attention_mask=attention_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_hidden_mask=encoder_hidden_mask,
            output_length=target_latents.shape[-1],
            timestep=timestep,
        ).sample

        target_latents = scheduler.step(
            model_output=noise_pred,
            timestep=t,
            sample=target_latents,
            return_dict=False,
            omega=diffusion_params.omega_scale,
            generator=random_generators[0] if random_generators else None,
        )[0]

        trajectory.append(target_latents.detach().clone())

    return list(reversed(trajectory))
