import torch
from diffusers.pipelines.stable_diffusion_3.pipeline_stable_diffusion_3 import (
    retrieve_timesteps,
)
from tqdm import tqdm

from src.schedulers import get_direct_scheduler
from src.utils.structures import DiffusionOut, DiffusionParams


@torch.no_grad()
def build_pivot_trajectory(
    model,
    target_latents,
    encoder_text_hidden_states,
    text_attention_mask,
    speaker_embds,
    lyric_token_ids,
    lyric_mask,
    diffusion_params: DiffusionParams,
) -> DiffusionOut:
    bsz = encoder_text_hidden_states.shape[0]

    scheduler = get_direct_scheduler(diffusion_params.scheduler_type)
    frame_length = target_latents.shape[-1]

    timesteps, num_inference_steps = retrieve_timesteps(
        scheduler,
        num_inference_steps=diffusion_params.num_steps,
        device=model.device,
        timesteps=None,
    )
    print(timesteps)
    timesteps[0] = timesteps[-1]
    timesteps = torch.roll(timesteps, shifts=-1)
    print(timesteps)

    attention_mask = torch.ones(
        bsz,
        frame_length,
        device=model.device,
        dtype=model.dtype,
    )

    encoder_hidden_states, encoder_hidden_mask = model.encode(
        encoder_text_hidden_states,
        text_attention_mask,
        speaker_embds,
        lyric_token_ids,
        lyric_mask,
    )

    trajectory = [target_latents.detach().clone()]
    model_outs = []
    scheduler._step_index = diffusion_params.num_steps - 1

    for t in tqdm(
        reversed(timesteps),
        total=num_inference_steps,
        desc="Building pivot trajectory...",
    ):
        timestep = t.expand(target_latents.shape[0])

        noise_pred = model.decode(
            hidden_states=target_latents,
            attention_mask=attention_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_hidden_mask=encoder_hidden_mask,
            output_length=target_latents.shape[-1],
            timestep=timestep,
        ).sample
        model_outs.append(noise_pred)

        target_latents = scheduler.step(
            model_output=-noise_pred,
            timestep=t,
            sample=target_latents,
            return_dict=False,
            omega=diffusion_params.omega_scale,
        )[0]
        scheduler._step_index -= 2

        trajectory.append(target_latents.detach().clone())

    return DiffusionOut(
        trajectory=list(reversed(trajectory)), model_outs=list(reversed(model_outs))
    )
