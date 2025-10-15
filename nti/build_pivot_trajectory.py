import torch
import utils.diffusion_utils as diffusion_utils
from tqdm import tqdm
from diffusers.pipelines.stable_diffusion_3.pipeline_stable_diffusion_3 import (
    retrieve_timesteps,
)

@torch.no_grad()
def build_pivot_trajectory(
    ace_step_transformer,
    target_latents,
    encoder_text_hidden_states,
    text_attention_mask,
    speaker_embds,
    lyric_token_ids,
    lyric_mask,
    random_generators=None,
    infer_steps=60,
    omega_scale=10.0,
    scheduler_type="euler_inverse",
):
    bsz = encoder_text_hidden_states.shape[0]

    if not scheduler_type.endswith('inverse'):
        raise ValueError("Reverse scheduler need for building pivot trajectory")
    
    scheduler = diffusion_utils.get_scheduler(scheduler_type)
    frame_length = target_latents.shape[-1]

    timesteps, num_inference_steps = retrieve_timesteps(
        scheduler,
        num_inference_steps=infer_steps,
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
            omega=omega_scale,
            generator=random_generators[0] if random_generators else None,
        )[0]

        trajectory.append(target_latents.detach().clone())

    return list(reversed(trajectory))
