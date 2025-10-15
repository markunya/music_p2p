import torch
import torchaudio
import utils.diffusion_utils
import nti
from pipelines.base_p2p_pipeline import BaseAceStepP2PEditPipeline
from nti.build_pivot_trajectory import build_pivot_trajectory
from tqdm import tqdm
from diffusers.pipelines.stable_diffusion_3.pipeline_stable_diffusion_3 import (
    retrieve_timesteps,
)
from nti.null_text_inversion import null_text_optimization

def music2noise(
    pipeline: BaseAceStepP2PEditPipeline,
    path: str,
    lyrics: str,
    tags: str,
    num_steps: int = 800,
    guidance_scale: float = 15.0,
    guidance_interval: float = 0.5,
    debug_mode: bool = False,
    audio_save_path: str = None 
):
    device = pipeline.ace_step_transformer.device
    dtype  = pipeline.ace_step_transformer.dtype
    encoder_text_hidden_states, text_attention_mask = pipeline.get_text_embeddings([tags])

    speaker_embeds = torch.zeros(1, 512).to(device=device, dtype=dtype)
    lyric_token_idx, lyric_mask = pipeline.prepare_lyric_tokens([lyrics], debug=False)

    wav, sr = torchaudio.load(path)
    latents, _ = pipeline.music_dcae.encode(wav.unsqueeze(0).to(device), sr=sr)

    trajectory = build_pivot_trajectory(
        pipeline.ace_step_transformer,
        target_latents=latents,
        encoder_text_hidden_states=encoder_text_hidden_states,
        text_attention_mask=text_attention_mask,
        speaker_embds=speaker_embeds,
        lyric_token_ids=lyric_token_idx,
        lyric_mask=lyric_mask,
        random_generators=None,
        infer_steps=num_steps,
        omega_scale=10.0,
        scheduler_type="euler_inverse"
    )
    if debug_mode:
        latents_rec = pipeline.diffusion_process(
            input_latents=trajectory[0],
            encoder_text_hidden_states=encoder_text_hidden_states,
            text_attention_mask=text_attention_mask,
            speaker_embds=speaker_embeds,
            lyric_token_ids=lyric_token_idx,
            lyric_mask=lyric_mask,
            random_generators=None,
            guidance_scale=1.0,
            infer_steps=num_steps,
            omega_scale=10.0,
            scheduler_type="euler",
        )

        tqdm.write(f"MAE after building pivot (no guidance): {(latents_rec - latents).abs().mean()}")

    scheduler = utils.diffusion_utils.get_scheduler('euler')
    timesteps, _ = retrieve_timesteps(
        scheduler,
        num_steps,
        device=device,
    )

    null_embeddings_per_step, losses_per_step = null_text_optimization(
        pipeline.ace_step_transformer,
        trajectory=trajectory,
        timesteps=timesteps,
        scheduler=scheduler,
        encoder_text_hidden_states=encoder_text_hidden_states,
        text_attention_mask=text_attention_mask,
        speaker_embds=speaker_embeds,
        guidance_scale=guidance_scale,
        guidance_interval=guidance_interval,
        lyric_token_ids=lyric_token_idx,
        lyric_mask=lyric_mask
    )

    if debug_mode:
        tqdm.write(f"Losses per step: {losses_per_step}")

        latents_rec2 = pipeline.diffusion_process(
            input_latents=trajectory[0],
            encoder_text_hidden_states=encoder_text_hidden_states,
            text_attention_mask=text_attention_mask,
            speaker_embds=speaker_embeds,
            lyric_token_ids=lyric_token_idx,
            lyric_mask=lyric_mask,
            random_generators=None,
            guidance_scale=guidance_scale,
            guidance_interval=guidance_interval,
            infer_steps=num_steps,
            omega_scale=10.0,
            scheduler_type="euler",
            null_embeddings_per_step=null_embeddings_per_step
        )

        tqdm.write(f"MAE after null text optimization (guidance): {(latents_rec2 - latents).abs().mean()}")
        pipeline.latents2audio(latents_rec2, save_path=audio_save_path)
    
    return trajectory[0], null_embeddings_per_step