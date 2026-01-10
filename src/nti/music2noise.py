import torch
import torchaudio
from diffusers.pipelines.stable_diffusion_3.pipeline_stable_diffusion_3 import (
    retrieve_timesteps,
)

from src.logging import utils as logging
from src.nti.build_pivot_trajectory import build_pivot_trajectory
from src.nti.null_text_inversion import null_text_optimization
from src.pipelines.base_p2p_pipeline import BaseAceStepP2PEditPipeline
from src.schedulers import get_direct_scheduler
from src.utils.structures import DiffusionParams, InvertedMusicData, Prompt


def music2noise(
    pipeline: BaseAceStepP2PEditPipeline,
    music_path: str,
    prompt: Prompt,
    diffusion_params: DiffusionParams,
    debug_mode: bool = False,
    audio_save_path: str = None,
) -> InvertedMusicData:
    device = pipeline.ace_step_transformer.device
    dtype = pipeline.ace_step_transformer.dtype
    encoder_text_hidden_states, text_attention_mask = pipeline.get_text_embeddings(
        [prompt.tags]
    )

    speaker_embeds = torch.zeros(1, 512).to(device=device, dtype=dtype)
    lyric_token_idx, lyric_mask = pipeline.prepare_lyric_tokens(
        [prompt.lyrics], debug=False
    )

    wav, sr = torchaudio.load(music_path)
    latents, _ = pipeline.music_dcae.encode(wav.unsqueeze(0).to(device), sr=sr)

    trajectory = build_pivot_trajectory(
        pipeline.ace_step_transformer,
        target_latents=latents,
        encoder_text_hidden_states=encoder_text_hidden_states,
        text_attention_mask=text_attention_mask,
        speaker_embds=speaker_embeds,
        lyric_token_ids=lyric_token_idx,
        lyric_mask=lyric_mask,
        diffusion_params=diffusion_params,
        random_generators=None,
    )
    if debug_mode:
        latents_rec = pipeline.diffusion_process(
            input_latents=trajectory[0],
            encoder_text_hidden_states=encoder_text_hidden_states,
            text_attention_mask=text_attention_mask,
            speaker_embds=speaker_embeds,
            lyric_token_ids=lyric_token_idx,
            lyric_mask=lyric_mask,
            diffusion_params=diffusion_params,
            random_generators=None,
        )

        logging.info(
            f"MAE after building pivot (no guidance): {(latents_rec - latents).abs().mean()}"
        )

    scheduler = get_direct_scheduler(diffusion_params.scheduler_type)
    timesteps, _ = retrieve_timesteps(
        scheduler,
        diffusion_params.num_steps,
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
        lyric_token_ids=lyric_token_idx,
        lyric_mask=lyric_mask,
        guidance_params=diffusion_params.guidance_params,
        omega_scale=diffusion_params.omega_scale,
    )

    if debug_mode:
        logging.debug(f"Losses per step: {losses_per_step}")

        latents_rec2 = pipeline.diffusion_process(
            input_latents=trajectory[0],
            encoder_text_hidden_states=encoder_text_hidden_states,
            text_attention_mask=text_attention_mask,
            speaker_embds=speaker_embeds,
            lyric_token_ids=lyric_token_idx,
            lyric_mask=lyric_mask,
            diffusion_params=diffusion_params,
            random_generators=None,
            null_embeddings_per_step=null_embeddings_per_step,
        )

        logging.debug(
            f"MAE after null text optimization (guidance): {(latents_rec2 - latents).abs().mean()}"
        )
        pipeline.latents2audio(latents_rec2, save_path=audio_save_path)

    return InvertedMusicData(
        noise=trajectory[0],
        null_embeds_per_step=null_embeddings_per_step,
        prompt=prompt,
        diffusion_params=diffusion_params,
    )
