from copy import deepcopy

import torch
import torchaudio
from diffusers.pipelines.stable_diffusion_3.pipeline_stable_diffusion_3 import (
    retrieve_timesteps,
)

from src.logging import utils as logging
from src.logging.writer import BaseWriter, DummyWriter
from src.nti.build_pivot_trajectory import build_pivot_trajectory
from src.nti.null_text_inversion import NullTextOptimization
from src.pipelines.base_p2p_pipeline import BaseAceStepP2PEditPipeline
from src.schedulers import get_direct_scheduler
from src.utils.structures import DiffusionParams, InvertedMusicData, Prompt


def music2noise(
    pipeline: BaseAceStepP2PEditPipeline,
    music_path: str,
    prompt: Prompt,
    diffusion_params: DiffusionParams,
    nti: NullTextOptimization,
    debug_mode: bool = False,
    writer: BaseWriter = DummyWriter(),
    audio_save_path: str = None,
) -> InvertedMusicData:
    device = pipeline.ace_step_transformer.device
    dtype = pipeline.ace_step_transformer.dtype
    encoder_text_hidden_states, text_attention_mask = pipeline.get_text_embeddings(
        [prompt.tags]
    )

    speaker_embeds = torch.zeros(1, 512).to(device=device, dtype=dtype)
    lyric_token_idx, lyric_mask = pipeline.prepare_lyric_tokens([prompt.lyrics])

    wav, sr = torchaudio.load(music_path)
    latents, _ = pipeline.music_dcae.encode(wav.unsqueeze(0).to(device), sr=sr)

    building_pivot_out = build_pivot_trajectory(
        pipeline.ace_step_transformer,
        target_latents=latents,
        encoder_text_hidden_states=encoder_text_hidden_states,
        text_attention_mask=text_attention_mask,
        speaker_embds=speaker_embeds,
        lyric_token_ids=lyric_token_idx,
        lyric_mask=lyric_mask,
        diffusion_params=diffusion_params,
    )
    pivot_trajectory = building_pivot_out.trajectory

    if debug_mode:
        diffusion_params_no_guidance = deepcopy(diffusion_params)
        diffusion_params_no_guidance.guidance_params.guidance_scale = 1.0

        diffusion_out = pipeline.diffusion_process(
            input_latents=pivot_trajectory[0],
            encoder_text_hidden_states=encoder_text_hidden_states,
            text_attention_mask=text_attention_mask,
            speaker_embds=speaker_embeds,
            lyric_token_ids=lyric_token_idx,
            lyric_mask=lyric_mask,
            diffusion_params=diffusion_params_no_guidance,
        )

        logging.debug(
            f"MAE after building pivot (no guidance): {(diffusion_out.trajectory[-1] - latents).abs().mean()}"
        )

        pred_wavs = pipeline.latents2audio(diffusion_out.trajectory[-1])
        pipeline.save_pred_wavs(pred_wavs, audio_save_path)
        for i, wav in enumerate(pred_wavs):
            writer.add_audio(f"m2n_no_guidance_{i}", wav.unsqueeze(0))

    scheduler = get_direct_scheduler(diffusion_params.scheduler_type)
    timesteps, _ = retrieve_timesteps(
        scheduler,
        diffusion_params.num_steps,
        device=device,
    )

    null_embeddings_per_step = nti.run(
        trajectory=pivot_trajectory,
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
        diffusion_out = pipeline.diffusion_process(
            input_latents=pivot_trajectory[0],
            encoder_text_hidden_states=encoder_text_hidden_states,
            text_attention_mask=text_attention_mask,
            speaker_embds=speaker_embeds,
            lyric_token_ids=lyric_token_idx,
            lyric_mask=lyric_mask,
            diffusion_params=diffusion_params,
            null_embeddings_per_step=null_embeddings_per_step,
        )

        logging.debug(
            f"MAE after null text optimization (guidance): {(diffusion_out.trajectory[-1] - latents).abs().mean()}"
        )

        pred_wavs = pipeline.latents2audio(diffusion_out.trajectory[-1])
        pipeline.save_pred_wavs(pred_wavs, audio_save_path)
        for i, wav in enumerate(pred_wavs):
            writer.add_audio(f"m2n_with_guidance_{i}", wav.unsqueeze(0))

    return InvertedMusicData(
        noise=pivot_trajectory[0],
        null_embeds_per_step=null_embeddings_per_step,
        prompt=prompt,
        diffusion_params=diffusion_params,
    )
