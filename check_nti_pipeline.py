import torch
import torchaudio
import numpy
from tqdm import tqdm
from pipeline import BaseAceStepP2PEditPipeline
import torch
import torch.nn.functional as F
import random
import utils
from loguru import logger
from tqdm import tqdm
from p2p.controllers import AttentionControl
from typing import List, Union, Dict, Tuple, Optional
from abc import ABC, abstractmethod
from acestep.pipeline_ace_step import ACEStepPipeline
from p2p.attention_processor import CustomerAttnProcessorWithP2PController2_0
from acestep.models.customer_attention_processor import CustomerAttnProcessor2_0
from diffusers.pipelines.stable_diffusion_3.pipeline_stable_diffusion_3 import (
    retrieve_timesteps,
)
from diffusers.utils.torch_utils import randn_tensor
from acestep.apg_guidance import (
    apg_forward,
    MomentumBuffer,
    cfg_forward,
    cfg_zero_star,
    cfg_double_condition_forward,
)
from acestep.cpu_offload import cpu_offload
from schedulers.flow_match_euler_inverse import FlowMatchEulerInverseDiscreteScheduler
from null_text_inversion import build_pivot_trajectory, null_text_optimization
import null_text_inversion

LYRICS="""
Her Majesty is a pretty nice girl
But she doesn't have a lot to say
Her Majesty is a pretty nice girl
But she changes from day to day
I wanna tell her that I love her a lot
But I gotta get a belly full of wine
Her Majesty is a pretty nice girl
Someday I'm gonna make her mine Oh yeah, someday I'm gonna make her mine 
"""
TAGS="acoustic, playful, light mood, short melody, humorous"

class LatentsFromNoisePipeline(ACEStepPipeline):
    def __init__(
            self,
            checkpoint_dir,
            dtype="bfloat16"
        ):
        super().__init__(
            checkpoint_dir,
            dtype=dtype
        )

        if not self.loaded:
            logger.warning("Checkpoint not loaded, loading checkpoint...")
            if self.quantized:
                self.load_quantized_checkpoint(self.checkpoint_dir)
            else:
                self.load_checkpoint(self.checkpoint_dir)

    @cpu_offload("ace_step_transformer")
    @torch.no_grad()
    def text2music_from_latents_diffusion_process(
        self,
        target_latents,
        encoder_text_hidden_states,
        text_attention_mask,
        speaker_embds,
        lyric_token_ids,
        lyric_mask,
        random_generators=None,
        infer_steps=100,
        guidance_scale=15.0,
        omega_scale=10.0,
        scheduler_type="heun",
        cfg_type="apg",
        guidance_interval=0.5,
        guidance_interval_decay=1.0,
        min_guidance_scale=3.0,
        null_embeddings_per_step=None,
    ):

        logger.info(
            "cfg_type: {}, guidance_scale: {}, omega_scale: {}".format(
                cfg_type, guidance_scale, omega_scale
            )
        )
        do_classifier_free_guidance = True
        if guidance_scale == 0.0 or guidance_scale == 1.0:
            do_classifier_free_guidance = False

        bsz = encoder_text_hidden_states.shape[0]

        scheduler = utils.diffusion_utils.get_scheduler(scheduler_type)
        frame_length = target_latents.shape[-1]

        timesteps, num_inference_steps = retrieve_timesteps(
            scheduler,
            num_inference_steps=infer_steps,
            device=self.device,
            timesteps=None,
        )

        attention_mask = torch.ones(bsz, frame_length, device=self.device, dtype=self.dtype)

        # guidance interval
        start_idx = int(num_inference_steps * ((1 - guidance_interval) / 2))
        end_idx = int(num_inference_steps * (guidance_interval / 2 + 0.5))
        logger.info(
            f"start_idx: {start_idx}, end_idx: {end_idx}, num_inference_steps: {num_inference_steps}"
        )

        momentum_buffer = MomentumBuffer()

        # P(speaker, text, lyric)
        encoder_hidden_states, encoder_hidden_mask = self.ace_step_transformer.encode(
            encoder_text_hidden_states,
            text_attention_mask,
            speaker_embds,
            lyric_token_ids,
            lyric_mask,
        )

        if null_embeddings_per_step is None:
            # P(null_speaker, null_text, null_lyric)
            encoder_hidden_states_null, _ = self.ace_step_transformer.encode(
                torch.zeros_like(encoder_text_hidden_states),
                text_attention_mask,
                torch.zeros_like(speaker_embds),
                torch.zeros_like(lyric_token_ids),
                lyric_mask,
            )

            null_embeddings_per_step = [encoder_hidden_states_null]*len(timesteps)

        traj = [target_latents.detach().clone()]

        for i, t in tqdm(enumerate(timesteps), total=num_inference_steps):
            # expand the latents if we are doing classifier free guidance
            latents = target_latents

            is_in_guidance_interval = start_idx <= i < end_idx
            if is_in_guidance_interval and do_classifier_free_guidance:
                # compute current guidance scale
                if guidance_interval_decay > 0:
                    # Linearly interpolate to calculate the current guidance scale
                    progress = (i - start_idx) / (
                        end_idx - start_idx - 1
                    )  # 归一化到[0,1]
                    current_guidance_scale = (
                        guidance_scale
                        - (guidance_scale - min_guidance_scale)
                        * progress
                        * guidance_interval_decay
                    )
                else:
                    current_guidance_scale = guidance_scale

                latent_model_input = latents
                timestep = t.expand(latent_model_input.shape[0])
                output_length = latent_model_input.shape[-1]
                # P(x|speaker, text, lyric)

                noise_pred_with_cond = self.ace_step_transformer.decode(
                    hidden_states=latent_model_input,
                    attention_mask=attention_mask,
                    encoder_hidden_states=encoder_hidden_states,
                    encoder_hidden_mask=encoder_hidden_mask,
                    output_length=output_length,
                    timestep=timestep,
                ).sample

                noise_pred_uncond = self.ace_step_transformer.decode(
                    hidden_states=latent_model_input,
                    attention_mask=attention_mask,
                    encoder_hidden_states=null_embeddings_per_step[i],
                    encoder_hidden_mask=encoder_hidden_mask,
                    output_length=output_length,
                    timestep=timestep,
                ).sample

                if cfg_type == "apg":
                    noise_pred = apg_forward(
                        pred_cond=noise_pred_with_cond,
                        pred_uncond=noise_pred_uncond,
                        guidance_scale=current_guidance_scale,
                        momentum_buffer=momentum_buffer,
                    )
                elif cfg_type == "cfg":
                    noise_pred = cfg_forward(
                        cond_output=noise_pred_with_cond,
                        uncond_output=noise_pred_uncond,
                        cfg_strength=current_guidance_scale,
                    )
                elif cfg_type == "cfg_star":
                    noise_pred = cfg_zero_star(
                        noise_pred_with_cond=noise_pred_with_cond,
                        noise_pred_uncond=noise_pred_uncond,
                        guidance_scale=current_guidance_scale,
                        i=i,
                        zero_steps=1,
                        use_zero_init=True,
                    )
            else:
                latent_model_input = latents
                timestep = t.expand(latent_model_input.shape[0])

                noise_pred = self.ace_step_transformer.decode(
                    hidden_states=latent_model_input,
                    attention_mask=attention_mask,
                    encoder_hidden_states=encoder_hidden_states,
                    encoder_hidden_mask=encoder_hidden_mask,
                    output_length=latent_model_input.shape[-1],
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
            traj.append(target_latents.detach().clone())

        return traj
    
    
    def prepare_lyric_tokens(self, lyrics: List[str], debug: bool = False):
        token_lists = [self.tokenize_lyrics(lr, debug=debug) for lr in lyrics]
        max_len = max(len(toks) for toks in token_lists)
        
        token_tensors = []
        for toks in token_lists:
            t = torch.tensor(toks, dtype=torch.long)
            pad_len = max_len - t.size(0)
            token_tensors.append(F.pad(t, (0, pad_len), value=0))
        
        lyric_token_ids = torch.stack(token_tensors, dim=0).to(self.device)
        lyric_mask = (lyric_token_ids != 0).long()
        
        return lyric_token_ids, lyric_mask
    
def main():
    device = torch.device('cuda')
    dtype = torch.float32
    prompts = [TAGS]
    lyrics = [LYRICS]
    pipeline = LatentsFromNoisePipeline(
        "../ACE_CHECKPOINTS",
        dtype=dtype
    )

    assert len(prompts) == len(lyrics)
    bsz = len(prompts)

    texts = prompts
    encoder_text_hidden_states, text_attention_mask = pipeline.get_text_embeddings(texts)

    speaker_embeds = torch.zeros(bsz, 512).to(device=device, dtype=dtype)
    lyric_token_idx, lyric_mask = pipeline.prepare_lyric_tokens(lyrics, debug=False)
    random_generators, _ = pipeline.set_seeds(bsz)
    NUM_STEPS = 500

    wav, sr = torchaudio.load('her_majesty.mp3')
    latents, _ = pipeline.music_dcae.encode(wav.unsqueeze(0).to('cuda'), sr=sr)

    trajectory = build_pivot_trajectory(
        pipeline.ace_step_transformer,
        target_latents=latents,
        encoder_text_hidden_states=encoder_text_hidden_states,
        text_attention_mask=text_attention_mask,
        speaker_embds=speaker_embeds,
        lyric_token_ids=lyric_token_idx,
        lyric_mask=lyric_mask,
        random_generators=random_generators,
        infer_steps=NUM_STEPS,
        omega_scale=10.0,
        scheduler_type="euler_inverse"
    )

    latents_rec = pipeline.text2music_from_latents_diffusion_process(
        target_latents=trajectory[0],
        encoder_text_hidden_states=encoder_text_hidden_states,
        text_attention_mask=text_attention_mask,
        speaker_embds=speaker_embeds,
        lyric_token_ids=lyric_token_idx,
        lyric_mask=lyric_mask,
        random_generators=random_generators,
        guidance_scale=1.0,
        infer_steps=NUM_STEPS,
        omega_scale=10.0,
        scheduler_type="euler",
    )

    tqdm.write(f"MAE: {(latents_rec[-1] - latents).abs().mean()}")

    scheduler = utils.diffusion_utils.get_scheduler('euler')
    timesteps, _ = retrieve_timesteps(
        scheduler,
        NUM_STEPS,
        device=device,
    )

    null_embeddings_per_step, losses_per_step = null_text_inversion.null_text_optimization(
        pipeline.ace_step_transformer,
        trajectory=trajectory,
        timesteps=timesteps,
        scheduler=scheduler,
        encoder_text_hidden_states=encoder_text_hidden_states,
        text_attention_mask=text_attention_mask,
        speaker_embds=speaker_embeds,
        lyric_token_ids=lyric_token_idx,
        lyric_mask=lyric_mask
    )

    reconstructed_traj = pipeline.text2music_from_latents_diffusion_process(
        target_latents=trajectory[0],
        encoder_text_hidden_states=encoder_text_hidden_states,
        text_attention_mask=text_attention_mask,
        speaker_embds=speaker_embeds,
        lyric_token_ids=lyric_token_idx,
        lyric_mask=lyric_mask,
        random_generators=random_generators,
        guidance_scale=15.0,
        infer_steps=NUM_STEPS,
        omega_scale=10.0,
        scheduler_type="euler",
        null_embeddings_per_step=null_embeddings_per_step
    )

    for latent, rec_latent in zip(trajectory, reconstructed_traj):
        tqdm.write(str((latent - rec_latent).abs().mean()))

    pipeline.latents2audio(
        reconstructed_traj[-1],
        save_path="/home/mabondarenko_4/music_p2p/check_gen"
    )

if __name__ == "__main__":
    main()
