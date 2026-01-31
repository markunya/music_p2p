import copy

import torch
import torch.nn.functional as F
from torch.optim import Adam
from tqdm import tqdm

from src.logging import utils as logging
from src.utils.diffusion_utils import (
    GuidanceParams,
    compute_current_guidance,
    mix_guidance,
)


class NullTextOptimization:
    def __init__(
        self, model, lr=1e-2, num_inner_steps=15, epsilon=1e-7, debug_mode=False
    ):
        self._model = model
        self._lr = lr
        self._num_inner_steps = num_inner_steps
        self._epsilon = epsilon
        self._debug_mode = debug_mode

    @torch.no_grad()
    def _encode_pair(
        self,
        encoder_text_hidden_states,
        text_attention_mask,
        speaker_embds,
        lyric_token_ids,
        lyric_mask,
    ):
        cond_emb, cond_mask = self._model.encode(
            encoder_text_hidden_states,
            text_attention_mask,
            speaker_embds,
            lyric_token_ids,
            lyric_mask,
        )
        null_emb, _ = self._model.encode(
            torch.zeros_like(encoder_text_hidden_states),
            text_attention_mask,
            torch.zeros_like(speaker_embds),
            torch.zeros_like(lyric_token_ids),
            lyric_mask,
        )
        return cond_emb, cond_mask, null_emb

    def _reset_scheduler_at_t(self, scheduler, t):
        idx = scheduler.index_for_timestep(t, scheduler.timesteps)
        scheduler._step_index = idx
        if hasattr(scheduler, "prev_derivative"):
            scheduler.prev_derivative = None
        if hasattr(scheduler, "dt"):
            scheduler.dt = None
        if hasattr(scheduler, "sample"):
            scheduler.sample = None

    def _mb_clone_detached(self, mb):
        if mb is None:
            return None
        snap = copy.deepcopy(mb)
        for k, v in snap.__dict__.items():
            if torch.is_tensor(v):
                snap.__dict__[k] = v.detach().clone()
        return snap

    def run(
        self,
        trajectory,
        timesteps,
        scheduler,
        encoder_text_hidden_states,
        text_attention_mask,
        speaker_embds,
        lyric_token_ids,
        lyric_mask,
        *,
        num_inner_steps=15,
        lr=1e-2,
        epsilon=1e-7,
        guidance_params: GuidanceParams,
        omega_scale=0.0,
    ) -> list[torch.Tensor]:
        device = self._model.device
        dtype = self._model.dtype

        bsz = encoder_text_hidden_states.shape[0]
        frame_length = trajectory[0].shape[-1]
        attention_mask = torch.ones(bsz, frame_length, device=device, dtype=dtype)

        cond_emb, cond_mask, base_null_emb = self._encode_pair(
            encoder_text_hidden_states,
            text_attention_mask,
            speaker_embds,
            lyric_token_ids,
            lyric_mask,
        )

        num_inference_steps = len(timesteps)
        start_idx = int(
            num_inference_steps * ((1 - guidance_params.guidance_interval) / 2)
        )
        end_idx = int(
            num_inference_steps * (guidance_params.guidance_interval / 2 + 0.5)
        )
        do_cfg = not (
            guidance_params.guidance_scale == 0.0
            or guidance_params.guidance_scale == 1.0
        )

        null_embeddings_per_step = []
        latent_cur = trajectory[0].to(device=device, dtype=dtype)

        for i, t in tqdm(
            enumerate(timesteps),
            total=num_inference_steps,
            desc="Null-text optimization...",
        ):
            latent_next = trajectory[i + 1].to(device=device, dtype=dtype)

            with torch.no_grad():
                noise_cond = self._model.decode(
                    hidden_states=latent_cur,
                    attention_mask=attention_mask,
                    encoder_hidden_states=cond_emb,
                    encoder_hidden_mask=cond_mask,
                    output_length=latent_cur.shape[-1],
                    timestep=t.expand(latent_cur.shape[0]),
                ).sample

            cur_scale, in_window = compute_current_guidance(
                i, start_idx, end_idx, guidance_params
            )

            if (not do_cfg) or (not in_window):
                null_embeddings_per_step.append(base_null_emb.detach().clone())

                with torch.no_grad():
                    self._reset_scheduler_at_t(scheduler, t)
                    latent_cur = scheduler.step(
                        model_output=noise_cond,
                        timestep=t,
                        sample=latent_cur,
                        return_dict=False,
                        omega=omega_scale,
                        generator=None,
                    )[0]
                continue

            null_emb = base_null_emb.clone().detach().requires_grad_(True)
            optimizer = Adam([null_emb], lr=lr * (1 - i / (2 * len(timesteps))))

            for j in range(num_inner_steps):
                self._reset_scheduler_at_t(scheduler, t)

                noise_null = self._model.decode(
                    hidden_states=latent_cur,
                    attention_mask=attention_mask,
                    encoder_hidden_states=null_emb,
                    encoder_hidden_mask=cond_mask,
                    output_length=latent_cur.shape[-1],
                    timestep=t.expand(latent_cur.shape[0]),
                ).sample

                noise_pred = mix_guidance(
                    cfg_type=guidance_params.type,
                    noise_cond=noise_cond,
                    noise_null=noise_null,
                    gscale=cur_scale,
                )

                lat_prev = scheduler.step(
                    model_output=noise_pred,
                    timestep=t,
                    sample=latent_cur,
                    return_dict=False,
                    omega=omega_scale,
                    generator=None,
                )[0]

                loss = F.mse_loss(lat_prev, latent_next)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                if loss.item() < epsilon:
                    break

            if self._debug_mode:
                logging.debug(f"NTI loss on step {i}: {loss.item()}")

            null_embeddings_per_step.append(null_emb.detach().clone())

            with torch.no_grad():
                self._reset_scheduler_at_t(scheduler, t)

                noise_null = self._model.decode(
                    hidden_states=latent_cur,
                    attention_mask=attention_mask,
                    encoder_hidden_states=null_emb,
                    encoder_hidden_mask=cond_mask,
                    output_length=latent_cur.shape[-1],
                    timestep=t.expand(latent_cur.shape[0]),
                ).sample

                noise_pred = mix_guidance(
                    cfg_type=guidance_params.type,
                    noise_cond=noise_cond,
                    noise_null=noise_null,
                    gscale=cur_scale,
                )

                latent_cur = scheduler.step(
                    model_output=noise_pred,
                    timestep=t,
                    sample=latent_cur,
                    return_dict=False,
                    omega=omega_scale,
                    generator=None,
                )[0]

        return null_embeddings_per_step
