import copy
import torch
import torch.nn.functional as F
from torch.optim import Adam
from tqdm import tqdm
from acestep.apg_guidance import MomentumBuffer
from utils.diffusion_utils import mix_guidance, compute_current_guidance

@torch.no_grad()
def _encode_pair(
    ace_step_transformer,
    encoder_text_hidden_states,
    text_attention_mask,
    speaker_embds,
    lyric_token_ids,
    lyric_mask
):
    cond_emb, cond_mask = ace_step_transformer.encode(
        encoder_text_hidden_states, text_attention_mask,
        speaker_embds, lyric_token_ids, lyric_mask
    )
    null_emb, _ = ace_step_transformer.encode(
        torch.zeros_like(encoder_text_hidden_states),
        text_attention_mask,
        torch.zeros_like(speaker_embds),
        torch.zeros_like(lyric_token_ids),
        lyric_mask
    )
    return cond_emb, cond_mask, null_emb

def _reset_scheduler_at_t(scheduler, t):
    idx = scheduler.index_for_timestep(t, scheduler.timesteps)
    scheduler._step_index = idx
    if hasattr(scheduler, "prev_derivative"):
        scheduler.prev_derivative = None
    if hasattr(scheduler, "dt"):
        scheduler.dt = None
    if hasattr(scheduler, "sample"):
        scheduler.sample = None

def _mb_clone_detached(mb):
    if mb is None:
        return None
    snap = copy.deepcopy(mb)
    for k, v in snap.__dict__.items():
        if torch.is_tensor(v):
            snap.__dict__[k] = v.detach().clone()
    return snap

def null_text_optimization(
    ace_step_transformer,
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
    epsilon=1e-5,
    cfg_type="apg",
    guidance_scale=15.0,
    guidance_interval=0.5,
    guidance_interval_decay=1.0,
    min_guidance_scale=3.0,
    omega_scale=10.0,
):
    momentum_buffer = MomentumBuffer() if cfg_type == "apg" else None

    device = ace_step_transformer.device
    dtype  = ace_step_transformer.dtype

    bsz = encoder_text_hidden_states.shape[0]
    frame_length = trajectory[0].shape[-1]
    attention_mask = torch.ones(bsz, frame_length, device=device, dtype=dtype)

    cond_emb, cond_mask, base_null_emb = _encode_pair(
        ace_step_transformer,
        encoder_text_hidden_states, text_attention_mask,
        speaker_embds, lyric_token_ids, lyric_mask
    )

    num_inference_steps = len(timesteps)
    start_idx = int(num_inference_steps * ((1 - guidance_interval) / 2))
    end_idx   = int(num_inference_steps * (guidance_interval / 2 + 0.5))
    do_cfg = not (guidance_scale == 0.0 or guidance_scale == 1.0)

    null_embeddings_per_step, losses_per_step = [], []
    latent_cur = trajectory[0].to(device=device, dtype=dtype)

    for i, t in tqdm(enumerate(timesteps), total=num_inference_steps, desc="Null-text optimization..."):
        latent_next = trajectory[i+1].to(device=device, dtype=dtype)

        with torch.no_grad():
            noise_cond = ace_step_transformer.decode(
                hidden_states=latent_cur,
                attention_mask=attention_mask,
                encoder_hidden_states=cond_emb,
                encoder_hidden_mask=cond_mask,
                output_length=latent_cur.shape[-1],
                timestep=t.expand(latent_cur.shape[0]),
            ).sample

        cur_scale, in_window = compute_current_guidance(
            i, start_idx, end_idx, guidance_scale, min_guidance_scale, guidance_interval_decay
        )

        if (not do_cfg) or (not in_window):
            null_embeddings_per_step.append(base_null_emb.detach().clone())
            losses_per_step.append(float('nan'))

            with torch.no_grad():
                _reset_scheduler_at_t(scheduler, t)
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
        optimizer = Adam([null_emb], lr=lr * (1 - i / len(timesteps)))

        for j in range(num_inner_steps):
            _reset_scheduler_at_t(scheduler, t)

            noise_null = ace_step_transformer.decode(
                hidden_states=latent_cur,
                attention_mask=attention_mask,
                encoder_hidden_states=null_emb,
                encoder_hidden_mask=cond_mask,
                output_length=latent_cur.shape[-1],
                timestep=t.expand(latent_cur.shape[0]),
            ).sample

            mb_snapshot = _mb_clone_detached(momentum_buffer) if cfg_type == "apg" else None
            noise_pred = mix_guidance(
                cfg_type=cfg_type,
                noise_cond=noise_cond,
                noise_null=noise_null,
                gscale=cur_scale,
                momentum_buffer=mb_snapshot,
                i=i
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

        null_embeddings_per_step.append(null_emb.detach().clone())
        losses_per_step.append(loss.item())

        with torch.no_grad():
            _reset_scheduler_at_t(scheduler, t)

            noise_null = ace_step_transformer.decode(
                hidden_states=latent_cur,
                attention_mask=attention_mask,
                encoder_hidden_states=null_emb,
                encoder_hidden_mask=cond_mask,
                output_length=latent_cur.shape[-1],
                timestep=t.expand(latent_cur.shape[0]),
            ).sample

            noise_pred = mix_guidance(
                cfg_type=cfg_type,
                noise_cond=noise_cond,
                noise_null=noise_null,
                gscale=cur_scale,
                momentum_buffer=momentum_buffer,
                i=i
            )

            latent_cur = scheduler.step(
                model_output=noise_pred,
                timestep=t,
                sample=latent_cur,
                return_dict=False,
                omega=omega_scale,
                generator=None,
            )[0]

    return null_embeddings_per_step, losses_per_step
