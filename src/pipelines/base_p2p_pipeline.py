import random
from typing import List, Optional

import torch
import torch.nn.functional as F
from acestep.apg_guidance import MomentumBuffer
from acestep.cpu_offload import cpu_offload
from acestep.models.customer_attention_processor import CustomerAttnProcessor2_0
from acestep.pipeline_ace_step import ACEStepPipeline
from diffusers.pipelines.stable_diffusion_3.pipeline_stable_diffusion_3 import (
    retrieve_timesteps,
)
from diffusers.utils.torch_utils import randn_tensor
from tqdm import tqdm

from src.p2p.attention_processor import CustomerAttnProcessorWithP2PController2_0
from src.p2p.controllers import AttentionControl
from src.schedulers import get_direct_scheduler
from src.utils import diffusion_utils, logging
from src.utils.structures import DiffusionParams, Prompt


class BaseAceStepP2PEditPipeline(ACEStepPipeline):
    def __init__(
        self,
        checkpoint_dir,
        controller: Optional[AttentionControl] = None,
        blocks_to_inject_idxs=None,
        dtype="bfloat16",
    ):
        super().__init__(checkpoint_dir, dtype=dtype)
        self.controller: Optional[AttentionControl] = controller

        if not self.loaded:
            logging.info("AceStep checkpoint not loaded, loading checkpoint...")
            if self.quantized:
                self.load_quantized_checkpoint(self.checkpoint_dir)
            else:
                self.load_checkpoint(self.checkpoint_dir)

        self.blocks_to_inject_idxs = blocks_to_inject_idxs
        if self.blocks_to_inject_idxs is None:
            self.blocks_to_inject_idxs = list(range(24))

    @cpu_offload("text_encoder_model")
    def get_text_embeddings(self, texts, text_max_length=128):
        inputs = self.text_tokenizer(
            texts,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=text_max_length,
        )
        inputs = {key: value.to(self.device) for key, value in inputs.items()}
        if self.text_encoder_model.device != self.device:
            self.text_encoder_model.to(self.device)
        with torch.no_grad():
            outputs = self.text_encoder_model(**inputs)
            last_hidden_states = outputs.last_hidden_state
        attention_mask = inputs["attention_mask"]
        return last_hidden_states, attention_mask

    @cpu_offload("text_encoder_model")
    def get_text_embeddings_null(
        self, texts, text_max_length=128, tau=0.01, l_min=8, l_max=10
    ):
        inputs = self.text_tokenizer(
            texts,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=text_max_length,
        )
        inputs = {key: value.to(self.device) for key, value in inputs.items()}
        if self.text_encoder_model.device != self.device:
            self.text_encoder_model.to(self.device)

        def forward_with_temperature(inputs, tau=0.01, l_min=8, l_max=10):
            handlers = []

            def hook(module, input, output):
                output[:] *= tau
                return output

            for i in range(l_min, l_max):
                handler = (
                    self.text_encoder_model.encoder.block[i]
                    .layer[0]
                    .SelfAttention.q.register_forward_hook(hook)
                )
                handlers.append(handler)

            with torch.no_grad():
                outputs = self.text_encoder_model(**inputs)
                last_hidden_states = outputs.last_hidden_state

            for hook in handlers:
                hook.remove()

            return last_hidden_states

        last_hidden_states = forward_with_temperature(inputs, tau, l_min, l_max)
        return last_hidden_states

    def register_controller(self):
        for i in self.blocks_to_inject_idxs:
            block = self.ace_step_transformer.transformer_blocks[i]
            block.cross_attn.set_processor(
                CustomerAttnProcessorWithP2PController2_0(self.controller)
            )

    def set_diffusion_step_to_controller(self, step):
        self.controller.set_diffusion_step(step)

    def unregister_controller(self):
        for i in self.blocks_to_inject_idxs:
            block = self.ace_step_transformer.transformer_blocks[i]
            block.cross_attn.set_processor(CustomerAttnProcessor2_0())

    @cpu_offload("ace_step_transformer")
    @torch.no_grad()
    def diffusion_process(
        self,
        input_latents,
        encoder_text_hidden_states,
        text_attention_mask,
        speaker_embds,
        lyric_token_ids,
        lyric_mask,
        diffusion_params: DiffusionParams,
        random_generators=None,
        null_embeddings_per_step=None,
    ):
        logging.info("Diffusion params:")
        logging.log_structure(diffusion_params)

        guidance_params = diffusion_params.guidance_params
        if (
            guidance_params.guidance_scale == 0.0
            or guidance_params.guidance_scale == 1.0
        ):
            do_classifier_free_guidance = False
        else:
            do_classifier_free_guidance = True

        bsz = encoder_text_hidden_states.shape[0]

        scheduler = get_direct_scheduler(diffusion_params.scheduler_type)
        frame_length = input_latents.shape[-1]

        timesteps, num_inference_steps = retrieve_timesteps(
            scheduler,
            num_inference_steps=diffusion_params.num_steps,
            device=self.device,
            timesteps=None,
        )

        attention_mask = torch.ones(
            bsz, frame_length, device=self.device, dtype=self.dtype
        )

        # guidance interval
        start_idx = int(
            num_inference_steps * ((1 - guidance_params.guidance_interval) / 2)
        )
        end_idx = int(
            num_inference_steps * (guidance_params.guidance_interval / 2 + 0.5)
        )
        logging.info(
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

            null_embeddings_per_step = [encoder_hidden_states_null] * len(timesteps)

        target_latents = input_latents

        for i, t in tqdm(enumerate(timesteps), total=num_inference_steps):
            # expand the latents if we are doing classifier free guidance
            latents = target_latents
            (
                current_guidance_scale,
                is_in_guidance_interval,
            ) = diffusion_utils.compute_current_guidance(
                i, start_idx, end_idx, guidance_params
            )

            if is_in_guidance_interval and do_classifier_free_guidance:
                latent_model_input = latents
                timestep = t.expand(latent_model_input.shape[0])
                output_length = latent_model_input.shape[-1]

                # P(x|speaker, text, lyric)
                self.register_controller()
                noise_pred_with_cond = self.ace_step_transformer.decode(
                    hidden_states=latent_model_input,
                    attention_mask=attention_mask,
                    encoder_hidden_states=encoder_hidden_states,
                    encoder_hidden_mask=encoder_hidden_mask,
                    output_length=output_length,
                    timestep=timestep,
                ).sample
                self.unregister_controller()

                noise_pred_uncond = self.ace_step_transformer.decode(
                    hidden_states=latent_model_input,
                    attention_mask=attention_mask,
                    encoder_hidden_states=null_embeddings_per_step[i],
                    encoder_hidden_mask=encoder_hidden_mask,
                    output_length=output_length,
                    timestep=timestep,
                ).sample

                noise_pred = diffusion_utils.mix_guidance(
                    cfg_type=guidance_params.type,
                    noise_cond=noise_pred_with_cond,
                    noise_null=noise_pred_uncond,
                    gscale=current_guidance_scale,
                    momentum_buffer=momentum_buffer,
                    i=i,
                )

            else:
                latent_model_input = latents
                timestep = t.expand(latent_model_input.shape[0])

                self.register_controller()
                noise_pred = self.ace_step_transformer.decode(
                    hidden_states=latent_model_input,
                    attention_mask=attention_mask,
                    encoder_hidden_states=encoder_hidden_states,
                    encoder_hidden_mask=encoder_hidden_mask,
                    output_length=latent_model_input.shape[-1],
                    timestep=timestep,
                ).sample
                self.unregister_controller()

            target_latents = scheduler.step(
                model_output=noise_pred,
                timestep=t,
                sample=target_latents,
                return_dict=False,
                omega=diffusion_params.omega_scale,
                generator=random_generators[0] if random_generators else None,
            )[0]

            if self.controller is not None:
                self.set_diffusion_step_to_controller(i + 1)
                target_latents = self.controller.step_callback(target_latents)

        return target_latents

    def prepare_lyric_tokens(self, lyrics: List[str], debug: bool = False, max_len=512):
        token_lists = [self.tokenize_lyrics(lr, debug=debug) for lr in lyrics]
        if max_len is None:
            max_len = max(len(toks) for toks in token_lists)

        token_tensors = []
        for toks in token_lists:
            t = torch.tensor(toks, dtype=torch.long)
            pad_len = max_len - t.size(0)
            token_tensors.append(F.pad(t, (0, pad_len), value=0))

        lyric_token_ids = torch.stack(token_tensors, dim=0).to(self.device)
        lyric_mask = (lyric_token_ids != 0).long()

        return lyric_token_ids, lyric_mask

    def forward(
        self,
        input_latents: torch.Tensor,
        null_embeds_per_step: List[torch.Tensor],
        tags: List[str],
        lyrics: List[str],
        diffusion_params: DiffusionParams,
        format: str = "wav",
        lora_name_or_path: str = "none",
        lora_weight: float = 1.0,
        save_path: Optional[str] = None,
        debug: bool = False,
    ):
        assert len(tags) == len(lyrics), "There must be same amount of tags and lyrics"
        bsz = len(tags)

        self.load_lora(lora_name_or_path, lora_weight)

        encoder_text_hidden_states, text_attention_mask = self.get_text_embeddings(tags)

        speaker_embeds = torch.zeros(bsz, 512).to(self.device).to(self.dtype)
        lyric_token_idx, lyric_mask = self.prepare_lyric_tokens(lyrics, debug=debug)

        target_latents = self.diffusion_process(
            input_latents=input_latents,
            null_embeddings_per_step=null_embeds_per_step,
            encoder_text_hidden_states=encoder_text_hidden_states,
            text_attention_mask=text_attention_mask,
            speaker_embds=speaker_embeds,
            lyric_token_ids=lyric_token_idx,
            lyric_mask=lyric_mask,
            random_generators=None,
            diffusion_params=diffusion_params,
        )

        output_paths = self.latents2audio(
            latents=target_latents,
            save_path=save_path,
            format=format,
        )

        self.cleanup_memory()
        return output_paths

    def text_to_music(
        self,
        prompt: Prompt,
        diffusion_params: DiffusionParams,
        duration: int = -1,
        input_latents: Optional[torch.Tensor] = None,
        null_embeds_per_step: Optional[List[torch.Tensor]] = None,
        lora_name_or_path: str = "none",
        lora_weight: float = 1.0,
        save_path: Optional[str] = None,
        debug_mode: bool = False,
    ) -> List[str]:
        tags = [prompt.tags]
        lyrics = [prompt.lyrics]

        if duration <= 0:
            duration = random.uniform(30.0, 240.0)
            logging.info(f"Random audio duration: {duration}")

        if input_latents is None:
            frame_length = int(duration * 44100 / 512 / 8)
            input_latents = randn_tensor(
                shape=(1, 8, 16, frame_length),
                generator=None,
                device=self.device,
                dtype=self.dtype,
            )

        return self.forward(
            input_latents,
            null_embeds_per_step,
            tags,
            lyrics,
            diffusion_params,
            lora_name_or_path=lora_name_or_path,
            lora_weight=lora_weight,
            save_path=save_path,
            debug=debug_mode,
        )
