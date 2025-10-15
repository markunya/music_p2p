import torch
import torch.nn.functional as F
import random
import utils
from tqdm import tqdm
from p2p.controllers import AttentionControl
from typing import List, Optional
from acestep.pipeline_ace_step import ACEStepPipeline
from p2p.attention_processor import CustomerAttnProcessorWithP2PController2_0
from acestep.models.customer_attention_processor import CustomerAttnProcessor2_0
from diffusers.pipelines.stable_diffusion_3.pipeline_stable_diffusion_3 import (
    retrieve_timesteps,
)
from diffusers.utils.torch_utils import randn_tensor
from acestep.apg_guidance import (
    MomentumBuffer,
)
from acestep.cpu_offload import cpu_offload
import utils.diffusion_utils

class BaseAceStepP2PEditPipeline(ACEStepPipeline):
    def __init__(
            self,
            checkpoint_dir,
            controller: Optional[AttentionControl] = None,
            blocks_to_inject_idxs=None,
            dtype="bfloat16"
        ):
        super().__init__(
            checkpoint_dir,
            dtype=dtype
        )
        self.controller: Optional[AttentionControl] = controller

        if not self.loaded:
            tqdm.write("Checkpoint not loaded, loading checkpoint...")
            if self.quantized:
                self.load_quantized_checkpoint(self.checkpoint_dir)
            else:
                self.load_checkpoint(self.checkpoint_dir)

        self.blocks_to_inject_idxs = blocks_to_inject_idxs
        if self.blocks_to_inject_idxs is None:
            self.blocks_to_inject_idxs = list(range(24))

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
        random_generators=None,
        infer_steps=100,
        guidance_scale=15.0,
        omega_scale=10.0,
        scheduler_type="euler",
        cfg_type="apg",
        guidance_interval=0.5,
        guidance_interval_decay=1.0,
        min_guidance_scale=3.0,
        null_embeddings_per_step=None
    ):

        tqdm.write(
            "cfg_type: {}, guidance_scale: {}, omega_scale: {}".format(
                cfg_type, guidance_scale, omega_scale
            )
        )

        do_classifier_free_guidance = True
        if guidance_scale == 0.0 or guidance_scale == 1.0:
            do_classifier_free_guidance = False

        bsz = encoder_text_hidden_states.shape[0]

        scheduler = utils.diffusion_utils.get_scheduler(scheduler_type)
        frame_length = input_latents.shape[-1]

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
        tqdm.write(
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

        target_latents = input_latents

        for i, t in tqdm(enumerate(timesteps), total=num_inference_steps):
            # expand the latents if we are doing classifier free guidance
            latents = target_latents
            current_guidance_scale, is_in_guidance_interval = utils.diffusion_utils.compute_current_guidance(
                i, start_idx, end_idx, guidance_scale, min_guidance_scale, guidance_interval_decay
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

                noise_pred = utils.diffusion_utils.mix_guidance(
                    cfg_type=cfg_type,
                    noise_cond=noise_pred_with_cond,
                    noise_null=noise_pred_uncond,
                    gscale=current_guidance_scale,
                    momentum_buffer=momentum_buffer,
                    i=i
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
                omega=omega_scale,
                generator=random_generators[0] if random_generators else None,
            )[0]

            if self.controller is not None:
                self.set_diffusion_step_to_controller(i+1)
                target_latents = self.controller.step_callback(target_latents)

        return target_latents
    
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
    
    def forward(
        self,
        input_latents: torch.Tensor,
        null_embeds_per_step: List[torch.Tensor],
        prompts: List[str],
        lyrics: List[str],
        format: str = "wav",
        infer_step: int = 100,
        guidance_scale: float = 15.0,
        scheduler_type: str = "euler",
        cfg_type: str = "apg",
        omega_scale: int = 10.0,
        guidance_interval: float = 0.5,
        guidance_interval_decay: float = 1.0,
        min_guidance_scale: float = 3.0,
        lora_name_or_path: str = "none",
        lora_weight: float = 1.0,
        save_path: str = None,
        debug: bool = False,
    ):
        assert len(prompts) == len(lyrics)
        bsz = len(prompts)

        self.load_lora(lora_name_or_path, lora_weight)

        texts = prompts
        encoder_text_hidden_states, text_attention_mask = self.get_text_embeddings(texts)

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
            infer_steps=infer_step,
            guidance_scale=guidance_scale,
            omega_scale=omega_scale,
            scheduler_type=scheduler_type,
            cfg_type=cfg_type,
            guidance_interval=guidance_interval,
            guidance_interval_decay=guidance_interval_decay,
            min_guidance_scale=min_guidance_scale,
        )

        output_paths = self.latents2audio(
            latents=target_latents,
            save_path=save_path,
            format=format,
        )

        self.cleanup_memory()
        return output_paths
    