import torch
import torch.nn.functional as F
import random
from loguru import logger
from tqdm import tqdm
from controllers import AttentionControl
from typing import List, Union, Dict, Tuple, Optional
from p2p_utils import get_time_words_attention_alpha
from abc import ABC, abstractmethod
from acestep.pipeline_ace_step import ACEStepPipeline
from attention_processor import CustomerAttnProcessorWithP2PController2_0
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
from acestep.schedulers.scheduling_flow_match_euler_discrete import (
    FlowMatchEulerDiscreteScheduler,
)
from acestep.schedulers.scheduling_flow_match_heun_discrete import (
    FlowMatchHeunDiscreteScheduler,
)
from acestep.schedulers.scheduling_flow_match_pingpong import (
    FlowMatchPingPongScheduler,
)

class BaseAceStepP2PEditPipeline(ACEStepPipeline, ABC):
    def __init__(
            self,
            checkpoint_dir,
            controller_cls,
            blocks_to_inject_idxs=None,
            dtype="bfloat16"
        ):
        super().__init__(
            checkpoint_dir,
            dtype=dtype
        )
        self.controller_cls = controller_cls

        if not self.loaded:
            logger.warning("Checkpoint not loaded, loading checkpoint...")
            if self.quantized:
                self.load_quantized_checkpoint(self.checkpoint_dir)
            else:
                self.load_checkpoint(self.checkpoint_dir)

        self.blocks_to_inject_idxs = blocks_to_inject_idxs
        if self.blocks_to_inject_idxs is None:
            self.blocks_to_inject_idxs = list(range(24))

    def register_controller(self, controller):
        for i in self.blocks_to_inject_idxs:
            block = self.ace_step_transformer.transformer_blocks[i]
            block.cross_attn.set_processor(
                CustomerAttnProcessorWithP2PController2_0(controller)
            )

    def get_controller(self) -> AttentionControl:
        idx = self.blocks_to_inject_idxs[0]
        block = self.ace_step_transformer.transformer_blocks[idx]
        return block.cross_attn.processor.controller

    def set_diffusion_step_to_controller(self, step):
        self.get_controller().set_diffusion_step(step)

    def unregister_controller(self):
        for i in self.blocks_to_inject_idxs:
            block = self.ace_step_transformer.transformer_blocks[i]
            block.cross_attn.set_processor(CustomerAttnProcessor2_0())

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
        infer_steps=60,
        guidance_scale=15.0,
        omega_scale=10.0,
        scheduler_type="euler",
        cfg_type="apg",
        zero_steps=1,
        use_zero_init=True,
        guidance_interval=0.5,
        guidance_interval_decay=1.0,
        min_guidance_scale=3.0,
        oss_steps=[],
        encoder_text_hidden_states_null=None,
        use_erg_lyric=False,
        use_erg_diffusion=False,
        guidance_scale_text=0.0,
        guidance_scale_lyric=0.0,
    ):

        logger.info(
            "cfg_type: {}, guidance_scale: {}, omega_scale: {}".format(
                cfg_type, guidance_scale, omega_scale
            )
        )
        do_classifier_free_guidance = True
        if guidance_scale == 0.0 or guidance_scale == 1.0:
            do_classifier_free_guidance = False

        do_double_condition_guidance = False
        if (
            guidance_scale_text is not None
            and guidance_scale_text > 1.0
            and guidance_scale_lyric is not None
            and guidance_scale_lyric > 1.0
        ):
            do_double_condition_guidance = True
            logger.info(
                "do_double_condition_guidance: {}, guidance_scale_text: {}, guidance_scale_lyric: {}".format(
                    do_double_condition_guidance,
                    guidance_scale_text,
                    guidance_scale_lyric,
                )
            )

        bsz = encoder_text_hidden_states.shape[0]

        if scheduler_type == "euler":
            scheduler = FlowMatchEulerDiscreteScheduler(
                num_train_timesteps=1000,
                shift=3.0,
            )
        elif scheduler_type == "heun":
            scheduler = FlowMatchHeunDiscreteScheduler(
                num_train_timesteps=1000,
                shift=3.0,
            )
        elif scheduler_type == "pingpong":
            scheduler = FlowMatchPingPongScheduler(
                num_train_timesteps=1000,
                shift=3.0,
            )

        frame_length = target_latents.shape[-1]

        if len(oss_steps) > 0:
            infer_steps = max(oss_steps)
            scheduler.set_timesteps
            timesteps, num_inference_steps = retrieve_timesteps(
                scheduler,
                num_inference_steps=infer_steps,
                device=self.device,
                timesteps=None,
            )
            new_timesteps = torch.zeros(len(oss_steps), dtype=self.dtype, device=self.device)
            for idx in range(len(oss_steps)):
                new_timesteps[idx] = timesteps[oss_steps[idx] - 1]
            num_inference_steps = len(oss_steps)
            sigmas = (new_timesteps / 1000).float().cpu().numpy()
            timesteps, num_inference_steps = retrieve_timesteps(
                scheduler,
                num_inference_steps=num_inference_steps,
                device=self.device,
                sigmas=sigmas,
            )
            logger.info(
                f"oss_steps: {oss_steps}, num_inference_steps: {num_inference_steps} after remapping to timesteps {timesteps}"
            )
        else:
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

        def forward_encoder_with_temperature(self, inputs, tau=0.01, l_min=4, l_max=6):
            handlers = []

            def hook(module, input, output):
                output[:] *= tau
                return output

            for i in range(l_min, l_max):
                handler = self.ace_step_transformer.lyric_encoder.encoders[
                    i
                ].self_attn.linear_q.register_forward_hook(hook)
                handlers.append(handler)

            encoder_hidden_states, encoder_hidden_mask = (
                self.ace_step_transformer.encode(**inputs)
            )

            for hook in handlers:
                hook.remove()

            return encoder_hidden_states

        # P(speaker, text, lyric)
        encoder_hidden_states, encoder_hidden_mask = self.ace_step_transformer.encode(
            encoder_text_hidden_states,
            text_attention_mask,
            speaker_embds,
            lyric_token_ids,
            lyric_mask,
        )

        if use_erg_lyric:
            # P(null_speaker, text_weaker, lyric_weaker)
            encoder_hidden_states_null = forward_encoder_with_temperature(
                self,
                inputs={
                    "encoder_text_hidden_states": (
                        encoder_text_hidden_states_null
                        if encoder_text_hidden_states_null is not None
                        else torch.zeros_like(encoder_text_hidden_states)
                    ),
                    "text_attention_mask": text_attention_mask,
                    "speaker_embeds": torch.zeros_like(speaker_embds),
                    "lyric_token_idx": lyric_token_ids,
                    "lyric_mask": lyric_mask,
                },
            )
        else:
            # P(null_speaker, null_text, null_lyric)
            encoder_hidden_states_null, _ = self.ace_step_transformer.encode(
                torch.zeros_like(encoder_text_hidden_states),
                text_attention_mask,
                torch.zeros_like(speaker_embds),
                torch.zeros_like(lyric_token_ids),
                lyric_mask,
            )

        encoder_hidden_states_no_lyric = None
        if do_double_condition_guidance:
            # P(null_speaker, text, lyric_weaker)
            if use_erg_lyric:
                encoder_hidden_states_no_lyric = forward_encoder_with_temperature(
                    self,
                    inputs={
                        "encoder_text_hidden_states": encoder_text_hidden_states,
                        "text_attention_mask": text_attention_mask,
                        "speaker_embeds": torch.zeros_like(speaker_embds),
                        "lyric_token_idx": lyric_token_ids,
                        "lyric_mask": lyric_mask,
                    },
                )
            # P(null_speaker, text, no_lyric)
            else:
                encoder_hidden_states_no_lyric, _ = self.ace_step_transformer.encode(
                    encoder_text_hidden_states,
                    text_attention_mask,
                    torch.zeros_like(speaker_embds),
                    torch.zeros_like(lyric_token_ids),
                    lyric_mask,
                )

        def forward_diffusion_with_temperature(
            self, hidden_states, timestep, inputs, tau=0.01, l_min=15, l_max=20
        ):
            handlers = []

            def hook(module, input, output):
                output[:] *= tau
                return output

            for i in range(l_min, l_max):
                handler = self.ace_step_transformer.transformer_blocks[
                    i
                ].attn.to_q.register_forward_hook(hook)
                handlers.append(handler)
                handler = self.ace_step_transformer.transformer_blocks[
                    i
                ].cross_attn.to_q.register_forward_hook(hook)
                handlers.append(handler)

            sample = self.ace_step_transformer.decode(
                hidden_states=hidden_states, timestep=timestep, **inputs
            ).sample

            for hook in handlers:
                hook.remove()

            return sample

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

                noise_pred_with_only_text_cond = None
                if (
                    do_double_condition_guidance
                    and encoder_hidden_states_no_lyric is not None
                ):
                    noise_pred_with_only_text_cond = self.ace_step_transformer.decode(
                        hidden_states=latent_model_input,
                        attention_mask=attention_mask,
                        encoder_hidden_states=encoder_hidden_states_no_lyric,
                        encoder_hidden_mask=encoder_hidden_mask,
                        output_length=output_length,
                        timestep=timestep,
                    ).sample

                if use_erg_diffusion:
                    noise_pred_uncond = forward_diffusion_with_temperature(
                        self,
                        hidden_states=latent_model_input,
                        timestep=timestep,
                        inputs={
                            "encoder_hidden_states": encoder_hidden_states_null,
                            "encoder_hidden_mask": encoder_hidden_mask,
                            "output_length": output_length,
                            "attention_mask": attention_mask,
                        },
                    )
                else:
                    noise_pred_uncond = self.ace_step_transformer.decode(
                        hidden_states=latent_model_input,
                        attention_mask=attention_mask,
                        encoder_hidden_states=encoder_hidden_states_null,
                        encoder_hidden_mask=encoder_hidden_mask,
                        output_length=output_length,
                        timestep=timestep,
                    ).sample

                if (
                    do_double_condition_guidance
                    and noise_pred_with_only_text_cond is not None
                ):
                    noise_pred = cfg_double_condition_forward(
                        cond_output=noise_pred_with_cond,
                        uncond_output=noise_pred_uncond,
                        only_text_cond_output=noise_pred_with_only_text_cond,
                        guidance_scale_text=guidance_scale_text,
                        guidance_scale_lyric=guidance_scale_lyric,
                    )

                elif cfg_type == "apg":
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
                        zero_steps=zero_steps,
                        use_zero_init=use_zero_init,
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
                generator=random_generators[0],
            )[0]

            self.set_diffusion_step_to_controller(i+1)
            target_latents = self.get_controller().step_callback(target_latents)

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
        format: str = "wav",
        audio_duration: float = 60.0,
        prompts: List[str] = None,
        lyrics: List[str] = None,
        infer_step: int = 60,
        guidance_scale: float = 15.0,
        scheduler_type: str = "euler",
        cfg_type: str = "apg",
        omega_scale: int = 10.0,
        guidance_interval: float = 0.5,
        guidance_interval_decay: float = 0.0,
        min_guidance_scale: float = 3.0,
        use_erg_tag: bool = True,
        use_erg_lyric: bool = True,
        use_erg_diffusion: bool = True,
        oss_steps: str = None,
        guidance_scale_text: float = 0.0,
        guidance_scale_lyric: float = 0.0,
        lora_name_or_path: str = "none",
        lora_weight: float = 1.0,
        save_path: str = None,
        debug: bool = False,
    ):
        assert len(prompts) == len(lyrics)
        bsz = len(prompts)
        random_generators, _ = self.set_seeds(bsz)

        self.load_lora(lora_name_or_path, lora_weight)

        if isinstance(oss_steps, str) and len(oss_steps) > 0:
            oss_steps = list(map(int, oss_steps.split(",")))
        else:
            oss_steps = []

        texts = prompts
        encoder_text_hidden_states, text_attention_mask = self.get_text_embeddings(texts)

        encoder_text_hidden_states_null = None
        if use_erg_tag:
            encoder_text_hidden_states_null = self.get_text_embeddings_null(texts)

        # not support for released checkpoint
        speaker_embeds = torch.zeros(bsz, 512).to(self.device).to(self.dtype)

        lyric_token_idx, lyric_mask = self.prepare_lyric_tokens(lyrics, debug=debug)

        if audio_duration <= 0:
            audio_duration = random.uniform(30.0, 240.0)
            logger.info(f"random audio duration: {audio_duration}")

        frame_length = int(audio_duration * 44100 / 512 / 8)
        target_latents = randn_tensor(
            shape=(1, 8, 16, frame_length),
            generator=random_generators[0],
            device=self.device,
            dtype=self.dtype,
        )
        target_latents = target_latents.expand(bsz, -1, -1, -1)

        target_latents = self.text2music_from_latents_diffusion_process(
                    target_latents=target_latents,
                    encoder_text_hidden_states=encoder_text_hidden_states,
                    text_attention_mask=text_attention_mask,
                    speaker_embds=speaker_embeds,
                    lyric_token_ids=lyric_token_idx,
                    lyric_mask=lyric_mask,
                    random_generators=random_generators,
                    infer_steps=infer_step,
                    guidance_scale=guidance_scale,
                    omega_scale=omega_scale,
                    scheduler_type=scheduler_type,
                    cfg_type=cfg_type,
                    guidance_interval=guidance_interval,
                    guidance_interval_decay=guidance_interval_decay,
                    min_guidance_scale=min_guidance_scale,
                    oss_steps=oss_steps,
                    encoder_text_hidden_states_null=encoder_text_hidden_states_null,
                    use_erg_lyric=use_erg_lyric,
                    use_erg_diffusion=use_erg_diffusion,
                    guidance_scale_text=guidance_scale_text,
                    guidance_scale_lyric=guidance_scale_lyric,
            )

        output_paths = self.latents2audio(
            latents=target_latents,
            target_wav_duration_second=audio_duration,
            save_path=save_path,
            format=format,
        )

        # Clean up memory after generation
        self.cleanup_memory()

        return output_paths
    
    @abstractmethod
    def __call__(self, *args, **kwargs):
        raise NotImplementedError

class LyricsP2PEditPipeline(BaseAceStepP2PEditPipeline):
    def __init__(
            self,
            checkpoint_dir,
            controller_cls,
            blocks_to_inject_idxs: List[int] = None,
            dtype="bfloat16",
        ):
        super().__init__(
            checkpoint_dir,
            controller_cls,
            blocks_to_inject_idxs, 
            dtype
        )

    def __call__(
        self,
        src_lyrics: str,
        tgt_lyrics: List[str],
        genre_tags: str,
        duration: float = -1,
        guidance_scale: float = 15.0,
        infer_steps=60,
        scheduler_type: str = "euler",
        save_path: Optional[str] = None,
        controller_kwargs: Optional[Dict] = None
    ):
        if controller_kwargs is None:
            controller_kwargs = {}

        controller = self.controller_cls(**controller_kwargs)
        self.register_controller(controller)
        output_paths = self.forward(
            audio_duration=duration,
            infer_step=infer_steps,
            prompts=[genre_tags for _ in range(len(tgt_lyrics) + 1)],
            lyrics=[src_lyrics] + tgt_lyrics,
            scheduler_type=scheduler_type,
            guidance_scale=guidance_scale,
            save_path=save_path,
        )
        self.unregister_controller()
        return output_paths

class TagsP2PEditPipeline(BaseAceStepP2PEditPipeline):
    def __init__(
            self,
            checkpoint_dir,
            controller_cls,
            blocks_to_inject_idxs: List[int] = None,
            dtype="bfloat16",
        ):
        super().__init__(
            checkpoint_dir,
            controller_cls,
            blocks_to_inject_idxs,
            dtype
        )

    def __call__(
        self,
        src_tags: str,
        tgt_tags: List[str],
        lyrics: str,
        duration: float = -1,
        src_audio_path: str = None,
        cross_replace_steps: Union[float, Dict[str, Tuple[float, float]]] = 1.0,
        guidance_scale: float = 15.0,
        infer_steps=60,
        scheduler_type: str = "euler",
        save_path: Optional[str] = None,
        controller_kwargs: Optional[Dict] = None
    ):
        if controller_kwargs is None:
            controller_kwargs = {}

        controller = self.controller_cls(**controller_kwargs)
        self.register_controller(controller)
        output_paths = self.forward(
            audio_duration=duration,
            infer_step=infer_steps,
            prompts=[src_tags] + tgt_tags,
            lyrics=[lyrics for _ in range(len(tgt_tags) + 1)],
            scheduler_type=scheduler_type,
            guidance_scale=guidance_scale,
            save_path=save_path,
        )
        self.unregister_controller()
        return output_paths

