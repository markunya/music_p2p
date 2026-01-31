from typing import List

import torch

from src.pipelines.base_p2p_pipeline import BaseAceStepP2PEditPipeline
from src.utils.structures import DiffusionParams, Prompt


class LyricsP2PEditPipeline(BaseAceStepP2PEditPipeline):
    def __call__(
        self,
        noise: torch.Tensor,
        null_embeds_per_step: List[torch.Tensor],
        src_prompt: Prompt,
        tgt_prompt: Prompt,
        diffusion_params: DiffusionParams,
        check_baseline: bool = False,
        debug_mode: bool = False,
        save_path: str | None = None,
    ):
        tags = [src_prompt.tags, src_prompt.tags]
        lyrics = [src_prompt.lyrics, tgt_prompt.lyrics]
        if check_baseline:
            lyrics.append(tgt_prompt.lyrics)
            tags.append(src_prompt.tags)

        diffusion_out = self.forward(
            input_latents=noise,
            null_embeds_per_step=null_embeds_per_step,
            tags=tags,
            lyrics=lyrics,
            diffusion_params=diffusion_params,
            save_path=save_path,
            debug=debug_mode,
        )

        return diffusion_out
