from typing import List

import torch

from src.pipelines.base_p2p_pipeline import BaseAceStepP2PEditPipeline
from src.utils.structures import DiffusionParams, Prompt


class TagsP2PEditPipeline(BaseAceStepP2PEditPipeline):
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
        lyrics = [src_prompt.lyrics, src_prompt.lyrics]
        tags = [src_prompt.tags, tgt_prompt.tags]
        if check_baseline:
            lyrics.append(src_prompt.lyrics)
            tags.append(tgt_prompt.tags)

        diffusion_out = self.forward(
            input_latents=noise,
            null_embeds_per_step=null_embeds_per_step,
            tags=tags,
            lyrics=lyrics,
            diffusion_params=diffusion_params,
            debug=debug_mode,
            save_path=save_path,
        )

        return diffusion_out
