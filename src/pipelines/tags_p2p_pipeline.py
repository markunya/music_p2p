import torch
from typing import List, Optional

from src.pipelines.base_p2p_pipeline import BaseAceStepP2PEditPipeline
from src.p2p.controllers import AttentionControl

class TagsP2PEditPipeline(BaseAceStepP2PEditPipeline):
    def __init__(
            self,
            checkpoint_dir,
            controller: AttentionControl = None,
            blocks_to_inject_idxs: List[int] = None,
            dtype="bfloat16",
        ):
        super().__init__(
            checkpoint_dir,
            controller,
            blocks_to_inject_idxs,
            dtype
        )

    def __call__(
        self,
        noise: torch.Tensor,
        null_embeds_per_step: List[torch.Tensor],
        src_tags: str,
        tgt_tags: List[str],
        lyrics: str,
        guidance_scale: float = 15.0,
        guidance_interval: float = 0.5,
        infer_steps=60,
        scheduler_type: str = "euler",
        save_path: Optional[str] = None
    ):
        output_paths = self.forward(
            input_latents=noise,
            null_embeds_per_step=null_embeds_per_step,
            infer_step=infer_steps,
            tags=[src_tags] + tgt_tags + tgt_tags,
            lyrics=[lyrics for _ in range(2 * len(tgt_tags) + 1)],
            scheduler_type=scheduler_type,
            guidance_scale=guidance_scale,
            guidance_interval=guidance_interval,
            save_path=save_path,
        )
        return output_paths
