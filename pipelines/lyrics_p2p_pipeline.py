from typing import List, Optional
from pipelines.base_p2p_pipeline import BaseAceStepP2PEditPipeline
from p2p.controllers import AttentionControl

class LyricsP2PEditPipeline(BaseAceStepP2PEditPipeline):
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
        src_lyrics: str,
        tgt_lyrics: List[str],
        genre_tags: str,
        duration: float = -1,
        guidance_scale: float = 15.0,
        infer_steps=60,
        scheduler_type: str = "euler",
        save_path: Optional[str] = None
    ):
        output_paths = self.forward(
            audio_duration=duration,
            infer_step=infer_steps,
            prompts=[genre_tags for _ in range(2 * len(tgt_lyrics) + 1)],
            lyrics=[src_lyrics] + tgt_lyrics + tgt_lyrics,
            scheduler_type=scheduler_type,
            guidance_scale=guidance_scale,
            save_path=save_path,
        )
        return output_paths
