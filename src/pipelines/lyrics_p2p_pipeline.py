from typing import List, Optional

from src.pipelines.base_p2p_pipeline import BaseAceStepP2PEditPipeline
from src.p2p.controllers import AttentionControl
from src.utils.diffusion_utils import DiffusionParams

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
        tags: str,
        diffusion_params: DiffusionParams,
        duration: float = -1,
        save_path: Optional[str] = None
    ):
        output_paths = self.forward(
            audio_duration=duration,
            tags=[tags for _ in range(2 * len(tgt_lyrics) + 1)],
            lyrics=[src_lyrics] + tgt_lyrics + tgt_lyrics,
            diffusion_params=diffusion_params,
            save_path=save_path,
        )
        return output_paths
