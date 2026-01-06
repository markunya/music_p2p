import os
import warnings

import hydra
from hydra.utils import instantiate

from src.utils import logging
from src.utils.utils import set_random_seed
from src.utils.structures import DiffusionParams
from acestep.pipeline_ace_step import ACEStepPipeline

warnings.filterwarnings("ignore", category=UserWarning)

@hydra.main(version_base=None, config_path="src/configs", config_name="generate")
def main(config):
    set_random_seed(config.generate.seed)

    diffusion_params: DiffusionParams = instantiate(config.diffusion_params)

    exp_dir = os.path.join(config.edit.save_dir, config.generate.exp_name)
    os.makedirs(exp_dir)

    pipeline = ACEStepPipeline(
        checkpoint_dir=config.generate.checkpoint_dir
    )

    latents_rec2 = pipeline.diffusion_process(
            input_latents=trajectory[0],
            encoder_text_hidden_states=encoder_text_hidden_states,
            text_attention_mask=text_attention_mask,
            speaker_embds=speaker_embeds,
            lyric_token_ids=lyric_token_idx,
            lyric_mask=lyric_mask,
            diffusion_params=diffusion_params,
        )
    
    pipeline.latents2audio(latents_rec2, save_path=exp_dir)

if __name__ == "__main__":
    main()
