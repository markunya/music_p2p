import os
import warnings

import hydra
from hydra.utils import instantiate

from src.utils import logging
from src.nti.music2noise import music2noise
from src.utils.utils import set_random_seed
from src.utils.structures import DiffusionParams, Prompt, dump
from src.pipelines.base_p2p_pipeline import BaseAceStepP2PEditPipeline

warnings.filterwarnings("ignore", category=UserWarning)

@hydra.main(version_base=None, config_path="src/configs", config_name="invert_music")
def main(config):
    set_random_seed(config.invert_music.seed)

    prompt: Prompt = instantiate(config.prompt)
    diffusion_params: DiffusionParams = instantiate(config.diffusion_params)

    exp_dir = os.path.join(config.invert_music.save_dir, config.invert_music.exp_name)
    os.makedirs(exp_dir)

    pipeline = BaseAceStepP2PEditPipeline(
        checkpoint_dir=config.edit.checkpoint_dir,
    )

    inverted_music_data = music2noise(
        pipeline=pipeline,
        music_path=config.invert_music.music_path,
        prompt=prompt,
        diffusion_params=diffusion_params,
        debug_mode=config.invert_music.debug_mode,
        audio_save_path=exp_dir
    )

    out_path = os.path.join(exp_dir, 'inverted_music_data.pth')
    dump(inverted_music_data, out_path)
    logging.info(f"Music successfully inverted and saved to {out_path}")

if __name__ == "__main__":
    main()
