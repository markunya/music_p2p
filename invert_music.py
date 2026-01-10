import os
import warnings

import hydra
from hydra.utils import instantiate

from src.logging import utils as logging
from src.nti.music2noise import music2noise
from src.pipelines.base_p2p_pipeline import BaseAceStepP2PEditPipeline
from src.utils.structures import DiffusionParams, Prompt, dump
from src.utils.utils import set_random_seed, setup_exp_dir

warnings.filterwarnings("ignore", category=UserWarning)


@hydra.main(version_base=None, config_path="src/configs", config_name="invert_music")
def main(config):
    set_random_seed(config.invert_music.seed)

    prompt: Prompt = instantiate(config.prompt)
    diffusion_params: DiffusionParams = instantiate(config.diffusion_params)

    exp_dir = setup_exp_dir(config.invert_music)

    pipeline = BaseAceStepP2PEditPipeline(
        checkpoint_dir=config.edit.checkpoint_dir,
    )

    inverted_music_data = music2noise(
        pipeline=pipeline,
        music_path=config.invert_music.music_path,
        prompt=prompt,
        diffusion_params=diffusion_params,
        debug_mode=config.invert_music.debug_mode,
        audio_save_path=exp_dir,
    )

    out_path = os.path.join(exp_dir, "inverted_music_data.pth")
    dump(inverted_music_data, out_path)
    logging.info(f"Music successfully inverted and saved to {out_path}")


if __name__ == "__main__":
    main()
