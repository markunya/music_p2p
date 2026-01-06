import os
import warnings

import hydra
from hydra.utils import instantiate

from src.utils import logging
from src.nti.music2noise import music2noise
from src.utils.utils import set_random_seed
from src.utils.structures import DiffusionParams, P2PTaskParams, dump
from src.pipelines.lyrics_p2p_pipeline import LyricsP2PEditPipeline
from src.pipelines.tags_p2p_pipeline import TagsP2PEditPipeline
from src.p2p.controllers import AttentionControl

warnings.filterwarnings("ignore", category=UserWarning)

@hydra.main(version_base=None, config_path="src/configs", config_name="edit")
def main(config):
    set_random_seed(config.edit.seed)

    controller: AttentionControl = instantiate(config.controller)
    diffusion_params: DiffusionParams = instantiate(config.diffusion_params)
    p2p_task_params: P2PTaskParams = instantiate(config.p2p_task_params)

    exp_dir = os.path.join(config.edit.save_dir, config.edit.exp_name)
    os.makedirs(exp_dir)

    if controller.__class__.__name__.find('Tags') != -1:
        pipeline_cls = TagsP2PEditPipeline
    else:
        pipeline_cls = LyricsP2PEditPipeline

    pipeline = pipeline_cls(
        checkpoint_dir=config.edit.checkpoint_dir,
        controller=controller,
    )

    if config.task_params.invert_music_path is not None:
        pass # generate music
    elif config.task_params.music_path is not None:
        pass # read inverted noise with params
    else:
        audio_save_path = os.path.join(exp_dir, 'music2noise')
        os.makedirs(audio_save_path)

        inverted_music_data = music2noise(
            pipeline=pipeline,
            path=p2p_task_params.music_path,
            prompt=config.p2p_task_params.src,
            diffusion_params=diffusion_params,
            debug_mode=config.edit.debug_mode,
            audio_save_path=audio_save_path
        )
        out_path = os.path.join(audio_save_path, 'inverted_music_data.pth')
        dump(inverted_music_data, out_path)
        logging.info(f"Music successfully inverted and saved to {out_path}")

if __name__ == "__main__":
    main()
