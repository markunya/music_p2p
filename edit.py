import os
import warnings

import hydra
from hydra.utils import instantiate

from src.nti.music2noise import music2noise
from src.utils.utils import set_random_seed
from src.utils.diffusion_utils import DiffusionParams
from src.utils.p2p_utils import P2PTaskParams
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

        noise, null_embeds_per_step = music2noise(
            pipeline=pipeline,
            path=p2p_task_params.music_path,
            lyrics=p2p_task_params.src.lyrics,
            tags=p2p_task_params.src.tags,
            num_steps=diffusion_params.num_steps,
            guidance_scale=diffusion_params.guidance_scale,
            guidance_interval=diffusion_params.guidance_interval,
            debug_mode=config.edit.debug_mode,
            audio_save_path=audio_save_path
        )

        data = {
            "lyrics": p2p_task_params.src.lyrics,
            "tags": p2p_task_params.src.tags,
            "noise": noise,
            "null_embeds_per_step": null_embeds_per_step,
            "num_steps": config.diffusion_params.num_steps,
            "guidance_scale": config.diffusion_params.guidance_scale,
            "guidance_interval": config.diffusion_params.guidance_interval
        }



if __name__ == "__main__":
    main()
