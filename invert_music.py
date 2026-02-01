import os
import warnings
from dataclasses import dataclass

import hydra
from hydra.core.config_store import ConfigStore
from omegaconf import MISSING

from src.logging import utils as logging
from src.logging.writer import setup_writer
from src.nti.music2noise import music2noise
from src.nti.null_text_inversion import NullTextOptimization
from src.pipelines.base_p2p_pipeline import BaseAceStepP2PEditPipeline
from src.utils.structures import BaseScriptConfig, DiffusionParams, Prompt, dump
from src.utils.utils import set_random_seed, setup_exp_dir

warnings.filterwarnings("ignore", category=UserWarning)


@dataclass
class InvertMusicConfig(BaseScriptConfig):
    learning_rate: float = 1e-3
    num_inner_steps: int = 15
    epsilon: float = 1e-7

    music_path: str = MISSING

    prompt: Prompt = MISSING
    diffusion_params: DiffusionParams = MISSING


cs = ConfigStore.instance()
cs.store(name="invert_music", node=InvertMusicConfig)


@hydra.main(version_base=None, config_path="src/configs", config_name="invert_music")
def main(cfg: InvertMusicConfig):
    set_random_seed(cfg.seed)

    exp_dir = setup_exp_dir(cfg)
    writer = setup_writer(cfg)

    pipeline = BaseAceStepP2PEditPipeline(
        checkpoint_dir=cfg.checkpoint_dir, debug_mode=cfg.debug_mode, writer=writer
    )

    nti = NullTextOptimization(
        model=pipeline.ace_step_transformer,
        lr=cfg.learning_rate,
        num_inner_steps=cfg.num_inner_steps,
        epsilon=cfg.epsilon,
        debug_mode=cfg.debug_mode,
        writer=writer,
    )

    inverted_music_data = music2noise(
        pipeline=pipeline,
        music_path=cfg.music_path,
        prompt=cfg.prompt,
        diffusion_params=cfg.diffusion_params,
        nti=nti,
        writer=writer,
        debug_mode=cfg.debug_mode,
        audio_save_path=exp_dir,
    )

    out_path = os.path.join(exp_dir, "inverted_music_data.pth")
    dump(inverted_music_data, out_path)
    logging.info(f"Music successfully inverted and saved to {out_path}")


if __name__ == "__main__":
    main()
