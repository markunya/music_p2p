import warnings
from dataclasses import dataclass

import hydra
from hydra.core.config_store import ConfigStore
from omegaconf import MISSING

from src.logging import utils as logging
from src.pipelines.base_p2p_pipeline import BaseAceStepP2PEditPipeline
from src.utils.structures import BaseScriptConfig, DiffusionParams, Prompt
from src.utils.utils import set_random_seed, setup_exp_dir

warnings.filterwarnings("ignore", category=UserWarning)


@dataclass
class GenerateConfig(BaseScriptConfig):
    duration: int = -1
    latents_path: str | None = None

    prompt: Prompt = MISSING
    diffusion_params: DiffusionParams = MISSING


cs = ConfigStore.instance()
cs.store(name="generate", node=GenerateConfig)


@hydra.main(version_base=None, config_path="src/configs", config_name="generate")
def main(cfg: GenerateConfig):
    set_random_seed(cfg.seed)

    prompt = cfg.prompt
    diffusion_params = cfg.diffusion_params

    exp_dir = setup_exp_dir(cfg)

    pipeline = BaseAceStepP2PEditPipeline(
        checkpoint_dir=cfg.checkpoint_dir, debug_mode=cfg.debug_mode
    )
    output_paths = pipeline.text_to_music(
        prompt=prompt,
        diffusion_params=diffusion_params,
        duration=cfg.duration,
        save_path=exp_dir,
    )

    logging.info(f"Music successfully generated and saved to: {output_paths}")


if __name__ == "__main__":
    main()
