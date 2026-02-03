from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime
from enum import Enum, auto

from omegaconf import DictConfig, OmegaConf

from src.logging import utils as logging


class ExperimentMode(Enum):
    Online = auto()
    Offline = auto()


@dataclass
class CometMLConfig:
    project_name: str
    workspace: str | None
    run_name: str | None
    mode: ExperimentMode


class BaseWriter(ABC):
    @abstractmethod
    def set_step(self, step):
        raise NotImplementedError

    @abstractmethod
    def add_scalar(self, scalar_name, scalar):
        raise NotImplementedError

    @abstractmethod
    def add_scalars(self, scalars):
        raise NotImplementedError

    @abstractmethod
    def add_audio(self, audio_name, audio, sample_rate=None):
        raise NotImplementedError

    @abstractmethod
    def add_image(self, image_name, image):
        raise NotImplementedError


class DummyWriter(BaseWriter):
    def set_step(self, step):
        pass

    def add_scalar(self, scalar_name, scalar):
        pass

    def add_scalars(self, scalars):
        pass

    def add_audio(self, audio_name, audio, sample_rate=None):
        pass

    def add_image(self, image_name, image):
        pass


class CometMLWriter(BaseWriter):
    """
    Class for experiment tracking via CometML.

    See https://www.comet.com/docs/v2/.
    """

    def __init__(self, project_config):
        try:
            import comet_ml

            from src.utils.structures import to_dict

            comet_ml.login()

            config = project_config.writer
            match config.mode:
                case ExperimentMode.Offline:
                    exp_class = comet_ml.OfflineExperiment
                case ExperimentMode.Online:
                    exp_class = comet_ml.Experiment
                case _:
                    raise ValueError(f"Invalid mode value: {config.mode}")

            self.exp = exp_class(
                project_name=config.project_name,
                workspace=config.workspace,
                experiment_key=None,
                log_code=False,
                log_graph=False,
                auto_metric_logging=False,
                auto_param_logging=False,
            )
            self.exp.set_name(config.run_name)
            self.exp.log_parameters(parameters=to_dict(project_config))

            self.comel_ml = comet_ml

        except ImportError:
            logging.warning("For use comet_ml install it via \n\t pip install comet_ml")

        self.step = 0
        self.timer = datetime.now()

    def set_step(self, step: int):
        """
        Define current step and mode for the tracker.

        Calculates the difference between method calls to monitor

        Args:
            step (int): current step.
        """
        previous_step = self.step
        self.step = step
        if step == 0:
            self.timer = datetime.now()
        else:
            duration = datetime.now() - self.timer
            self.add_scalar(
                "steps_per_sec", (self.step - previous_step) / duration.total_seconds()
            )
            self.timer = datetime.now()

    def add_scalar(self, scalar_name, scalar):
        """
        Log a scalar to the experiment tracker.

        Args:
            scalar_name (str): name of the scalar to use in the tracker.
            scalar (float): value of the scalar.
        """
        self.exp.log_metrics(
            {scalar_name: scalar},
            step=self.step,
        )

    def add_scalars(self, scalars):
        """
        Log several scalars to the experiment tracker.

        Args:
            scalars (dict): dict, containing scalar name and value.
        """
        self.exp.log_metrics(
            {scalar_name: scalar for scalar_name, scalar in scalars.items()},
            step=self.step,
        )

    def add_image(self, image_name, image):
        """
        Log an image to the experiment tracker.

        Args:
            image_name (str): name of the image to use in the tracker.
            image (Path | Tensor | ndarray | list[tuple] | Image): image
                in the CometML-friendly format.
        """
        self.exp.log_image(image_data=image, name=image_name, step=self.step)

    def add_audio(self, audio_name, audio, sample_rate=48_000):
        """
        Log an audio to the experiment tracker.

        Args:
            audio_name (str): name of the audio to use in the tracker.
            audio (Path | ndarray): audio in the CometML-friendly format.
            sample_rate (int): audio sample rate.
        """
        audio = audio.detach().cpu().numpy().T
        self.exp.log_audio(
            file_name=audio_name, audio_data=audio, sample_rate=sample_rate
        )


def setup_writer(cfg):
    if cfg.writer is None:
        return DummyWriter()
    if isinstance(cfg.writer, DictConfig):
        writer_cfg = OmegaConf.to_object(cfg.writer)
    if isinstance(writer_cfg, CometMLConfig):
        return CometMLWriter(cfg)
    raise ValueError(f"Not supportd writer config type: {type(writer_cfg)}")
