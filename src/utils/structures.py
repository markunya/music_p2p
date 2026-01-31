from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, auto
from typing import Any, Type, TypeVar

import cattrs
import torch
from omegaconf import MISSING

from src.schedulers import SchedulerType

T = TypeVar("T")

converter = cattrs.Converter()

converter.register_unstructure_hook(Enum, lambda e: e.name)
converter.register_structure_hook(Enum, lambda v, t: t[v])


def to_dict(obj: Any) -> dict:
    return converter.unstructure(obj)


def from_dict(cls: Type[T], data: dict) -> T:
    return converter.structure(data, cls)


def dump(obj: Any, path: str):
    torch.save(to_dict(obj), path)


def load(cls: Type[T], path: str, device: torch.device = None) -> T:
    dict_ = torch.load(path, device)
    return from_dict(dict_)


@dataclass
class BaseScriptConfig:
    checkpoint_dir: str = MISSING
    save_dir: str = MISSING
    exp_name: str = MISSING
    debug_mode: bool = False
    seed: int = 1


@dataclass
class Prompt:
    lyrics: str
    tags: str


@dataclass
class P2PTaskParams:
    music_path: str | None
    inverted_music_path: str | None
    src: Prompt
    tgt: Prompt


class CfgType(Enum):
    APG = auto()
    CFG = auto()
    CFG_STAR = auto()


@dataclass
class GuidanceParams:
    type: CfgType
    guidance_scale: float
    guidance_interval: float
    guidance_interval_decay: float
    min_guidance_scale: float


@dataclass
class DiffusionParams:
    num_steps: int
    guidance_params: GuidanceParams
    omega_scale: float
    scheduler_type: SchedulerType


@dataclass
class InvertedMusicData:
    prompt: Prompt
    diffusion_params: DiffusionParams
    noise: torch.Tensor
    null_embeds_per_step: list[torch.Tensor] | None


@dataclass
class DiffusionOut:
    trajectory: list[torch.Tensor]
    model_outs: list[torch.Tensor]
