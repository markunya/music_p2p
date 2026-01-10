import abc
from typing import List

import torch


class StepCallbackBase(abc.ABC):
    @abc.abstractmethod
    def __call__(
        self,
        x_t: torch.Tensor,
        attention_store: List[torch.Tensor],
        diffusion_step: int,
    ):
        raise NotImplementedError


class SkipSteps:
    def __init__(self, skip_steps_num: int):
        self.skip_steps_num = int(skip_steps_num)

    @torch.no_grad()
    def __call__(
        self, x_t: torch.Tensor, attention_store, diffusion_step: int
    ) -> torch.Tensor:
        if diffusion_step < self.skip_steps_num:
            ref = x_t[:1]
            return ref.expand_as(x_t)
        return x_t


class LyricsLocalBlendTimeOnly(StepCallbackBase):
    def __init__(
        self,
        lyrics_len: int,
        threshold: float = 0.4,
        pool_k: int = 3,
        start_diffusion_step: int = 50,
        mask: List[int] = None,
    ):
        self.lyrics_len = int(lyrics_len)
        self.threshold = threshold
        self.pool_k = pool_k
        self.start_diffusion_step = start_diffusion_step
        self.mask = mask

    @torch.no_grad()
    def __call__(self, x_t: torch.Tensor, attention_store, diffusion_step: int):
        if self.mask is None:
            return x_t
        # deep_maps = [m for m in attention_store[-5:] if m is not None]
        # if not deep_maps:
        #     return x_t

        # maps = torch.cat(deep_maps, dim=1)[0]
        # maps = maps.max(dim=0).values

        # N = maps.shape[-1]
        # lyr_len = min(self.lyrics_len, N)

        # base = maps[:, N-lyr_len:N].mean(dim=-1)
        # p10 = torch.quantile(base, 0.10, dim=-1 if base.dim()==2 else 0, keepdim=True)
        # p90 = torch.quantile(base, 0.90, dim=-1 if base.dim()==2 else 0, keepdim=True)
        # mask = ((base - p10) / (p90 - p10 + 1e-6)).clamp(0, 1)
        # mask = F.avg_pool1d(
        #     mask.unsqueeze(0),
        #     kernel_size=17,
        #     padding=8,
        #     stride=1
        # ).squeeze(0)
        # mask = (mask >= 0.5).float()

        x_ref = x_t[:1]
        if isinstance(self.mask, list):
            self.mask = torch.tensor(self.mask, device=x_t.device)
        return x_ref + self.mask * (x_t - x_ref)
