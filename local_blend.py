import abc
import torch
import torch.nn.functional as F
from typing import List

class LocalBlend(abc.ABC):
    @abc.abstractmethod
    def __call__(
        self,
        x_t: torch.Tensor,
        attention_store: List[torch.Tensor]
    ):
        raise NotImplementedError

class LyricsLocalBlendTimeOnly(LocalBlend):
    def __init__(
            self,
            lyrics_len: int,
            threshold: float = 0.4,
            pool_k: int = 11,
            start_diffusion_step: int = 40
        ):
        self.lyrics_len = int(lyrics_len)
        self.threshold = threshold
        self.pool_k = pool_k
        self.start_diffusion_step = start_diffusion_step

    @torch.no_grad()
    def __call__(self, x_t: torch.Tensor, attention_store, diffusion_step: int):
        if diffusion_step < self.start_diffusion_step:
            return x_t
        deep_maps = [m for m in attention_store[len(attention_store)//2:] if m is not None]
        if not deep_maps:
            return x_t

        maps = torch.cat(deep_maps, dim=1)[0]
        maps = maps.max(dim=0).values

        N = maps.shape[-1]
        lyr_len = min(self.lyrics_len, N)

        base = maps[:, N-lyr_len:N].mean(dim=-1)
        p10 = torch.quantile(base, 0.10, dim=-1 if base.dim()==2 else 0, keepdim=True)
        p90 = torch.quantile(base, 0.90, dim=-1 if base.dim()==2 else 0, keepdim=True)
        base = ((base - p10) / (p90 - p10 + 1e-6)).clamp(0, 1)
        mask = (base >= self.threshold).float()
        mask = F.max_pool1d(
            mask.unsqueeze(1),
            kernel_size=self.pool_k,
            stride=1,
            padding=self.pool_k // 2
        ).squeeze(1)
        print(mask)

        x_ref = x_t[:1]
        return x_ref + mask * (x_t - x_ref)

