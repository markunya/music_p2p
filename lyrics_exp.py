import os
import torch
import random
import numpy as np
from typing import List
from dataclasses import dataclass
from p2p.controllers import AttentionReplaceTags
from pipelines.tags_p2p_pipeline import TagsP2PEditPipeline
from p2p.step_callback import SkipSteps
import torch.nn.functional as F


def main():
    BATCH_N = 3
    SPEAKER_LEN = 1

    pipe = TagsP2PEditPipeline(
        '../ACE_CHECKPOINTS',
        blocks_to_inject_idxs=None,
        dtype='float32'
    )

    chp = torch.load('stay.pt')

    new_tags = 'electronic, dance, edm, future house, melodic, energetic, female vocal, uplifting, club, emotional, modern, synth bass, drop, rhythmic, catchy hook, remix'
    new_tags = 'electronic, dance, edm, future house, melodic, energetic, female vocal, uplifting, club, emotional, modern, synth bass, drop, rhythmic, catchy hook, remix'

    controller = AttentionReplaceTags(
        prompts=[chp['tags'], new_tags],
        tokenizer=pipe.text_tokenizer,
        step_callback=None,
        num_diffusion_steps=400,
        diffusion_step_start=1,
        diffusion_step_end=200
    )
    pipe.controller = controller

    output_dir = os.path.join("outputs_tags", "stay")
    os.makedirs(output_dir, exist_ok=True)

    noise = chp['noise'].repeat(BATCH_N, 1, 1, 1)

    null_embeds_per_step = chp['null_embeds_per_step']
    for i in range(len(null_embeds_per_step)):
        null_embeds_per_step[i] = null_embeds_per_step[i].repeat(BATCH_N, 1, 1)

    tok = pipe.text_tokenizer

    src_tok = tok([chp['tags']], padding="longest", truncation=True, return_tensors="pt")
    tgt_tok = tok([new_tags],   padding="longest", truncation=True, return_tensors="pt")

    tags_old_len = int(src_tok.input_ids.shape[1])
    tags_new_len = int(tgt_tok.input_ids.shape[1])
    delta = tags_new_len - tags_old_len

    if delta > 0:
        insert_idx = SPEAKER_LEN + tags_old_len

        for i in range(len(null_embeds_per_step)):
            emb = null_embeds_per_step[i]
            B, S, D = emb.shape
            insert_idx_clamped = max(0, min(insert_idx, S))

            pad_block = torch.zeros(B, delta, D, device=emb.device, dtype=emb.dtype)
            emb = torch.cat([emb[:, :insert_idx_clamped, :],
                             pad_block,
                             emb[:, insert_idx_clamped:, :]], dim=1)
            null_embeds_per_step[i] = emb

    output_paths = pipe(
        noise=noise,
        null_embeds_per_step=null_embeds_per_step,
        src_tags=chp['tags'],
        infer_steps=400,
        tgt_tags=[new_tags],
        lyrics=chp['lyrics'],
        save_path=output_dir,
    )

    print(f"Saved to: {output_paths}")


if __name__ == "__main__":
    main()
