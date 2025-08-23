import os
import torch
import random
import numpy as np
from typing import List
from local_blend import LyricsLocalBlendTimeOnly
from dataclasses import dataclass
from controllers import AttentionReplaceLyrics, AttentionReplaceTags, AttentionStore
from pipeline import LyricsP2PEditPipeline, TagsP2PEditPipeline
from acestep.models.lyrics_utils.lyric_tokenizer import VoiceBpeTokenizer
from seq_aligner import tokenize_lyrics

def setup_seed(seed):
    random.seed(seed)                
    np.random.seed(seed)
    torch.manual_seed(seed)        
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed) 
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

@dataclass
class TagsExpParams:
    identifier: str
    src_tags: str
    tgt_tags: str
    lyrics: str
    seed: int
    
Tags_EXP_PARAMS_LIST = \
[
]

EXPERIMENT_INJECTION_CONFIGS = {
    # "second_half": list(range(12, 24)),
    # "late_only": list(range(18, 24)),
    # "sparse_4_fh": [10, 15, 20, 23],
    # "center_4": [10, 11, 12, 13],
    # "center_6": [9, 10, 11, 12, 13, 14],
    # "center_4_+sparse": [10, 11, 12, 13, 16, 19, 22],
    # "quarter_3": [12, 13, 14, 15, 16, 17],
    # "quarter_4": [18, 19, 20, 21, 22, 23],
    # "quarter_2_3": [6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17],
    # "sparse_middle": [8, 12, 16]
    # "sparse_middlev2": [8, 10, 12],
    # "sparse_middlev3": [10, 12],
    # "sparse_middlev4": [6, 9, 12],
    # "sparse_more_early": [7, 9],
    "no_injection": [0]
}

def main():
    tokenizer = VoiceBpeTokenizer()

    pipe = TagsP2PEditPipeline(
        '../ACE_CHECKPOINTS',
        AttentionStore,
        blocks_to_inject_idxs=None,
        dtype='float32'
    )
    
    for exp_name, block_idxs in EXPERIMENT_INJECTION_CONFIGS.items():
        print(f"Running experiment: {exp_name} with blocks {block_idxs}")

        pipe.blocks_to_inject_idxs = block_idxs

        for tags_param in TAGS_EXP_PARAMS_LIST:
            setup_seed(tags_param.seed)

            output_dir = os.path.join("outputs", exp_name, tags_param.identifier)
            os.makedirs(output_dir, exist_ok=True)
            
            tags_len = len(tokenize_tags(tags_param.src_lyrics, tokenizer))

            output_paths = pipe(
                src_tags=tags_param.src_tags,
                tgt_tags=[tags_param.tgt_tags],
                lyrics=tags_param.lyrics,
                duration=60,
                controller_kwargs={
                    # 'prompts': [lyrics_param.src_lyrics, lyrics_param.tgt_lyrics],
                    # 'tokenizer': tokenizer,
                    'num_diffusion_steps': 60,
                    # 'local_blend': LyricsLocalBlendTimeOnly(lyrics_len, mask=lyrics_param.mask)
                },
                save_path=output_dir,
            )

            print(f"Saved to: {output_paths}")

if __name__ == "__main__":
    main()
