import os
import torch
import random
import numpy as np
from typing import List
from dataclasses import dataclass
from music_p2p.p2p.controllers import AttentionReplaceTags
from pipeline import TagsP2PEditPipeline
from music_p2p.p2p.step_callback import SkipSteps

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
    
TAGS_EXP_PARAMS_LIST = \
[
    TagsExpParams(
        identifier="soul",
        lyrics="[verse]\nNeon lights they flicker bright\nCity hums in dead of night\nRhythms pulse through concrete veins\nLost in echoes of refrains\n\n[verse]\nBassline groovin' in my chest\nHeartbeats match the city's zest\nElectric whispers fill the air\nSynthesized dreams everywhere\n\n[chorus]\nTurn it up and let it flow\nFeel the fire let it grow\nIn this rhythm we belong\nHear the night sing out our song",
        src_tags="rock, hip - hop, orchestral, bass, male voice",
        tgt_tags = "phonk, trap, dark, heavy bass, pitched vocals",
        seed=1
    ),
    TagsExpParams(
        identifier="cuba",
        lyrics = "[verse]\nSun dips low the night ignites\nBassline hums with gleaming lights\nElectric guitar singing tales so fine\nIn the rhythm we all intertwine\n\n[verse]\nDrums beat steady calling out\nPercussion guides no room for doubt\nElectric pulse through every vein\nDance away every ounce of pain\n\n[chorus]\nFeel the rhythm feel the flow\nLet the music take control\nBassline deep electric hum\nIn this night we're never numb\n\n[bridge]\nStars above they start to glow\nEchoes of the night's soft glow\nElectric strings weave through the air\nIn this moment none compare\n\n[verse]\nHeartbeats sync with every tone\nLost in music never alone\nElectric tales of love and peace\nIn this groove we find release",
        src_tags="male voice, Cuban music, salsa, son, Afro-Cuban, traditional Cuban",
        tgt_tags="female voice, Latin pop, EDM, sidechain, bright synths, wide chorus",
        seed=11
    ),
    TagsExpParams(
        identifier="nightclub",
        lyrics = "Burning in motion, set me alight!\nEvery heartbeat turns into a fight!\nCaged in rhythm, chained in time!\nLove’s a battle— You're Mine!  You're Mine!",
        src_tags="Nightclubs, dance parties, female voice, workout playlists, radio broadcasts",
        tgt_tags="Nightclubs, dance parties, male voice, chill playlists, radio broadcasts",
        seed=12
    ),
]

EXPERIMENT_INJECTION_CONFIGS = {
    # "second_half": list(range(12, 24)),
    # "late_only": list(range(18, 24)),
    # "sparse_4_fh": [10, 15, 20, 23],
    # "center_4": [10, 11, 12, 13],
    # "center_6": [9, 10, 11, 12, 13, 14],
    "full": list(range(24)),
    # "center_4_+sparse": [10, 11, 12, 13, 16, 19, 22],
    # "quarter_3": [12, 13, 14, 15, 16, 17],
    # "quarter_4": [18, 19, 20, 21, 22, 23],
    # "quarter_2_3": [6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17],
    # "sparse_middle": [8, 12, 16]
    # "sparse_middlev2": [8, 10, 12],
    # "sparse_middlev3": [10, 12],
    # "sparse_middlev4": [6, 9, 12],
    # "sparse_more_early": [7, 9],
    # "no_injection": []
}

def main():
    pipe = TagsP2PEditPipeline(
        '../ACE_CHECKPOINTS',
        blocks_to_inject_idxs=None,
        dtype='float32'
    )
    
    for exp_name, block_idxs in EXPERIMENT_INJECTION_CONFIGS.items():
        print(f"Running experiment: {exp_name} with blocks {block_idxs}")

        pipe.blocks_to_inject_idxs = block_idxs

        for tags_param in TAGS_EXP_PARAMS_LIST:
            setup_seed(tags_param.seed)

            controller = AttentionReplaceTags(
                prompts=[tags_param.src_tags, tags_param.tgt_tags],
                tokenizer=pipe.text_tokenizer,
                step_callback=None,
                num_diffusion_steps=60,
                diffusion_step_start=1,
                diffusion_step_end=30
            )
            pipe.controller = controller

            output_dir = os.path.join("outputs_tags", exp_name, tags_param.identifier)
            os.makedirs(output_dir, exist_ok=True)

            output_paths = pipe(
                src_tags=tags_param.src_tags,
                tgt_tags=[tags_param.tgt_tags],
                lyrics=tags_param.lyrics,
                duration=60,
                save_path=output_dir,
            )

            print(f"Saved to: {output_paths}")

if __name__ == "__main__":
    main()
