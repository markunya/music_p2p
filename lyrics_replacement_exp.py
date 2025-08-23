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
class LyricsExpParams:
    identifier: str
    src_lyrics: str
    tgt_lyrics: str
    tags: str
    seed: int
    mask: List[int]

LYRICS_EXP_PARAMS_LIST = \
[
    # LyricsExpParams(
    #     identifier="epic_metal",
    #     src_lyrics="Running through the night, the shadows chase me down!",
    #     tgt_lyrics="Gunning through the fight, I wear a silver crown!",
    #     tags="metal, heavy metal, electric guitar, double bass, fast tempo, aggressive, heroic, dark energy",
    #     seed=12
    # ),
    # LyricsExpParams(
    #     identifier="dreamy_ambient",
    #     src_lyrics="Falling into dreams, the stars begin to glow!",
    #     tgt_lyrics="Calling through the streams, I feel the river flow!",
    #     tags="ambient, downtempo, chillout, soft pads, slow bpm, atmospheric, emotional, dreamy",
    #     seed=22
    # ),
    LyricsExpParams(
        identifier="dance_pop",
        src_lyrics="Dancing in the fire, the rhythm takes control!",
        tgt_lyrics="Prancing with desire, it’s burning through my soul!",
        tags="pop, dance pop, energetic, catchy, synths, upbeat, female vocal, lively",
        seed=3,
        mask=[0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.]
    ),
    # LyricsExpParams(
    #     identifier="electro_house",
    #     src_lyrics="Riding on the wind, my heart begins to soar!",
    #     tgt_lyrics="Sliding on the ice, my feet begin to roar!",
    #     tags="electronic, house, electro house, synthesizer, drums, bass, percussion, fast, energetic, uplifting",
    #     seed=42
    # ),
    # LyricsExpParams(
    #     identifier="dark_cinematic",
    #     src_lyrics="Sinking in the storm, I try to hold my ground!",
    #     tgt_lyrics="Blinking through the swarm, I hear a haunting sound!",
    #     tags="cinematic, dark ambient, horror, tension, drones, eerie textures, slow build",
    #     seed=52
    # ),
    LyricsExpParams(
        identifier="synthwave",
        src_lyrics="Flying past the sun, my rocket leaves the Earth!",
        tgt_lyrics="Crying from the gun, I question what it's worth!",
        tags="synthwave, retro wave, 80s synths, analog sounds, spacey, nostalgic, futuristic",
        seed=6,
        mask=[0., 0., 0., 0., 0., 0., 0., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 0., 0., 0., 0., 0., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1.]
    ),
    # LyricsExpParams(
    #     identifier="punk_rock",
    #     src_lyrics="Crashing through the gate, my engine starts to scream!",
    #     tgt_lyrics="Flashing into fate, I’m chasing every dream!",
    #     tags="rock, punk rock, distorted guitar, anthemic, fast tempo, rebellion, energetic, live drums",
    #     seed=7
    # ),
    # LyricsExpParams(
    #     identifier="lofi_chill",
    #     src_lyrics="Hiding from the light, the silence wraps me tight!",
    #     tgt_lyrics="Gliding through the night, I vanish out of sight!",
    #     tags="lo-fi, lo-fi hip hop, mellow, vinyl crackle, soft drums, introspective, night vibes",
    #     seed=82
    # ),
    # LyricsExpParams(
    #     identifier="alt_rock",
    #     src_lyrics="Burning like a flame, the passion never dies!",
    #     tgt_lyrics="Turning from the shame, I see through all the lies!",
    #     tags="alternative rock, emotional rock, melodic, guitar-driven, introspective, powerful chorus",
    #     seed=92
    # ),
    LyricsExpParams(
        identifier="psytrance",
        src_lyrics="Surfing on the wave, the ocean sings to me!",
        tgt_lyrics="Swerving from the wave, I rewrite destiny!",
        tags="psytrance, trance, progressive, high bpm, driving bassline, hypnotic, euphoric, tribal elements",
        seed=10,
        mask=[1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
                1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
                1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
                1., 1., 1., 1., 1., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
                0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
                0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
                0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
                0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
                0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.]
    ),
]

LONG_LYRICS_EXP_PARAMS_LIST=\
[
    LyricsExpParams(
        identifier="soul",
        src_lyrics="[verse]\nNeon lights they flicker bright\nCity hums in dead of night\nRhythms pulse through concrete veins\nLost in echoes of refrains\n\n[verse]\nBassline groovin' in my chest\nHeartbeats match the city's zest\nElectric whispers fill the air\nSynthesized dreams everywhere\n\n[chorus]\nTurn it up and let it flow\nFeel the fire let it grow\nIn this rhythm we belong\nHear the night sing out our song\n\n[verse]\nGuitar strings they start to weep\nWake the soul from silent sleep\nEvery note a story told\nIn this night we’re bold and gold\n\n[bridge]\nVoices blend in harmony\nLost in pure cacophony\nTimeless echoes timeless cries\nSoulful shouts beneath the skies\n\n[verse]\nKeyboard dances on the keys\nMelodies on evening breeze\nCatch the tune and hold it tight\nIn this moment we take flight",
        tgt_lyrics="[verse]\nNeon signs they shimmer light\nCity breathes in endless night\nRhythms surge through iron veins\nFound in shadows of refrains\n\n[verse]\nBassline beating in my chest\nHeartbeats chase the city's quest\nElectric murmurs paint the air\nDigitized dreams rising there\n\n[chorus]\nCrank it loud and let it go\nFeel the fever let it show\nIn this rhythm we survive\nHear the midnight come alive\n\n[verse]\nGuitar strings they start to cry\nWake the soul that hides inside\nEvery chord a secret told\nIn this night we’re fierce and bold\n\n[bridge]\nVoices clash in symphony\nLost inside cacophony\nEndless echoes endless highs\nRestless shouts beneath the skies\n\n[verse]\nKeyboard dances through the breeze\nHarmony on twilight seas\nCatch the song and hold it near\nIn this moment we break clear",
        tags="rock, hip - hop, orchestral, bass, drums, electric guitar, piano, synthesizer, violin, viola, cello, fast, energetic, motivational, inspirational, empowering",
        seed=10,
        mask=None
    ),
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

    pipe = LyricsP2PEditPipeline(
        '../ACE_CHECKPOINTS',
        AttentionStore,
        blocks_to_inject_idxs=None,
        dtype='float32'
    )
    
    for exp_name, block_idxs in EXPERIMENT_INJECTION_CONFIGS.items():
        print(f"Running experiment: {exp_name} with blocks {block_idxs}")

        pipe.blocks_to_inject_idxs = block_idxs

        for lyrics_param in LONG_LYRICS_EXP_PARAMS_LIST:
            setup_seed(lyrics_param.seed)

            output_dir = os.path.join("outputs", exp_name, lyrics_param.identifier)
            os.makedirs(output_dir, exist_ok=True)
            
            lyrics_len = len(tokenize_lyrics(lyrics_param.src_lyrics, tokenizer))

            output_paths = pipe(
                src_lyrics=lyrics_param.src_lyrics,
                tgt_lyrics=[lyrics_param.tgt_lyrics],
                genre_tags=lyrics_param.tags,
                duration=60,
                controller_kwargs={
                    # 'prompts': [lyrics_param.src_lyrics, lyrics_param.tgt_lyrics],
                    # 'tokenizer': tokenizer,
                    'num_diffusion_steps': 60,
                    'local_blend': LyricsLocalBlendTimeOnly(lyrics_len, mask=lyrics_param.mask)
                },
                save_path=output_dir,
            )

            print(f"Saved to: {output_paths}")

if __name__ == "__main__":
    main()
