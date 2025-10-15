import torch
import os
from pipelines.base_p2p_pipeline import BaseAceStepP2PEditPipeline
import torch
import argparse
from tqdm import tqdm
from nti.music2noise import music2noise

def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Run the ACE-Step Prompt-to-Prompt preprocessing pipeline.\n\n"
            "The script reads lyrics, tags, and music from the input directory, "
            "converts the music to noise using `music2noise`, and saves the resulting "
            "latent tensors and metadata to an output file."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        "--input_path",
        type=str,
        required=True,
        help=(
            "Путь к входной папке с исходными материалами.\n\n"
            " Требования к содержимому папки:\n"
            "  - lyrics.txt — текст песни (в кодировке UTF-8);\n"
            "  - tags.txt — теги в свободной форме (жанр, настроение и т.п.);\n"
            "  - music.mp3 — исходный музыкальный трек.\n\n"
            "Пример:\n"
            "  input_dir/\n"
            "  ├── lyrics.txt\n"
            "  ├── tags.txt\n"
            "  └── music.mp3"
        )
    )

    parser.add_argument(
        "--out_path",
        type=str,
        required=True,
        help=(
            "Путь к выходному .pt файлу, куда будет сохранён результат "
            "(например, '/home/user/output/music2noise_data.pt')."
        )
    )

    parser.add_argument(
        "--checkpoint_dir",
        type=str,
        required=True,
        help=(
            "Путь к директории с весами модели ACE-Step. "
            "По умолчанию '../ACE_CHECKPOINTS'."
        )
    )

    args = parser.parse_args()

    required_files = ["lyrics.txt", "tags.txt", "music.mp3"]
    missing = [f for f in required_files if not os.path.exists(os.path.join(args.input_path, f))]
    if missing:
        parser.error(
            f"В папке {args.input_path} отсутствуют необходимые файлы: {', '.join(missing)}.\n"
            f"Папка должна содержать:\n"
            f"  - lyrics.txt\n  - tags.txt\n  - music.mp3"
        )

    return args


def main():
    args = parse_args()

    dtype = torch.float32
    pipeline = BaseAceStepP2PEditPipeline(
        args.checkpint_dir,
        dtype=dtype
    )

    lyrics_path = f"{args.input_path}/lyrics.txt"
    tags_path = f"{args.input_path}/tags.txt"
    music_path = f"{args.input_path}/music.mp3"

    with open(lyrics_path, "r", encoding="utf-8") as f:
        lyrics = f.read()
    with open(tags_path, "r", encoding="utf-8") as f:
        tags = f.read() 

    num_steps = 400
    guidance_scale = 15.0
    guidance_interval = 0.5

    noise, null_embeds_per_step = music2noise(
        pipeline=pipeline,
        path=music_path,
        lyrics=lyrics,
        tags=tags,
        num_steps=num_steps,
        guidance_scale=guidance_scale,
        guidance_interval=guidance_interval,
        debug_mode=True,
        audio_save_path="/home/mabondarenko_4/music_p2p/check_gen"
    )

    data = {
        "lyrics": lyrics,
        "tags": tags,
        "noise": noise,
        "null_embeds_per_step": null_embeds_per_step,
        "num_steps": num_steps,
        "guidance_scale": guidance_scale,
        "guidance_interval": guidance_interval
    }
        
    torch.save(data, args.out_path)

if __name__ == "__main__":
    main()
