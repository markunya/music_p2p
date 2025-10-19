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
            "Запускает препроцессинг пайплайна ACE-Step Prompt-to-Prompt.\n\n"
            "Скрипт считывает текст песни, теги и аудиофайл из входной директории, "
            "преобразует музыку в латентное шумовое представление с помощью функции `music2noise` "
            "и сохраняет полученные тензоры и метаданные в выходной файл."
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
        "--num_steps",
        type=int,
        required=False,
        default=400,
        help=(
            "Количество шагов диффузионного процесса"
        )
    )

    parser.add_argument(
        "--guidance_scale",
        type=float,
        required=False,
        default=15.0,
    )

    parser.add_argument(
        "--guidance_interval",
        type=float,
        required=False,
        default=0.5,
    )

    parser.add_argument(
        "--checkpoint_dir",
        type=str,
        required=True,
        help=(
            "Путь к директории с весами модели ACE-Step. "
        )
    )

    parser.add_argument(
        "--debug_mode",
        type=bool,
        required=False,
        default=False,
        help=(
            "Позволяет запустить оптимизацию в режиме дебага, когда логируется дополнительная инфомрация"
        )
    )

    parser.add_argument(
        "--audio_save_path",
        type=str,
        required=False,
        default='./',
        help=(
            "Параметр для режима отладки. Путь куда сохранить аудио сгенерированное из шума полученного после null text optimization."
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
        args.checkpoint_dir,
        dtype=dtype
    )

    lyrics_path = f"{args.input_path}/lyrics.txt"
    tags_path = f"{args.input_path}/tags.txt"
    music_path = f"{args.input_path}/music.mp3"

    with open(lyrics_path, "r", encoding="utf-8") as f:
        lyrics = f.read()
    with open(tags_path, "r", encoding="utf-8") as f:
        tags = f.read() 

    noise, null_embeds_per_step = music2noise(
        pipeline=pipeline,
        path=music_path,
        lyrics=lyrics,
        tags=tags,
        num_steps=args.num_steps,
        guidance_scale=args.guidance_scale,
        guidance_interval=args.guidance_interval,
        debug_mode=args.debug_mode,
        audio_save_path=args.audio_save_path
    )

    data = {
        "lyrics": lyrics,
        "tags": tags,
        "noise": noise,
        "null_embeds_per_step": null_embeds_per_step,
        "num_steps": args.num_steps,
        "guidance_scale": args.guidance_scale,
        "guidance_interval": args.guidance_interval
    }
        
    torch.save(data, args.out_path)

if __name__ == "__main__":
    main()
