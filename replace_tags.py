import os
import torch
import argparse
from tqdm import tqdm

from src.p2p.controllers import AttentionReplaceTags
from src.pipelines.tags_p2p_pipeline import TagsP2PEditPipeline

def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Редактирование музыки заменой тегов с помощью Prompt-to-Prompt"
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        "--noise_path",
        type=str,
        required=True,
        help=(
            "Путь к шуму полученному из invert_music.py."
        )
    )

    parser.add_argument(
        "--new_tags_path",
        type=str,
        required=True,
        help="Путь к текстовому файлу (UTF-8) с целевыми тегами. Содержимое будет считано целиком."
    )

    parser.add_argument(
        "--out_dir_path",
        type=str,
        required=True,
        help=(
            "Путь к директории в которую будет сохранен результат редактирования."
        )
    )

    parser.add_argument(
        '--diffusion_step_start',
        type=float,
        required=False,
        default=0.0,
        help=(
            "Шаг на котором начинается процесс замены карт внимания"
        )
    )

    parser.add_argument(
        '--diffusion_step_end',
        type=float,
        required=False,
        default=0.5,
        help=(
            "Шаг на котором заканчивается процесс замены карт внимания"
        )
    )

    parser.add_argument(
        "--checkpoint_dir",
        type=str,
        required=True,
        help=(
            "Путь к директории с весами модели ACE-Step. "
        )
    )

    args = parser.parse_args()
    return args


def main():
    args = parse_args()

    pipe = TagsP2PEditPipeline(
        '../ACE_CHECKPOINTS',
        blocks_to_inject_idxs=None,
        dtype='float32'
    )

    chp = torch.load(args.noise_path)

    num_diffusion_steps = chp['num_steps']
    diffusion_step_start = min(max(int(num_diffusion_steps * args.diffusion_step_start), 1), num_diffusion_steps)
    diffusion_step_end = min(max(int(num_diffusion_steps * args.diffusion_step_end), 1), num_diffusion_steps)
    guidance_scale = chp['guidance_scale']
    guidance_interval = chp['guidance_interval']

    tqdm.write(
        "P2P process params:\n"
        f" - num_diffusion_steps={num_diffusion_steps}\n"
        f" - diffusion_step_start={diffusion_step_start}\n"
        f" - diffusion_step_end={diffusion_step_end}\n"
        f" - guidance_scale={guidance_scale}\n"
        f" - guidance_interval={guidance_interval}\n"
    )

    with open(args.new_tags_path, "r", encoding="utf-8") as f:
        new_tags = f.read().strip()

    controller = AttentionReplaceTags(
        prompts=[chp['tags'], new_tags],
        tokenizer=pipe.text_tokenizer,
        step_callback=None,
        num_diffusion_steps=num_diffusion_steps,
        diffusion_step_start=diffusion_step_start,
        diffusion_step_end=diffusion_step_end
    )
    pipe.controller = controller

    os.makedirs(args.out_dir_path, exist_ok=True)

    noise = chp['noise'].repeat(3, 1, 1, 1).to(pipe.device)

    null_embeds_per_step = chp['null_embeds_per_step']
    for i in range(len(null_embeds_per_step)):
        null_embeds_per_step[i] = null_embeds_per_step[i].repeat(3, 1, 1).to(pipe.device)

    output_paths = pipe(
        noise=noise,
        null_embeds_per_step=null_embeds_per_step,
        src_tags=chp['tags'],
        infer_steps=num_diffusion_steps,
        tgt_tags=[new_tags],
        lyrics=chp['lyrics'],
        save_path=args.out_dir_path,
        guidance_scale=guidance_scale,
        guidance_interval=guidance_interval
    )

    info_path = os.path.join(args.out_dir_path, "edit_info.txt")
    with open(info_path, "w", encoding="utf-8") as f:
        f.write(f"src_tags={chp['tags']}\n")
        f.write(f"tgt_tags={new_tags}\n")
        f.write(f"lyrics={chp['lyrics']}")
        
    tqdm.write(f"Saved to: {output_paths}")


if __name__ == "__main__":
    main()
