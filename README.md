# 🎵 Music Editing with Prompt-to-Prompt

Этот проект посвящён редактированию музыки с использованием подхода **Prompt-to-Prompt (P2P)**, адаптированного под аудиодомен.
Метод позволяет изменять отдельные элементы музыкального произведения — жанр, инструменты или текст песни — без переобучения модели, сохраняя общую структуру и стиль трека.
Работа выполнена на основе модели **ACE-Step** и методов **Null-Text Inversion** для инверсии аудио в латентное пространство.

## ⚙️ Настройка окружения

```bash
git clone https://github.com/markunya/music_p2p.git
cd music_p2p
conda create -n music_p2p python=3.10.18 ffmpeg
conda activate music_p2p
pip install -r requirements.txt
```

## 🔧 Скрипты

* `invert_music.py` — инверсия аудио с помощью **Null Text Inversion**
* `replace_tags.py` — замена тегов (жанр, инструменты, стиль)
* `replace_lyrics.py` — замена текста песни (лирики)

## 🎧 Прослушивание примеров

Для прослушивания сэмплов и повторения экспериментов запустите ячейки в ноутбуке:
`examples.ipynb`
