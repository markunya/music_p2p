from typing import Optional
from dataclasses import dataclass

@dataclass
class Prompt:
    lyrics: str
    tags: str

@dataclass
class P2PTaskParams:
    music_path: Optional[str]
    inverted_music_path: Optional[str]
    src: Prompt
    tgt: Prompt

def is_special_token(token: str):
    return token.startswith('<') and token.endswith('>')
