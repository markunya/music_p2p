import json
from tqdm import tqdm
from typing import Any
from src.utils.structures import to_dict

def info(msg: str):
    tqdm.write(f"[INFO] {msg}")

def debug(msg: str):
    tqdm.write(f"[DEBUG] {msg}")

def warning(msg: str):
    tqdm.write(f"[WARNING] {msg}")

def error(msg: str):
    tqdm.write(f"[ERROR] {msg}")

def log_structure(obj: Any):
    tqdm.write(json.dumps(to_dict(obj), indent=2, ensure_ascii=False))
