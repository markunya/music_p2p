import numpy as np
import torch
from typing import Union, Tuple, Optional, Dict

def is_special_token(token: str):
    return token.startswith('<') and token.endswith('>')
