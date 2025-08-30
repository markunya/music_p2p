import torch
from tqdm import tqdm
import numpy as np
from difflib import SequenceMatcher
import re
from acestep.models.lyrics_utils.lyric_tokenizer import VoiceBpeTokenizer
from typing import List
from p2p_utils import is_special_token

structure_pattern = re.compile(r"\[.*?\]")
lyric_tokenizer = VoiceBpeTokenizer()

def tokenize_lyrics(lyrics, debug=False):
        lines = lyrics.split("\n")
        lyric_token_idx = [261]
        for line in lines:
            line = line.strip()
            if not line:
                lyric_token_idx += [2]
                continue

            lang = "en"

            try:
                if structure_pattern.match(line):
                    token_idx = lyric_tokenizer.encode(line, "en")
                else:
                    token_idx = lyric_tokenizer.encode(line, lang)
                if debug:
                    toks = lyric_tokenizer.batch_decode(
                        [[tok_id] for tok_id in token_idx]
                    )
                    tqdm.write(str(toks))
                lyric_token_idx = lyric_token_idx + token_idx + [2]
            except Exception as e:
                print("tokenize error", e, "for line", line, "major_language", lang)
        return lyric_token_idx

def tokenize_tags(tags, tokenizer):
    input_ids = tokenizer(
            tags,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=256,
    )['input_ids'].tolist()
    return [[tokenizer.decode(idx) for idx in input_id] for input_id in input_ids]

def get_lyrics_replacement_mapper_(idx_src, idx_tgt):
    sm = SequenceMatcher(None, idx_src, idx_tgt)
    m, n = len(idx_src), len(idx_tgt)
    M = np.zeros((m, n), dtype=float)

    for opcode, i1, i2, j1, j2 in sm.get_opcodes():
        if opcode == 'equal':
            for k in range(i2 - i1):
                M[i1 + k, j1 + k] = 1.0
        else:
            src_inds = range(i1, i2)
            tgt_inds = range(j1, j2)
            if not src_inds or not tgt_inds:
                continue
            weight = 1.0 / len(tgt_inds)
            for i in src_inds:
                for j in tgt_inds:
                    M[i, j] = weight

    row_sums = M.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1.0
    M = M / row_sums

    return torch.from_numpy(M)

def get_lyrics_replacement_mapper(prompts):
    src_tokens = lyric_tokenizer.tokenizer.decode(tokenize_lyrics(prompts[0]), skip_special_tokens=False).split(' ')
    Ms = []

    for i in range(1, len(prompts)):
        tgt_tokens_i = lyric_tokenizer.tokenizer.decode(tokenize_lyrics(prompts[i]), skip_special_tokens=False).split(' ')
        M = get_lyrics_replacement_mapper_(src_tokens, tgt_tokens_i)
        Ms.append(M)

    max_len = max(max(M.shape[1] for M in Ms), Ms[0].shape[0])

    padded = []
    for M in Ms:
        m, n = M.shape
        M_pad = torch.zeros((max_len, max_len), dtype=M.dtype, device=M.device)
        M_pad[:m, :n] = M
        for i in range(max_len):
            if M_pad[i].sum() == 0:
                M_pad[i, i] = 1.0
        padded.append(M_pad)

    edit_mask = torch.ones(len(padded), max_len) # not supported yet
    return torch.stack(padded, dim=0), edit_mask

def get_tags_replacement_mapper_(src_tokens: List[str], tgt_tokens: List[str]):
    M = torch.zeros(len(src_tokens), len(tgt_tokens))
    src_tag_toks = []
    tgt_tag_toks = []
    edit_mask = torch.zeros(len(tgt_tokens))

    src_l = 0   
    tgt_l = 0
    while True:
        while src_l < len(src_tokens) \
                and not is_special_token(src_tokens[src_l]) \
                and src_tokens[src_l] != ',':
            src_tag_toks.append(src_tokens[src_l])
            src_l += 1

        while tgt_l < len(tgt_tokens) \
                and not is_special_token(tgt_tokens[tgt_l]) \
                and tgt_tokens[tgt_l] != ',':
            tgt_tag_toks.append(tgt_tokens[tgt_l])
            tgt_l += 1

        if src_tag_toks != tgt_tag_toks:
            M[src_l-len(src_tag_toks):src_l,tgt_l-len(tgt_tag_toks):tgt_l] = 1 / len(tgt_tag_toks)
            edit_mask[tgt_l-len(tgt_tag_toks):tgt_l] = 1
        else:
            k = len(src_tag_toks)
            M[src_l-len(src_tag_toks):src_l,tgt_l-len(tgt_tag_toks):tgt_l] = torch.eye(k, k)

        src_tag_toks.clear()
        tgt_tag_toks.clear()

        if src_l < len(src_tokens) and src_tokens[src_l] == ',':
            M[src_l, tgt_l] = 1

        src_l += 1
        tgt_l += 1

        if src_l >= len(src_tokens) or is_special_token(src_tokens[src_l]):
            break

    return M, edit_mask

def get_tags_replacement_mapper(prompts: List[str], tokenizer):
    src_comma_count = prompts[0].count(',')
    for i in range(1, len(prompts)):
        tgt_i_comma_count = prompts[i].count(',')
        assert src_comma_count == tgt_i_comma_count, f"Src tags amount must be equal to tgt {i} tags amount"

    tokens = tokenize_tags(prompts, tokenizer)
    Ms = []
    edit_masks = []
    for i in range(1, len(tokens)):
        M, edit_mask = get_tags_replacement_mapper_(tokens[0], tokens[i])
        Ms.append(M)
        edit_masks.append(edit_mask)
        
    return torch.stack(Ms, dim=0), torch.stack(edit_masks, dim=0)
