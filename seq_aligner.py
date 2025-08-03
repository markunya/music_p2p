import torch
import numpy as np
from difflib import SequenceMatcher
import re
from acestep.models.lyrics_utils.lyric_tokenizer import VoiceBpeTokenizer

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
                lyric_token_idx = lyric_token_idx + token_idx + [2]
            except Exception as e:
                print("tokenize error", e, "for line", line, "major_language", lang)
        return lyric_token_idx

def get_replacement_mapper_(idx_src, idx_tgt):
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



def get_replacement_mapper(prompts, tokenizer):
    src_tokens = tokenizer.tokenizer.decode(tokenize_lyrics(prompts[0]), skip_special_tokens=False).split(' ')
    Ms = []

    for i in range(1, len(prompts)):
        tgt_tokens_i = tokenizer.tokenizer.decode(tokenize_lyrics(prompts[i]), skip_special_tokens=False).split(' ')
        M = get_replacement_mapper_(src_tokens, tgt_tokens_i)
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

    return torch.stack(padded, dim=0)
