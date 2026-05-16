"""Part 2 BiLSTM-CRF 推断脚本：加载 .pt，对 test.txt 解码并生成预测文件。

输出：
  p2/part2_crf_chn_test.txt
  p2/part2_crf_eng_test.txt
"""

from __future__ import annotations

import os
import sys
import argparse
import time
import torch
from torch.utils.data import DataLoader

HERE = os.path.dirname(os.path.abspath(__file__))
PJ_ROOT = os.path.dirname(HERE)
if PJ_ROOT not in sys.path:
    sys.path.insert(0, PJ_ROOT)

from data_utils import (  # noqa: E402
    load_tokens_only,
    write_predictions,
    ner_path,
)
from part2_crf import BiLSTM_CRF  # noqa: E402


def predict(language: str, src_name: str, device, batch_size: int = 64):
    short = "chn" if language == "Chinese" else "eng"
    ckpt_path = os.path.join(HERE, f"part2_crf_{short}.pt")
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    word2id = ckpt["word2id"]
    char2id = ckpt.get("char2id", None)
    id2tag = ckpt["id2tag"]
    cfg = ckpt["config"]
    lower = ckpt["lower"]
    use_char = bool(cfg.get("use_char_cnn", False)) and char2id is not None
    max_word_len = int(cfg.get("max_word_len", 20))

    model = BiLSTM_CRF(vocab_size=len(word2id), num_tags=len(id2tag),
                       emb_dim=cfg["emb_dim"], hid_dim=cfg["hid_dim"],
                       dropout=cfg["dropout"], pad_idx=0,
                       n_chars=len(char2id) if use_char else 0,
                       char_emb_dim=cfg.get("char_emb_dim", 30),
                       char_out_dim=cfg.get("char_out_dim", 30)).to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    src_path = ner_path(language, src_name)
    raw_tokens = load_tokens_only(src_path)

    # 构造 word ids 与（可选）char ids
    input_ids = []
    char_ids_per_sent = []
    for sent in raw_tokens:
        toks = [t.lower() if lower else t for t in sent]
        input_ids.append([word2id.get(t, 1) for t in toks])
        if use_char:
            cs = []
            for w in sent:
                w_trunc = w[:max_word_len]
                cs.append([char2id.get(c, 1) for c in w_trunc])
            char_ids_per_sent.append(cs)

    preds_per_sent: list = [None] * len(input_ids)
    t0 = time.time()
    with torch.no_grad():
        for i in range(0, len(input_ids), batch_size):
            chunk_ids = input_ids[i:i + batch_size]
            L = max(len(x) for x in chunk_ids)
            B = len(chunk_ids)
            x_t = torch.zeros(B, L, dtype=torch.long)
            mask_t = torch.zeros(B, L, dtype=torch.bool)
            for j, x in enumerate(chunk_ids):
                x_t[j, :len(x)] = torch.tensor(x, dtype=torch.long)
                mask_t[j, :len(x)] = True
            x_t = x_t.to(device); mask_t = mask_t.to(device)

            if use_char:
                chunk_chars = char_ids_per_sent[i:i + batch_size]
                W = max(max((len(c) for c in cs), default=1) for cs in chunk_chars)
                W = max(W, 1)
                ch_t = torch.zeros(B, L, W, dtype=torch.long)
                for j, cs in enumerate(chunk_chars):
                    for k, c in enumerate(cs):
                        if c:
                            ch_t[j, k, :len(c)] = torch.tensor(c, dtype=torch.long)
                ch_t = ch_t.to(device)
                decoded = model.decode(x_t, mask_t, char_ids=ch_t)
            else:
                decoded = model.decode(x_t, mask_t)
            for j, seq in enumerate(decoded):
                preds_per_sent[i + j] = [id2tag[t] for t in seq]
    dt = time.time() - t0

    out_path = os.path.join(HERE, f"part2_crf_{short}_{os.path.splitext(src_name)[0]}.txt")
    write_predictions(src_path, preds_per_sent, out_path)
    print(f"[{language}] decoded {len(input_ids)} sents in {dt:.1f}s -> {out_path} (char_cnn={use_char})")
    return out_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--src", default="test.txt")
    args = parser.parse_args()
    device = torch.device(args.device)
    for lang in ("Chinese", "English"):
        print("=" * 30, lang, "=" * 30)
        predict(lang, args.src, device)


if __name__ == "__main__":
    main()
