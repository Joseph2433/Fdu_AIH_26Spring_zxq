"""Part 3 BERT-CRF 推断脚本：加载 .pt，对 test.txt 解码并生成预测文件。

输出：
  p3/part3_chn_test.txt
  p3/part3_eng_test.txt
"""

from __future__ import annotations

import os
import sys
import argparse
import time
from typing import List

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
PJ_ROOT = os.path.dirname(HERE)
if PJ_ROOT not in sys.path:
    sys.path.insert(0, PJ_ROOT)

from data_utils import (  # noqa: E402
    load_tokens_only,
    write_predictions,
    ner_path,
)
from part3_transformer_crf import (  # noqa: E402
    BertCRF,
    chunk_encode,
    ensure_bert_model,
    is_local_bert_ready,
)
from transformers import AutoTokenizer  # noqa: E402


def predict(language: str, src_name: str, device, batch_size: int = 32):
    short = "chn" if language == "Chinese" else "eng"
    ckpt = torch.load(os.path.join(HERE, f"part3_{short}.pt"),
                      map_location=device, weights_only=False)
    bert_path = ckpt.get("bert_path")
    if not bert_path or not is_local_bert_ready(bert_path):
        bert_path = ensure_bert_model(language)
    id2tag = ckpt["id2tag"]
    cfg = ckpt["config"]

    tokenizer = AutoTokenizer.from_pretrained(bert_path)
    model = BertCRF(
        bert_path,
        num_tags=len(id2tag),
        dropout=cfg["dropout"],
        bert_trainable_layers=int(cfg.get("bert_trainable_layers", 4)),
    ).to(device)
    missing, unexpected = model.load_state_dict(ckpt["state_dict"], strict=False)
    unexpected = [k for k in unexpected if not k.startswith("bert.")]
    if unexpected:
        raise RuntimeError(f"Unexpected checkpoint keys: {unexpected}")
    model.eval()

    src_path = ner_path(language, src_name)
    raw_tokens = load_tokens_only(src_path)

    # 对每个原句切 chunk 并保留映射，逐 chunk 推断后拼回原句长度
    sent_chunk_map: List[List[dict]] = []  # 每句对应一组 chunk
    flat_chunks = []                       # batch 用：所有 chunks 平摊
    flat_chunk_meta = []                   # (sent_idx, w_start, w_end)
    for s_idx, words in enumerate(raw_tokens):
        chunks = chunk_encode(words, tokenizer)
        sent_chunk_map.append(chunks)
        for ch in chunks:
            flat_chunks.append(ch)
            flat_chunk_meta.append((s_idx, ch["word_slice"][0], ch["word_slice"][1]))

    # 准备每个原句的预测槽
    preds_per_sent: List[List[str]] = [[None] * len(words) for words in raw_tokens]

    t0 = time.time()
    with torch.no_grad():
        for i in range(0, len(flat_chunks), batch_size):
            batch_chunks = flat_chunks[i:i + batch_size]
            sub_max = max(c["input_ids"].size(0) for c in batch_chunks)
            word_max = max(c["first_idx"].size(0) for c in batch_chunks)
            B = len(batch_chunks)
            input_ids = torch.zeros(B, sub_max, dtype=torch.long)
            attn = torch.zeros(B, sub_max, dtype=torch.long)
            first_idx = torch.zeros(B, word_max, dtype=torch.long)
            word_mask = torch.zeros(B, word_max, dtype=torch.bool)
            for j, c in enumerate(batch_chunks):
                s = c["input_ids"].size(0); w = c["first_idx"].size(0)
                input_ids[j, :s] = c["input_ids"]
                attn[j, :s] = c["attention_mask"]
                first_idx[j, :w] = c["first_idx"]
                word_mask[j, :w] = True
            input_ids = input_ids.to(device); attn = attn.to(device)
            first_idx = first_idx.to(device); word_mask = word_mask.to(device)
            decoded = model.decode(input_ids, attn, first_idx, word_mask)
            for j, seq in enumerate(decoded):
                s_idx, w_start, w_end = flat_chunk_meta[i + j]
                tags = [id2tag[t] for t in seq]
                # 防御性：长度对齐到 (w_end - w_start)
                if len(tags) != (w_end - w_start):
                    tags = tags[: (w_end - w_start)] + ["O"] * max(0, (w_end - w_start) - len(tags))
                for k, tg in enumerate(tags):
                    preds_per_sent[s_idx][w_start + k] = tg
    dt = time.time() - t0

    # 兜底：若仍有 None（理论不会），填 O
    for sent in preds_per_sent:
        for i, t in enumerate(sent):
            if t is None:
                sent[i] = "O"

    out_path = os.path.join(HERE, f"part3_{short}_{os.path.splitext(src_name)[0]}.txt")
    write_predictions(src_path, preds_per_sent, out_path)
    print(f"[{language}] decoded {len(raw_tokens)} sents in {dt:.1f}s -> {out_path}")
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
