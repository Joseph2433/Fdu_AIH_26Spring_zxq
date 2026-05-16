"""Part 3: BERT + 手写 CRF 命名实体识别。

- 中文: bert-base-chinese (字符级)
- 英文: bert-base-cased  (subword，按 word_ids 取每词首子词)
- CRF 层: 共享 crf_layer.LinearChainCRF
- 长句处理: 按子词数 chunk，每 chunk ≤ 510，CRF 在 word-level emission 上整句解码

- 仅解冻 BERT 最后 4 层，训练 CRF + classifier head
- AdamW + linear warmup (10% steps)
- epochs:  Chinese 15, English 20
- 模型保存到 part3_<lang>.pt（仅保存 head + crf 权重 + 配置；BERT 路径单独记录）

运行：
    cd p3
    python part3_transformer_crf.py
"""

from __future__ import annotations

import os
import sys
import time
import math
import argparse
from typing import List, Tuple, Dict

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

HERE = os.path.dirname(os.path.abspath(__file__))
PJ_ROOT = os.path.dirname(HERE)
if PJ_ROOT not in sys.path:
    sys.path.insert(0, PJ_ROOT)

from data_utils import (  # noqa: E402
    LANG2LABELS,
    LANG2SCHEME,
    build_legal_masks,
    load_corpus,
    ner_path,
)
from crf_layer import LinearChainCRF  # noqa: E402

from transformers import AutoTokenizer, AutoModel, get_linear_schedule_with_warmup

LANG2HF_REPO = {
    "Chinese": "google-bert/bert-base-chinese",
    "English": "google-bert/bert-base-cased",
}

LANG2BERT = {
    "Chinese": os.path.join(PJ_ROOT, "pretrained", "bert-base-chinese"),
    "English": os.path.join(PJ_ROOT, "pretrained", "bert-base-cased"),
}

MAX_SUBWORD = 510  # 留 [CLS] [SEP] 各 1


def is_local_bert_ready(model_dir: str) -> bool:
    """检查本地 BERT 目录是否具备 tokenizer/config/权重核心文件。"""
    if not os.path.isdir(model_dir):
        return False
    required = ["config.json", "vocab.txt"]
    has_required = all(os.path.isfile(os.path.join(model_dir, name)) for name in required)
    has_weight = any(
        os.path.isfile(os.path.join(model_dir, name))
        for name in ("model.safetensors", "pytorch_model.bin")
    )
    return has_required and has_weight


def ensure_bert_model(language: str) -> str:
    """优先使用 pj2/pretrained 下的 BERT；缺失时从 Hugging Face 下载。"""
    model_dir = LANG2BERT[language]
    if is_local_bert_ready(model_dir):
        return model_dir

    repo_id = LANG2HF_REPO[language]
    os.makedirs(model_dir, exist_ok=True)
    print(f"[{language}] local BERT not found or incomplete: {model_dir}")
    print(f"[{language}] downloading {repo_id} from Hugging Face -> {model_dir}")
    try:
        os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
        from huggingface_hub import snapshot_download
        snapshot_download(
            repo_id=repo_id,
            local_dir=model_dir,
            allow_patterns=[
                "config.json",
                "vocab.txt",
                "tokenizer.json",
                "tokenizer_config.json",
                "special_tokens_map.json",
                "model.safetensors",
                "pytorch_model.bin",
            ],
        )
    except Exception as exc:
        raise RuntimeError(
            f"Cannot prepare pretrained model {repo_id}. "
            f"Please check network access or manually place it under {model_dir}."
        ) from exc

    if not is_local_bert_ready(model_dir):
        raise RuntimeError(f"Downloaded BERT files are incomplete under {model_dir}")
    return model_dir

# ------------------ 子词对齐：返回每个 chunk 的 input_ids / attn / first_subword_indices ------------------
def chunk_encode(words: List[str], tokenizer) -> List[Dict]:
    """把 word 列表切成多个 chunk，每个 chunk 的子词数 ≤ MAX_SUBWORD（不含 CLS/SEP）。

    返回 list[dict]，每 dict 包含：
        input_ids   : LongTensor (S+2,)
        attention   : LongTensor (S+2,)
        first_idx   : LongTensor (Wc,)，指向 input_ids 中每个 word 的"首子词"位置（已加上 CLS 偏移）
        word_slice  : (start_word, end_word) 该 chunk 对应原 word 列表的切片，左闭右开
    """
    # 先一次性 tokenize 所有 word，每个 word 对应若干 subword
    sub_lists: List[List[int]] = []
    for w in words:
        ids = tokenizer.encode(w, add_special_tokens=False)
        if len(ids) == 0:
            # 空 token（如纯空格），用 [UNK] 兜底
            ids = [tokenizer.unk_token_id]
        sub_lists.append(ids)

    chunks = []
    cur_ids: List[int] = []
    cur_first_idx: List[int] = []  # word 首子词在 cur_ids 中的位置（不含 CLS 偏移）
    cur_w_start = 0
    w_idx = 0
    while w_idx < len(words):
        sub = sub_lists[w_idx]
        # 若一个 word 的子词就超长（极端），强制截断到 MAX_SUBWORD
        if len(sub) > MAX_SUBWORD:
            sub = sub[:MAX_SUBWORD]
        if len(cur_ids) + len(sub) > MAX_SUBWORD:
            # 关闭当前 chunk
            chunks.append(_pack_chunk(cur_ids, cur_first_idx, cur_w_start, w_idx, tokenizer))
            cur_ids, cur_first_idx = [], []
            cur_w_start = w_idx
        cur_first_idx.append(len(cur_ids))
        cur_ids.extend(sub)
        w_idx += 1
    if cur_ids:
        chunks.append(_pack_chunk(cur_ids, cur_first_idx, cur_w_start, len(words), tokenizer))
    return chunks


def _pack_chunk(ids: List[int], first_idx: List[int],
                w_start: int, w_end: int, tokenizer) -> Dict:
    cls = tokenizer.cls_token_id
    sep = tokenizer.sep_token_id
    full = [cls] + ids + [sep]
    first = [i + 1 for i in first_idx]   # +1 是因为前面加了 [CLS]
    return {
        "input_ids": torch.tensor(full, dtype=torch.long),
        "attention_mask": torch.ones(len(full), dtype=torch.long),
        "first_idx": torch.tensor(first, dtype=torch.long),
        "word_slice": (w_start, w_end),
    }


# ------------------ Dataset ------------------
class BertNERDataset(Dataset):
    def __init__(self, sentences, tokenizer, tag2id: Dict[str, int]):
        self.items = []  # list of (chunks_list, all_tag_ids)
        for sent in sentences:
            words = [w for w, _ in sent]
            tags = [t for _, t in sent]
            tag_ids = [tag2id[t] for t in tags]
            chunks = chunk_encode(words, tokenizer)
            # 把每个 chunk 切出对应的 tag 子段
            for ch in chunks:
                s, e = ch["word_slice"]
                ch["tag_ids"] = torch.tensor(tag_ids[s:e], dtype=torch.long)
            # 训练样本就是「单 chunk」，每个 chunk 独立做 CRF（多 chunk 会被分别训练，这是常见做法）
            self.items.extend(chunks)

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        return self.items[i]


def bert_collate(batch):
    """把 chunks 拼 batch：以子词数对齐 input_ids/mask/first_idx，以 word 数对齐 tags/word_mask。"""
    B = len(batch)
    sub_max = max(item["input_ids"].size(0) for item in batch)
    word_max = max(item["first_idx"].size(0) for item in batch)
    input_ids = torch.zeros(B, sub_max, dtype=torch.long)
    attn = torch.zeros(B, sub_max, dtype=torch.long)
    first_idx = torch.zeros(B, word_max, dtype=torch.long)
    tag_ids = torch.zeros(B, word_max, dtype=torch.long)
    word_mask = torch.zeros(B, word_max, dtype=torch.bool)
    for i, item in enumerate(batch):
        s = item["input_ids"].size(0)
        w = item["first_idx"].size(0)
        input_ids[i, :s] = item["input_ids"]
        attn[i, :s] = item["attention_mask"]
        first_idx[i, :w] = item["first_idx"]
        tag_ids[i, :w] = item["tag_ids"]
        word_mask[i, :w] = True
    return input_ids, attn, first_idx, tag_ids, word_mask


# ------------------ Model ------------------
class BertCRF(nn.Module):
    def __init__(self, bert_path: str, num_tags: int, dropout: float = 0.1,
                 illegal_trans_mask=None, illegal_start_mask=None, illegal_end_mask=None,
                 bert_trainable_layers: int = 4):
        super().__init__()
        self.bert = AutoModel.from_pretrained(bert_path)
        self.bert_trainable_layers = int(bert_trainable_layers)
        for p in self.bert.parameters():
            p.requires_grad = False
        if self.bert_trainable_layers > 0:
            encoder = getattr(self.bert, "encoder", None)
            layers = getattr(encoder, "layer", None) if encoder is not None else None
            if layers is None:
                raise RuntimeError("BERT encoder layers not found; cannot unfreeze last layers")
            for layer in layers[-self.bert_trainable_layers:]:
                for p in layer.parameters():
                    p.requires_grad = True
        hidden = self.bert.config.hidden_size
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden, num_tags)
        self.crf = LinearChainCRF(
            num_tags,
            illegal_trans_mask=illegal_trans_mask,
            illegal_start_mask=illegal_start_mask,
            illegal_end_mask=illegal_end_mask,
        )

    def emissions(self, input_ids, attn, first_idx, word_mask):
        if any(p.requires_grad for p in self.bert.parameters()):
            out = self.bert(input_ids=input_ids, attention_mask=attn).last_hidden_state
        else:
            with torch.no_grad():
                out = self.bert(input_ids=input_ids, attention_mask=attn).last_hidden_state
        # 取每词首子词
        B, S, H = out.shape
        W = first_idx.size(1)
        # gather
        idx = first_idx.unsqueeze(-1).expand(-1, -1, H)        # (B, W, H)
        word_h = torch.gather(out, dim=1, index=idx)           # (B, W, H)
        word_h = self.dropout(word_h)
        em = self.fc(word_h)                                   # (B, W, T)
        return em

    def loss(self, input_ids, attn, first_idx, tag_ids, word_mask):
        em = self.emissions(input_ids, attn, first_idx, word_mask)
        return self.crf.nll_loss(em, tag_ids, word_mask, reduction="mean")

    @torch.no_grad()
    def decode(self, input_ids, attn, first_idx, word_mask):
        em = self.emissions(input_ids, attn, first_idx, word_mask)
        return self.crf.decode(em, word_mask)


# ------------------ Eval helper ------------------
def evaluate_loader(model: BertCRF, loader: DataLoader, id2tag: List[str], device) -> float:
    from sklearn.metrics import f1_score
    model.eval()
    ys, ps = [], []
    for input_ids, attn, first_idx, tag_ids, word_mask in loader:
        input_ids = input_ids.to(device); attn = attn.to(device)
        first_idx = first_idx.to(device); word_mask = word_mask.to(device)
        decoded = model.decode(input_ids, attn, first_idx, word_mask)
        for i, seq in enumerate(decoded):
            l = len(seq)
            ps.extend([id2tag[t] for t in seq])
            ys.extend([id2tag[t.item()] for t in tag_ids[i, :l]])
    labels = id2tag[1:]
    return float(f1_score(ys, ps, labels=labels, average="micro", zero_division=0))


# ------------------ Train ------------------
def train_one_lang(language: str, epochs: int, batch_size: int,
                   lr_head: float, weight_decay: float,
                   warmup_ratio: float, dropout: float,
                   device: torch.device, seed: int = 42) -> dict:
    torch.manual_seed(seed)
    np.random.seed(seed)

    bert_path = ensure_bert_model(language)
    labels = LANG2LABELS[language]
    tag2id = {t: i for i, t in enumerate(labels)}

    tokenizer = AutoTokenizer.from_pretrained(bert_path)

    train_sents = load_corpus(ner_path(language, "train.txt"))
    val_sents = load_corpus(ner_path(language, "validation.txt"))
    print(f"[{language}] bert={bert_path}  train={len(train_sents)}  val={len(val_sents)}")

    train_ds = BertNERDataset(train_sents, tokenizer, tag2id)
    val_ds = BertNERDataset(val_sents, tokenizer, tag2id)
    print(f"[{language}] train chunks={len(train_ds)}  val chunks={len(val_ds)}")

    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                          collate_fn=bert_collate, num_workers=0)
    val_dl = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                        collate_fn=bert_collate, num_workers=0)

    # Task 2B: 注入 BMES/BIO 合法约束
    legal_init, legal_trans, legal_end = build_legal_masks(labels, LANG2SCHEME[language])
    illegal_init = (~legal_init).astype("float32")
    illegal_trans = (~legal_trans).astype("float32")
    illegal_end = (~legal_end).astype("float32")
    print(f"[{language}] CRF constraint: blocked init={int(illegal_init.sum())}/{len(labels)} "
          f"trans={int(illegal_trans.sum())}/{len(labels)**2} end={int(illegal_end.sum())}/{len(labels)}")

    model = BertCRF(bert_path, num_tags=len(labels), dropout=dropout,
                    illegal_trans_mask=illegal_trans,
                    illegal_start_mask=illegal_init,
                    illegal_end_mask=illegal_end,
                    bert_trainable_layers=4).to(device)

    # 冻结 BERT 其余层，仅优化最后 4 层 + 分类头与 CRF。
    bert_params = list(model.bert.named_parameters())
    trainable_bert_params = [(n, p) for n, p in bert_params if p.requires_grad]
    head_params = (
        list(model.fc.named_parameters())
        + [("crf." + n, p) for n, p in model.crf.named_parameters()]
    )
    no_decay = ("bias", "LayerNorm.weight")
    optim_groups = [
        {"params": [p for n, p in trainable_bert_params if not any(nd in n for nd in no_decay)],
         "lr": 1e-5, "weight_decay": weight_decay},
        {"params": [p for n, p in trainable_bert_params if any(nd in n for nd in no_decay)],
         "lr": 1e-5, "weight_decay": 0.0},
        {"params": [p for n, p in head_params if not any(nd in n for nd in no_decay)],
         "lr": lr_head, "weight_decay": weight_decay},
        {"params": [p for n, p in head_params if any(nd in n for nd in no_decay)],
         "lr": lr_head, "weight_decay": 0.0},
    ]
    opt = torch.optim.AdamW(optim_groups)
    total_steps = epochs * max(len(train_dl), 1)
    sched = get_linear_schedule_with_warmup(opt, int(total_steps * warmup_ratio), total_steps)

    best_f1, best_state = -1.0, None
    t0 = time.time()
    for ep in range(1, epochs + 1):
        model.train()
        total = 0.0
        n = 0
        for batch in train_dl:
            input_ids, attn, first_idx, tag_ids, word_mask = [b.to(device) for b in batch]
            loss = model.loss(input_ids, attn, first_idx, tag_ids, word_mask)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            total += loss.item()
            n += 1
        f1 = evaluate_loader(model, val_dl, labels, device)
        avg_loss = total / max(n, 1)
        elapsed = time.time() - t0
        print(f"  epoch {ep:02d} | loss {avg_loss:.4f} | val micro-F1 {f1:.4f} | "
              f"elapsed {elapsed:.0f}s")
        if f1 > best_f1:
            best_f1 = f1
            best_state = {
                k: v.detach().cpu().clone()
                for k, v in model.state_dict().items()
                if k.startswith("bert.encoder.layer.") or not k.startswith("bert.")
            }

    print(f"[{language}] best val micro-F1 = {best_f1:.4f}")
    return {
        "language": language,
        "tag2id": tag2id,
        "id2tag": labels,
        "bert_path": bert_path,
        "config": {"dropout": dropout, "bert_trainable_layers": 4},
        "state_dict": best_state,
        "best_f1": best_f1,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--lang", default="both", choices=["Chinese", "English", "both"])
    args = parser.parse_args()
    device = torch.device(args.device)
    print(f"device={device}")

    plan = {
        "Chinese": dict(epochs=15, batch_size=16, lr_head=1e-3,
                        weight_decay=0.01, warmup_ratio=0.1, dropout=0.1),
        "English": dict(epochs=20, batch_size=16, lr_head=1e-3,
                        weight_decay=0.01, warmup_ratio=0.1, dropout=0.1),
    }
    langs = ["Chinese", "English"] if args.lang == "both" else [args.lang]
    for lang in langs:
        print("=" * 30, lang, "=" * 30)
        ckpt = train_one_lang(lang, device=device, **plan[lang])
        short = "chn" if lang == "Chinese" else "eng"
        out = os.path.join(HERE, f"part3_{short}.pt")
        torch.save(ckpt, out)
        print(f"saved -> {out}")


if __name__ == "__main__":
    main()
