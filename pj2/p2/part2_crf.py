"""Part 2: BiLSTM-CRF 命名实体识别（PyTorch + 手写 CRF 层）。

- Encoder: Embedding + 1 层 BiLSTM + Linear 投影到 tag 数 (emission)
- CRF: 共享自实现的 crf_layer.LinearChainCRF
- 训练:
    Adam, lr=1e-3
    动态 padding，按句长排序近似 bucket
    epochs:  Chinese 30, English 15  (英文数据规模更大)
- 模型保存到 part2_crf_<lang>.pt（state_dict + 配置 + vocab + tag2id）

运行:
    cd p2
    python part2_crf.py
"""

from __future__ import annotations

import os
import sys
import time
import math
import argparse
import zipfile
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
    build_vocab,
    ner_path,
)
from crf_layer import LinearChainCRF  # noqa: E402


def ensure_glove_file(glove_path: str) -> str:
    """优先使用本地 GloVe；若缺失则从 Hugging Face 下载并解压 100d 文件。"""
    if os.path.isfile(glove_path):
        return glove_path

    glove_dir = os.path.dirname(glove_path)
    os.makedirs(glove_dir, exist_ok=True)
    zip_name = "glove.6B.zip"
    zip_path = os.path.join(glove_dir, zip_name)
    print(f"[GloVe] local file not found: {glove_path}")

    if not os.path.isfile(zip_path):
        print(f"[GloVe] downloading stanfordnlp/glove/{zip_name} from Hugging Face -> {zip_path}")
        try:
            os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
            from huggingface_hub import hf_hub_download
            downloaded = hf_hub_download(
                repo_id="stanfordnlp/glove",
                filename=zip_name,
                repo_type="model",
                local_dir=glove_dir,
            )
            zip_path = downloaded
        except Exception as exc:
            raise RuntimeError(
                f"Cannot prepare GloVe file. Please check network access or manually place "
                f"glove.6B.100d.txt under {glove_dir}."
            ) from exc

    print(f"[GloVe] extracting glove.6B.100d.txt from {zip_path}")
    try:
        with zipfile.ZipFile(zip_path) as zf:
            zf.extract("glove.6B.100d.txt", glove_dir)
    except Exception as exc:
        raise RuntimeError(f"Failed to extract glove.6B.100d.txt from {zip_path}") from exc

    if not os.path.isfile(glove_path):
        raise RuntimeError(f"GloVe extraction finished but file is still missing: {glove_path}")
    return glove_path

# ------------------ Dataset ------------------
class NERDataset(Dataset):
    def __init__(self, sentences, word2id: Dict[str, int], tag2id: Dict[str, int],
                 lower: bool = False, char2id: Dict[str, int] = None,
                 max_word_len: int = 20):
        """char2id 非 None 时，每条样本同时返回 char_ids；None 表示不启用 char-CNN。"""
        self.use_char = char2id is not None
        self.char2id = char2id
        self.max_word_len = max_word_len
        self.data = []
        for sent in sentences:
            raw_toks = [w for w, _ in sent]                        # 用原始大小写做 char
            toks = [w.lower() if lower else w for w in raw_toks]
            tags = [t for _, t in sent]
            x = [word2id.get(t, 1) for t in toks]                  # 1 = <UNK>
            y = [tag2id[t] for t in tags]
            if self.use_char:
                # 截断到 max_word_len，给每个词构造 char id 列表（不做填充，留 collate 里做）
                cs = []
                for w in raw_toks:
                    w_trunc = w[:max_word_len]
                    cs.append([char2id.get(c, 1) for c in w_trunc])  # 1 = <UNK_CHAR>
                self.data.append((x, y, cs))
            else:
                self.data.append((x, y))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        return self.data[i]


def collate(batch):
    """无 char-CNN 路径。"""
    lens = [len(x) for x, _ in batch]
    L = max(lens)
    B = len(batch)
    xs = torch.zeros(B, L, dtype=torch.long)
    ys = torch.zeros(B, L, dtype=torch.long)
    mask = torch.zeros(B, L, dtype=torch.bool)
    for i, (x, y) in enumerate(batch):
        l = len(x)
        xs[i, :l] = torch.tensor(x, dtype=torch.long)
        ys[i, :l] = torch.tensor(y, dtype=torch.long)
        mask[i, :l] = True
    return xs, ys, mask


def collate_char(batch):
    """带 char-CNN 路径。pad 词维度 + 字符维度。"""
    lens = [len(x) for x, _, _ in batch]
    L = max(lens)
    B = len(batch)
    W = max(max((len(c) for c in cs), default=1) for _, _, cs in batch)
    W = max(W, 1)
    xs = torch.zeros(B, L, dtype=torch.long)
    ys = torch.zeros(B, L, dtype=torch.long)
    chars = torch.zeros(B, L, W, dtype=torch.long)   # 0 = <PAD_CHAR>
    mask = torch.zeros(B, L, dtype=torch.bool)
    for i, (x, y, cs) in enumerate(batch):
        l = len(x)
        xs[i, :l] = torch.tensor(x, dtype=torch.long)
        ys[i, :l] = torch.tensor(y, dtype=torch.long)
        mask[i, :l] = True
        for j, c in enumerate(cs):
            if c:
                chars[i, j, :len(c)] = torch.tensor(c, dtype=torch.long)
    return xs, ys, chars, mask


# ------------------ Char-CNN ------------------
class CharCNN(nn.Module):
    """Ma & Hovy 2016: char embedding → 1D Conv (kernel=3) → ReLU → max-pool over chars。"""

    def __init__(self, n_chars: int, char_emb_dim: int = 30, out_dim: int = 30,
                 kernel_size: int = 3, dropout: float = 0.5, padding_idx: int = 0):
        super().__init__()
        self.embed = nn.Embedding(n_chars, char_emb_dim, padding_idx=padding_idx)
        self.conv = nn.Conv1d(char_emb_dim, out_dim, kernel_size=kernel_size,
                              padding=kernel_size // 2)
        self.dropout = nn.Dropout(dropout)
        self.out_dim = out_dim

    def forward(self, char_ids: torch.Tensor) -> torch.Tensor:
        # char_ids: (B, L, W)
        B, L, W = char_ids.shape
        emb = self.embed(char_ids)                   # (B, L, W, C)
        emb = self.dropout(emb)
        emb = emb.view(B * L, W, -1).transpose(1, 2)  # (B·L, C, W)
        conv = torch.relu(self.conv(emb))             # (B·L, out, W)
        pooled, _ = conv.max(dim=2)                   # (B·L, out)
        return pooled.view(B, L, self.out_dim)        # (B, L, out)


# ------------------ Task 1A: GloVe 加载 ------------------
def load_glove_embeddings(glove_path: str, word2id: Dict[str, int],
                           emb_dim: int = 100, lower: bool = True,
                           verbose: bool = True) -> Tuple[np.ndarray, int]:
    """读取 GloVe txt，按 word2id 顺序返回 (V, emb_dim) 矩阵 + 命中数。

    - 每行格式：word v1 v2 ... v100
    - 词不在 GloVe 中：保持 nn.Embedding 的随机初值（这里返回 NaN，由调用方填）
    - <PAD> id=0 → 全 0；<UNK> id=1 → GloVe 中所有未命中词的均值（更稳的 OOV 表示）
    """
    V = len(word2id)
    embedding_matrix = np.full((V, emb_dim), np.nan, dtype=np.float32)
    embedding_matrix[0] = 0.0   # <PAD>
    n_hit = 0
    glove_vec_sum = np.zeros(emb_dim, dtype=np.float64)
    glove_vec_count = 0
    with open(glove_path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.rstrip().split(" ")
            if len(parts) != emb_dim + 1:
                continue
            word = parts[0]
            vec = np.asarray(parts[1:], dtype=np.float32)
            glove_vec_sum += vec
            glove_vec_count += 1
            key = word.lower() if lower else word
            wid = word2id.get(key, -1)
            if wid >= 0 and np.isnan(embedding_matrix[wid, 0]):
                embedding_matrix[wid] = vec
                n_hit += 1
    # <UNK> 用所有 GloVe 向量均值兜底
    if glove_vec_count > 0:
        unk_vec = (glove_vec_sum / glove_vec_count).astype(np.float32)
        if "<UNK>" in word2id:
            embedding_matrix[word2id["<UNK>"]] = unk_vec
    if verbose:
        print(f"[GloVe] {glove_path} | hit {n_hit}/{V} ({100*n_hit/V:.1f}%) | "
              f"<UNK>=mean of {glove_vec_count} vectors")
    return embedding_matrix, n_hit


# ------------------ Model ------------------
class BiLSTM_CRF(nn.Module):
    def __init__(self, vocab_size: int, num_tags: int, emb_dim: int = 100,
                 hid_dim: int = 128, dropout: float = 0.5, pad_idx: int = 0,
                 illegal_trans_mask=None, illegal_start_mask=None, illegal_end_mask=None,
                 n_chars: int = 0, char_emb_dim: int = 30, char_out_dim: int = 30,
                 pretrained_emb: np.ndarray = None, freeze_emb: bool = False):
        """n_chars > 0 时启用 Char-CNN，feature 维度 = emb_dim + char_out_dim。
        pretrained_emb (V, emb_dim) 非 None 时用其覆盖随机初始化（NaN 位置保持随机）。
        """
        super().__init__()
        self.use_char = n_chars > 0
        self.embed = nn.Embedding(vocab_size, emb_dim, padding_idx=pad_idx)
        if pretrained_emb is not None:
            assert pretrained_emb.shape == (vocab_size, emb_dim), pretrained_emb.shape
            with torch.no_grad():
                rand_w = self.embed.weight.detach().clone()
                pre = torch.from_numpy(pretrained_emb)
                # NaN 位置 (GloVe 未命中) 保留随机权重，命中位置覆盖
                hit_mask = ~torch.isnan(pre).any(dim=-1, keepdim=True)
                pre_filled = torch.where(hit_mask, pre, rand_w)
                self.embed.weight.copy_(pre_filled)
            if freeze_emb:
                self.embed.weight.requires_grad = False
        if self.use_char:
            self.char_cnn = CharCNN(n_chars, char_emb_dim=char_emb_dim,
                                    out_dim=char_out_dim, dropout=dropout)
            lstm_in = emb_dim + char_out_dim
        else:
            self.char_cnn = None
            lstm_in = emb_dim
        self.lstm = nn.LSTM(lstm_in, hid_dim, num_layers=1,
                            bidirectional=True, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hid_dim * 2, num_tags)
        self.crf = LinearChainCRF(
            num_tags,
            illegal_trans_mask=illegal_trans_mask,
            illegal_start_mask=illegal_start_mask,
            illegal_end_mask=illegal_end_mask,
        )

    def emissions(self, x, mask, char_ids=None):
        emb = self.embed(x)                           # (B, L, E)
        if self.use_char:
            assert char_ids is not None
            char_feat = self.char_cnn(char_ids)       # (B, L, char_out)
            emb = torch.cat([emb, char_feat], dim=-1)
        # 用 pack 让 LSTM 忽略 padding（提速 + 让 padding 不污染 hidden）
        lens = mask.long().sum(1).cpu()
        packed = nn.utils.rnn.pack_padded_sequence(emb, lens, batch_first=True, enforce_sorted=False)
        out, _ = self.lstm(packed)
        out, _ = nn.utils.rnn.pad_packed_sequence(out, batch_first=True, total_length=x.size(1))
        out = self.dropout(out)
        return self.fc(out)

    def loss(self, x, y, mask, char_ids=None):
        em = self.emissions(x, mask, char_ids=char_ids)
        return self.crf.nll_loss(em, y, mask, reduction="mean")

    @torch.no_grad()
    def decode(self, x, mask, char_ids=None):
        em = self.emissions(x, mask, char_ids=char_ids)
        return self.crf.decode(em, mask)


# ------------------ Train loop ------------------
def evaluate_loader(model: BiLSTM_CRF, loader: DataLoader, id2tag: List[str],
                    device, use_char: bool) -> float:
    """计算 micro F1（不含 O，与 check.py 同算法）。"""
    from sklearn.metrics import f1_score
    model.eval()
    ys, ps = [], []
    for batch in loader:
        if use_char:
            x, y, chars, mask = batch
            x = x.to(device); chars = chars.to(device); mask = mask.to(device)
            preds = model.decode(x, mask, char_ids=chars)
        else:
            x, y, mask = batch
            x = x.to(device); mask = mask.to(device)
            preds = model.decode(x, mask)
        for i, seq in enumerate(preds):
            l = len(seq)
            ps.extend([id2tag[t] for t in seq])
            ys.extend([id2tag[t.item()] for t in y[i, :l]])
    labels = id2tag[1:]  # exclude 'O'
    return float(f1_score(ys, ps, labels=labels, average="micro", zero_division=0))


def train_one_lang(language: str, epochs: int, batch_size: int, lr: float,
                   emb_dim: int, hid_dim: int, dropout: float,
                   device: torch.device, use_char_cnn: bool = False,
                   char_emb_dim: int = 30, char_out_dim: int = 30,
                   max_word_len: int = 20, seed: int = 42,
                   glove_path: str = None, freeze_emb: bool = False) -> dict:
    torch.manual_seed(seed)
    np.random.seed(seed)

    labels = LANG2LABELS[language]
    tag2id = {t: i for i, t in enumerate(labels)}
    lower = (language == "English")

    train_sents = load_corpus(ner_path(language, "train.txt"))
    val_sents = load_corpus(ner_path(language, "validation.txt"))

    word2id = build_vocab(train_sents, min_freq=1, lower=lower,
                          specials=("<PAD>", "<UNK>"))
    char2id = None
    if use_char_cnn:
        # 字符词表：从原始 token (含原始大小写) 抽所有字符
        from collections import Counter
        cc = Counter()
        for sent in train_sents:
            for w, _ in sent:
                for ch in w[:max_word_len]:
                    cc[ch] += 1
        char2id = {"<PAD_CHAR>": 0, "<UNK_CHAR>": 1}
        for c, _ in cc.most_common():
            char2id[c] = len(char2id)
        print(f"[{language}] char vocab={len(char2id)}")

    print(f"[{language}] vocab={len(word2id)}, tags={len(labels)}, "
          f"train={len(train_sents)}, val={len(val_sents)}, char_cnn={use_char_cnn}")

    train_ds = NERDataset(train_sents, word2id, tag2id, lower=lower,
                          char2id=char2id, max_word_len=max_word_len)
    val_ds = NERDataset(val_sents, word2id, tag2id, lower=lower,
                        char2id=char2id, max_word_len=max_word_len)
    cf = collate_char if use_char_cnn else collate
    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                          collate_fn=cf, num_workers=0)
    val_dl = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                        collate_fn=cf, num_workers=0)

    # Task 2B: 构造 BMES/BIO 合法转移掩码注入 CRF 层
    legal_init, legal_trans, legal_end = build_legal_masks(labels, LANG2SCHEME[language])
    illegal_init = (~legal_init).astype("float32")
    illegal_trans = (~legal_trans).astype("float32")
    illegal_end = (~legal_end).astype("float32")
    print(f"[{language}] CRF constraint: blocked init={int(illegal_init.sum())}/{len(labels)} "
          f"trans={int(illegal_trans.sum())}/{len(labels)**2} end={int(illegal_end.sum())}/{len(labels)}")

    # Task 1A: 加载 GloVe（仅英文）
    pretrained_emb = None
    if glove_path is not None:
        glove_path = ensure_glove_file(glove_path)
        pretrained_emb, _ = load_glove_embeddings(glove_path, word2id,
                                                   emb_dim=emb_dim, lower=lower)

    model = BiLSTM_CRF(vocab_size=len(word2id), num_tags=len(labels),
                       emb_dim=emb_dim, hid_dim=hid_dim, dropout=dropout,
                       pad_idx=0,
                       illegal_trans_mask=illegal_trans,
                       illegal_start_mask=illegal_init,
                       illegal_end_mask=illegal_end,
                       n_chars=len(char2id) if use_char_cnn else 0,
                       char_emb_dim=char_emb_dim,
                       char_out_dim=char_out_dim,
                       pretrained_emb=pretrained_emb,
                       freeze_emb=freeze_emb).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    best_f1, best_state = -1.0, None
    t_start = time.time()
    for ep in range(1, epochs + 1):
        model.train()
        total = 0.0
        n_batches = 0
        for batch in train_dl:
            if use_char_cnn:
                x, y, chars, mask = batch
                x, y, chars, mask = x.to(device), y.to(device), chars.to(device), mask.to(device)
                loss = model.loss(x, y, mask, char_ids=chars)
            else:
                x, y, mask = batch
                x, y, mask = x.to(device), y.to(device), mask.to(device)
                loss = model.loss(x, y, mask)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            total += loss.item()
            n_batches += 1
        f1 = evaluate_loader(model, val_dl, labels, device, use_char=use_char_cnn)
        avg_loss = total / max(n_batches, 1)
        elapsed = time.time() - t_start
        print(f"  epoch {ep:02d} | loss {avg_loss:.4f} | "
              f"val micro-F1 {f1:.4f} | elapsed {elapsed:.0f}s")
        if f1 > best_f1:
            best_f1 = f1
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    print(f"[{language}] best val micro-F1 = {best_f1:.4f}")
    return {
        "language": language,
        "tag2id": tag2id,
        "id2tag": labels,
        "word2id": word2id,
        "char2id": char2id,
        "lower": lower,
        "config": {
            "emb_dim": emb_dim, "hid_dim": hid_dim, "dropout": dropout,
            "use_char_cnn": use_char_cnn,
            "char_emb_dim": char_emb_dim, "char_out_dim": char_out_dim,
            "max_word_len": max_word_len,
            "use_glove": glove_path is not None,
            "freeze_emb": freeze_emb,
        },
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

    glove_path = os.path.join(PJ_ROOT, "pretrained", "glove", "glove.6B.100d.txt")
    plan = {
        "Chinese": dict(epochs=30, batch_size=64, lr=1e-3, emb_dim=100, hid_dim=128,
                        dropout=0.5, use_char_cnn=False, glove_path=None),
        "English": dict(epochs=10, batch_size=64, lr=1e-3, emb_dim=100, hid_dim=200,
                        dropout=0.5, use_char_cnn=True,
                        glove_path=glove_path),  # Task 1A: GloVe + Task 1B: Char-CNN
    }
    langs = ["Chinese", "English"] if args.lang == "both" else [args.lang]
    for lang in langs:
        print("=" * 30, lang, "=" * 30)
        ckpt = train_one_lang(lang, device=device, **plan[lang])
        short = "chn" if lang == "Chinese" else "eng"
        out = os.path.join(HERE, f"part2_crf_{short}.pt")
        torch.save(ckpt, out)
        print(f"saved -> {out}")


if __name__ == "__main__":
    main()
