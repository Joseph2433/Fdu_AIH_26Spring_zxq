"""共享数据 IO 与标签集，供 p1/p2/p3/bonus 复用。

- 数据格式：每行 `token<空格>tag`，句子之间空行；validation 与 test 同格式。
- 评测脚本 NER/check.py 按行号对齐 gold / pred，预测文件必须保持相同行数与空行位置。
- 注意：check.py 在 Windows 上未指定 encoding，读 utf-8 数据会 UnicodeDecodeError；
  本仓库 evaluate() 使用 sklearn.classification_report 与 check.py 同算法做 micro-F1，
  评测结果与 check.py 在 Linux 上的输出一致。
"""

from __future__ import annotations

import os
from typing import List, Sequence, Tuple

# 与 NER/check.py 中的两个 sorted_labels 完全一致，第 0 个固定为 'O'。
LABELS_ENG: List[str] = [
    "O",
    "B-PER", "I-PER",
    "B-ORG", "I-ORG",
    "B-LOC", "I-LOC",
    "B-MISC", "I-MISC",
]

LABELS_CHN: List[str] = [
    "O",
    "B-NAME", "M-NAME", "E-NAME", "S-NAME",
    "B-CONT", "M-CONT", "E-CONT", "S-CONT",
    "B-EDU", "M-EDU", "E-EDU", "S-EDU",
    "B-TITLE", "M-TITLE", "E-TITLE", "S-TITLE",
    "B-ORG", "M-ORG", "E-ORG", "S-ORG",
    "B-RACE", "M-RACE", "E-RACE", "S-RACE",
    "B-PRO", "M-PRO", "E-PRO", "S-PRO",
    "B-LOC", "M-LOC", "E-LOC", "S-LOC",
]

LANG2LABELS = {"Chinese": LABELS_CHN, "English": LABELS_ENG}
LANG2SCHEME = {"Chinese": "BMES", "English": "BIO"}


# ---------------- 标签合法性约束（BMES / BIO） ----------------
def parse_tag(tag: str) -> Tuple[str, str]:
    """'B-NAME' -> ('B', 'NAME')；'O' -> ('O', '')"""
    if tag == "O":
        return ("O", "")
    if "-" in tag:
        p, t = tag.split("-", 1)
        return (p, t)
    return (tag, "")


def build_legal_masks(labels: List[str], scheme: str):
    """构造布尔掩码 (legal_init[T], legal_trans[T,T], legal_end[T])。

    BMES（中文 Resume NER）：
        起始: O / B-X / S-X 合法（M-X / E-X 非法）
        结束: O / E-X / S-X 合法（B-X / M-X 是不完整实体，非法）
        转移:
            O   → {O, B-Y, S-Y}
            B-X → {M-X, E-X}      仅同类型
            M-X → {M-X, E-X}      仅同类型
            E-X → {O, B-Y, S-Y}
            S-X → {O, B-Y, S-Y}
    BIO（英文 CoNLL-2003）：
        起始: O / B-X 合法（I-X 非法）
        结束: 任意均合法
        转移: I-X 仅可由 B-X 或 I-X（同类型 X）后接；其他位置任意
    """
    import numpy as np
    T = len(labels)
    legal_init = np.zeros(T, dtype=bool)
    legal_trans = np.zeros((T, T), dtype=bool)
    legal_end = np.zeros(T, dtype=bool)
    parsed = [parse_tag(t) for t in labels]

    if scheme == "BMES":
        for i, (p, _) in enumerate(parsed):
            if p in ("O", "B", "S"):
                legal_init[i] = True
            if p in ("O", "E", "S"):
                legal_end[i] = True
        for i, (pp, pt) in enumerate(parsed):
            for j, (cp, ct) in enumerate(parsed):
                if pp in ("O", "E", "S"):
                    legal_trans[i, j] = cp in ("O", "B", "S")
                elif pp in ("B", "M"):
                    legal_trans[i, j] = (cp in ("M", "E")) and (ct == pt)
    elif scheme == "BIO":
        for i, (p, _) in enumerate(parsed):
            if p in ("O", "B"):
                legal_init[i] = True
            legal_end[i] = True  # BIO 任何位置均可结束
        for i, (pp, pt) in enumerate(parsed):
            for j, (cp, ct) in enumerate(parsed):
                if cp == "I":
                    legal_trans[i, j] = (pp in ("B", "I")) and (pt == ct)
                else:
                    legal_trans[i, j] = True
    else:
        raise ValueError(f"unknown scheme: {scheme}")
    return legal_init, legal_trans, legal_end


def project_root() -> str:
    """返回 pj2 根目录的绝对路径（即 data_utils.py 所在目录）。"""
    return os.path.dirname(os.path.abspath(__file__))


def ner_path(language: str, fname: str) -> str:
    """拼接 NER/<language>/<fname> 的绝对路径。"""
    return os.path.join(project_root(), "NER", language, fname)


def load_corpus(path: str) -> List[List[Tuple[str, str]]]:
    """读取 NER 数据文件，按空行切句返回 [[(token, tag), ...], ...]。"""
    sentences: List[List[Tuple[str, str]]] = []
    cur: List[Tuple[str, str]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if line.strip() == "":
                if cur:
                    sentences.append(cur)
                    cur = []
                continue
            parts = line.split(" ")
            if len(parts) < 2:
                continue
            token = " ".join(parts[:-1])
            tag = parts[-1]
            cur.append((token, tag))
    if cur:
        sentences.append(cur)
    return sentences


def load_tokens_only(path: str) -> List[List[str]]:
    """与 load_corpus 同样切句，但只取每行首列 token，用于 test.txt 兼容。"""
    sentences: List[List[str]] = []
    cur: List[str] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if line.strip() == "":
                if cur:
                    sentences.append(cur)
                    cur = []
                continue
            parts = line.split(" ")
            cur.append(parts[0])
    if cur:
        sentences.append(cur)
    return sentences


def _looks_like_tag(value: str) -> bool:
    return value == "O" or value.startswith(("B-", "I-", "M-", "E-", "S-"))


def write_predictions(orig_path: str, pred_sentences: Sequence[Sequence[str]], out_path: str) -> None:
    """按原文件结构（含空行）输出预测，保证行数与原文件完全对齐。

    pred_sentences[i] 长度需与原文件第 i 段非空行 token 数一致。
    """
    os.makedirs(os.path.dirname(os.path.abspath(out_path)) or ".", exist_ok=True)
    sent_idx = 0
    tok_idx = 0
    with open(orig_path, "r", encoding="utf-8") as fin, open(out_path, "w", encoding="utf-8") as fout:
        for line in fin:
            stripped = line.rstrip("\n")
            if stripped.strip() == "":
                fout.write("\n")
                if tok_idx > 0:
                    sent_idx += 1
                    tok_idx = 0
                continue
            parts = stripped.split(" ")
            token = " ".join(parts[:-1]) if len(parts) >= 2 and _looks_like_tag(parts[-1]) else stripped
            tag = pred_sentences[sent_idx][tok_idx]
            fout.write(f"{token} {tag}\n")
            tok_idx += 1
    return


def build_vocab(sentences, min_freq: int = 1, lower: bool = False,
                specials: Tuple[str, ...] = ("<PAD>", "<UNK>")) -> dict:
    """从 token 列表构建 vocab；返回 {token: id}。sentences 既可是 (tok,tag) 也可纯 token。"""
    from collections import Counter
    cnt: Counter = Counter()
    for sent in sentences:
        for item in sent:
            tok = item[0] if isinstance(item, tuple) else item
            cnt[tok.lower() if lower else tok] += 1
    vocab = {sp: i for i, sp in enumerate(specials)}
    for tok, c in cnt.most_common():
        if c < min_freq:
            break
        if tok in vocab:
            continue
        vocab[tok] = len(vocab)
    return vocab


def evaluate(language: str, gold_path: str, pred_path: str) -> float:
    """与 NER/check.py 同算法（sklearn classification_report，labels=labels[1:]，digits=4）。

    打印完整 report 并返回 micro avg F1。
    """
    from sklearn import metrics
    import warnings
    warnings.filterwarnings("ignore")
    labels = LANG2LABELS[language]
    y_true: List[str] = []
    y_pred: List[str] = []
    with open(gold_path, "r", encoding="utf-8") as g, open(pred_path, "r", encoding="utf-8") as p:
        g_lines = g.readlines()
        p_lines = p.readlines()
    n = min(len(g_lines), len(p_lines))
    for i in range(n):
        if g_lines[i].strip() == "":
            continue
        gp = g_lines[i].rstrip("\n").split(" ")
        pp = p_lines[i].rstrip("\n").split(" ")
        y_true.append(gp[-1])
        y_pred.append(pp[-1])
    report = metrics.classification_report(
        y_true=y_true, y_pred=y_pred, labels=labels[1:], digits=4
    )
    print(report)
    micro_f1 = metrics.f1_score(
        y_true=y_true, y_pred=y_pred, labels=labels[1:], average="micro"
    )
    return float(micro_f1)


# --------- 简单 sanity check（可单独运行） ---------
if __name__ == "__main__":
    for lang in ("Chinese", "English"):
        train = load_corpus(ner_path(lang, "train.txt"))
        val = load_corpus(ner_path(lang, "validation.txt"))
        n_tok = sum(len(s) for s in train)
        n_tag_set = set(t for s in train for _, t in s)
        print(f"[{lang}] train sents={len(train)}, tokens={n_tok}, "
              f"val sents={len(val)}, observed_tags={len(n_tag_set)}, "
              f"label_set={len(LANG2LABELS[lang])}")
        assert n_tag_set.issubset(set(LANG2LABELS[lang])), (
            f"unexpected tags: {n_tag_set - set(LANG2LABELS[lang])}")
    print("data_utils self-check OK")
